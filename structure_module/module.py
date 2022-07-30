import torch
import torch.nn as nn
from .layers import (
    InvariantPointAttention,
    PreModule,
    Transition,
    BackboneUpdate,
    AngleResNet,
)
from .rigid import Rigid, Rotation
from .coordinate import (
    torsion_angles_to_frames,
    frames_and_literature_positions_to_atom14_pos,
)
from .utils import dict_multimap, exists
from .residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from .primitive import LayerNorm


class StructureModule(nn.Module):
    def __init__(
        self,
        c_s,  # channel dim of single_rep
        c_z,  # channel dim of pair_rep
        c_ipa,  # hidden channel dim of ipa
        c_resnet,  # hidden channel dim of resnet
        no_head_ipa,  # num of head in ipa
        no_qk_points,  # num of qk points in ipa
        no_v_points,  # num of v points in ipa
        dropout_rate,
        no_blocks,  # num of structure module blocks
        no_transition_layers,  # num of transition layers in resnet
        no_resnet_blocks,  # num of blocks in resnet
        no_angles,  # num of angles to generate in resnet
        trans_scale_factor,  # scale of c_s hidden dim
        epsilon,  # small number used in angle resnet normalization
        inf,  # large num for mask
        *args,
        **kwargs,
    ) -> None:
        """
        The structure module, AF2 supplyment Algorithm 20 Structure module.

        Notes:
            My implementation is different from AF2's:
                I remove the loss terms from StructureModule for clearity.
                An independent module is for loss.

        """
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_head_ipa = no_head_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor

        ############### Modules ###############

        self.pre_module = PreModule(
            c_s,
            c_z,
        )
        self.IPA = InvariantPointAttention(
            c_s,
            c_z,
            c_ipa,
            no_head_ipa,
            no_qk_points,
            no_v_points,
            inf=inf,
            eps=epsilon,
        )  # Algorithm 22 Invariant point attention (IPA)
        self.ipa_dropout = nn.Dropout(c_s)  # Algorithm 20 Structure module line 7
        self.ipa_layer_norm = LayerNorm(c_s)  # Algorithm 20 Structure module line 7
        self.transition = Transition(
            c_s, no_transition_layers, dropout_rate
        )  # Algorithm 20 Structure module line 8 - 9
        self.bb_update = BackboneUpdate(c_s)  # Algorithm 20 Structure module line 10
        self.angle_resnet = AngleResNet(
            c_s,
            c_resnet,
            no_resnet_blocks,
            no_angles,
            eps=epsilon,
        )  # Algorithm 20 Structure module line 11

    def forward(
        self,
        s,
        z,
        aatype,
        mask=None,
    ):
        """

        Args:
            s (torch.Tensor): single_rep, [*, N_res, C_s]
            z (torch.Tensor): pair_rep, [*, N_res, C_z]
            aatype (torch.Tensor): amino acid type of residues [*, N_res]
            mask (torch.Tensor, optional): the sequence mask. Defaults to None. [*, N_res]

        Returns (dict):
            a dict of outputs including:
                s_out (torch.Tensor): [*, N_res, C_s]
                coord,
                torsion_angle,
                frames,

        """
        s_init, s, z = self.pre_module(s, z)  # Algorithm 20 Structure module line 1-3

        rigids = Rigid.identity()

        outputs = []
        for iter in range(self.self.no_blocks):
            s = self.IPA(
                s,
                z,
                rigids,
                mask,
            )
            s = self.ipa_layer_norm(
                self.ipa_dropout(s)
            )  # Algorithm 20 Structure module line 7
            s = self.transition(s)
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # convert quaternion-format rigid to rotation matrix one.
            # This step is unnecessary. only to hew as close as possible to AF2.
            backb_to_global = Rigid(
                Rotation(rigids.get_rots().get_rot_mats()),
                trans=rigids.get_trans()
            )
            backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)

            # we use two num to represent the 7 angles, i.e., spi, phi, omega, x1, x2, x3, x4
            # 'angles' use the (x,y) in the unit circle denote the angle.
            # (*, N, 7, 2)
            unnormalized_angles, angles = self.angle_resnet(s, s_init)

            # convert the angles to global frame.
            # Algorithm 24 Compute all atom coordinates 1 - 10
            # all_frames_to_global contains 8 frames,
            # the backb_to_global and the 7 angles frames.
            # Atoms in these frames are transformed to global coordinates with them.
            all_frames_to_global = self.torsion_angles_to_frames(
                backb_to_global, angles, aatype
            )

            # compute x,y,z coordinates of 14 atoms in a residue.

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )
            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            preds = {
                "scaled_rigids": scaled_rigids,
                "side_chain_frames": all_frames_to_global,
                "unormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
            }
            outputs.append(preds)

            if iter < self.no_blocks - 1:
                rigids = rigids.stop_rot_gradient()

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def torsion_angles_to_frames(
        self,
        backb_to_global,
        angles,
        aatypes,
    ):
        self._init_residue_constants(angles.dtype, angles.device)
        return torsion_angles_to_frames(
            backb_to_global, angles, aatypes, self.default_frames
        )

    def frames_and_literature_positions_to_atom14_pos(
        self,
        frames,
        aatypes,
    ):
        self._init_residue_constants(aatypes.dtype, aatypes.device)
        return frames_and_literature_positions_to_atom14_pos(
            frames,
            aatypes,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

    def _init_residue_constants(self, float_dtype, device):
        """
        lazy init residue constants for atom coordinates prediction.

        21 types of amino acids (20 + 1 for unknown).
        14 atoms in a residue, differs by amino acid type.
        alphafol2 defines 8 frame groups.
            - The backbone frame,
            - 3 backbone torsion angle frame (phi, psi, omega),
            - 4 side chain torsion angles frame(x1, x2, x3, x4).)

        Specifically, the following constants would be intialized:
            - default_frame_transformations [21, 8, 4, 4],
            - restype_atom14_to_rigid_group, [21, 14],
                                            a table tells an atom belongs to whcih rigid groups.
                                            See Alphafold2 Supplement Table 2.
            - atom_mask, [21, 14], a 0/1 table where 0 means the atom does not exist in the amino acid type,
                                    and thus should be masked.
            - atom_local_positions, the local positions of the 14 atoms in the residue,
                                    the local coordinates are defined in the aforementioned 8 frames.

        """
        if not exists(self.default_frames):
            self.default_frames = torch.tensor(
                restype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if not exists(self.group_idx):
            self.group_idx = torch.tensor(
                restype_atom14_to_rigid_group,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if not exists(self.atom_mask):
            self.atom_mask = torch.tensor(
                restype_atom14_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )

        if not exists(self.lit_positions):
            self.lit_positions = torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
            restype_rigid_group_default_frame,
