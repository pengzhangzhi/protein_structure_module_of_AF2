import torch
import torch.nn as nn


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

        self.pre_module = PreModule()  # Algorithm 20 Structure module line 1 - 3
        self.IPA = IPA()  # Algorithm 22 Invariant point attention (IPA)
        self.ipa_dropout = nn.Dropout()  # Algorithm 20 Structure module line 7
        self.ipa_layer_norm = nn.LayerNorm()  # Algorithm 20 Structure module line 7
        self.transition = Transition()  # Algorithm 20 Structure module line 8 - 9
        self.bb_update = BackboneUpdate()  # Algorithm 20 Structure module line 10
        self.angle_resnet = AngleResNet()  # Algorithm 20 Structure module line 11

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
        s_init = s
        s = self.pre_module(s, z, aatype)  # Algorithm 20 Structure module line 1-3

        rigids = Rigid.identity()

        outputs = []
        for iter in range(self.self.no_blocks):
            s = self.IPA()
            s = self.ipa_layer_norm(
                self.ipa_dropout(s)
            )  # Algorithm 20 Structure module line 7
            s = self.transition(s)
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # convert quaternion-format rigid to rotation matrix one.
            backb_to_global = Rigid()
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
            all_frames_to_global = self.torsion_angle_to_global(
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
            
            if iter < self.no_blocks - 1 :
                rigids = rigids.stop_rot_gradient()
                
        outputs = dict_multimap(torch.stack, outputs) 
        outputs["single"] = s

        return outputs
