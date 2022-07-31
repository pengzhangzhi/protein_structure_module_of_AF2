import torch
from .rigid import Rigid, Rotation


def torsion_angles_to_frames(
    backb_to_global: Rigid,
    angles: torch.Tensor,
    aatypes: torch.Tensor,
    default_frames: torch.Tensor,
):
    """
    convert the predicted torsion angles to frame transformation.
    see Algorithm 24 Compute all atom coordinates 1 - 10.

    Args:
        backb_to_global: the backbone to global transformation.
        angles [*, N, 7, 2]: the predicted torsion angles.
                            N: number of residues,
                            7: number of predicted torsion angles,
                            2: (x,y) representation of an angle.
        aatypes [*, N]: the amino acid types.
        default_frames [21, 8, 4, 4]: the default frame transformations to backbone.

    Returns:
        the 8 frame transformation to global.
    """
    # [*, N, 8]
    default_angle_frames_to_bb = Rigid.from_tensor_4x4(default_frames[aatypes])[..., 1:]
    default_omega_to_bb, default_phi_frame_to_bb, default_psi_frame_to_bb = (
        default_angle_frames_to_bb[..., 0],
        default_angle_frames_to_bb[..., 1],
        default_angle_frames_to_bb[..., 2],
    )
    default_x1_to_bb = default_angle_frames_to_bb[..., 3]
    default_x2_to_x1 = default_angle_frames_to_bb[..., 4]
    default_x3_to_x2 = default_angle_frames_to_bb[..., 5]
    default_x4_to_x3 = default_angle_frames_to_bb[..., 6]

    # [*, N, 7]
    pred_angle_frames: Rigid = makeRotX(angles)
    omega_frame_to_bb, phi_frame_to_bb, psi_frame_to_bb = (
        pred_angle_frames[..., 0],
        pred_angle_frames[..., 1],
        pred_angle_frames[..., 2],
    )
    x1_frame_to_bb, x2_frame_to_x1, x3_frame_to_x2, x4_frame_to_x3 = (
        pred_angle_frames[..., 3],
        pred_angle_frames[..., 4],
        pred_angle_frames[..., 5],
        pred_angle_frames[..., 6],
    )
    omega_to_global = backb_to_global.compose(
        default_omega_to_bb.compose(omega_frame_to_bb)
    )
    phi_to_global = backb_to_global.compose(
        default_phi_frame_to_bb.compose(phi_frame_to_bb)
    )
    psi_to_global = backb_to_global.compose(
        default_psi_frame_to_bb.compose(psi_frame_to_bb)
    )
    x1_to_global = backb_to_global.compose(default_x1_to_bb.compose(x1_frame_to_bb))

    x2_to_global = x1_to_global.compose(default_x2_to_x1.compose(x2_frame_to_x1))
    x3_to_global = x2_to_global.compose(default_x3_to_x2.compose(x3_frame_to_x2))
    x4_to_global = x3_to_global.compose(default_x4_to_x3.compose(x4_frame_to_x3))

    
    return Rigid.cat(
        [
            backb_to_global.unsqueeze(-1),
            omega_to_global.unsqueeze(-1),
            phi_to_global.unsqueeze(-1),
            psi_to_global.unsqueeze(-1),
            x1_to_global.unsqueeze(-1),
            x2_to_global.unsqueeze(-1),
            x3_to_global.unsqueeze(-1),
            x4_to_global.unsqueeze(-1),
        ],
        dim=-1,
    )


def frames_and_literature_positions_to_atom14_pos(
    frames: Rigid,
    aatypes: torch.Tensor,
    group_idx: torch.Tensor,
    atom_mask: torch.Tensor,
    lit_positions: torch.Tensor,
):
    """
    map atom literature postions to global coordinates with the frame transformation.
    see Algorithm 24 Compute all atom coordinates 11.

    Args:
        frames [*, N, 8]: the 8 frame transformations to global.
        aatypes [*, N]: the amino acid types.
        default_frames [21, 8]: the default frame transformations to backbone.
        group_idx [21, 14]: the group index of atoms.
        atom_mask [21, 14]: the atom mask of the amino acid.
        lit_positions [21, 14, 3]: the literature positions of atoms.

    Returns:
        the predicted atom positions. [*, N, 14, 3]

    Note: the practical implementation is more complicated than the supplement for Two preprocessing steps.
            1. group mask.  Atoms belong to the pre-defined group should be predicted,
                            while others should be ignored.
            2. atom mask.   Different amino acid consists of different atoms.

    """
    # [*, N, 14]
    group_mask = group_idx[aatypes]
    # [*, N, 14]
    atom_mask = atom_mask[aatypes]
    # [*, N, 14, 3]
    lit_positions = lit_positions[aatypes]

    # derive the transformation of the 14 atom
    num_frames = frames.shape[-1]
    group_mask = torch.nn.functional.one_hot(group_mask.long(), num_classes=num_frames)
    # [*, N, 14, 8]
    atom_to_global = frames[..., None, :] * group_mask
    # [*, N, 14]
    atom_to_global = atom_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))
    # transform local coordinates to global coordinates
    global_postions = atom_to_global.apply(lit_positions)

    # apply atom mask
    pred_postions = atom_mask.unsqueeze(-1) * global_postions

    return pred_postions


def makeRotX(alpha: torch.Tensor) -> Rigid:
    """
    Make a transformation that rotates around the x-axis by alpha.
    Args:
        alpha [*,2]: rotation angle represented by x,y.  ||alpha|| = 1.
    Returns (Rigid):
        [*,]. A transformation that rotates around the x-axis by alpha.
    """
    assert alpha.shape[-1] == 2, "alpha must be a 2D tensor"
    assert torch.allclose((alpha ** (2.0)).sum(-1), torch.tensor(1.0)), "alpha must be a unit vector"

    a1 = alpha[..., 0]
    a2 = alpha[..., 1]
    batch_dim = alpha.shape[:-1]
    rot = torch.zeros((*batch_dim, 3, 3), dtype=alpha.dtype, device=a1.device)
    rot[..., 0, 0] = 1
    rot[..., 1, 1], rot[..., 2, 2] = a1, a1
    rot[..., 1, 2], rot[..., 2, 1] = -a2, a2
    trans = torch.zeros((*batch_dim, 3))

    # the output interface.
    return Rigid(rots=Rotation(rot_mats=rot), trans=trans)
