from __future__ import annotations
from typing import Tuple, Any, Sequence, Callable, Optional
import numpy as np
import torch
from utils import exists
from functools import reduce


def rot_matmul(
    a: torch.Tensor, 
    b: torch.Tensor
) -> torch.Tensor:
    """
        Performs matrix multiplication of two rotation matrix tensors. Written
        out by hand to avoid AMP downcasting.

        Args:
            a: [*, 3, 3] left multiplicand
            b: [*, 3, 3] right multiplicand
        Returns:
            The product ab
    """
    row_1 = torch.stack(
        [
            a[..., 0, 0] * b[..., 0, 0]
            + a[..., 0, 1] * b[..., 1, 0]
            + a[..., 0, 2] * b[..., 2, 0],
            a[..., 0, 0] * b[..., 0, 1]
            + a[..., 0, 1] * b[..., 1, 1]
            + a[..., 0, 2] * b[..., 2, 1],
            a[..., 0, 0] * b[..., 0, 2]
            + a[..., 0, 1] * b[..., 1, 2]
            + a[..., 0, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )
    row_2 = torch.stack(
        [
            a[..., 1, 0] * b[..., 0, 0]
            + a[..., 1, 1] * b[..., 1, 0]
            + a[..., 1, 2] * b[..., 2, 0],
            a[..., 1, 0] * b[..., 0, 1]
            + a[..., 1, 1] * b[..., 1, 1]
            + a[..., 1, 2] * b[..., 2, 1],
            a[..., 1, 0] * b[..., 0, 2]
            + a[..., 1, 1] * b[..., 1, 2]
            + a[..., 1, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )
    row_3 = torch.stack(
        [
            a[..., 2, 0] * b[..., 0, 0]
            + a[..., 2, 1] * b[..., 1, 0]
            + a[..., 2, 2] * b[..., 2, 0],
            a[..., 2, 0] * b[..., 0, 1]
            + a[..., 2, 1] * b[..., 1, 1]
            + a[..., 2, 2] * b[..., 2, 1],
            a[..., 2, 0] * b[..., 0, 2]
            + a[..., 2, 1] * b[..., 1, 2]
            + a[..., 2, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )

    return torch.stack([row_1, row_2, row_3], dim=-2)


def rot_vec_mul(
    r: torch.Tensor, 
    t: torch.Tensor
) -> torch.Tensor:
    """
        Applies a rotation to a vector. Written out by hand to avoid transfer
        to avoid AMP downcasting.

        Args:
            r: [*, 3, 3] rotation matrices
            t: [*, 3] coordinate tensors
        Returns:
            [*, 3] rotated coordinates
    """
    x = t[..., 0]
    y = t[..., 1]
    z = t[..., 2]
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )

    
def identity_rot_mats(
    batch_dims: Tuple[int], 
    dtype: Optional[torch.dtype] = None, 
    device: Optional[torch.device] = None, 
    requires_grad: bool = True,
) -> torch.Tensor:
    rots = torch.eye(
        3, dtype=dtype, device=device, requires_grad=requires_grad
    )
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)

    return rots


def identity_trans(
    batch_dims: Tuple[int], 
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None, 
    requires_grad: bool = True,
) -> torch.Tensor:
    trans = torch.zeros(
        (*batch_dims, 3), 
        dtype=dtype, 
        device=device, 
        requires_grad=requires_grad
    )
    return trans


def identity_quats(
    batch_dims: Tuple[int], 
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None, 
    requires_grad: bool = True,
) -> torch.Tensor:
    quat = torch.zeros(
        (*batch_dims, 4), 
        dtype=dtype, 
        device=device, 
        requires_grad=requires_grad
    )

    with torch.no_grad():
        quat[..., 0] = 1

    return quat


_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}


def _to_mat(pairs):
    mat = np.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat


_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
        Converts a quaternion to a rotation matrix.

        Args:
            quat: [*, 4] quaternions
        Returns:
            [*, 3, 3] rotation matrices
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = quat.new_tensor(_QTR_MAT, requires_grad=False)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(
    rot: torch.Tensor,
):
    if(rot.shape[-2:] != (3, 3)):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot 

    k = [
        [ xx + yy + zz,      zy - yz,      xz - zx,      yx - xy,],
        [      zy - yz, xx - yy - zz,      xy + yx,      xz + zx,],
        [      xz - zx,      xy + yx, yy - xx - zz,      yz + zy,],
        [      yx - xy,      xz + zx,      yz + zy, zz - xx - yy,]
    ]

    k = (1./3.) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)

    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]


_QUAT_MULTIPLY = np.zeros((4, 4, 4))
_QUAT_MULTIPLY[:, :, 0] = [[ 1, 0, 0, 0],
                          [ 0,-1, 0, 0],
                          [ 0, 0,-1, 0],
                          [ 0, 0, 0,-1]]

_QUAT_MULTIPLY[:, :, 1] = [[ 0, 1, 0, 0],
                          [ 1, 0, 0, 0],
                          [ 0, 0, 0, 1],
                          [ 0, 0,-1, 0]]

_QUAT_MULTIPLY[:, :, 2] = [[ 0, 0, 1, 0],
                          [ 0, 0, 0,-1],
                          [ 1, 0, 0, 0],
                          [ 0, 1, 0, 0]]

_QUAT_MULTIPLY[:, :, 3] = [[ 0, 0, 0, 1],
                          [ 0, 0, 1, 0],
                          [ 0,-1, 0, 0],
                          [ 1, 0, 0, 0]]

_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]


def quat_multiply(quat1, quat2):
    """Multiply a quaternion by another quaternion."""
    mat = quat1.new_tensor(_QUAT_MULTIPLY)
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    return torch.sum(
        reshaped_mat *
        quat1[..., :, None, None] *
        quat2[..., None, :, None],
        dim=(-3, -2)
      )


def quat_multiply_by_vec(quat, vec):
    """Multiply a quaternion by a pure-vector quaternion."""
    mat = quat.new_tensor(_QUAT_MULTIPLY_BY_VEC)
    reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    return torch.sum(
        reshaped_mat *
        quat[..., :, None, None] *
        vec[..., None, :, None],
        dim=(-3, -2)
    )


def invert_rot_mat(rot_mat: torch.Tensor):
    return rot_mat.transpose(-1, -2)


def invert_quat(quat: torch.Tensor):
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    inv = quat_prime / torch.sum(quat ** 2, dim=-1, keepdim=True)
    return inv


class Rotation(object):
    """An immutable ADT represent the rotation in 3D space.
        The ADT can be instantiated with either a rotation matrix or a quaternion.
    Abstract Functions:
        AF(rotation_matrix) = a rotation obj.
        AF(quaternion) = a rotation obj.

    Reprsentation Invariant:
        I(rotation_matrix) =  rotation_matrix([*, 3, 3]) is a 3x3 matrix.
        I(quaternion) = quaternion ([*, 4]) is a 4-vector.

    Safety from Representation Exposure:
        Defensive copy is used to prevent modification of the rotation matrix or quaternion.

    """

    def __init__(
        self,
        rot_mats: torch.Tensor = None,
        quaternion: torch.Tensor = None,
        normalize_quaternion: bool = True,
    ) -> None:
        """instantiate a rotation object with either a rotation matrix or a quaternion.
        Args:
            rot_mats [*, 3, 3]: a 3x3 matrix.
            quaternion[*, 4]: a 4-vector.
            normalize_quaternion: if True and the quaternion is specified, normalize the quaternion.
        """
        assert (exists(rot_mats) and exists(quaternion)) or (
            not exists(rot_mats) and not exists(quaternion)
        ), "Exactly one of rot_mats and quaternion must be specified"

        self._rot_mats = rot_mats
        self._quaternion = quaternion
        self.__check_repr_invariant()
        self._to_float_32()
        if normalize_quaternion and exists(self._quaternion):
            self._quaternion = self._quaternion / torch.linalg.norm(
                self._quaternion, dim=-1, keepdim=True
            )

    def __check_repr_invariant(self) -> None:
        assert (exists(self._rot_mats) and self._rot_mats.shape[-2:] == (3, 3)) or (
            not exists(self._rot_mats)
            and exists(self._quaternion)
            and self._quaternion.shape[-1] == 4
        ), "Invalid representation of rotation matrix or quaternion"

    def _to_float_32(self):
        """Force full-precision.
        Convert the rotation representation to float32.
        """

        if exists(self._rot_mats):
            self._rot_mats = self._rot_mats.to(dtype=torch.float32)
        else:
            self._quaternion = self._quaternion.to(dtype=torch.float32)

    @staticmethod
    def identity(
        shape, dtype=None, device=None, requires_grad=True, fmt="quat"
    ) -> Rotation:
        """
        return an identity rotation.
        """
        if fmt == "quat":
            quaternion = torch.tensor(
                [1, 0, 0, 0], dtype=dtype, device=device, requires_grad=requires_grad
            )
            quaternion = quaternion.reshape(*((1,) * len(shape) + (4,)))
            quaternion_batch = quaternion.repeat(*shape, -1)
            rotation = Rotation(quaternion=quaternion_batch, normalize_quaternion=False)
        elif fmt == "rot_mat":
            identity_rotation_matrix = torch.eye(
                3, dtype=dtype, device=device, requires_grad=requires_grad
            )
            identity_rotation_matrix = identity_rotation_matrix.reshape(
                *((1,) * len(shape)), 3, 3)
            identity_rotation_matrix = identity_rotation_matrix.repeat(*shape, -1, -1)
            rotation = Rotation(rot_mats=identity_rotation_matrix)
        else:
            raise ValueError(f"Unknown rotation format: {fmt}")
        return rotation

    def __getitem___(self, index) -> Rotation:
        """
        return a slice of the rotation.
        Args:
            index: the index of the slice. e.g., 1, (1,2), or slice(1,2)
        Returns:
            a rotation object.
        """
        if not isinstance(index,tuple):
            index = (index,)
        if exists(self._quaternion):
            return Rotation(quaternion=self._quaternion[index],normalize_quaternion=False)
        else:
            return Rotation(rot_mats=self._rot_mats[index])
    
    
    

    def __mul__(self, other: torch.Tensor) -> Rotation:
        """
        pointwise multiplication of the rotation with the `other` tensor.
        Args:
            other: a tensor [*,], with similar shape as the rotation.
        Returns:
            a rotation object.
        """
        if exists(self._quaternion):
            quaternion = self._quaternion * other[..., None]
            return Rotation(quaternion=quaternion)
        else:
            matrix = self._rot_mats * other[..., None, None]
            return Rotation(rot_mats=matrix)
        

    def __rmul__(self, other: torch.Tensor) -> Rotation:
        """
        pointwise multiplication of the rotation with the `other` tensor.
        Args:
            other: a tensor [*,].
        Returns:
            a rotation object.
        """
        return self.__mul__(other)
    

    @property
    def shape(self) -> torch.Size:
        """return the shape of the rotation obj.
        The shape is defined as the batch dim.
        For the representation of rotation matrix [*,3,3]
        or quanternion [*,4], the shape returns [*,].
        """
        if exists(self._quaternion):
            return self._quaternion.shape[:-1]
        else:
            return self._rot_mats.shape[:-2]
        

    @property
    def dtype(self) -> torch.dtype:
        """return the dtype of the rotation obj.
        The dtype is defined as the dtype of the rotation matrix or quaternion.
        """
        if exists(self._quaternion):
            return self._quaternion.dtype
        else:
            return self._rot_mats.dtype
        

    @property
    def requires_grad(self) -> bool:
        """return whether the rotation obj requires grad."""
        if exists(self._quaternion):
            return self._quaternion.requires_grad
        else: 
            return self._rot_mats.requires_grad

    @property
    def device(self) -> torch.device:
        """return the device of the rotation obj.
        The device is defined as the device of the rotation matrix or quaternion.
        """
        if exists(self._quaternion):
            return self._quaternion.device
        else:
            return self._rot_mats.device

    def get_rot_mats(self) -> torch.Tensor:
        """return the rotation matrix."""
        if exists(self._rot_mats):
            return self._rot_mats.clone().detach()
        else:
            return self._quaternion_to_mat(self._quaternion)

    def get_quaternion(self) -> torch.Tensor:
        """return the quaternion."""
        if exists(self._quaternion):
            return self._quaternion.clone().detach()
        else:
            return self._mat_to_quaternion(self._rot_mats)
    
    def _mat_to_quaternion(self, mat: torch.Tensor) -> torch.Tensor:
        """ convert rotation matrix to quaternion. """
        assert mat.shape[-2:] == (3, 3), "Invalid rotation matrix"
        quaternion = torch.zeros(mat.shape[:-2] + (4,), dtype=mat.dtype, device=mat.device)
        quaternion[..., 0] = torch.sqrt(1 + mat[..., 0, 0] - mat[..., 1, 1] - mat[..., 2, 2]) / 2
        quaternion[..., 1] = (mat[..., 0, 1] + mat[..., 1, 0]) / (4 * quaternion[..., 0])
        quaternion[..., 2] = (mat[..., 0, 2] + mat[..., 2, 0]) / (4 * quaternion[..., 0])
        quaternion[..., 3] = (mat[..., 2, 1] - mat[..., 1, 2]) / (4 * quaternion[..., 0])
        return quaternion
    def _quaternion_to_mat(self, quaternion: torch.Tensor) -> torch.Tensor:
        """ convert quaternion to rotation matrix. """
        assert quaternion.shape[-1] == 4, "Invalid quaternion"
        rot_mat = torch.zeros(quaternion.shape[:-1] + (3, 3), dtype=quaternion.dtype, device=quaternion.device)
        rot_mat[..., 0, 0] = 1 - 2 * (quaternion[..., 1] ** 2 + quaternion[..., 2] ** 2)
        rot_mat[..., 0, 1] = 2 * (quaternion[..., 0] * quaternion[..., 1] - quaternion[..., 3] * quaternion[..., 2])
        rot_mat[..., 0, 2] = 2 * (quaternion[..., 0] * quaternion[..., 2] + quaternion[..., 3] * quaternion[..., 1])
        rot_mat[..., 1, 0] = 2 * (quaternion[..., 0] * quaternion[..., 1] + quaternion[..., 3] * quaternion[..., 2])
        rot_mat[..., 1, 1] = 1 - 2 * (quaternion[..., 0] ** 2 + quaternion[..., 2] ** 2)
        rot_mat[..., 1, 2] = 2 * (quaternion[..., 1] * quaternion[..., 2] - quaternion[..., 3] * quaternion[..., 0])
        rot_mat[..., 2, 0] = 2 * (quaternion[..., 0] * quaternion[..., 2] - quaternion[..., 3] * quaternion[..., 1])
        rot_mat[..., 2, 1] = 2 * (quaternion[..., 1] * quaternion[..., 2] + quaternion[..., 3] * quaternion[..., 0])
        rot_mat[..., 2, 2] = 1 - 2 * (quaternion[..., 0] ** 2 + quaternion[..., 1] ** 2)
        return rot_mat
    

    def get_cur_rot(self) -> torch.Tensor:
        """return the current rotation representation.
        #Warning: this method suffer from resentation exposure.
        """
        return self._rot_mats

    def compose_q_update_vec(
        self,
        q_update_vec: torch.Tensor,
        normalize_quats: bool = True,
    ) -> Rotation:
        """compose the quaternion update vector with the current quaternion.
        Args:
            q_update_vec [*, 3]: a quaternion update vector whose last three columns are x,y,z,
                                such that (1, x, y, z) is the disired quaternion update.
            normalize_quats: if True, normalize the quaternion.
        Returns:
            a rotation object.
        """
        quats = self.get_quats()
        new_quats = quats + quat_multiply_by_vec(quats, q_update_vec)
        return Rotation(
            rot_mats=None, 
            quats=new_quats, 
            normalize_quats=normalize_quats,
        )

    def compose_r(self, r: Rotation) -> Rotation:
        """compose the rotation `r`` with the current rotation in the rotation matrix format.
        Args:
            rot_mats (Rotation): a rotation matrix.
        Returns:
            a rotation object.
        """
        ...

    def compose_q(self, q: Rotation, normalize_quats: bool = True) -> Rotation:
        """compose the rotation `q` with the current rotation in the quaternion format.
        Args:
            q (Rotation): a quaternion.
        Returns:
            a rotation object.
        """
        ...

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """apply the rotation to the `pts` tensor.
        Args:
            pts [*, 3]: a tensor of points.
        Returns:
            [*, 3], the rotated tensor of points.
        """
        ...

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """apply the inverse rotation to the `pts` tensor.
        Args:
            pts [*, 3]: a tensor of points.
        Returns:
            [*, 3], the rotated tensor of points.
        """
        ...

    def invert(self) -> Rotation:
        """return the inverse rotation."""
        ...
         
        
    def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rotation:
        """
            Apply a Tensor -> Tensor function to underlying rotation tensors,
            mapping over the rotation dimension(s). Can be used e.g. to sum out
            a one-hot batch dimension.

            Args:
                fn:
                    A Tensor -> Tensor function to be mapped over the Rotation 
            Returns:
                The transformed Rotation object
        """ 
        if(self._rot_mats is not None):
            rot_mats = self._rot_mats.view(self._rot_mats.shape[:-2] + (9,))
            rot_mats = torch.stack(
                list(map(fn, torch.unbind(rot_mats, dim=-1))), dim=-1
            )
            rot_mats = rot_mats.view(rot_mats.shape[:-1] + (3, 3))
            return Rotation(rot_mats=rot_mats, quats=None)
        elif(self._quats is not None):
            quats = torch.stack(
                list(map(fn, torch.unbind(self._quats, dim=-1))), dim=-1
            )
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")
        
        

    # The following methods mimic tensor interface.

    def unqueeze(self, dim: int = 0) -> Rotation:
        """return a rotation object with the dimension `dim` expanded.
        Analogous to torch.unsqueeze.
        """
        if exists(self._quaternion):
            return Rotation(quaternion=self._quaternion.unsqueeze(dim))
        else:
            return Rotation(rot_mats=self._rot_mats.unsqueeze(dim))
    @staticmethod
    def cat(rs: Sequence[Rotation], dim: int = 0) -> Rotation:
        """return the concatenation of sequence of rotation object in the dimension `dim`.
        Analogous to `torch.cat``.
        """
        if exists(rs[0].__rot_mats):
            mats = torch.cat([rot.__rot_mats for rot in rs], dim=dim)
            return Rotation(rot_mats=mats)
        else:
            quats = torch.cat([rot.__quaternions for rot in rs], dim=dim)
            return Rotation(quaternion=quats)
    def cuda(
        self,
    ) -> Rotation:
        """return a rotation object on the GPU.
        Analogous to `torch.cuda`.
        """
        if exists(self._rot_mats):
            return Rotation(self._rot_mats.to('cuda'))
        else:
            return Rotation(self._quaternion.to('cuda'))

    def to(
        self, deice: Optional[torch.device], dtype: Optional[torch.dtype]
    ) -> Rotation:
        """return a rotation object on the device.
        Analogous to `torch.to`.
        """
        if exists(self._rot_mats):
            return Rotation(self._rot_mats.to(deice, dtype))
        else:
            return Rotation(self._quaternion.to(deice, dtype))

    def detach(self) -> Rotation:
        """return a rotation object without gradient.
        Analogous to `torch.detach`.
        """
        if exists(self._rot_mats):
            return Rotation(self._rot_mats.detach())
        else:
            return Rotation(self._quaternion.detach())
            


class Rigid(object):
    """

    An immutable ADT representing rigid body transformation.
    Abstract Function:
        Abstract Functions:
        AF(rotation, tranlation) = a rigid transmation.

    Reprsentation Invariant:
        rotation is a Rotation  class object.
        translation is a tensor [*,3].

    Safety from Representation Exposure:
        rotation is immutable.
        Defensive copy is used to prevent modification of the rotation matrix or quaternion.


    """

    def __init__(self, rots: Rotation = None, trans: torch.Tensor = None) -> None:
        """instantialize the rigid transformation with at least one of `rots` or `trans`.
            If no specified, they will be intialized as identity rotation of translation.
        Args:
            rots (Rotation): _description_
            trans (torch.Tensor): _description_
        """
        ...

    def __check_rep(self):
        self.rots.__check_rep()
        assert self._trans.shape[-1] == 3, " the translation should be 3D."

    @staticmethod
    def identity(
        shape, dtype=None, device=None, requires_grad=True, fmt="quat"
    ) -> Rigid:
        """
        return an identity Rigid.
        """
        ...

    def __getitem___(self, index) -> Rigid:
        """
        return a slice of the Rigid.
        Args:
            index: the index of the slice. e.g., 1, (1,2), or slice(1,2)
        Returns:
            a Rigid object.
        """
        ...

    def __mul__(self, other: torch.Tensor) -> Rigid:
        """
        pointwise multiplication of the Rigid with the `other` tensor.
        Args:
            other: a tensor [*,].
        Returns:
            a Rigid object.
        """
        ...

    def __rmul__(self, other: torch.Tensor) -> Rigid:
        """
        pointwise multiplication of the Rigid with the `other` tensor.
        Args:
            other: a tensor [*,].
        Returns:
            a Rigid object.
        """
        ...

    @property
    def shape(self) -> torch.Size:
        """return the shape of the Rigid obj.
        The shape is defined as the shared dimensions of rotation and translation.
        """
        ...

    @property
    def dtype(self) -> torch.dtype:
        """return the dtype of the Rigid obj.
        The dtype is defined as the dtype of the rotation and translation.
        """
        ...

    @property
    def requires_grad(self) -> bool:
        """return whether the Rigid obj requires grad."""
        ...

    @property
    def device(self) -> torch.device:
        """return the device of the Rigid obj.
        The device is defined as the device of the rotation and translation.
        """
        ...

    def get_rots(self) -> Rotation:
        """return the rotation of the rigid."""
        ...

    def get_trans(self) -> torch.Tensor:
        """return the translation of the rigid."""
        ...

    def compose_q_update_vec(self, q_update_vec: torch.Tensor) -> Rigid:
        """compose the transformation of quaternion update vector with the current rigid.
        Args:
            q_update_vec [*, 6]: In the last dim, the first 3 columns notes the roation update vector (x,y,z),
                                the last 3 columns notes the 3D translation update vector.
        Returns:
            a Rigid object.
        """
        ...

    def compose(self, r: Rigid) -> Rigid:
        """compose the current rigid with another rigid `r`.
        Args:
            r (Rigid): a rigid object.
        Returns:
            a Rigid object.
        """
        ...

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """apply the rigid to the `pts` tensor.
        Args:
            pts [*, 3]: a tensor of points.
        Returns:
            [*, 3], the rotated tensor of points.
        """
        ...

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """apply the inverse rigid to the `pts` tensor.
        Args:
            pts [*, 3]: a tensor of points.
        Returns:
            [*, 3], the rotated tensor of points.
        """
        ...

    def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rigid:
        """return a rigid object with the tensor mapped by the function `fn`.
        Args:
            fn: a function that takes a tensor and returns a tensor.

        """
        ...

    def to_tensor_4x4(self) -> torch.Tensor:
        """return the 4x4 matrix representation of the rigid.

        Returns:
            [*, 4, 4] a 4x4 tensor.
        """
        ...

    @staticmethod
    def from_tensor_4x4(t: torch.Tensor) -> Rigid:
        """return a rigid object from the 4x4 matrix representation of the rigid.

        Args:
            t [*, 4, 4]: a 4x4 tensor.
        Returns:
            a Rigid object.
        """
        ...

    def to_tensor_7x7(self) -> torch.Tensor:
        """return the quanterinon representation of the rigid.

        Returns:
            [*, 7] a quanterinon tensor, the first 4 columns denote rotation,
                the last 3 columns are the translation.
        """
        ...

    @staticmethod
    def from_tensor_7x7(
        t: torch.Tensor,
        normalize_quats: bool = False,
    ) -> Rigid:
        """return a rigid object from the quanterinon representation of the rigid.

        Args:
            t [*, 7]: a quanterinon tensor, the first 4 columns denote rotation,
                the last 3 columns are the translation.
        Returns:
            a Rigid object.
        """
        ...

    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor,
        origin: torch.Tensor,
        p_xy_plane: torch.Tensor,
        eps: float = 1e-8,
    ) -> Rigid:
        """
        construct a rigid from the 3 points.
        See Alphafold2 supplement algorithm 21 for the details.
        Take the main chain frame as example:
            The input three points are: N (p_neg_x_axis), Ca (origin), C (p_xy_plane).
            Ca is the origin of the constructed coordinate.
            vector Ca->N denote the negtive x axis.
            The y axis should be perpendicular to the x axis
            and the resulting xy plane should include atom C (p_xy_plane).

        Args:
            p_neg_x_axis [*, 3]: a point on the negative x axis.
            origin [*, 3]: a point on the origin.
            p_xy_plane [*, 3]: a point on the xy plane.
        Returns:
            a Rigid object.
        """
        ...

    def unsequence(self, dim) -> Rigid:
        """Analogous to torch.unsqueeze. The dimension is relative to the
        shared dimensions of the rotation/translation.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed transformation.
        """
        ...

    @staticmethod
    def cat(ts: Sequence[Rigid], dim: int = 0) -> Rigid:
        """
        Concatenate a sequence of Rigid objects.
        Args:
            ts (Sequence[Rigid]): a sequence of Rigid objects.
            dim (int): the dimension to concatenate.
        Returns:
            a Rigid object.
        """
        ...

    def apply_rot_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rigid:
        """return a rigid object with the rotation tensor mapped by the function `fn`.
        Args:
            fn: a function that takes a tensor and returns a tensor.

        Returns:
            a Rigid object.


        """
        ...

    def apply_trans_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rigid:
        """return a rigid object with the translation tensor mapped by the function `fn`.
        Args:
            fn: a function that takes a tensor and returns a tensor.

        Returns:
            a Rigid object.

        """
        ...

    def scale_translation(self, trans_scale_factor: float) -> Rigid:
        """scale the translation by a const factor.

        Args:
            trans_scale_factor (float): a const factor

        Returns:
            Rigid: the rigid with scaled translation
        """
        ...

    def stop_rot_gradient(self) -> Rigid:
        """detach the gradient of rotation matrix for stable training.
        Returns:
            a Rigid object.
        """
        ...

    def cuda(self) -> Rigid:
        """move the rigid to GPU.
        Returns:
            a Rigid object.
        """
        ...

    @staticmethod
    def make_transform_from_reference(n_xyz, ca_xyz, c_xyz, eps=1e-20):
        """
        Returns a transformation object from reference coordinates.

        Note that this method does not take care of symmetries. If you
        provide the atom positions in the non-standard way, the N atom will
        end up not at [-0.527250, 1.359329, 0.0] but instead at
        [-0.527250, -1.359329, 0.0]. You need to take care of such cases in
        your code.

        Args:
            n_xyz: A [*, 3] tensor of nitrogen xyz coordinates.
            ca_xyz: A [*, 3] tensor of carbon alpha xyz coordinates.
            c_xyz: A [*, 3] tensor of carbon xyz coordinates.
        Returns:
            A transformation object. After applying the translation and
            rotation to the reference backbone, the coordinates will
            approximately equal to the input coordinates.
        """

        ...
