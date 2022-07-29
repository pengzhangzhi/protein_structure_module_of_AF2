from __future__ import annotations
from typing import Tuple, Any, Sequence, Callable, Optional
import numpy as np
import torch
from utils import exists


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

        self.__rot_mats = torch.clone(rot_mats)
        self.__quaternion = torch.clone(quaternion)
        self.__check_repr_invariant()

    def __check_repr_invariant(self) -> None:
        assert (exists(self.__rot_mats) and self.__rot_mats.shape[-2:] == (3, 3)) or (
            not exists(self.__rot_mats)
            and exists(self.__quaternion)
            and self.__quaternion.shape[-1] == 4
        ), "Invalid representation of rotation matrix or quaternion"

    def _to_float_32(self):
        """Force full-precision.
        Convert the rotation representation to float32.
        """

        ...

    @staticmethod
    def identity(
        shape, dtype=None, device=None, requires_grad=True, fmt="quat"
    ) -> Rotation:
        """
        return an identity rotation.
        """
        ...

    def __getitem___(self, index) -> Rotation:
        """
        return a slice of the rotation.
        Args:
            index: the index of the slice. e.g., 1, (1,2), or slice(1,2)
        Returns:
            a rotation object.
        """
        ...

    def __mul__(self, other: Rotation) -> Rotation:
        """
        pointwise multiplication of the rotation with the `other` tensor.
        Args:
            other: a tensor [*,].
        Returns:
            a rotation object.
        """
        ...

    def __rmul__(self, other: Rotation) -> Rotation:
        """
        pointwise multiplication of the rotation with the `other` tensor.
        Args:
            other: a tensor [*,].
        Returns:
            a rotation object.
        """
        ...

    @property
    def shape(self) -> torch.Size:
        """return the shape of the rotation obj.
        The shape is defined as the batch dim.
        For the representation of rotation matrix [*,3,3]
        or quanternion [*,4], the shape returns [*,].
        """
        ...

    @property
    def dtype(self) -> torch.dtype:
        """return the dtype of the rotation obj.
        The dtype is defined as the dtype of the rotation matrix or quaternion.
        """
        ...

    @property
    def requires_grad(self) -> bool:
        """return whether the rotation obj requires grad."""
        ...

    @property
    def device(self) -> torch.device:
        """return the device of the rotation obj.
        The device is defined as the device of the rotation matrix or quaternion.
        """
        ...

    def get_rot_mats(self) -> torch.Tensor:
        """return the rotation matrix."""
        ...

    def get_quaternion(self) -> torch.Tensor:
        """return the quaternion."""
        ...

    def get_cur_rot(self) -> torch.Tensor:
        """return the current rotation representation.
        #Warning: this method suffer from resentation exposure.
        """
        ...

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
        ...

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
        """return a rotation object with the tensor mapped by the function `fn`.
        Analogous to `torch.map_location`.
        """
        ...

    # The following methods mimic tensor interface.

    def unqueeze(self, dim: int = 0) -> Rotation:
        """return a rotation object with the dimension `dim` expanded.
        Analogous to torch.unsqueeze.
        """
        ...

    @staticmethod
    def cat(rs: Sequence[Rotation], dim: int = 0) -> Rotation:
        """return the concatenation of sequence of rotation object in the dimension `dim`.
        Analogous to `torch.cat``.
        """
        ...

    def cuda(
        self,
    ) -> Rotation:
        """return a rotation object on the GPU.
        Analogous to `torch.cuda`.
        """
        ...

    def to(
        self, deice: Optional[torch.device], dtype: Optional[torch.dtype]
    ) -> Rotation:
        """return a rotation object on the device.
        Analogous to `torch.to`.
        """
        ...

    def detach(self) -> Rotation:
        """return a rotation object without gradient.
        Analogous to `torch.detach`.
        """
        ...


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
        """ instantialize the rigid transformation with at least one of `rots` or `trans`.
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

    
    def unsequence(self,dim) -> Rigid:
        """ Analogous to torch.unsqueeze. The dimension is relative to the
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
        
    def apply_trans_fn(self,fn: Callable[[torch.Tensor], torch.Tensor]) -> Rigid:
        """return a rigid object with the translation tensor mapped by the function `fn`.
        Args:
            fn: a function that takes a tensor and returns a tensor.

        Returns:
            a Rigid object.
             
        """
        ...
        
        
    def scale_translation(
        self,
        trans_scale_factor: float
    ) -> Rigid:
        """scale the translation by a const factor.

        Args:
            trans_scale_factor (float): a const factor

        Returns:
            Rigid: the rigid with scaled translation
        """
        ...
    def stop_rot_gradient(self) -> Rigid:
        """ detach the gradient of rotation matrix for stable training.
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
        
        
        