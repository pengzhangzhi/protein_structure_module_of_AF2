import math
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from structure_module.primitive import Linear, LayerNorm, ipa_point_weights_init_
from structure_module.rigid import Rigid
from structure_module.utils import flatten_final_dims, permute_final_dims
from einops import rearrange


class PreModule(nn.Module):
    """Algorithm 20 Structure module line 1 - 3
    """
    def __init__(self,
                 c_s,
                 c_z,
                 ):
        super().__init__()
        self.layer_norm_s = LayerNorm(c_s)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear = Linear(in_features=c_s, out_features=c_s, bias=True, init='default')

    def forward(self, s, z,):
        """
        Args:
            s (Tensor): single_rep, [*, N_res, C_s]
            z (Tensor): pair_rep, [*, N_res,N_res, C_z]

        Returns:
            s_init: [*, N_res, C_s]
            s: [*, N_res, C_z]
            z: [*, N_res, N_res, C_z]
            
        """
        s = self.layer_norm_s(s)
        s_init = s
        z = self.layer_norm_z(z)
        s = self.linear(s)
        return s_init, s, z
    
         


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    adpoted from openfold InvariantPointAttention.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        hpv = self.no_heads * self.no_v_points * 3

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z)

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        a = a + pt_att 
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # As DeepMind explains, this manual matmul ensures that the operation
        # happens in float32.
        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z.to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
            ).to(dtype=z.dtype)
        )

        return s
    
    
class Transition(nn.Module):
    def __init__(self,c_in, no_layers, dropout_rate):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransistionLayer(c_in) for _ in range(no_layers)
        ])
        
        self.post_layer_norm = LayerNorm(c_in)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, s):
        for l in self.layers:
            s = l(s)
        s = self.dropout(s)
        s = self.post_layer_norm(s)
        return s
    

class TransistionLayer(nn.Module):
    def __init__(self,c_in):
        super().__init__()
        self.linear_1 = Linear(c_in, c_in,init='relu')
        self.linear_2 = Linear(c_in, c_in,init='relu')
        self.linear_3 = Linear(c_in, c_in,init='final')
        self.relu = nn.ReLU()
    def forward(self, x):
        x_init = x
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        return x + x_init
    
    
class BackboneUpdate(nn.Module):
    def __init__(self,c_s):
        """
        Args:
            c_s (int): channels dim of the single representation
        """
        super().__init__()
        self.linear = Linear(c_s, 6,init='final')
        
    def forward(self, s):
        return self.linear(s)
    
    
class AngleResNet(nn.Module):
    def __init__(self,
                 c_s,
                 c_resnet,
                 no_resnet_blocks,
                 no_agnles,
                 eps=1e-6,
                 ):
        """

        Args:
            c_s (int): channels dim of the single representation
            c_resnet (int): channels dim of the resnet
            no_resnet_blocks (int): number of resnet blocks
            no_agnles (int):  number of angles
            epsilon (float): small constant for normalization.
        """
        super().__init__()
        self.no_angles = no_agnles
        self.relu = nn.ReLU()
        self.linear_init = Linear(c_s, c_resnet,)
        self.linear = Linear(c_s, c_resnet,)
        self.epsilon = eps
        self.layers = nn.ModuleList([
            AngleResNetBlock(c_resnet) for _ in range(no_resnet_blocks)
            ])
        self.linear_out = Linear(c_resnet, no_agnles*2)
        
    def forward(self,
                s: torch.Tensor,
                s_initial: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        Args:
            s (torch.Tensor): [*, C_s]
            s_initial (torch.Tensor): [*, C_s]
        Returns:
            (torch.Tensor): 
                        [*, no_angles, 2] predicted angle, represented by x, y coordinate.
                        The predicted x,y coordinate is normalized to the unit circle.
                        
        """
        s_initial = self.relu(s_initial)
        s_initial = self.linear_init(s_initial)
        
        s = self.relu(s)
        s = self.linear(s)
        
        s += s_initial
        
        for l in self.layers:
            s = l(s)
        s = self.relu(s)
        s = self.linear_out(s)
        # s = rearrange(
        #     s, '... (2 n) -> ... n 2',n = self.no_angles
        # )
        s = s.reshape(*(s.shape[:-1] + (-1,2)))
        un_normalized_angles = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s ** 2, dim=-1,keepdim=True), 
                min=self.epsilon
            )
        )
        normalized_angles = s / norm_denom
        
        return un_normalized_angles, normalized_angles
        
class AngleResNetBlock(nn.Module):
    def __init__(self, c_resnet):
        super().__init__()
        self.linear_1 = Linear(c_resnet, c_resnet,init='relu')
        self.linear_2 = Linear(c_resnet, c_resnet,init='final')
        self.relu = nn.ReLU()
    def forward(self, x):
        x_init = x
        x = self.relu(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x + x_init
        
        
        
        
    
