# further encapsulate the primitive pytorch layer for better performance.
import math
from typing import Callable, Optional
import torch
import torch.nn as nn
from torch import Tensor
import importlib
import numpy as np
from scipy.stats import truncnorm

from structure_module.utils import exists

deepspeed_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_installed:
    from deepspeed import comm as dist


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


class Linear(nn.Linear):
    """customized linear layer providing many choices of initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_features (int):
                            The final dimension of inputs to the layer
            out_features (int):
                            The final dimension of layer outputs
            bias (bool, optional):
                            Whether to learn an additive bias. Defaults to True.
            init (str, optional):
                                The initialization mode to use. Defaults to 'default'.
                                Choose from:
                                'default': LeCun fan-in truncated normal initialization
                                "relu": He initialization w/ truncated normal distribution
                                "glorot": Fan-average Glorot uniform initialization
                                "gating": Weights=0, Bias=1
                                "normal": Normal initialization with std=1/sqrt(fan_in)
                                "final": Weights=0, Bias=0

            init_fn (callable[[torch.Tensor,torch.Tensor],None], optional):
                                A customized function take weights and bias tensor as input for initalization.
                                Defaults to None.
            Notes:
                `init_fn` and `init` can not both be set, init_fn takes precedence.
        """
        super().__init__(in_features, out_features, bias)
        # assert that init and init_fn can not both be set
        assert (
            init_fn is None or init == "default"
        ), "init_fn and init can not both be set, init_fn takes precedence"

        if exists(init_fn):
            init_fn(self.weight, self.bias)

        # init bias to 0 by default.
        if exists(bias):
            with torch.no_grad():
                self.bias.zero_()

        if init == "default":
            he_normal_init_(self.weight)
        elif init == "final":
            nn.init.constant_(self.weight, 0)
        elif init == "gating":
            nn.init.constant_(self.weight, 0)
            nn.init.constant_(self.bias, 1)
        elif init == "glorot":
            nn.init.xavier_uniform_(self.weight, gain=1)
        elif init == "normal":
            nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        elif init == "relu":
            he_normal_init_(self.weight)
        else:
            raise NotImplemented(f'init mode "{init}" is not implemented')


class LayerNorm(nn.Module):
    """Layer normalization with pricision check.
    If input is bfloat16, the LayerNorm will be bfloat16.
    """

    def __init__(self, c_in, eps=1e-5):
        super().__init__()
        self.c_in = (c_in,)
        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))
        self.eps = eps

    def forward(self, x):
        d = x.dtype
        deepspeed_is_initialized = deepspeed_installed and dist.is_initialized()
        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.cuda.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(dtype=d),
                    self.bias.to(dtype=d),
                    self.eps,
                )

        else:
            out = nn.functional.layer_norm(x, self.c_in,self.weight, self.bias, self.eps)
        return out


if __name__ == "__main__":
    pass
