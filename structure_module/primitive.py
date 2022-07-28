# further encapsulate the primitive pytorch layer for better performance.
import torch
import torch.nn as nn
from torch import Tensor
import importlib

deepspeed_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_installed:
    from deepspeed import comm as dist


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, s, z):
        s = self.linear(s)
        return s, z


class LayerNorm(nn.Module):
    """ Layer normalization with pricision check.
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

                print(out.dtype)

        else:
            out = nn.functional.layer_norm(x, self.weight, self.bias, self.eps)
        return out


if __name__ == "__main__":
    pass
