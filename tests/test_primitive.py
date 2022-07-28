import unittest
import torch
from structure_module.primitive import LayerNorm, Linear


class Test_primitive(unittest.TestCase):
    def test_layernorm(self):
        layernorm = LayerNorm(100)
        x = torch.randn(10, 100).bfloat16()
        out = layernorm(x)
        assert out.dtype == torch.bfloat16
        
        
if __name__ == "__main__":
    unittest.main()
 