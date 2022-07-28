import unittest
import torch
from structure_module.primitive import LayerNorm, Linear


class Test_primitive(unittest.TestCase):
    def test_layernorm(self):
        layernorm = LayerNorm(100)
        x = torch.randn(10, 100).bfloat16()
        out = layernorm(x)
        assert out.dtype == torch.bfloat16
        
    def test_linear(self):
        """ test the initilization of Linear """
        x = torch.randn(10, 100)
        for init in ['default', 'final', 'gating', 'glorot', 'normal', 'relu']:
            linear = Linear(100, 200, init=init)
            out = linear(x)
            # assert the initalized weight and bias are correct
            print(linear.weight.data)
            print(linear.bias.data)
            
            # elif init == 'final':
            #     assert torch.allclose(linear.weight.data, torch.zeros(100, 200))
            #     assert torch.allclose(linear.bias.data, torch.zeros(200))
            # elif init == 'gating':
            #     assert torch.allclose(linear.weight.data, torch.zeros(100, 200))
            #     assert torch.allclose(linear.bias.data, torch.ones(200))
            # elif init == 'glorot':
            #     assert torch.allclose(linear.weight.data, torch.randn(100, 200))
            # elif init == 'normal':
            #     assert torch.allclose(linear.weight.data, torch.randn(100, 200))
            # elif init == 'relu':
            #     assert torch.allclose(linear.weight.data, torch.randn(100, 200))
            #     assert torch.allclose(linear.bias.data, torch.zeros(200))
            # else:
            #     raise NotImplemented(f'init mode "{init}" is not implemented')
            
        
if __name__ == "__main__":
    unittest.main()
 