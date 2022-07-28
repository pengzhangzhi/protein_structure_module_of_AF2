import unittest
import torch
from structure_module.layers import (
    PreModule,
    InvariantPointAttention,
    TransistionLayer,
    BackboneUpdate,
    AngleResNet,
    AngleResNetBlock,
    Transition,
)


class TestLayer(unittest.TestCase):
    def test_premodule(self):
        premodule = PreModule(c_s=100, c_z=200)
        s = torch.randn(10, 100)
        z = torch.randn(10, 200)
        s_init, s, z = premodule(s, z)
        assert s_init.shape == (10, 100)
        assert s.shape == (10, 100)
        assert z.shape == (10, 200)


# class TestIPA(unittest.TestCase):
#     def test_ipa(self):
#         ipa = InvariantPointAttention(
#             c_s=100, c_z=200, c_hidden=300, no_heads=4, no_qk_points=5, no_v_points=6
#         )
#         s = torch.randn(10, 100)
#         z = torch.randn(10, 200)
#         s = ipa(
#             s,
#             z,
#         )
#         pass

#     def test_equalvariance(self):
#         ipa = InvariantPointAttention(
#             c_s=100, c_z=200, c_hidden=300, no_heads=4, no_qk_points=5, no_v_points=6
#         )
#         s = torch.randn(10, 100)
#         z = torch.randn(10, 200)
#         # rotate the rigid body by 90 degrees the ouput should be the same.
#         s = ipa(
#             s,
#             z,
#         )
#         pass


class TestTransistionLayer(unittest.TestCase):
    def test_TransistionLayer(self):
        transistion_layer = TransistionLayer(100)
        x = torch.randn(10, 100)
        out = transistion_layer(x)

    def test_Transition(self):
        transition = Transition(100, 3, 0.1)
        x = torch.randn(10, 100)
        out = transition(x)


class TestBackboneUpdate(unittest.TestCase):
    def test_BackboneUpdate(self):
        backbone_update = BackboneUpdate(100)
        x = torch.randn(10, 100)
        out = backbone_update(x)
        assert out.shape[-1] == 6, "backbone update should output 6 features"


class TestAngleResNet(unittest.TestCase):
    def test_AngleResNetBlock(self):
        angle_resnet_block = AngleResNetBlock(
            100,
        )
        x = torch.randn(10, 100)
        out = angle_resnet_block(x)

    def test_AngleResNet(self):
        angle_resnet = AngleResNet(
            100,
            100,
            4,
            8,
        )
        x = torch.randn(10, 100)
        out = angle_resnet(x, x)
        assert out[0].shape[-2:] == (8, 2), "angle resnet should output 6 features"
        assert out[1].shape[-2:] == (8, 2), "angle resnet should output 6 features"
        assert torch.allclose(
            torch.sum(out[1] ** 2, dim=-1, keepdim=True),
            torch.ones(out[1].shape[:-1] + (1,)),
        ), "normalized angle represented in x,y should on the unit circle."


if __name__ == "__main__":
    unittest.main()
