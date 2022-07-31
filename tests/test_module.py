import unittest
import torch
from structure_module import StructureModule


class TestStructureModule(unittest.TestCase):
    def test_init(self):
        batch_size = 2
        n = 100
        c_s = 128
        c_z = 128
        c_ipa = 13
        c_resnet = 17
        no_heads_ipa = 6
        no_query_points = 4
        no_value_points = 4
        dropout_rate = 0.1
        no_layers = 3
        no_transition_layers = 3
        no_resnet_layers = 3
        ar_epsilon = 1e-6
        no_angles = 7
        trans_scale_factor = 10
        inf = 1e5
        


        structure_module = StructureModule(
        c_s,  # channel dim of single_rep
        c_z,  # channel dim of pair_rep
        c_ipa,  # hidden channel dim of ipa
        c_resnet,  # hidden channel dim of resnet
        no_heads_ipa,  # num of head in ipa
        no_query_points,  # num of qk points in ipa
        no_value_points,  # num of v points in ipa
        dropout_rate,
        no_layers,  # num of structure module blocks
        no_transition_layers,  # num of transition layers in resnet
        no_resnet_layers,  # num of blocks in resnet
        no_angles,  # num of angles to generate in resnet
        trans_scale_factor,  # scale of c_s hidden dim
        )
        s, z = torch.randn(batch_size,n, c_s), torch.randn(batch_size, n, n, c_z)
        aatype = torch.randint(low=0, high=21, size=(batch_size, n)).long()
        outputs = structure_module(s,z,aatype)
        