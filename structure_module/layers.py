import torch
import torch.nn as nn


class PreModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, z, aatype):
        return s, z, aatype
    
class IPA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, z, aatype):
        return s, z, aatype
    

class Transition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, z, aatype):
        return s, z, aatype
    
class BackboneUpdate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, z, aatype):
        return s, z, aatype
    
    
class AngleResNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s, z, aatype):
        return s, z, aatype
    
