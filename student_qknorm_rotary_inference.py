"""CPU inference variant of the trained QK-Norm rotary model.

Weights and architecture match student_qknorm_rotary. Rotary pairs use a
single complex multiplication, which reduces temporary tensors on CPU.
"""
import torch
from torch import nn

from student_qknorm_rotary import ReferenceRotary as TrainedRotary
from student_qknorm_rotary import RotaryBlock as TrainedBlock


class RotaryBlock(TrainedBlock):
    @staticmethod
    def rotate(tensor, cosine, sine):
        pairs = torch.view_as_complex(tensor.reshape(*tensor.shape[:-1], -1, 2))
        phase = torch.complex(cosine, sine)
        return torch.view_as_real(pairs * phase).flatten(-2)


class ReferenceRotary(TrainedRotary):
    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList(RotaryBlock(config) for _ in range(config['depth']))


def build_model(config):
    return ReferenceRotary(config)
