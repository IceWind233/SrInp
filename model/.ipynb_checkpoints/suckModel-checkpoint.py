import torch.nn as nn
import torch

from model import SoftmaxSplatting, DITN_Real

ditn_path = "./models/DITN_Real_x4.pth"
softmax_path = "./models/network-lf.pytorch"

class SuckModel(nn.Module):
    def __init__(self):
        self.ditn = DITN_Real().load_state_dict(torch.load(ditn_path, map_location='cpu'), strict=True)
        self.softmax = SoftmaxSplatting().load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(softmax_path).items()})

    def forward(self, tensor1, tensor2):
        with torch.set_grad_enabled(False):
            tensor_mean = self.softmax(tensor1, tensor2, [0.5])

            tensor1 = self.ditn(tensor1)
            tensor_mean = self.ditn(tensor_mean)
            tensor2 = self.ditn(tensor2)

        return [tensor1, tensor_mean, tensor2]
