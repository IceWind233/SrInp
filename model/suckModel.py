import torch.nn as nn
import torch

import numpy

from model import SoftmaxSplatting, DITN_Real

ditn_path = "./models/DITN_Real_x4.pth"
softmax_path = "./models/network-lf.pytorch"

class SuckModel(nn.Module):
    def __init__(self):
        super(SuckModel, self).__init__()

        self.ditn = DITN_Real()
        self.ditn.load_state_dict(torch.load(ditn_path, map_location='cpu'), strict=False)
        self.softmax = SoftmaxSplatting()
        self.softmax.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(softmax_path).items()})

        self.add_module('ditn', self.ditn)
        self.add_module('softmax', self.softmax)

    def forward(self, tensor1, tensor2):
        tensor_mean = self.softmax(tensor1.cuda(), tensor2.cuda(), [0.5])
        tensor_mean = tensor_mean[0].cpu()
        self.ditn = self.ditn.to(torch.device("cpu"))
        # torch.device("cpu")
        tensor1_hr = self.ditn(tensor1)

        tensor2_hr = self.ditn(tensor2)

        tensor_mean_hr = self.ditn(tensor_mean)

        return [tensor1_hr, tensor_mean_hr, tensor2_hr]
