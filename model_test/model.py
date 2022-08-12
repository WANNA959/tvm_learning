import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from tvm.contrib.download import download_testdata
import tvm
import tvm.relay
from pathlib import Path
import math
import os

def cnt_shape(x):
    x = math.ceil(x / 2 - 1)
    x = math.ceil(x / 2 - 1)
    return x

class CNN(nn.Module):
    def __init__(self, shape,out_c1=64,out_c2=128, middle_size = 128, output = 10, stride=1):
        super().__init__()
        self.shape = shape
        self.out_c2 = out_c2
        self.conv1 = nn.Conv2d(shape[1], out_c1, kernel_size=3, stride=stride, padding=1)
        self.pool1 = nn.AvgPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(out_c1, out_c2, kernel_size=3, stride=stride, padding=1)
        self.pool2 = nn.AvgPool2d(3, stride=2)
        self.linear1 = nn.Linear(shape[0]*out_c2*cnt_shape(shape[2])*cnt_shape(shape[3]),middle_size)
        self.linear2 = nn.Linear(middle_size, output)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(self.shape[0]*self.out_c2*cnt_shape(self.shape[2])*cnt_shape(self.shape[3]))
        x = self.linear1(x)
        x = self.linear2(x)
        return x

input = torch.randn(4,3,64,64)
net = CNN([4,3,64,64])
output = net(input)
print(output)

def create_onnx(dshape,onnx_name="onnx_tmp.onnx",**args):
    input_shape = dshape[0]
    onnx_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../models/"+onnx_name)

    net = CNN(input_shape,**args)

    input = torch.randn(*input_shape)
    torch.onnx.export(net, input, onnx_name,input_names=['input'],output_names=['output'])

    return onnx_name