import torch
import torch.nn as nn
from torch.nn import init
import functools
import math
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
from torchvision import models
import numpy as np
from collections import OrderedDict

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

################################################
# Encoder model
################################################
affine_par = True
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class ResNetFCN(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetFCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def Encoder(is_restore_from_imagenet=True, resnet_weight_path="./resnetweight/"):
    model = ResNetFCN(Bottleneck, [3, 4, 6, 3])
    if is_restore_from_imagenet:
        print("loading pretrained model (resnet50)")
        state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir=resnet_weight_path)
        model.load_state_dict(state_dict, strict=False) #, strict=False
    return model


################################################
# Segmentation model
################################################
class SegBranch(nn.Module):
    def __init__(self, n_features, up_scale, bn=True, act='relu'):
        super(SegBranch, self).__init__()
        m = []
        if (up_scale & (up_scale - 1)) == 0:  # Is scale = 2^n?
            for i in range(int(math.log(up_scale, 2))):
                m.append(nn.ConvTranspose2d(n_features, n_features // ( 2 ** (i+1) ),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=True))
                if bn:
                    m.append(nn.BatchNorm2d(n_features // ( 2 ** (i+1))))
                if act == 'relu':
                    m.append(nn.ReLU(True))

        else:
            raise NotImplementedError
        self.upsample = nn.Sequential(*m)
    def forward(self, x):




