# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 16:59
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

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


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResNet(nn.Module):

    def __init__(self):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # Modules
        self.conv1 = None
        self.bn1 = None
        self.relu = None
        self.maxpool = None

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

        self.init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0] * rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i] * rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        return input

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP_sum(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_sum, self).__init__()
        self.aspp1 = ASPP_module(inplanes, planes, rate[0])
        self.aspp2 = ASPP_module(inplanes, planes, rate[1])
        self.aspp3 = ASPP_module(inplanes, planes, rate[2])
        self.aspp4 = ASPP_module(inplanes, planes, rate[3])

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp1(x)
        x3 = self.aspp1(x)
        x4 = self.aspp1(x)
        x = x1 + x2 + x3 + x4

        return x


class DeeplabV2(ResNet):
    def __init__(self, in_channels, n_class, block, layers, pyramids):
        super(DeeplabV2, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, rate=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, rate=4)

        # self.aspp1 = ASPP_sum(2048, n_class, pyramids)
        # self.aspp2 = ASPP_sum(2048, n_class, pyramids)

        self.aspp1_first = ASPP_module(1024, n_class, pyramids[0])
        self.aspp2_first = ASPP_module(1024, n_class, pyramids[1])
        self.aspp3_first = ASPP_module(1024, n_class, pyramids[2])
        self.aspp4_first = ASPP_module(1024, n_class, pyramids[3])

        self.aspp1_second = ASPP_module(2048, n_class, pyramids[0])
        self.aspp2_second = ASPP_module(2048, n_class, pyramids[1])
        self.aspp3_second = ASPP_module(2048, n_class, pyramids[2])
        self.aspp4_second = ASPP_module(2048, n_class, pyramids[3])

        self.init_weight()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # x = self.layer3(x)
        # x = self.layer4(x)
        # x_first = self.aspp1(x)
        # x_second = self.aspp2(x)
        # x_first = F.interpolate(x_first, size=input.size()[2:], mode='bilinear', align_corners=True)
        # x_second = F.interpolate(x_second, size=input.size()[2:], mode='bilinear', align_corners=True)

        x_first = self.layer3(x)
        x_second = self.layer4(x_first)

        x1_first = self.aspp1_first(x_first)
        x2_first = self.aspp2_first(x_first)
        x3_first = self.aspp3_first(x_first)
        x4_first = self.aspp4_first(x_first)
        x_first = x1_first + x2_first + x3_first + x4_first

        x1_second = self.aspp1_second(x_second)
        x2_second = self.aspp2_second(x_second)
        x3_second = self.aspp3_second(x_second)
        x4_second = self.aspp4_second(x_second)
        x_second = x1_second + x2_second + x3_second + x4_second
        # x_first = F.interpolate(x_first, size=input.size()[2:], mode='bilinear', align_corners=True)
        # x_second = F.interpolate(x_second, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x_first, x_second

    def get_1x_lr_params(self):
        b = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        b = [self.aspp1, self.aspp2, self.aspp3, self.aspp4]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone_bn(self):
        self.bn1.eval()

        for m in self.layer1:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer2:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer3:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer4:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def DeeplabV2_101(in_channels, n_class, pretrained=True):

    model = DeeplabV2(in_channels, n_class=n_class, block=Bottleneck, layers=[3, 4, 23, 3], pyramids=[6, 12, 18, 24])

    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    x = torch.randn(2, 3, 512, 512)
    net = DeeplabV2_101(in_channels=3, n_class=6, pretrained=True)
    x = net(x)
    print(net)
    print(x.size())