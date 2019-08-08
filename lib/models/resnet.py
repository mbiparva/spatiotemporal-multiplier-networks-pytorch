import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
from torch.nn import Parameter

__all__ = ['ResNet3D', 'resnet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# =============================
# ********* 3D ResNet *********
# =============================
def conv1x3x3(in_planes, out_planes, stride=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.downsample = downsample
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def init_temporal(self, strategy):
        raise NotImplementedError


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, t_conv=False, bias=False):
        super(Bottleneck3D, self).__init__()
        self.t_conv, self.t_bn = t_conv, True
        self.downsample = downsample    # to have the pre-trained net loaded easily, just moved it up here.
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes,
                               kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=bias)
        self.bn2 = nn.BatchNorm3d(planes)
        if self.t_conv:
            self.conv2_temporal = nn.Conv3d(planes, planes,  # It seems base implementation uses kernel size of 3
                                            kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias)
            if self.t_bn:
                self.bn2_temporal = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.t_conv:
            out = self.conv2_temporal(out)
            if self.t_bn:
                out = self.bn2_temporal(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out

    def init_temporal(self, strategy):
        channels = len(self.conv2_temporal.weight)
        w_eye = np.eye(channels, dtype=np.float32)
        if strategy == 'center':
            f = np.array([0, 1, 0])
        elif strategy == 'difference':
            f = np.array([-1, 3, -1])
        elif strategy == 'average':
            f = np.ones(3)*1/3.
        else:
            raise NotImplementedError
        w_outer = np.multiply.outer(w_eye, f).astype(np.float32)
        w_expand = w_outer[:, :, :, np.newaxis, np.newaxis]     # BxCxTx1x1
        assert self.conv2_temporal.weight.shape == w_expand.shape
        self.conv2_temporal.weight.data = torch.from_numpy(w_expand)
        if self.conv2_temporal.bias is not None:
            self.conv2_temporal.bias.data.zero_()


class ResNet3D(nn.Module):
    def __init__(self, block, layers, **kwargs):
        super().__init__()
        in_channels, num_classes = kwargs['in_channels'], kwargs['num_classes']
        t_conv_layer = kwargs['temporal_conv_layer']
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=True)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = Block(block, self.inplanes, 64, layers[0], t_conv_layer=t_conv_layer)
        self.layer2 = Block(block, self.layer1.inplanes, 128, layers[1], stride=2, t_conv_layer=t_conv_layer)
        self.layer3 = Block(block, self.layer2.inplanes, 256, layers[2], stride=2, t_conv_layer=t_conv_layer)
        self.layer4 = Block(block, self.layer3.inplanes, 512, layers[3], stride=2, t_conv_layer=t_conv_layer)

        self.s_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.t_pool = nn.AdaptiveMaxPool3d(1)

        # self.fc_dropout = nn.Dropout3d(p=0.8, inplace=False)
        self.fc = nn.Conv3d(512 * block.expansion, num_classes, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self):
        raise NotImplementedError('use the two_stream wrapper network\' forward function')

    @staticmethod
    def get_mod_type(in_mod):
        mod_type = str(in_mod.__class__)[:-2].rpartition('.')[-1]

        return mod_type

    def get_param_mod(self, mod_filter=('Conv3d', 'BatchNorm3d', 'Conv2d')):
        param_mod_output = []
        for n, m in self.named_modules():
            mod_type = self.get_mod_type(m)
            if mod_type in mod_filter:
                param_mod_output.append({'name': n, 'type': mod_type, 'module': m})
            else:
                continue
        return param_mod_output

    def load_state_dict(self, pret_state_dict, strict=True):
        own_state_dict = self.state_dict()
        own_keys_iter = iter(own_state_dict.keys())
        pret_state_dict.pop('fc.weight')
        pret_state_dict.pop('fc.bias')
        for pret_key, pret_param in pret_state_dict.items():
            try:
                own_key = next(own_keys_iter)
            except StopIteration:
                break
            own_param = own_state_dict[own_key]
            if isinstance(pret_param, Parameter):
                pret_param = pret_param.data
            while not (own_param.shape == pret_param.shape or own_param.shape == pret_param[:, :, np.newaxis].shape):
                print(own_key.split('.')[-1], 'skipped')
                try:
                    own_key = next(own_keys_iter)
                except StopIteration:
                    break
                own_param = own_state_dict[own_key]
            pret_param = pret_param if own_param.shape == pret_param.shape else pret_param[:, :, np.newaxis]
            try:
                own_param.copy_(pret_param)
            except Exception:
                raise Exception('While copying the parameter named {} to {}, '
                                'whose dimensions in the model are {} and '
                                'whose dimensions in the checkpoint are {}.'.format(pret_key, own_key,
                                                                                    own_param.size(),
                                                                                    pret_param.size()))


class Block(nn.Module):
    def __init__(self, block, inplanes, planes, blocks, t_conv_layer, stride=1, bias=False):
        super(Block, self).__init__()
        self.inplanes, self.blocks = inplanes, blocks
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes*block.expansion, kernel_size=1, stride=(1, stride, stride), bias=bias),
                nn.BatchNorm3d(planes * block.expansion),
            )

        self.sblock_0 = block(self.inplanes, planes, stride, downsample, t_conv=True if t_conv_layer == 0 else False,
                              bias=bias)
        self.inplanes = planes * block.expansion
        for i in range(1, self.blocks):
            self.__setattr__('sblock_{}'.format(i), block(self.inplanes, planes,
                                                          t_conv=True if t_conv_layer == i else False))

    def forward(self, x):
        raise NotImplementedError


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet3D(Bottleneck3D, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pt = model_zoo.load_url(model_urls['resnet50'])
        pt.pop('conv1.weight')
        pt.pop('fc.weight')
        pt.pop('fc.bias')
        model_dict = model.state_dict()
        model_dict.update(pt)
        model.load_state_dict(model_dict)

    return model
