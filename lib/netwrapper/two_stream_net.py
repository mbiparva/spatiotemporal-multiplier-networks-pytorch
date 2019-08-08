import os
from utils.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import numpy as np

from scipy.io import loadmat

from models.resnet import resnet50


# noinspection PyProtectedMember
class StepLRestart(optim.lr_scheduler._LRScheduler):
    """The same as StepLR, but this one has restart.
    """
    def __init__(self, optimizer, step_size, restart_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.restart_size = restart_size
        assert self.restart_size > self.step_size
        self.gamma = gamma
        super(StepLRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** ((self.last_epoch % self.restart_size) // self.step_size)
                for base_lr in self.base_lrs]


class TwoStreamNet(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.spatial_stream, self.temporal_stream = None, None
        self.criterion, self.optimizer, self.scheduler = None, None, None

        self.create_load(device)

        self.setup_optimizer()

    def create_load(self, device):
        if cfg.PRETRAINED_MODE == 'ImageNet':
            self.create_net(imagenet_pretrained=True)
            self.init_temporal_convolution()
        elif cfg.PRETRAINED_MODE == 'ResNet_ST':
            self.create_net()
            self.load_st_pretrained()
            self.init_temporal_convolution()
        elif cfg.PRETRAINED_MODE == 'Custom':
            self.create_net()
            self.load(cfg.CUS_PT_PATH)
        else:
            pass

        self.spatial_stream = self.spatial_stream.to(device)
        self.temporal_stream = self.temporal_stream.to(device)

    def create_net(self, imagenet_pretrained=False):
        self.spatial_stream = resnet50(**{
            'pretrained': imagenet_pretrained,
            'in_channels': cfg.CHANNEL_INPUT_SIZE,
            'num_classes': cfg.NUM_CLASSES,
            'temporal_conv_layer': cfg.RST.TEMPORAL_CONVOLUTION_LAYER,
        })
        self.temporal_stream = resnet50(**{
            'pretrained': imagenet_pretrained,
            'in_channels': cfg.TEMPORAL_INPUT_SIZE,
            'num_classes': cfg.NUM_CLASSES,
            'temporal_conv_layer': cfg.RST.TEMPORAL_CONVOLUTION_LAYER,
        })

    def load_st_pretrained(self):
        self.map_mat_torch('spatial')
        self.map_mat_torch('temporal')

    def map_mat_torch(self, stream):
        pt_path = cfg.BASE_NET[stream]
        net_torch = self.spatial_stream if stream == 'spatial' else self.temporal_stream
        net_torch_gen = iter(net_torch.get_param_mod())

        net_mat = loadmat(pt_path, struct_as_record=False, squeeze_me=True)['net']
        net_lay, net_par = net_mat.layers, net_mat.params

        param_dict = {}
        for i in range(len(net_par)):
            param_dict[net_par[i].name] = net_par[i].value
        layer_dict = {}
        for i in range(len(net_lay)):
            layer_dict[net_lay[i].name] = {
                'inputs': net_lay[i].inputs,
                'outputs': net_lay[i].outputs,
                'type': net_lay[i].type,
                'param_names': list(net_lay[i].params)
                if isinstance(net_lay[i].params, np.ndarray) else [net_lay[i].params],
                'block_attributes': net_lay[i].block}
        for i in layer_dict.keys():
            #  possible matconvnet layer types are: {'BatchNorm', 'Conv', 'Loss', 'Pooling', 'ReLU', 'Sum'}
            layer_names, layer_type = layer_dict[i]['param_names'], layer_dict[i]['type'].rpartition('.')[-1]
            if len(layer_names):
                net_torch_param = next(net_torch_gen)
                while 'temporal' in net_torch_param['name']:
                    net_torch_param = next(net_torch_gen)   # skip temporal convolution
                assert layer_type.lower() in net_torch_param['type'].lower(), 'layer types do not match'
                # print(layer_names, net_torch_param['name'])
                # if 'fc' in net_torch_param['name']:  # CHECKME __ This must be modified for the extra FC layers
                #     net_torch_param = next(net_torch_gen)  # skip new fc layers
                #     continue
                net_torch_mod = net_torch_param['module']

                weight = param_dict[layer_names[0]]
                if not weight.ndim == 1:
                    weight = weight.transpose(range(weight.ndim-1, -1, -1))
                    if weight.ndim == 2:
                        # if 'fc' not in net_torch_param['name']:
                        weight = weight[:, :, np.newaxis, np.newaxis, np.newaxis]
                        # else:
                        #     weight = weight[:, :, np.newaxis, np.newaxis]   # since it fully spatio-temporal network.
                    elif weight.ndim == 4:
                        weight = weight[:, :, np.newaxis]
                    else:
                        raise Exception('it is a bug')
                weight = torch.from_numpy(weight)
                assert net_torch_mod.weight.shape == weight.shape
                net_torch_mod.weight.data = weight

                if len(layer_names) == 1:   # Conv wo bias
                    # assert net_torch_mod.bias is None, 'bias mismatch is encountered'
                    if net_torch_mod.bias is not None:
                        print(layer_names, 'does not have bias while the net has, skipped...')
                    continue

                bias = param_dict[layer_names[1]]
                bias = torch.from_numpy(bias)
                assert net_torch_mod.bias.shape == bias.shape
                net_torch_mod.bias.data = bias

                if len(layer_names) == 3:   # BN with running moments
                    assert layer_type == 'BatchNorm', 'only {} is addressed to have 3 parameters'.format(layer_type)
                    moments = param_dict[layer_names[2]]
                    r_m = torch.from_numpy(moments[:, 0])
                    assert net_torch_mod.running_mean.shape == r_m.shape
                    net_torch_mod.running_mean = r_m
                    r_v = torch.from_numpy(moments[:, 1])
                    assert net_torch_mod.running_var.shape == r_v.shape
                    net_torch_mod.running_var = r_v
                elif len(layer_names) < 3:
                    pass
                else:
                    raise Exception('NotExpected')

    def init_temporal_convolution(self):
        for i in range(1, 5):
            layer_spatial = self.spatial_stream.__getattr__('layer{}'.format(i))
            layer_temporal = self.temporal_stream.__getattr__('layer{}'.format(i))
            assert layer_spatial.blocks <= layer_temporal.blocks
            layer_spatial.__getattr__(
                'sblock_{}'.format(cfg.RST.TEMPORAL_CONVOLUTION_LAYER)
            ).init_temporal(cfg.RST.INIT_TEMPORAL_STRATEGY)
            layer_temporal.__getattr__(
                'sblock_{}'.format(cfg.RST.TEMPORAL_CONVOLUTION_LAYER)
            ).init_temporal(cfg.RST.INIT_TEMPORAL_STRATEGY)

    def setup_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(params=self.parameters(),
                                   lr=cfg.TRAIN.LR,
                                   weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                   momentum=cfg.TRAIN.MOMENTUM,
                                   nesterov=cfg.TRAIN.NESTEROV)

        if cfg.TRAIN.SCHEDULER_MODE:
            if cfg.TRAIN.SCHEDULER_TYPE == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=60, gamma=0.1)
            elif cfg.TRAIN.SCHEDULER_TYPE == 'step_restart':
                self.scheduler = StepLRestart(self.optimizer, step_size=4, restart_size=8, gamma=0.1)
            elif cfg.TRAIN.SCHEDULER_TYPE == 'multi':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                milestones=cfg.TRAIN.SCHEDULER_MULTI_MILESTONE,
                                                                gamma=0.1)
            elif cfg.TRAIN.SCHEDULER_TYPE == 'lambda':
                def lr_lambda(e): return 1 if e < 5 else .5 if e < 10 else .1 if e < 15 else .01
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            elif cfg.TRAIN.SCHEDULER_TYPE == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10,
                                                                      cooldown=0,
                                                                      verbose=True)
            else:
                raise NotImplementedError

    def schedule_step(self, stage, metric=None):
        if cfg.TRAIN.SCHEDULER_MODE:
            if stage == 'early' and cfg.TRAIN.SCHEDULER_TYPE in ['step', 'step_restart', 'multi', 'lambda']:
                self.scheduler.step()
            if stage == 'late' and cfg.TRAIN.SCHEDULER_TYPE == 'plateau':
                self.scheduler.step(metric.meters['loss'].avg)

    def save(self, file_path, e):
        torch.save(self.spatial_stream.state_dict(), os.path.join(file_path, 'spt_{:03d}.pth'.format(e)))
        torch.save(self.temporal_stream.state_dict(), os.path.join(file_path, 'tmp_{:03d}.pth'.format(e)))

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def forward(self, x_s, x_t):
        x_s = self.spatial_stream.conv1(x_s)
        x_s = self.spatial_stream.bn1(x_s)
        x_s = self.spatial_stream.relu(x_s)
        x_s = self.spatial_stream.maxpool(x_s)

        x_t = self.temporal_stream.conv1(x_t)
        x_t = self.temporal_stream.bn1(x_t)
        x_t = self.temporal_stream.relu(x_t)
        x_t = self.temporal_stream.maxpool(x_t)

        for i in range(1, 5):
            layer_spatial = self.spatial_stream.__getattr__('layer{}'.format(i))
            layer_temporal = self.temporal_stream.__getattr__('layer{}'.format(i))
            assert layer_spatial.blocks <= layer_temporal.blocks    # CHECKME: skipped for resnet 50 and 152
            for j in range(layer_spatial.blocks):
                x_s_res, x_t_res = None, None
                if cfg.RST.CROSS_STREAM_MOD_LAYER == j:
                    x_s_res = x_s                       # T -> S
                    x_s = x_s * x_t                     # Multiplicative Modulation

                x_s = layer_spatial.__getattr__('sblock_{}'.format(j))(x_s, residual=x_s_res)
                x_t = layer_temporal.__getattr__('sblock_{}'.format(j))(x_t, residual=x_t_res)

            for j in range(layer_temporal.blocks-layer_spatial.blocks):
                x_t = layer_temporal.__getattr__('sblock_{}'.format(layer_spatial.blocks+j))(x_t, residual=x_t_res)

        x_s = self.spatial_stream.s_pool(x_s)
        x_s = self.spatial_stream.t_pool(x_s)

        x_t = self.temporal_stream.s_pool(x_t)
        x_t = self.temporal_stream.t_pool(x_t)

        x_s = functional.relu(self.spatial_stream.fc(x_s), inplace=True)
        x_t = functional.relu(self.temporal_stream.fc(x_t), inplace=True)

        x_s = x_s.view(x_s.size(0), x_s.size(1))
        x_t = x_t.view(x_t.size(0), x_t.size(1))

        return x_s, x_t

    def loss_update(self, p, a, step=True):
        p_s, p_t = p
        loss = cfg.RST.LR_S_STREAM_MULT * self.criterion(p_s, a) + self.criterion(p_t, a)

        if step:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()
