from utils.config import cfg

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from data.temporal_sampling import TemporalSampler


class UCF101(Dataset):
    def __init__(self, mode, data_entities, spatial_trans):
        self.mode = mode
        self.annotations_path, self.images_path, self.flows_path = data_entities
        self.spatial_trans = spatial_trans

        self.valid_f25 = True if self.mode == 'valid' and cfg.RST.VALID_F25 else False

        self.temporal_sampler = TemporalSampler('f25' if self.valid_f25 else cfg.RST.FRAME_SAMPLING_METHOD)

        with open(os.path.join(self.annotations_path, 'annot0{}.json'.format(cfg.SPLIT_NO))) as fp:
            self.annotations = json.load(fp)
        self.class_labels = self.annotations['labels']
        self.annotations = self.annotations['training' if self.mode == 'train' else 'validation']

        self.indices = list(self.annotations.keys())  # [:100]
        if self.mode == 'valid':  # these have inconsistent video size so avoids mini-batching at validation
            for i in ['v_PommelHorse_g05_c01', 'v_PommelHorse_g05_c02',
                      'v_PommelHorse_g05_c03', 'v_PommelHorse_g05_c04']:
                try:
                    self.indices.remove(i)
                except ValueError:
                    continue
        if 'v_LongJump_g18_c03' in self.indices:    # a bug in the provided data set
            self.annotations['v_LongJump_g18_c03']['nframes'] -= 1

        self.images_only, self.flows_only = True, True

    def __getitem__(self, index):
        import gc

        uv = ['u', 'v']
        key = self.indices[index]
        i_annotation = self.annotations[key]
        nframes = i_annotation['nframes']
        i_annotation['label'] -= 1  # Fix MATLAB indexing for labels
        i_image_path = os.path.join(self.images_path, key)
        i_flow_path = self.flows_path

        images_list, flows_list = self.temporal_sampler.generate(key, nframes)

        images = self.load_images_list(images_list, i_image_path)
        assert min(images[0].size) == 256
        flows = self.load_flows_list(flows_list, i_flow_path)
        assert min(flows[0][0].size) == 256

        if cfg.RST.FRAME_RANDOMIZATION:
            for i in images:
                self.spatial_trans.randomize_parameters()
                images.append(self.spatial_trans(i, 'image'))
            for i in flows:
                of = []
                self.spatial_trans.randomize_parameters()
                for k, j in enumerate(i):
                    of.append(self.spatial_trans(j, 'flow_{}'.format(uv[k % 2])))
                flows.append(of)
        else:
            self.spatial_trans.randomize_parameters()
            images = [self.spatial_trans(i, 'image') for i in images]
            flows = [[
                self.spatial_trans(j, 'flow_{}'.format(uv[k % 2])) for k, j in enumerate(i)
            ] for i in flows]

        images, flows = self.pack_frames(images, flows)

        gc.collect()

        return images, flows, i_annotation

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def load_images_list(images_list, image_path):
        images = [Image.open(os.path.join(image_path, i)) for i in images_list]

        return images

    @staticmethod
    def load_flows_list(flows_list, flow_path):
        flows = [[Image.open(os.path.join(flow_path, j)) for j in i] for i in flows_list]

        return flows

    @staticmethod
    def pack_frames(images, flows):
        images_o, flows_o = [], []
        if not len(images) == 0:
            images_o = torch.stack(images).transpose(1, 0)
        if not len(flows) == 0:
            flows_o = torch.stack([torch.cat(i) for i in flows]).transpose(1, 0)
        return images_o, flows_o
