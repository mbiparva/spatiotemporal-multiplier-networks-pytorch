from utils.config import cfg

import os
import datetime

import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data
from data.data_set import UCF101
import data.spatial_transformation as Transformation

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def custom_collator(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    import re
    from torch._six import string_classes, int_classes
    import collections
    _use_shared_memory = True
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        # return {key: custom_collator([d[key] for d in batch]) for key in batch[0]}
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [custom_collator(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def ds_worker_init_fn(worker_id):
    np.random.seed(datetime.datetime.now().microsecond + worker_id)
    # np.random.seed(worker_id)


class DataContainer:
    def __init__(self, mode):
        self.dataset, self.dataloader = None, None
        self.mode = mode
        self.mode_cfg = cfg.get(self.mode.upper())

        self.create()

    def create(self):
        self.create_dataset()
        self.create_dataloader()

    def create_transform(self):
        h, w = cfg.SPATIAL_INPUT_SIZE
        assert h == w
        transformations_final = [
            Transformation.ToTensor(norm_value=1),  # 255
            Transformation.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[1, 1, 1])
            # Transformation.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])
            # Transformation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if self.mode == 'train':
            # transformations = [
            #     Transformation.CenterCrop(256),
            #     Transformation.Resize(h),
            # ]
            transformations = [
                # Transformation.RandomCornerCrop(256, crop_scale=(256, 224, 192, 168), border=0.25),
                Transformation.RandomCornerCrop(256, crop_scale=(0.66, 1.0), border=0.25),
                Transformation.Resize(h),   # This is necessary for async-resolution streaming
                Transformation.RandomHorizontalFlip()
            ]
        elif self.mode == 'valid':
            transformations = [
                Transformation.CenterCrop(256),
                Transformation.Resize(h),
            ]
        else:
            raise NotImplementedError

        return Transformation.Compose(
            transformations + transformations_final
        )

    def create_dataset(self):
        spatial_transform = self.create_transform()

        if cfg.DATASET_NAME == 'UCF101':
            self.dataset = UCF101(
                self.mode,
                [os.path.join(cfg.DATASET_DIR, cfg.DATASET_NAME, 'annotations'),
                 os.path.join(cfg.DATASET_DIR, cfg.DATASET_NAME, 'images'),
                 os.path.join(cfg.DATASET_DIR, cfg.DATASET_NAME, 'flows')],
                spatial_transform
            )
        else:
            raise NotImplementedError('Implement other dataset classes here')

    def create_dataloader(self):
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.mode_cfg.BATCH_SIZE,
                                     shuffle=self.mode_cfg.SHUFFLE,
                                     num_workers=4,
                                     collate_fn=custom_collator,
                                     pin_memory=False,
                                     drop_last=True,
                                     worker_init_fn=ds_worker_init_fn,
                                     )
