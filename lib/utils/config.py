"""Config file setting hyperparameters

This file specifies default config options. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.
"""

from easydict import EasyDict as edict
import os
import datetime

__C = edict()
cfg = __C   # from config.py import cfg


# ================
# GENERAL
# ================

# Set modes
__C.TRAINING = True
__C.VALIDATING = True

# Root directory of project
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data directory
__C.DATASET_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'dataset'))

# Model directory
__C.MODELS_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'lib', 'models'))

# Experiment directory
__C.EXPERIMENT_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'experiment'))

# Set meters to use for experimental evaluation
__C.METERS = ['loss', 'label_accuracy']

# Use GPU
__C.USE_GPU = True

# Default GPU device id
__C.GPU_ID = 0

# Number of epochs
__C.NUM_EPOCH = 100

# Dataset name
__C.DATASET_NAME = ('UCF101', )[0]

if __C.DATASET_NAME == 'UCF101':
    __C.SPLIT_NO = 1

    # Number of categories
    __C.NUM_CLASSES = 101

    # Official Base Pre-trained Networks
    __C.BASE_NET = {
        'spatial': os.path.join(__C.EXPERIMENT_DIR,
                                'base_pretrained_nets', 'ucf101-img-resnet50-split{}-dr0'.format(__C.SPLIT_NO)),
        'temporal': os.path.join(__C.EXPERIMENT_DIR,
                                 'base_pretrained_nets', 'ucf101-flow-resnet50-split{}-dr0.8'.format(__C.SPLIT_NO))}

# Normalize database samples according to some mean and std values
__C.DATASET_NORM = True

# Input data size
__C.SPATIAL_INPUT_SIZE = (112, 112)
__C.CHANNEL_INPUT_SIZE = 3
__C.TEMPORAL_INPUT_SIZE = 20

# FC Size
__C.NUM_FC = 4096

# Set parameters for snapshot and verbose routines
__C.MODEL_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
__C.SNAPSHOT = True
__C.SNAPSHOT_INTERVAL = 10
__C.VERBOSE = True
__C.VERBOSE_INTERVAL = 10
__C.VALID_INTERVAL = 1

# Network Architecture
__C.NET_ARCH = ('resnet', )[0]

# Pre-trained network
__C.PRETRAINED_MODE = (None, 'ImageNet', 'ResNet_ST', 'Custom')[2]

# Path to the custom pre-trained network
__C.CUS_PT_PATH = None
if __C.PRETRAINED_MODE == 'ResNet_ST':
    __C.CUS_PT_PATH = {
        'image': os.path.join(__C.EXPERIMENT_DIR, 'pretrained_nets',
                              'ucf101-img-resnet50-split{}-dr0.mat'.format(__C.SPLIT_NO)),
        'flow': os.path.join(__C.EXPERIMENT_DIR, 'pretrained_nets',
                             'ucf101-flow-resnet50-split{}-dr0.8.mat'.format(__C.SPLIT_NO))
    }
elif __C.PRETRAINED_MODE == 'Custom':
    __C.CUS_PT_PATH = {
        'image': os.path.join(__C.EXPERIMENT_DIR, 'snapshot', '20181010_124618_219443', 'spt_079.pt'),
        'flow': os.path.join(__C.EXPERIMENT_DIR, 'snapshot', '20181010_124618_219443', 'tmp_079.pt')
    }

# =============================
# Spatiotemporal ResNet options
# =============================
__C.RST = edict()

__C.RST.CROSS_STREAM_MOD_LAYER = 2
__C.RST.TEMPORAL_CONVOLUTION_LAYER = 1
__C.RST.INIT_TEMPORAL_STRATEGY = ('center', 'difference', 'average')[0]
__C.RST.FRAME_RANDOMIZATION = False
__C.RST.VALID_F25 = False
__C.RST.FRAME_SAMPLING_METHOD = ('uniform', 'temporal_stride', 'random', 'temporal_stride_random')[2]
__C.RST.NFRAMES_PER_VIDEO = 11
__C.RST.TEMPORAL_STRIDE = (5, 35)
__C.RST.LR_S_STREAM_MULT = 0.75

# ================
# Training options
# ================
if __C.TRAINING:
    __C.TRAIN = edict()

    # Images to use per minibatch
    __C.TRAIN.BATCH_SIZE = 16

    # Shuffle the dataset
    __C.TRAIN.SHUFFLE = True

    # Learning parameters are set below
    __C.TRAIN.LR = 1e-3
    __C.TRAIN.WEIGHT_DECAY = 1e-5
    __C.TRAIN.MOMENTUM = 0.90
    __C.TRAIN.NESTEROV = False
    __C.TRAIN.SCHEDULER_MODE = False
    __C.TRAIN.SCHEDULER_TYPE = ('step', 'step_restart', 'multi', 'lambda', 'plateau')[0]
    __C.TRAIN.SCHEDULER_MULTI_MILESTONE = [50, 75, 100]

# ================
# Validation options
# ================
if __C.VALIDATING:
    __C.VALID = edict()

    # Images to use per minibatch
    __C.VALID.BATCH_SIZE = __C.TRAIN.BATCH_SIZE

    # Shuffle the dataset
    __C.VALID.SHUFFLE = False
