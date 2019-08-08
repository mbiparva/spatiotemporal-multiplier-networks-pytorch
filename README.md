# Spatiotemporal Multiplier Networks for Video Action Recognition in PyTorch

This is a PyTorch implementation of the "Spatiotemporal Multiplier Networks for Video Action Recognition" paper by Christoph Feichtenhofer, Axel Pinz, Richard P. Wildes published in CVPR 2017. The official code released by Christoph can be found [here](https://github.com/feichtenhofer/st-resnet).

## Pre-trained Base Networks
Please download the pre-trained base networks provided by the official repository [here](https://github.com/feichtenhofer/st-resnet#models-st-mulnet). The current implementatio uses ResNet-50, so make sure you choose the network snapshot that matches best your dataset (UCF-101), network architecture (ResNet-50), and the dataset split number correctly.
You need to copy the downloaded pre-trained networks in experiment/base_pretrained_nets/ directory to be found by the network module.

## Datasets
You can download the RGB and Optical Flow frames for both UCF-101 and HMDB-51 at the official repository [here](https://github.com/feichtenhofer/st-resnet#models-st-mulnet). You just need to extract the zip files in the dataset directory such that it respect the following directory hierarchy so then the provided dataloader can easily find directories of different categories.

### Directory Hierarchy
Please make sure the downloaded dataset folders and files sit according to the following structure:

```
dataset
|    | UCF101
|    |    | images
│    │    │    | ApplyEyeMakeup  
│    │    │    | ApplyLipstick  
│    │    │    | ...  
|    |    | flows
│    │    │    | ApplyEyeMakeup  
│    │    │    | ApplyLipstick  
│    │    │    | ...  
|    |    | annotations
|    |    |    | annot01.json
|    |    |    | annot02.json
|    |    |    | annot03.json
|    |    |    | ucf101_splits
|    |    |    |    | trainlist01
|    |    |    |    | trainlist02
|    |    |    |    | ....
```
### JSON Annotation Generation
You need to create the annotations of each training and test splits using the script provided in the lib/utils/json_ucf.py. They need to be placed in the annotation folder as described above.

### Training and Validation Configuration
All of the configuration hyperparameters are set in the lib/utils/config.py. If you want to change them, simply edit the file with the settings you would like to.





