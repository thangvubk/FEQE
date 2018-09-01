# FEQE
Official implementation for Fast and Efficient Image Quality Enhancement via Desubpixel Convolutional Neural Networks
## Dependencies
- 1 Nvidia GPU (4h training on Titan Xp)
- ``Python3``
- ``tensorflow 1.10+``
- ``tensorlayer 1.9+``
- ``tensorboardX 1.4+``

## Datasets, models, and results
### Dataset
- Train: DIV2K (800 2K-resolution images)
- Valid: DIV2K (9 val images)
- Test: Set5, Set14, B100, Urban100
- Download [train+val+test](https://drive.google.com/file/d/1dyL6KxaBI8Aq7E3AnuIK-RODkqXUAfcF/view?usp=sharing) datasets
- Download [test only](https://drive.google.com/file/d/1bch29fFj5t7IwoNjceuK8lFM6-ivwrP5/view?usp=sharing) dataset
    
### Pretrained models
- Download [pretrained models]() including 1 PSNR-optimized model and 1 perception-optimized model
    
### Paper results
- Download [paper results]() in images of the test datasets

## Quick start
- Download test dataset and put into ``data/origin/`` directory
- Download pretrained model and put into ``check_point`` directory
- Run ``python test.py --dataset <DATASET_NAME>``
- Results will be saved into ``results/`` directory

## Training
- Download train+val+test dataset and put into ``data/origin directory``
- Pretrain with L1 loss: ``python train.py --phase pretrain --learning_rate 1e-4``
- Finetune on pretrained model with GAN: ``python train.py``
- Models with be saved into ``check_point/`` direcory

## Visualization
- Start tensorboard: ``tensorboard --logdir check_point``
- Enter: ``YOUR_IP:6006`` to your web browser.
- Tensorboard when finetuning on pretrained model should be similar to:

![Tensorboard](https://github.com/thangvubk/PESR/blob/master/docs/tensorboard.PNG)

![Tensorboard_imgs](https://github.com/thangvubk/PESR/blob/master/docs/tensorboard_img.PNG)

## Comprehensive testing
- Test perceptual model: follow [Quick start](#quick-start)
- Interpolate between perceptual model and PSNR model: ``python test.py --dataset <DATASET> --alpha <ALPHA>``  (with alpha being perceptual weight)
- Test perceptual quality: refer to [PIRM validation code](https://github.com/roimehrez/PIRM2018)

## Quantitative and Qualitative results
<p> RED and BLUE indicate best and second best respectively.</p>
<p align="center">
    <img src="https://github.com/thangvubk/PESR/blob/master/docs/quantitative.PNG">
    <img width="800" height="1200", src="https://github.com/thangvubk/PESR/blob/master/docs/qualitative.PNG">
</p>

## References
- [EDSR-pytorch](https://github.com/thstkdgus35/EDSR-PyTorch)
- [Relativistic-GAN](https://github.com/AlexiaJM/RelativisticGAN)
