# FEQE
Official implementation for Fast and Efficient Image Quality Enhancement via Desubpixel Convolutional Neural Networks
## Dependencies
- 1 Nvidia GPU (4h training on Titan Xp)
- ``Python3``
- ``tensorflow 1.10+``
- ``tensorlayer 1.9+``
- ``tensorboardX 1.4+``

## Download Datasets, models, and results
### Dataset
- Train: DIV2K (800 2K-resolution images)
- Valid: DIV2K (9 val images)
- Test: Set5, Set14, B100, Urban100
- Download [train+val+test](https://drive.google.com/file/d/1dyL6KxaBI8Aq7E3AnuIK-RODkqXUAfcF/view?usp=sharing) datasets
- Download [test only](https://drive.google.com/file/d/1bch29fFj5t7IwoNjceuK8lFM6-ivwrP5/view?usp=sharing) dataset
    
### Pretrained models
- Download [pretrained models](https://drive.google.com/file/d/1ok7-Y0Ldbyi9Ii0Cm3wTzMx8vPvt6zIR/view?usp=sharing) including 1 PSNR-optimized model and 1 perception-optimized model
- Download [pretrained VGG](https://drive.google.com/file/d/1KLZOwxW0KpQxRwwUepVYEi147UG9IRIx/view?usp=sharing) used for VGG loss
    
### Paper results
- Download [paper results]() in images of the test datasets

## Quick start
- Download [test only](https://drive.google.com/file/d/1bch29fFj5t7IwoNjceuK8lFM6-ivwrP5/view?usp=sharing) dataset dataset and put into ``data/origin/`` directory
- Download [pretrained models](https://drive.google.com/file/d/1ok7-Y0Ldbyi9Ii0Cm3wTzMx8vPvt6zIR/view?usp=sharing) and put into ``checkpoint/`` directory
- Run ``python test.py --dataset <DATASET_NAME>``
- Results will be saved into ``results/`` directory

## Training
- Download [train+val+test](https://drive.google.com/file/d/1dyL6KxaBI8Aq7E3AnuIK-RODkqXUAfcF/view?usp=sharing) datasets dataset and put into ``data/origin/`` directory
- Download [pretrained VGG](https://drive.google.com/file/d/1KLZOwxW0KpQxRwwUepVYEi147UG9IRIx/view?usp=sharing) and put into ``pretrained_vgg/`` directory
- Pretrain with MSE loss on scale 2: ``python train.py --checkpoint checkpoint/mse_s2 --alpha_vgg 0 --scale 2 --phase pretrain``
- Finetune with MSE loss on scale 4 (FEQE-P): ``python main.py --checkpoint checkpoint/mse_s4 --alpha_vgg 0 --pretrained_model checkpoint_test/mse_s2/model.ckpt``
- Finetune with full loss on scale 4: ``python main.py --checkpoint checkpoint/full_s4 ---pretrained_model checkpoint_test/mse_s4/model.ckpt``
- All Models with be saved into ``check_point/`` direcory

## Visualization
- Start tensorboard: ``tensorboard --logdir check_point``
- Enter: ``YOUR_IP:6006`` to your web browser.
- Result ranges should be similar to:

![Tensorboard](https://github.com/thangvubk/PESR/blob/master/docs/tensorboard.PNG)

## Comprehensive testing
- Test FEQE model (defaults): follow [Quick start](#quick-start)
- Test FEQE-P model: ``python test.py --dataset <DATASET> --model_path <FEQE-P path>``
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
