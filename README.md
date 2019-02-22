# FEQE
Official implementation for [Fast and Efficient Image Quality Enhancement via Desubpixel Convolutional Neural Networks](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Vu_Fast_and_Efficient_Image_Quality_Enhancement_via_Desubpixel_Convolutional_Neural_ECCVW_2018_paper.pdf), ECCV workshop 2018
## Citation
Please cite our project if it is helpful for your research
```
@InProceedings{Vu_2018_ECCV_Workshops},
author = {Vu, Thang and Van Nguyen, Cao and Pham, Trung X. and Luu, Tung M. and Yoo, Chang D.},
title = {Fast and Efficient Image Quality Enhancement via Desubpixel Convolutional Neural Networks},
booktitle = {The European Conference on Computer Vision (ECCV) Workshops},
month = {September},
year = {2018}
}
```

<p align="center">
    <img src="https://github.com/thangvubk/FEQE/blob/master/docs/P_results.PNG">
</p> 
<p align="center">
    Comparison of proposed FEQE with other state-of-the-art super-resolution and enhancement methods
</p>
<p align="center">
    <img src="https://github.com/thangvubk/FEQE/blob/master/docs/net.PNG">
</p> 
<p align="center">
    Network architecture
</p>
<p align="center">
    <img src="https://github.com/thangvubk/FEQE/blob/master/docs/sub-des.PNG">
</p> 
<p align="center">
    Proposed desubpixel
</p>



## PIRM 2018 challenge results (super-resolution on mobile devices task)

<p align="center">
    <img src="https://github.com/thangvubk/FEQE/blob/master/docs/PIRM.PNG">
</p> 
<p align="center">
    TEAM_ALEX placed the first in overall benchmark score. Refer to <a href="http://ai-benchmark.com/challenge.html">PIRM 2018</a> for details.
</p>

## Dependencies
- 1 Nvidia GPU (4h training on Titan Xp)
- ``Python3``
- ``tensorflow 1.10+``
- ``tensorlayer 1.9+``
- ``tensorboardX 1.4+``

## Download datasets, models, and results
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
- Download [paper results](https://drive.google.com/file/d/1KMpp_6Rp4XmRCQxdRIRpC1XdBLS4WrcS/view?usp=sharing) (images) of the test datasets

## Project layout (recommended)
```
FEQE/
├── checkpoint
│   ├── FEQE
│   └── FEQE-P
├── data
│   ├── DIV2K_train_HR
│   ├── DIV2K_valid_HR_9
│   └── test_benchmark
├── docs
├── model
├── results
└── vgg_pretrained
    └── imagenet-vgg-verydeep-19.mat
```
## Quick start
1. Download [test only](https://drive.google.com/file/d/1bch29fFj5t7IwoNjceuK8lFM6-ivwrP5/view?usp=sharing) dataset dataset and put into ``data/`` directory
2. Download [pretrained models](https://drive.google.com/file/d/1ok7-Y0Ldbyi9Ii0Cm3wTzMx8vPvt6zIR/view?usp=sharing) and put into ``checkpoint/`` directory
3. Run ``python test.py --dataset <DATASET_NAME>``
4. Results will be saved into ``results/`` directory

## Training
1. Download [train+val+test](https://drive.google.com/file/d/1dyL6KxaBI8Aq7E3AnuIK-RODkqXUAfcF/view?usp=sharing) datasets dataset and put into ``data/`` directory
2. Download [pretrained VGG](https://drive.google.com/file/d/1KLZOwxW0KpQxRwwUepVYEi147UG9IRIx/view?usp=sharing) and put into ``vgg_pretrained/`` directory
3. Pretrain with MSE loss on scale 2: ``python train.py --checkpoint checkpoint/mse_s2 --alpha_vgg 0 --scale 2 --phase pretrain``
4. Finetune with MSE loss on scale 4 (FEQE-P): ``python main.py --checkpoint checkpoint/mse_s4 --alpha_vgg 0 --pretrained_model checkpoint_test/mse_s2/model.ckpt``
5. Finetune with full loss on scale 4: ``python main.py --checkpoint checkpoint/full_s4 ---pretrained_model checkpoint_test/mse_s4/model.ckpt``
6. All Models with be saved into ``checkpoint/`` direcory

## Visualization
1. Start tensorboard: ``tensorboard --logdir checkpoint``
2. Enter: ``YOUR_IP:6006`` to your web browser.
3. Result ranges should be similar to:

![Tensorboard](https://github.com/thangvubk/FEQE/blob/master/docs/tensorboard.gif)

## Comprehensive testing
1. Test FEQE model (defaults): follow [Quick start](#quick-start)
2. Test FEQE-P model: ``python test.py --dataset <DATASET> --model_path <FEQE-P path>``
3. Test perceptual quality: refer to [PIRM validation code](https://github.com/roimehrez/PIRM2018)

## Quantitative and Qualitative results
<p align="center">
    <img src="https://github.com/thangvubk/FEQE/blob/master/docs/quan.PNG">
</p> 
<p align="center">
    PSNR/SSIM/Perceptual-Index comparison. Red indicates the best results
</p>

<p align="center">
    <img src="https://github.com/thangvubk/FEQE/blob/master/docs/time.PNG">
</p> 
<p align="center">
    Running time comparison. Red indicates the best results
</p>

<p align="center">
    <img src="https://github.com/thangvubk/FEQE/blob/master/docs/qual.PNG">
</p> 
<p align="center">
    Qualitative comparison
</p>
