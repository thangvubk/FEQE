from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 500
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 300

## train set location
config.TRAIN.hr_img_path = 'data/DIV2K_train_HR/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'data/DIV2K_valid_HR/'

## checkpoint
config.checkpoint = 'checkpoint/SRGAN'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
