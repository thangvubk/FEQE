from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 50 #16
config.TRAIN.lr_init = 5e-4 #1e-4
config.TRAIN.beta1 = 0.9

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 500
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 300

## train set location
#config.TRAIN.hr_img_path = 'data_enhance/dped/iphone/training_data/iphone'
config.TRAIN.hq_img_path = 'data_enhance/dped/iphone/training_data/canon'
config.TRAIN.lq_img_path = 'data_enhance/dped/iphone/training_data/iphone'

config.VALID = edict()
## test set location
config.VALID.hq_img_path = 'data_enhance/dped/iphone/test_data/patches/100-canon'
config.VALID.lq_img_path = 'data_enhance/dped/iphone/test_data/patches/100-iphone'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
