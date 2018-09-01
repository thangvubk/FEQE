import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
from model import *
from utils import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse
import pdb


parser = argparse.ArgumentParser()

# data
parser.add_argument('--train_path', type=str, default='./data/DIV2K_train_HR')
parser.add_argument('--valid_path', type=str, default='./data/DIV2K_valid_HR_9')
parser.add_argument('--scale', type=int, default=4,
                    help='downsample scale')
# Train ops
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--decay_every', type=int, default=1200,
                    help='epoch to decay learning rate')
parser.add_argument('--phase', type=str, default='train',
                    help='train or pretrain')

# Model
parser.add_argument('--downsample_type', type=str, default='desubpixel')
parser.add_argument('--upsample_type', type=str, default='subpixel')
parser.add_argument('--conv_type', type=str, default='default')
parser.add_argument('--body_type', type=str, default='resnet')
parser.add_argument('--n_feats', type=int, default=16,
                    help='number of convolution feats')
parser.add_argument('--n_blocks', type=int, default=20,
                    help='number of residual block if body_type=resnet')
parser.add_argument('--n_groups', type=int, default=0,
                    help='number of residual group if body_type=res_in_res')
parser.add_argument('--n_convs', type=int, default=0,
                    help='number of conv layers if body_type=conv')
parser.add_argument('--n_squeezes', type=int, default=0,
                    help='number of squeeze blocks if body_type=squeeze')
parser.add_argument('--pretrained_model', type=str, default='', 
                    help='if specified, fine tune on pretrained model')

# Loss
parser.add_argument('--alpha_mse', type=float, default=1)
parser.add_argument('--alpha_vgg', type=float, default=1e-4)
parser.add_argument('--vgg_dir', type=str, default='vgg_pretrained/imagenet-vgg-verydeep-19.mat',
                    help='vvg model for vgg loss')
# Logging
parser.add_argument('--checkpoint', type=str, default='checkpoint',
                    help='save logs and models')
parser.add_argument('--eval_every', type=int, default=20)
args = parser.parse_args()

print('############################################################')
print('# Image Super Resolution - PIRM2018 - TEAM_ALEX            #')
print('# Implemented by Thang Vu, thangvubk@gmail.com             #')
print('############################################################')
print('')
print('_____________YOUR SETTINGS_____________')
for arg in vars(args):
    print("%20s: %s" %(str(arg), str(getattr(args, arg))))
print('')

def train():
    ## create folders to save trained model
    tl.files.exists_or_mkdir(args.checkpoint)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_npy = os.path.join(args.train_path, 'train_hr.npy')
    valid_hr_npy = os.path.join(args.valid_path, 'valid_hr.npy')
    valid_lr_npy = os.path.join(args.valid_path, 'X{}_valid_lr.npy'.format(args.scale))

    if os.path.exists(train_hr_npy) and os.path.exists(valid_hr_npy) and os.path.exists(valid_lr_npy):
        print('Loading data...')
        train_hr_imgs = np.load(train_hr_npy)
        valid_hr_imgs = np.load(valid_hr_npy)
        valid_lr_imgs = np.load(valid_lr_npy)
    else:
        print('Data bin is not created. Creating data bin...')
        train_hr_img_list = sorted(tl.files.load_file_list(path=args.train_path, regx='.*.png', printable=False))
        valid_hr_img_list = sorted(tl.files.load_file_list(path=args.valid_path, regx='.*.png', printable=False))
        train_hr_imgs = np.array(tl.vis.read_images(train_hr_img_list, path=args.train_path, n_threads=32))
        valid_hr_imgs = np.array(tl.vis.read_images(valid_hr_img_list, path=args.valid_path, n_threads=16))
        valid_lr_imgs = tl.prepro.threading_data(valid_hr_imgs, fn=downsample_fn, scale=args.scale)
        np.save(train_hr_npy, train_hr_imgs)
        np.save(valid_hr_npy, valid_hr_imgs)
        np.save(valid_lr_npy, valid_lr_imgs)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_lr = tf.placeholder('float32', [None, None, None, 3], name='t_lr')
    t_hr = tf.placeholder('float32', [None, None, None, 3], name='t_hr')

    # some options are mutual exclusive, check model for detail
    opt = {
        'n_feats': args.n_feats,
        'n_blocks': args.n_blocks,
        'n_groups': args.n_groups,
        'n_convs': args.n_convs,
        'n_squeezes': args.n_squeezes,
        'downsample_type': args.downsample_type,
        'upsample_type': args.upsample_type,
        'conv_type': args.conv_type,
        'body_type': args.body_type,
        'scale': args.scale
    }
    
    print('Loading model...')
    t_sr = FEQE(t_lr, opt)

    ## Load VGG net
    vgg_dir = args.vgg_dir
    if not os.path.exists(vgg_dir):
        print('Not found vgg19 pretrained.')
        return

    CONTENT_LAYER = 'relu5_4'
    with tf.variable_scope('VGG'):
        sr_vgg = vgg19(vgg_dir, preprocess(t_sr * 255))
        hr_vgg = vgg19(vgg_dir, preprocess(t_hr * 255))

    # Count number of parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Total number of trainable parameters: %d" % total_parameters)

    ####========================== Loss function ==========================###
    #mse_loss = args.alpha_mse*tl.cost.absolute_difference_error(t_sr, t_hr, is_mean=True)
    mse_loss = args.alpha_mse*tl.cost.mean_squared_error(t_sr, t_hr, is_mean=True)

    with tf.variable_scope('VGG_loss'):
        vgg_loss = args.alpha_vgg*tl.cost.mean_squared_error(sr_vgg[CONTENT_LAYER], hr_vgg[CONTENT_LAYER], is_mean=True) if args.alpha_vgg != 0 else tf.constant(0.0)

    g_loss = vgg_loss  + mse_loss

    #==========================Training ops==================================
    g_vars = tl.layers.get_variables_with_name('Generator', True, True)
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(args.learning_rate, trainable=False)
    g_optim = tf.train.AdamOptimizer(lr_v).minimize(g_loss, var_list=g_vars)


    #===========================PSNR and SSIM================================
    t_psnr = tf.image.psnr(t_sr, t_hr, max_val=1.0)
    t_ssim = tf.image.ssim_multiscale(t_sr, t_hr, max_val=1.0)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    if args.phase == 'pretrain':
        body_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Generator/body'))
    else:
        global_saver = tf.train.Saver()
        if args.pretrained_model != '':
            body_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Generator/body')) #TODO change to Generator/body
            body_saver.restore(sess, args.pretrained_model)

    ###=========================Tensorboard=============================###
    writer = SummaryWriter(os.path.join(args.checkpoint, 'result'))
    tf.summary.FileWriter(os.path.join(args.checkpoint, 'graph'), sess.graph)
    best_score, best_epoch = -1, -1

    ###========================= Training ====================###
    for epoch in range(1, args.n_epochs + 1):
        ## update learning rate
        if epoch % args.decay_every == 0:
            new_lr_decay = 0.5**(epoch // args.decay_every)
            sess.run(tf.assign(lr_v, args.learning_rate * new_lr_decay))
            log = " ** new learning rate: %f " % (args.learning_rate * new_lr_decay)
            print(log)

        # ids to shuffle batches
        ids = np.random.permutation(len(train_hr_imgs))

        epoch_time = time.time()
        num_batches = len(train_hr_imgs)//args.batch_size
        # running_loss = 0
        total_vgg_loss, total_mse_loss, total_g_loss = 0, 0, 0
        running_loss = np.zeros(3)

        for i in tqdm(range(num_batches)):
            hr = tl.prepro.threading_data(train_hr_imgs[ids[i*args.batch_size:(i+1)*args.batch_size]],
                                          fn=crop_sub_imgs_fn, is_random=True)
            lr = tl.prepro.threading_data(hr, fn=downsample_fn, scale=args.scale)
            [lr, hr] = normalize([lr, hr])

            ## update G
            errG, errL, errV, _ = sess.run([g_loss, mse_loss, vgg_loss, g_optim], {t_lr: lr, t_hr: hr})

            running_loss += [errG, errL, errV]
        
        avr_loss = running_loss/num_batches
        log = "[*] Epoch: [%2d/%2d], g_loss: %.6f, mse_loss: %.6f, vgg_loss: %.6f" % \
              (epoch, args.n_epochs, avr_loss[0], avr_loss[1], avr_loss[2])
        print(log)

        writer.add_scalar('G_total_Loss', avr_loss[0], epoch)
        writer.add_scalar('MSE_Loss', avr_loss[1], epoch)
        writer.add_scalar('VGG_Loss', avr_loss[2], epoch)

        #=============Valdating==================#
        running_loss = 0
        if (epoch % args.eval_every == 0):
            print('Validating...')
            val_psnr = 0
            val_ssim = 0
            score = 0
            for i in tqdm(range(len(valid_hr_imgs))):

                hr = valid_hr_imgs[i]
                lr = valid_lr_imgs[i]

                [lr, hr] = normalize([lr, hr])

                hr_ex = np.expand_dims(hr, axis=0)
                lr_ex = np.expand_dims(lr, axis=0)

                psnr, ssim,  sr_ex = sess.run([t_psnr, t_ssim, t_sr], {t_lr: lr_ex, t_hr: hr_ex})
                sr = np.squeeze(sr_ex)
                
                #pdb.set_trace()
                update_tensorboard(epoch, writer, i, lr, sr, hr)

                val_psnr += psnr
                val_ssim += ssim
                loss = 0
                running_loss += loss

                # score referred to https://github.com/aiff22/ai-challenge
                score += (psnr-26.5) + (ssim-0.94)*100

            #global_saver.save(sess, os.path.join(args.checkpoint, 'model_{}.ckpt'.format(epoch)))

            val_psnr = val_psnr/len(valid_hr_imgs)
            val_ssim = val_ssim/len(valid_hr_imgs)
            score = score/len(valid_hr_imgs)
            avr_loss = running_loss/len(valid_hr_imgs)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                print('Saving new best model')

                if args.phase == 'pretrain':
                    body_saver.save(sess, os.path.join(args.checkpoint, 'body.ckpt'))
                else:
                    global_saver.save(sess, os.path.join(args.checkpoint, 'model.ckpt'))
            print('Validate score: %.4f. Best: %.4f at epoch %d' %(score, best_score, best_epoch))
            writer.add_scalar('Validate PSNR', val_psnr, epoch)
            writer.add_scalar('Validate SSIM', val_ssim, epoch)
            writer.add_scalar('Validate score', score, epoch)
            writer.add_scalar('Best val score', best_score, epoch)
            writer.add_scalar('Validation Loss', avr_loss, epoch)

if __name__ == '__main__':
    train()

