from __future__ import print_function
import tensorflow as tf
from scipy import misc
import numpy as np
import utils
import os
import argparse
import sys
from tensorflow.python.client import timeline

sys.path.append('../')

# Choose GPU manually in code
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

## --------- Change test parameters below -----------

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint', type=str, default='checkpoint')
parser.add_argument('--sample_type', type=str, default='subpixel')
parser.add_argument('--conv_type', type=str, default='default')
parser.add_argument('--body_type', type=str, default='resnet')
parser.add_argument('--n_feats', type=int, default=16)
parser.add_argument('--n_blocks', type=int, default=32)
parser.add_argument('--n_groups', type=int, default=0)
parser.add_argument('--n_convs', type=int, default=0)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--eval_every', type=int, default=20)
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--train_path', type=str, default='./data/DIV2K_train_HR')
parser.add_argument('--valid_path', type=str, default='./data/DIV2K_valid_HR_1')
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--model', type=str, default='train')
args = parser.parse_args()

from model import SRGAN_g as test_model              # import your model definition as "test_model"
model_location = "../checkpoint/" + args.model + "/model.ckpt"    # specify the location of your saved pre-trained model (ckpt file)

print('Evaluate model: ', model_location)
compute_PSNR_SSIM = False
compute_running_time = True

if __name__ == "__main__":

    # print("\n-------------------------------------\n")
    # print("Image Super-resolution task\n")
    #
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # np.warnings.filterwarnings('ignore')
    #
    # ###############################################################
    # #  1 Produce .pb model file that will be used for validation  #
    # ###############################################################
    # opt = {
    #     'n_feats': args.n_feats,
    #     'n_blocks': args.n_blocks,
    #     'n_groups': args.n_groups,
    #     'n_convs': args.n_convs,
    #     'sample_type': args.sample_type,
    #     'conv_type': args.conv_type,
    #     'body_type': args.body_type,
    #     'scale': args.scale
    # }
    #
    # print("Saving pre-trained model as .pb file")
    #
    # g = tf.Graph()
    # with g.as_default(), tf.Session() as sess:
    #
    #     image_ = tf.placeholder(tf.float32, shape=(1, None, None, 3), name="input")
    #     out_ = tf.identity(test_model(image_, opt), name="output")
    #
    #     saver = tf.train.Saver()
    #     saver.restore(sess, model_location)
    #
    #     output_graph_def = tf.graph_util.convert_variables_to_constants(
    #         sess, g.as_graph_def(), "input,output".split(",")
    #     )
    #
    #     tf.train.write_graph(output_graph_def, 'models_converted', 'model.pb', as_text=False)
    #
    # print("Model was successfully saved!")
    # print("\n-------------------------------------\n")
    # sess.close()
    #
    #
    # if compute_PSNR_SSIM:
    #
    #     #######################################
    #     #  2 Computing PSNR / MS-SSIM scores  #
    #     #######################################
    #
    #     tf.reset_default_graph()
    #     config = None
    #
    #     with tf.Session(config=config) as sess:
    #
    #         print("\rLoading pre-trained model")
    #
    #         with tf.gfile.FastGFile("models_converted/model.pb", 'rb') as f:
    #
    #             graph_def = tf.GraphDef()
    #             graph_def.ParseFromString(f.read())
    #             tf.import_graph_def(graph_def, name='')
    #
    #             x_ = sess.graph.get_tensor_by_name('input:0')
    #             out_ = sess.graph.get_tensor_by_name('output:0')
    #
    #         y_ = tf.placeholder(tf.float32, [1, None, None, 3])
    #         #h_ = tf.placeholder(tf.int32)
    #         #w_ = tf.placeholder(tf.int32)
    #
    #         # Remove boundaries (16px) from the produced and target images
    #
    #         #output_crop_ = tf.clip_by_value(tf.image.crop_to_bounding_box(out_, 16, 16, h_, w_), 0.0, 1.0)
    #         #target_crop_ = tf.clip_by_value(tf.image.crop_to_bounding_box(y_, 16, 16, h_, w_), 0.0, 1.0)
    #
    #         psnr_ = tf.image.psnr(out_, y_, max_val=1.0)
    #         ssim_ = tf.image.ssim_multiscale(out_, y_, max_val=1.0)
    #
    #         print("Computing PSNR/SSIM scores....")
    #
    #         ssim_score = 0.0
    #         psnr_score = 0.0
    #         validation_images = os.listdir("div2k/original/")
    #         num_val_images = len(validation_images)
    #
    #         for j in range(num_val_images):
    #
    #             print("\rImage %d / %d" % (j + 1, num_val_images), end='')
    #             image = misc.imread("div2k/original/" + validation_images[j])
    #
    #             image_bicubic = misc.imresize(image, 0.25, interp="bicubic")
    #             image_bicubic = misc.imresize(image_bicubic, 4.0, interp="bicubic")
    #
    #             image_bicubic = np.float32(np.reshape(image_bicubic, [1, image_bicubic.shape[0], image_bicubic.shape[1], 3])) / 255
    #             image_target = np.float32(np.reshape(image, [1, image.shape[0], image.shape[1], 3])) / 255
    #
    #             h = image.shape[0] - 32
    #             w = image.shape[1] - 32
    #
    #             #[psnr, ssim] = sess.run([psnr_, ssim_], feed_dict={x_: image_bicubic, y_: image_target, h_: h, w_: w})
    #             [psnr, ssim] = sess.run([psnr_, ssim_], feed_dict={x_: image_bicubic, y_: image_target})
    #
    #             psnr_score += psnr / num_val_images
    #             ssim_score += ssim / num_val_images
    #
    #         print("\r\r\r")
    #         print("Scores | PSNR: %.4g, MS-SSIM: %.4g" % (psnr_score, ssim_score))
    #         print("\n-------------------------------------\n")
    #         sess.close()


    if compute_running_time:

        ##############################
        #  3 Computing running time  #
        ##############################

        print("Evaluating model speed")
        print("This can take a few minutes\n")

        tf.reset_default_graph()

        print("Testing pre-trained baseline SRCNN model")
        avg_time_baseline, max_ram = utils.compute_running_time_basedline("superres", "models_pretrained/div2k_srcnn.pb", "div2k/HD_res/")

        tf.reset_default_graph()

        print("Testing provided model")
        # avg_time_solution, max_ram = utils.compute_running_time("superres", "models_converted/model.pb", "div2k/HD_res/")
        avg_time_solution, max_ram = utils.compute_running_time("superres", "models_converted/quantized.pb",
                                                                "div2k/HD_res/")

        # Calculate number of parameters
        reader = tf.train.NewCheckpointReader(model_location)
        param_map = reader.get_variable_to_shape_map()
        total_count = 0
        for k, v in param_map.items():
            if 'Momentum' not in k and 'global_step' not in k:
                temp = np.prod(v)
                total_count += temp
                # print('%s: %s => %d' % (k, str(v), temp))

        print('Total Param Count: %d' % total_count)
        print("Baseline SRCNN time, ms: ", avg_time_baseline)
        print("Test model time, ms: ", avg_time_solution)
        print("Speedup ratio (baseline, ms / solution, ms): %.4f" % (float(avg_time_baseline) / avg_time_solution))
        print("Approximate RAM consumption (HD image): " + str(max_ram) + " MB")
