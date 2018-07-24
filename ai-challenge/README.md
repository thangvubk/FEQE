# Perceptual Image Enhancement on Smartphones

### [[Challenge Website]](http://ai-benchmark.org), [[PIRM 2018]](https://www.pirm2018.org/)

<br/>

This repository provides a guideline and code to convert your pre-trained models into an appropriate submission format for [AI Challenge](http://ai-benchmark.org) organized in conjunction with ECCV 2018 Conference.<br/><br/>


#### 1. Prerequisites

- [TensorFlow (>=1.8)](https://www.tensorflow.org/install/)
- Python (3.5+) packages: [scipy](https://pypi.org/project/scipy/), [numpy](https://pypi.org/project/numpy/), [psutil](https://pypi.org/project/psutil/)<br/><br/>


#### 2. General model requirements

- Your model should be able to process images of <b>arbitrary resolution</b>
- It should require no more than <b>3.5GB of RAM</b> while processing HD-resolution ```[1280x720px]``` photos
- Maximum model size: <b>100MB</b>
- Should be saved as <b>Tensorflow .pb file</b></br></br>


#### 3. Model conversion and validation

Your model should be implemented as a function that takes one single input and produces one output:

- <b>Input:</b>&nbsp; Tensorflow 4-dimensional tensor of shape ```[batch_size, image_height, image_width, 3]```</br><sub>In the Super-Resolution task, input images are already <b>bicubically interpolated (x4)</b> and have the same size as target high-resolution photos</br></sub>
- <b>Output:</b>&nbsp; Same as input
- <b>Values:</b>&nbsp; The values of both original and processed images should lie in the <b>```interval [0, 1]```</b>

Here is a valid SRCNN function provided for you as a reference:</br>

<sub>

```bash
def srcnn(images):

    with tf.variable_scope("generator"):

        weights = {
          'w1': tf.Variable(tf.random_normal([9, 9, 3, 64], stddev=1e-3), name='w1'),
          'w2': tf.Variable(tf.random_normal([5, 5, 64, 32], stddev=1e-3), name='w2'),
          'w3': tf.Variable(tf.random_normal([5, 5, 32, 3], stddev=1e-3), name='w3')
        }

        biases = {
          'b1': tf.Variable(tf.zeros([64]), name='b1'),
          'b2': tf.Variable(tf.zeros([32]), name='b2'),
          'b3': tf.Variable(tf.zeros([1]), name='b3')
        }

        conv1 = tf.nn.relu(tf.nn.conv2d(images, weights['w1'], strides=[1,1,1,1], padding='SAME') + biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1,1,1,1], padding='SAME') + biases['b2'])
        conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1,1,1,1], padding='SAME') + biases['b3']

    return tf.nn.tanh(conv3) * 0.58 + 0.5
```
</sub>

To test and convert your pre-trained models, run the following scripts:

- <b>Track A,</b> Image Super-Resolution:&nbsp; <b>```evaluate_super_resolution.py```</b>
- <b>Track B,</b> Image Enhancement:&nbsp; <b>```evaluate_enhancement.py```</b>

You need to modify two lines in the headers of the above scripts:

```bash
from <model_file> import <your_model> as test_model
model_location = "path/to/your/saved/pre-trained/model"
```

Here, <b>```model_file.py```</b> should be a python file containing your model definition, <b>```your_model```</b> is the actual function that defines your model, and <b>```model_location```</b> points to your saved pre-trained model file.</br>

After running these scripts, they will:

1. save your model as ```model.pb``` file stored in ```models_pretrained/``` folder
2. compute PSNR/SSIM scores on a subset of validation images/patches
3. compute running time and <b>estimated</b> RAM consumption for HD-resolution images</br></br>

#### 4. Provided pre-trained models

Apart from the validation scripts, we also provide you several pre-trained models that can be restored and validated using the same scripts. In all cases, model architectures are defined in the ```models.py``` file.

<b>Super-resolution task:</b>

1. SRCNN, function: ```srcnn```, pre-trained model: ```models_pretrained/div2k_srcnn```
2. ResNet with one residual block, function: ```resnet_6_16```, pre-trained model: ```models_pretrained/div2k_resnet_6_16```
3. VGG-19, function: ```vgg_19```, pre-trained model: ```models_pretrained/div2k_vgg19_vdsr.ckpt```

<b>Image Enhancement task:</b>

1. SRCNN, function: ```srcnn```, pre-trained model: ```models_pretrained/dped_srcnn```
2. ResNet with 4 residual blocks, function: ```resnet_12_64```, pre-trained model: ```models_pretrained/dped_resnet_12_64```
3. ResNet with 2 residual blocks, function: ```resnet_8_32```, pre-trained model: ```models_pretrained/dped_resnet_8_32```</br></br>

#### 5. Team registration and model submission

To register your team, send an email to <b>```ai.mobile.challenge@gmail.com```</b> with the following information:

```bash
Email Subject:  AI Mobile Challenge Registration

Email Text:     Team Name
                Team Member 1 (Name, Surname, Affiliation)
                Team Member 2 (Name, Surname, Affiliation)
                ....
```

To validate your model, send an email indicating the ```track```, ```team id``` and the corresponding ```model.pb``` file:

```bash
Email Subject:  [Track X] [Team ID] [Team Name] Submission

Email Text:     Link to model.pb file
```
You are allowed to send up to ```2 submissions per day``` for each track. The leaderboard will show the results of your last successful submission. Please make sure that the results provided by our validation scripts are meaningful before sending your submission files.</br></br>


#### 6. Scoring formulas

The performance of your solution will be assessed based on three metrics: its speed compared to a baseline network, its fidelity score measured by PSNR, and its perceptual score computed based on MS-SSIM metric. Since PSNR and SSIM scores do not always objectively reflect image quality, during the test phase we will conduct a user study where your final submissions will be evaluated by a large number of people, and the resulting MOS Scores will replace MS-SSIM results. The total score of your solution will be calculated as a weighted average of the previous scores:

```bash
TotalScore = α * (PSNR_solution - PSNR_baseline) + β * (SSIM_solution - SSIM_baseline) + γ * min(Time_baseline / Time_solution, 4) 
```

We will use three different validation tracks for evaluating your results. Score A is giving preference to solution with the highest fidelity (PSNR) score, score B is aimed at the solution providing the best visual results (MS-SSIM/MOS scores), and score C is targeted at the best balance between the speed and perceptual/quantitative performance. For each track, we will use the above scoring formula but with different coefficients:

<b>Track A (Super-Resolution):</b>

- ```PSNR_baseline``` = 26.5, ```SSIM_baseline``` = 0.94
- (```α```, ```β```, ```γ```): &nbsp; score A - (4, 100, 1); &nbsp; score B - (1, 400, 1); &nbsp; score C - (2, 200, 1.5)

<b>Track B (Image Enhancement):</b>

- ```PSNR_baseline``` = 21, ```SSIM_baseline``` = 0.9
- (```α```, ```β```, ```γ```): &nbsp; score A - (4, 100, 2); &nbsp; score B - (1, 400, 2); &nbsp; score C - (2, 200, 2.9)</br></br>


#### 7. Other remarks

- Note that the provided code is used only for preliminary model validation, while all final numbers will be obtained by us by testing all submissions on the test parts of the datasets (accuracy) and on the same hardware (speed)

- To check the above RAM requirements, we will run your submissions on a GPU with 3.5GB of RAM.<br/> In case this won't be enough for your model, it will be disqualified from the final validation stage
