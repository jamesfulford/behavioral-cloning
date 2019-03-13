# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model begins with a normalization layer which crops incoming images, then
mean-centers the images about 0, then normalizes values to be about -0.5, 0.5.
The latter two practices are known to improve the outcomes of optmization
algorithms.

My model mimics the NVIDIA self-driving car architecture to an extent, in that
the number of filters and some dimensions for the beginning CNN layers are
equal or similar. I have 5 convolutional layers, with RELU activations are used
in each convolutional layer to introduce non-linearity. After each layer is
a dropout layer with a 10% drop probability. I have heard that low dropouts can
have some value in convolutions (forces NN to consider other connections), but
a high drop probability would lose a lot of data each time the convolution is
applied to the input image, potentially dropping too much data (which, in our
case, is comparable to learning to drive without glasses).

After each of the 4 dense fully-connected layers, a 50% drop dropout is applied.
This is to reduce overfitting and reliance on particular inputs.

The output layer consists of 1 node. There is no dropout applied to the output
layer, as that would defeat the purpose of the model.

(Doing so would cause the backpropagation mechanism to 'punish' the model when
behind the final dropout layer the model was right.)

#### 2. Attempts to reduce overfitting in the model

As previously mentioned, dropout layers were thouroughly employed. This
differentiates this architecture from the Nvidia architecture previously
mentioned.

The model was trained on a dataset separate from the validation set to reduce
the liklihood of overfitting during training. The model was tested by running it
through the simulator and ensuring that the vehicle could stay on the track.

Please see video.mp4 for evidence of the success of this run.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I opted to use the training data provided instead of generating my own.

(This was for technical reasons - lag with simulator made manual driving
near impossible to do well.)

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to adopt a known
architecture and improve upon it.

Earlier iterations of my training script are in earlier commits. Starting with
the simplest network possible (just output layer), I built the code surrounding
the model and found the resulting model to be lacking in terms of mean squared
error and in terms of performance. It seemed like it didn't know how to turn
right. (This prompted me to include the flipped images as well)

Over time, I moved to mimic the classic LeNet hidden-layer architecture (outputs
and inputs differed). Thinking was that the best way for lower layers to detect
the edges of the road would be to use a convolutional layer so detection would
not vary much throughout the image. Several dense layers would be required to
properly interpret these detections, so several of those formed the top layer.

My final model technically mimics the Nvidia structure, but is rather similar in
the convolutional-then-dense approach. I took out the max pooling stages from
LeNet, as pooling is a trick to reduce parameters by essentially discarding
data, which is not necessary with an abundance of data. Dropout layers help
with generalization of a model, so I introduced those as well. When I saw that
the general size of the LeNet model caused it to underfit (high mean squared
error, more epochs did not yield improvements), adding more layers to match the
Nvidia model was only wise.

At the end of the process, the vehicle is able to drive autonomously around the
track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:


Convolution2D(24, (5, 5), strides=(2, 2), activation="relu")
Dropout(0.1)
Convolution2D(36, (5, 5), strides=(2, 2), activation="relu")
Dropout(0.1)
Convolution2D(48, (5, 5), strides=(2, 2), activation="relu")
Dropout(0.1)
Convolution2D(64, (3, 3), activation="relu")
Dropout(0.1)
Convolution2D(64, (3, 3), activation="relu")
Dropout(0.1)

Flatten()

Dense(100)
Dropout(0.5)
Dense(50)
Dropout(0.5)
Dense(10)
Dropout(0.5)
Dense(1))  # output laye

#### 3. Creation of the Training Set & Training Process

As previously stated, I used the provided data set for training. I did not
generate my own data. (Would have loved to, but data quality would have been
poor due to technical restraints.)

Before training, I provided my model with the original and flipped (augmented)
images (original and negated angles, respectively) for the left, center, and
right cameras on the vehicle. As previously mentioned, I preprocessed this data
by cropping, mean-shifting, and normalizing.

For each epoch and batch, I shuffled the training and validation data. This
helps avoid the optimizer from falling into the same local minima or issues
with every training. Stochasticity helps the process.

I used this training data for training the model. The validation set helped
determine if the model was over or under fitting. The validation loss value
flattened out at about 10 epochs.

The adam optimizer tunes the learning rate for me, so I made no adjustments to
that parameter.
