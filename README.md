# **Behavioral Cloning** 

## Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[image1]: ./res/model_plot.png "Model Visualization"
[training1]: ./res/Figure_1.png "Training Visualization"
[input]: ./res/input_img.png "Normal Image"
[flipped]: ./res/input_flipped.png "Flipped Image"
[clipped]: ./res/input_clipped.png "Cropped Image"

## Rubric Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

model.py lines 92-114

My model starts with a cropping layer to remove the bottom where you can see the hood of the car (bottom 20px) and the top where the environment is and no additional information regarding the road (top 50px). Then the data is normalized in the model using a Keras lambda layer (model.py line 96).
The data is fed into my convolutional neural network followed by fully connected layers and Dropout layers(model.py 108-114).

My convolutional neural network (CNN) consists of 3 convolutional layers with 5x5 filter sizes with a stride of 2, followed by 2 convolutional layers with 3x3 filter size with a stride of 1 and depths between 24 and 64 (model.py lines 101-105).

The model includes RELU layers between each convolutional layer to introduce nonlinearity (model.py line 101-105).

After the CNN I flatten the input to 8448 inputs and forwarded to the first fully connected layer (FCL). Next up, two Droplayers with a dropping rate of 0.5, with a FCL inbetween. The model finishes with two FCLs ensuring that the predicted steering angle has an output size of 1.

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py lines 110 112). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 79-80). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 119).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving for two laps counter-clockwise and one lap clockwise. Using also the side cameras of the vehicle with a steering angle correction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start out with the architecture from [NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), proven to be able to self drive, and then empirically adapt the network. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is as described above in section(An appropriate model architecture has been employed).

Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

Then I repeated this process in the opposite direction to mitigate the left turn bias gained by using a closed track driving counter-clockwise.

To augment the data set, I also flipped images and angles thinking that this would prevent the model from overfitting and also increasing the training and validation sample size.

After the collection process, I had about 40000 number of data points.
I didn't preprocess the data as this is done by the first two layers in the model.

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
