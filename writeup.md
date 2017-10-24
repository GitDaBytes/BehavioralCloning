**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/pilotnet.png "Model Visualization"
[image2]: ./images/center_example.jpg "Center Road"
[image3]: ./images/correct1.jpg "Recovery Image"
[image4]: ./images/correct2.jpg "Recovery Image"
[image5]: ./images/correct3.jpg "Recovery Image"
[image6]: ./images/flip1.jpg "Normal Image"
[image7]: ./images/flip2.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 showing the automated car running the model to navigate track 1
* cloning.ipynb is a jupyter notebook that I used to tweak my model building and training process.

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I tried several model architectures, including a very basic non-convolutional model to start, then moving to LeNet 5 type net. I really wanted to try and build a network from scratch rather than take a pre-trained network. I had read about the NVIDIA network online, which documented the actual archnitecture they used in a real car, and thought it would make sense to try and reproduce that, knowing it worked "in the wild". 

The NVIDIA blog that documents their architecture can be seen here: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

While they do state the top level architecture of their model, they do not get specific as to the activations they use, or the training the model. As I knew it would be made to work, I decided to give it a try.

My model consists of a copping layer, a normalizational layer, three convolutional layers (5x5 kernel, 2x2 stride) two convolutional layers (3x3 kernel, 1x1 stride), followed by three fully connected layers and an output layer with a single neuron for the steering command. 

My input resoluton was different to the NVIDIA network (my images were larger). I played with different activations including tanh, sigmoid, relu, but after a lot of reading, I found several papers stating that elu activations were proving quite successful as a replacement for relu as they can stop individual neurons from "dying out" during training. I therefore tried elu activations on each layer to introduce non-linearity to the model, and added a Keras lambda layer to normalize the data to between zero and 1 (centered at 0.5). Further, I read that He Normalization was becoming popular, so I also added that as the weight initializer and found it to improve training.

####2. Attempts to reduce overfitting in the model

As this is a moderately deep network, I added Batch Normalization layers inbetween every layer of the nework (except the final output layer) to ensure the weights did not explode or die out. I also added dropout layers to the flat / Dense / Fully Connected layers (except for the final output layer).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. I took all the data I wanted to use for training, and shuffled it all. I then took 20% of the now shuffled data for validation use. During the training process, the training data was further shuffled to ensure more consistent training.


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The primary goal of this project was to get the car around track 1 without crossing the lane lines (leaving the road). I wanted to teach my model to turn the steering wheel to the right if the car got too far to the left of the lane, and turn to the left if the car gets too far to the right of its lane.

For details about how I created the training data, see the next section. 

It should be noted, that by default the simulator wanted keyboard input to steer the car. This was a problem as a keyboard only allows you to turn the wheel or not (there is no find grain way to control steering with a keyboard). So, I purchased a Logitech F310 game controller and used that to control the car. This provided me the ability to keep the car more centered in the track as well as turn the car in a more controlled manner. On corners, I needed to record a picture and steering angle that match accurately. This could only be done with an analogue controller. With a keyboard you can only turn corners by rapidly presing and releasing a key to wobble your way around the turn. This results in some corner images recorded with full steering lock, and some with zero steering angle. This is very bad and will not teach your model to drive.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the NVIDIA PilotNet architecture as closely as I could. My assumption was that NVIDIA has put in the leg work to find a good architecture for the driving space, so my first attempt should be to see if I can build and train a model to take advantage of their development work.

The model uses a Keras 2D cropping layer, then a Keras normalization layer followed by a five convolutional layers and then three fully connected layers, and then a final output layer. The convolutional layers will primarily identify patterns in the car's video feed (lines, circles, curves etc.) whereas the fully connected layers decide how to steer based on the output from the convolutional layers. This said, as we train the whole network together, it is impossible to say that there isn't some cross over and that the convolutional part of the network isn't helping with the steering and vice versa.

In order to gauge how well the model was working, the training data was shuffled, then split with 20% of the data being used for validation. I split my image and steering angle data into a training and validation set. During training, the goal is to see that the training loss and validation loss are descending together, and that there is not too much descrepency between the validation loss and training loss. 

To try to prevent overfitting, I added dropout inbetween all of the fully connected layers. I also added Batch Normalization layers between all layers of the network (except the output layer).

The final step was to run the simulator to see how well the car was driving around track one. On my earlier attempts, I found that the car only wanted to turn left. This is not surprising as the track is mostly made of left turns... I had taught the car to turn left only! I then retrained the model by augmenting the data set. One such way I did this was to flip all my images (and the corresponding steering angles) so I had similar number of left and right turns as examples. I next found that the car did not know how to recover if it got too far off center of the road. I did one lap whereby I would drive to the edge of the track, start a recording, drive to the center of the track and stop the recording. I did this several times to teach the model how to correct when close to the edge of the road. Other augmentation was performed as detailed below.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I found the ultimate drive around track 1 to be very stable and the car managed to remain very centered in its lane.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

1. Input of 320x160 image in YUV format
2. 2D Cropping Layer removing 60 pixels from the top and 20 from the bottom (to remove unneeded image of dash and scenery)
3. Normalization layer, amending image values to 0-1 centered at 0.5 as we are using ELU activations
4. Convolutional Layer 5x5 Kernel and 2x2 Stride ELU activation with He Normalization
5. Batch Normalization Layer
6. Convolutional Layer 5x5 Kernel and 2x2 Stride ELU activation with He Normalization
7. Batch Normalization Layer
8. Convolutional Layer 5x5 Kernel and 2x2 Stride ELU activation with He Normalization
9. Batch Normalization Layer
10. Convolutional Layer 3x3 Kernel and 1x1 Stride ELU activation with He Normalization
11. Batch Normalization Layer
12. Convolutional Layer 3x3 Kernel and 1x1 Stride ELU activation with He Normalization
13. Batch Normalization Layer
14. Flatten Layer
15. Dropout Layer (0.5)
16. Fully Connected Layer (100 units) - ELU activation with He Normalization
17. Batch Normalization Layer
18. Dropout Layer (0.5)
19. Fully Connected Layer (50 units) - ELU activation with He Normalization
20. Batch Normalization Layer
21. Dropout Layer (0.5)
22. Fully Connected Layer (10 units) - ELU activation with He Normalization
23. Fully Connected Layer (1 units) - OUTPUT

Here is a visualization of the architecture (Batch Normalization and Dropout Layers not shown). 

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to turn back towards the center of the road if it got too close to the edge. These images show what a recovery looks like starting from the left curb to the center:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I also flipped images and angles thinking that this would make up for all of the left hand turns in track 1, by adding an equal amount of right hand turns. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 4,857 number of data points. I then preprocessed this data by following all steps below in this order:

* iterating through each of these images and loading the corresponding images for the left and right camera in the car. For the left images, I took the steering angle reading from the center channel and added 0.2 to it, for the right camera images, I subtracted 0.2 from it. I now have three times as much data (4857 x 3 = 14,571).
* I iterate through all images and randomly apply none, one or more of: fake shadows, random jitter in the y axis, and change the global image brightness to the images.
* Finally to compensate for all the left turns in the track, I flip every image and invert its steerin angle giving a total of 14,571 x 2 = 29,142 images for training.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 with a batch size of 256 as evidenced by watching the loss drop. At around epoch 15 I observed that the MSE started to flatten out. I used an adam optimizer so that manually training the learning rate wasn't necessary.
