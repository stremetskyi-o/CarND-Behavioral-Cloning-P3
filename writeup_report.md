# **Behavioral Cloning**
## Project Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[loss]: writeup-img/loss.png "Mean squared error loss"
[act1]: writeup-img/act1.jpg "Image from track 1"
[act2]: writeup-img/act2.jpg "Image from track 2"
[act1_conv1]: writeup-img/out_act1_conv2d_1.jpg "Activation of the first convolution layer"
[act2_conv1]: writeup-img/out_act2_conv2d_1.jpg "Activation of the first convolution layer"

---

### 1. Writeup

This writeup describes research and implementation steps that were taken to address project rubric points. Where 
appropriate steps are backed up by additional information in the form of images, tables, etc.

### 2. Data collection and augmentation

Images for model training and validation were collected using the supplied driving simulator.
The initial dataset consists of one direct lap and one reverse lap of the first track. While driving, I was trying to keep the vehicle at the road center. After testing first models, the vehicle was able to stay at the road center on the straight runs and small turns but failed to follow sharper turns. I have analyzed the dataset and concluded that around 82% of images are of straight parts of the road
with the angle in the range [-3, 3] degrees. So I have expanded the dataset with another lap of the first track with only turns recorded. Then I have added images of one lap of the second track to help the model to generalize better, increase the percentage of turns in the dataset
and allow the model to cope with the second track. I have also used images from the left and right cameras with an angle bias of +-2 degrees to provide a stronger signal of avoiding road edges which helped the simulated vehicle to pass the curves more precisely. Later the bias was increased to 5.

Dataset image examples:

![act1]![act2]

The dataset summary is the following:
* Dataset has `21042` images of shape `(160, 320, 3)`.
* Labels mean is `-0.008` and the standard deviation is `0.27`.
* Straight road images amount is `21%`.

The dataset provides enough images with more turns than straight runs; the mean angle value is close to zero.

It's possible to effectively augment dataset by flipping the images and inverting the angles which I did with NumPy ```fliplr()``` method.
The augmented dataset doubled in size to 42084 images.

To prevent overfitting dataset was shuffled and split to training and validation subsets sized to 80% and 20% respectively.
Memory usage was minimized by using Python generators to retain only the list of image paths and labels at all times and load images as needed.

### 3. Model architecture

Architecture is based on DAVE-2 System architecture[1].
The architecture was selected based on its small size and the fact it was proven effective for this particular task.

The pre-processing steps are:
* Cropping the input to select only the road part of the image and resizing it to 66x200 px. 
* Normalizing image using the following equation: `x = x / 128 - 1`.

The model consists from the next layers:   

| Layer (type)              | Filter, stride    | Output Shape      | Param # |
|---------------------------|-------------------|-------------------|---------|
| Input                     |                   | (160, 320, 3)     |
| cropping2d_1 (Cropping2D) |                   | (71, 320, 3)      |
| resizing (Lambda)         |                   | (66, 200, 3)      |
| normalizing (Lambda)      |                   | (66, 200, 3)      |
| conv2d_1 (Conv2D)         | 5x5, 2            | (31, 98, 24)      | 1824
| conv2d_2 (Conv2D)         | 5x5, 2            | (14, 47, 36)      | 21636
| conv2d_3 (Conv2D)         | 5x5, 2            | (5, 22, 48)       | 43248
| conv2d_4 (Conv2D)         | 3x3, 1            | (3, 20, 64)       | 27712
| conv2d_5 (Conv2D)         | 3x3, 1            | (1, 18, 64)       | 36928
| flatten_1 (Flatten)       |                   | (1152)            |
| dense_1 (Dense)           |                   | (100)             | 115300
| dense_2 (Dense)           |                   | (50)              | 5050
| dense_3 (Dense)           |                   | (10)              | 510
| dense_4 (Dense)           |                   | (1)               | 11
||
| **Total**                                   | | **252,219**

Convolution and fully connected layers use ReLU activations to introduce non-linearity.

### 4. Model training

#### 4.1. Training original model

For the training I have selected the Adam optimizer, the property I was minimizing is *mean squared error*.
The batch size was set to 32, so the model would update frequently and backpropagation based on a smaller amount of images
to reduce overfitting. Also callbacks where added to save the best model found and to interrupt training if there is no significant improvement
for 2 epochs. The number of epochs was set to 15. The next figure shows the training and validation errors for every epoch.

![loss]

The validation loss did not improve in 13th and 14th epochs and fitting was terminated early with the final model saved after
epoch #12 with validation loss value of 0.0033.

#### 4.2. Additional training

The model was tested against both tracks and was able to guide the vehicle without incidents; however, on some curves
vehicle was crossing the road shoulders. To improve the driving, I have decided against retraining the whole model and worked
on improving one from the previous step.

I have selected to freeze the convolution layers' weights as the image features won't be changing and only the signal strength they produce.
To avoid driving on the shoulders, I have increased angle bias for left and right cameras to 5 degrees. 

After freezing, the weights had the following distribution:

```
Trainable params: 120,871
Non-trainable params: 131,348
```

I have decreased the number of epochs to 5 and applied the same callbacks for saving the best model and early stopping.
The final model trained for 3 epochs and had validation loss about 0.0012.

### 5. Results

The model successfully navigated vehicle on track 1 and track 2: video for [track 1](video.mp4) and [track 2](video_track_2.mp4).
To go further, I have also tested the tracks in reverse and at different speeds:
* **track 1:** the vehicle can navigate the track in both directions and at a max speed of 30 mph.
* **track 2:** the vehicle can navigate on the direct track without problems and with some issues on the reversed lap at speed between 9-15 mph. Some of the features that may cause problems are shadows and mountains which are close to the road. 
 
 *Note: the speed can be adjusted in the drive.py script*

### 6. Discussion

To investigate the issues of the model and to overview which features are being used by the model I have chosen to visualize the internal state of the network.
For the images from the first and second track, I have visualized activations of the first convolution layer.

Original image:

![act1]

conv2d_1 activations:

![act1_conv1]

For the image of the first track, activations are mostly edges of the road which shows that the model successfully picked up important features without specific labeling.

Original image:

![act2] 

conv2d_1 activations:

![act2_conv1]

Second track image also detected road edges as well as lane line in the center. However, there are also strong activations for shadows
and mountains beyond the road edge, which I believe can be the reason for worse performance on the reversed track 2.

Based on the tests and internal state visualizations, the model can benefit from additional training on data with more shadows or different lighting conditions. 

It was proven effective to adjust driving angles by using transfer learning techniques and to only adjust fully connected layers.
However, to continue training the model with additional data, it's better to adjust all of the weights.
Some angle averaging may be used to improve the overall driving performance, and for higher speeds and sharper turns, there is a need to control speed or throttle besides the steering angle.

---

[1]: Bojarski, Mariusz, et al. "End to end learning for self-driving cars." arXiv preprint arXiv:1604.07316 (2016).