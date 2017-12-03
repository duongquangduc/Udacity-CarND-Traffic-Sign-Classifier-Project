# **Traffic Sign Recognition** 

## Writeup

###  This is the second project of term 1 in Self-Driving Car Nanodegree Program by Udacity
---
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/histogram.png "Histogram"
[image4]: ./examples/germany-road-signs-pedestrians-crossing.png "Traffic Sign 1: Pedestrians Crossing"
[image5]: ./examples/germany-road-signs-slippery.png "Traffic Sign 2: Slippery Road"
[image6]: ./examples/germany-speed-limit-sign-60.png "Traffic Sign 3: Speed Limit 60"
[image7]: ./examples/germany-end-speed-limit-sign-60.png "Traffic Sign 4: End Speed Limit 60"
[image8]: ./examples/germany-road-signs-wild-animals.png "Traffic Sign 5: Wild Animals"

---
### Writeup / README

#### 1. Provide a Writeup / README 

You're reading it! and here is a link to my [project code](https://github.com/duongquangduc/Udacity-CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32,32,3)
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set:
* First, I printed 50 random images from the training set.
* Second, I printed a histogram showing the number of classes and them numbers of images of each class.

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to shuffle the images by using shuffle from sklearn.utils.
Then I preprocessed the data by normalizing it with function normalize_data() as a way to improve the accuracy of the trained model. I got accuracy rate over 93% after several epochs, so in this project I just use one method for improving the accuracy. 

Though in my future work, I will use other methods as the suggestions from the lectures such as experimenting with different network models, changing the dimensions of the LeNet layers, adding regularization like dropout or L2 regularization, tuning hyperparameters, improving data processing with normalization and zero mean, or augmenting the training data by rotating, shifting images, changing colors, etc. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 1     	| 1x1 stride, VALID padding, output = 28x28x6 	|
| RELU			|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 14x14x6   |
| Convolution 2  	    | 1x1 stride, VALID padding, output = 10x10x16  |
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 5x5x16    |
| Flatten				| output = 400									|
| Fully connected		| input = 400, output = 120       	            |
| RELU					|												|
| Fully connected		| input = 120, output = 84       	            |
| RELU					|												|
| Fully connected		| input = 84, output = 10       	            |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a g2.2xlarge GPU machine in AWS. I trained the model with 20 epochs, batch_size of 128, and learning rate of 0.001.

For the optimizer, first, I used softmax_cross_entropy_with_logits, then applied tf.reduce_mean() to calculate the mean of elemtent, and finally use tf.train.AdamOptimizer().

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.1 %
* test set accuracy of 90.5 %

I think this accuracy acquired by the following factors:
* The architecture of the neural networks: LeNet-5
* The preprocessing method
* The number of epochs is good enough

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of 90.5%

Please check the report for more details.
Here are the results of the prediction:


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:				| 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Please check the report for more details.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
