# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/visualization.png "Visualization"
[image2]: ./writeup_images/filters.png "Filter sample"
[image3]: ./writeup_images/random_noise.jpg "Random Noise"

[image4]: ./writeup_images/tablak.jpg "Amusing image 1"
[image5]: ./writeup_images/tablak2.jpg "Amusing image 2"
[image6]: ./writeup_images/tabla1.png "Traffic Sign 1"
[image7]: ./writeup_images/tabla2.png "Traffic Sign 2"
[image8]: ./writeup_images/tabla3.png "Traffic Sign 3"
[image9]: ./writeup_images/tabla4.png "Traffic Sign 4"
[image10]: ./writeup_images/tabla5.png "Traffic Sign 5"
[image11]: ./writeup_images/tabla6.png "Traffic Sign 6"
[image12]: ./writeup_images/tabla7.png "Traffic Sign 7"
[image13]: ./writeup_images/tabla8.png "Traffic Sign 8"

[image14]: ./writeup_images/topk.png "Top 5 guesses"
[image15]: ./writeup_images/featuremap.png "Featuremaps"
[image16]: ./writeup_images/featuremap2.png "Featuremaps"
[image17]: ./writeup_images/feature_blue.png "Featuremaps"
[image18]: ./writeup_images/feature_stop.png "Featuremaps"
[image19]: ./writeup_images/feature_rw.png "Featuremaps"

[image20]: ./writeup_images/feature_eonp.png "Featuremaps"
[image21]: ./writeup_images/feature_eonp2.png "Featuremaps"

[image22]: ./writeup_images/end_of_no_passing.png "Featuremaps"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/esp32wrangler/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used the implicit numpy shape variable, and the built-in `set` collection type to calculate summary statistics of the traffic
signs data set:
```
Number of training examples = 34799
Number of training examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43
```

#### 2. Include an exploratory visualization of the dataset.

See the exploratory visualization of the data set below. It is a mosaic showing one random sample for each image class, with the numerical identifier and the number of examples in the dataset annotated. There is also a histogram for the number of examples, which shows that this dataset is very unbalanced - there are as few as 180 images for one classes, while over 2000 for others. This will cause issues later on...

![visualization][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I experimented with different normalizations for the images:
- a simple "-128/128" normalization to map the pixels to the [-1..1] range
- an even simpler "/255" normalization to the [0..1] range (results seemed about the same as the [-1..1] normalization)
- converting the images to greyscale (with 1 or 3 channels)
   - the 1 channel approach would have allowed me to use the LeNet architecture without change (for the channel 1 case), but I think color is very important in traffic sign recognition, so I didn't feel this was the right way to go
   - the 3 channel greyscale would have given some resistance to the network against changes in hue and saturation of the input, which it did, but not to a level to justify the drastically increased training time
- boosting the color saturation with conversion to HSV color space, multiplying the saturation value and then converting back to RGB
    - my hope was that this would help the recognizer spot the colors easier, but it didn't seem to matter
- converting to HLS and HSV color spaces and feeding that to the network
    - this worked well for the lane recognition, but ultimately it is a linear transformation that the network was able to discover on its own
- "stretching" the image over the entire [-1..1] range, increasing contrast as needed
    - this didn't actually add any information to the image that was there already, and it actually made the results worse
- adding gaussian noise
    - my hope was that this would add some more randomness and help prevent overtraining, but the regularization and dropouts helped more

Because of the unbalanced dataset, I added a simple algorithm to add new, distorted (rotated, perspective distorted, affine transformed) version of the images in the underrepresented classes. This, unfortunately did not yield a noticeable positive impact. I also tried a more sophisticated approach using the keras ImageGenerator and the imbalanced-learn library, but I couldn't find the right parameters for this to have a significant impact on the results, so I abandoned this approach.

I finally ended up using an approach where for each image, I insert three images into the training dataset, one original, one with gaussian noise and one converted to 3 channel greyscale. This particular combination had a significant positive impact on the results. 

Then when I tuned the LeNet implementation, adding dropouts and regularizations and fixing all the silly bugs, I realized that the robust neural network gives good results even with just the basic data set and basic normalization. So I removed this step and just normalize the pixel values to the [-1..1] range.

Here is an example of the 3 different prefiltered versions for some images (greyscale - gaussian noise - original):

![alt text][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

After experimenting with different combinations of the layers, and different sizing for each layer, I failed to find architectures that yielded significantly better results than the basic LeNet approach with dropouts:

The first one is the original LeNet architecture with dropouts added:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| C1 Convolution 3x3   	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| C2 Convolution 3x3	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flattening		    | flatten to 1D, 400 elements					|
| F1 Fully connected	| outputs 120  									|
| RELU					|												|
| Dropout   	      	| 0.7 keep probability             				|
| F2 Fully connected	| outputs 84  									|
| RELU					|												|
| Dropout   	      	| 0.7 keep probability             				|
| F3 Fully connected	| outputs 43 final object classes   			|
| 						|												|
 
I even tried a very heavy version, scaling up each level to 2-3 times the channels, but the results were the same or worse:
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| C1 Convolution 3x3   	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18   				|
| C2 Convolution 3x3	| 1x1 stride, valid padding, outputs 10x10x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48   				|
| Flattening		    | flatten to 1D, 1200 elements					|
| F1 Fully connected	| outputs 240  									|
| RELU					|												|
| F2 Fully connected	| outputs 84  									|
| RELU					|												|
| F3 Fully connected	| outputs 43 final object classes   			|
| 						|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the Adam Optimizer to minimize the cross entropy of the softmax output of the network. I also added a regularization term that uses the `tf.nn.l2_loss` function to add the weights of all the convolution and fully connected layers to the loss function. This helped prevent overtraining.

I settled on the batch size of 128, training rate of 0.003, dropout rate of 0.7 and regulatization beta of 0.0015 after much experimentation. 

Improvements plateaued after about 20 epochs, so that is what I configured. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started out with the LeNet architecture as-is on the recommendation of the training course. I first experimented with the image preprocessing, without touching any other parameters. I got to around 90-92% accuracy on the validation data set with this approach, which was just under the rubic target. See more details above in the first chapter.

Once I felt I reached a plateu with the preprocessing, I tried different layer sizes for processing the image. I doubled or tripled the parameters intuitively, but this did not yield a major breakthrough.

Next I added the dropout layers and the regularization term which promptly pushed the model over the 93% edge.

Finally I started tuning the batch size, learning rate, dropout rate and regularization beta. These, as well as some bug fixes in the code yielded incremental improvements in learning speed and final accuracy, reaching around 95% on the validation dataset.

This translated into an around 94% result on the test dataset.

At this point I briefly went back to the image processing step and realized that what I done was not necessary and simplified it to a simple levelling. 

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 95% 
* test set accuracy of 94%

After submitting the work, I put the image balancer and image generator back to see how far I can get with the fully tuned neural network and these tools (see https://github.com/esp32wrangler/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_with_augmentation.ipynb). Suprisingly, the results on the Test dataset were worse, not better. I was hoping that the results on my Hungarian dataset would at least improve, but that got worse as well. Perhaps with better tuning of the image generator variables I could get better results, which is an excercise for the future.

I also started researching more modern architectures targeted specifically to sign recognition, and found some promising candidates, such as the paper "Trafﬁc-Sign Detection and Classiﬁcation Under
Challenging Conditions: A Deep Neural Network Based Approach" from Uday Kamal and his team, and tried to implement it as LeNetUday, but it did not work at all out of the door, and I not have a chance yet to try to figure out the bugs yet...

According to this table: http://benchmark.ini.rub.de/?section=gtsrb&subsection=results&subsubsection=ijcnn , human performance is around 98.8% on this type of data. So there is still a lot of room to improve to reach that, but I'm already very impressed with the relative ease and speed I was able to develop such a decent recognizer. If I were to attempt to write a manual algorithm similar to the lane follower example earlier in the course, it would have taken a lot longer, and would have probably yielded worse (albeit much more predictably bad) results.

I plan to look at more modern architectures and play with them to get closer to the human accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

As these are European standardized traffic signs, I decided to look for some snapshots from Hungary. I found some amusing images on the web:

 ![alt text][image4]  ![alt text][image5]   

On a more serious note... Here are some cropped signs from random pictures I found on the interwebs:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10] ![alt text][image11]
![alt text][image12]

These are all fairly good quality images, but suffer from some rotation and perspective distortion, so they are not trivial to identify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction of the model for this dataset:

![image14]

The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. This result is numerically worse than the test dataset, but due to the sample size it is hard to say anything for sure.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The weakness of the original dataset clearly shows, both the miss of the "No passing" sign and the near-miss of the 20 km/h sign are related to the two most under-represented classes in the dataset, the 20 km/h sign with only 180 examples, and the "End of no passing" sign with 210. Even though on average the model is pretty good, it is clearly not reliable when it comes to these particular signs (and they are also underrepresented in the validation and test datasets, supporting a false trust in the model in light of the validation and test results). With a curated, carefully balanced test dataset, a better understanding could be achieved of the quality of the model.

In addition to the above, the model also had some doubts about the Stop sign, which is more surprising and harder to explain with my level of knowledge.

For the others, the softmax values show a very strong peak (or a single point), indicating that the model is certain or near certain about what signs they are.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I chose to examine the image that confused the classifier, the "no passing" sign. I also included an image of the blue "go straight or turn right" sign, to look at any indications that color is playing a role.

Looking at the feature maps of the conv1 layer, the edges become very prominent, and there is a clear distinction between the black car and red car in all the feature maps, meaning that color (or at least saturation) does play a role in the output. But comparing the (originally) red stop sign's face with the blue turn sign's face, it is clear that colors are not preserved as such.

Conv1 layer:

![image15]
![image17]
![image18]


Looking at the conv2 layer, things are becoming very abstract, it is impossible to make out the pattern. I was hoping to at least see some recurring bit patterns for the two little cars on the sign, but I couldn't find any. 

Conv2 layer for "no passing":

![image16]

Comparing the "No passing" and "End of no passing" images, and their conv1 counterparts, it is interesting how the cars virtually float above the cross hashing. The difference in the feature maps is more subtle on the conv1 layer than the difference between the original images.

On the conv2 layer however, the cross hashing is extremely prominent on most of the feature maps, meaning that the confusion arises in the fully connected deep layers downstream. 


![image12]
![image22]:

![image15]
![image20]:
![image16]
![image21]: 



I also found an image for a roadworks sign (from a Lego set, to keep things interesting), which the classifier considered a strong candidate for the stop sign, and took a look at its feature map:

![image19]
![image18]

I can't really see the similarity...