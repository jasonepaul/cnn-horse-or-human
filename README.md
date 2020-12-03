# cnn-horse-or-human
An Exploration of Computer Vision and Training of CNNs

This was a university course project with subsequent improvements. It is fundamentally
a machine learning classification problem where the target variable is binary, representing whether an image 
has a human or horse.

The project uses the well-known "horse vs human" dataset made available for 
learning computer vision algorithms. Credit for the dataset goes to Laurence Moroney (lmoroney@gmail.com / 
laurencemoroney.com). Images are 300 x 300 pixels in 3 colour channels with various species of 
horses and diversity of humans represented. Images were available in the following pre-organized categories:

|         | Training | Test | Total |
| :------ | --------:| ----------:| -----:|
| horses  | 500      | 128        | 628   |
| humans  | 527      | 128        | 655   |
| total   | 1027     | 256        | 1283  |

Sample human and horse images:

|![](.\data\horse-or-human\train\humans\human01-15.png)|![](.\data\horse-or-human\train\horses\horse02-0.png)

A convolutional neural network (CNN) was chosen as the classification model. "Training" data was divided into 
80% training and 20% validation during the model fitting process. The CNN was built using the 
TensorFlow (Keras) Sequential class.

The first model constructed used 11 layers (8 filter, 1 flatten, 2 fully connected dense) for 2.2 million trainable 
parameters. Regularization (L2) and dropout was used to try to reduce over-fitting of the training data. The resulting 
best accuracy on the test data was 83.6%.

Subsequent to completion of the course, I revisited this project and built a second model that incorporated the
VGG16 (very deep) CNN architecture. (This design was used to win an Imagenet competition in 2014.) My idea was to
experiment with transfer learning—taking a successful model made by others on a classification problem and using
this on an entirely different classification problem. I usedTensorFlow's VGG16 CNN model with the pre-trained
weights for the filter layers, and added fully connected layers that were subsequently trained to my problem.
The resulting model had 16 layers (13 filter, 1 flatten, 2 fully connected dense) for 14.7 million trainable
parameters. This turned out to be highly successful with an accuracy on the test data of 99.6%.
