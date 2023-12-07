# Getting-Started-with-Pytorch
In this notebook I will showcase a convoluted neural network model that achieves 99.6% accuracy on the MNIST Handwritten Digit problem. This model is built using Pytorch Lightning. This package is great for beginners and experts alike as it offers simple yet powerful APIs.

<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*DQSLKyHw2eAkM385mG-29g.png" alt="header.png" width="1000" >

> **_MNIST Image from [Wikipedia](https://commons.wikimedia.org/wiki/File:MnistExamples.png)_**

<br><br>

## Abstract
This project is based on Mnist dataset, and the architecture used in this project can be find through the [link](https://medium.com/@BrendanArtley/mnist-keras-simple-cnn-99-6-731b624aee7f). Although this article was implemented this architecture on Keras, I re-write it in Pytorch lightning. Pytorch is considered as an academic tool in AI so it is so much better to learn Pytorch from skratch.

<br><br>

## Table of Contents
- <a href='#requirements'>Requirements</a>
- <a href='#training-linear-regression'> Training RFC</a>
- <a href='#performance'>Performance</a>
- <a href='#references'>Reference</a>

<br>

## Requirements
This project does not need any specific requirements. The dataset, Mnist, also is available in `torchvision.datasets`.
<br><br>


### Dataset and pre-processing

Data augmentation is extremely important. For image data, it means we can artificially increase the number of images our model sees.This is achieved by Rotating the Image, Flipping the Image, Zooming the Image, Changing light conditions, Cropping it etc.
<br>


## Architecture
In order to build a strong Deep neural network, we should go through the following steps:
1. Add Convolutional Layers — Building blocks of ConvNets and what do the heavy computation
2. Add Pooling Layers — Steps along image — reduces params and decreases likelihood of overfitting
3. Add Batch Normalization Layer — Scales down outliers, and forces NN to not relying too much on a Particular Weight
4. Add Dropout Layer — Regularization Technique that randomly drops a percentage of neurons to avoid overfitting (usually 20% — 50%)
5. Add Flatten Layer — Flattens the input as a 1D vector
6. Add Output Layer — Units equals number of classes. Sigmoid for Binary Classification, Softmax in case of Multi-Class Classification.
7. Add Dense Layer — Fully connected layer which performs a linear operation on the layer’s input
```
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last',
                 input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' ))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
```


<img src="./images/skewness.gif" alt="skewness.gif" width="100%" >

- *The Histogram plot corresponding to each features.*
**[Blue Plots: acceptable**
**, Orange Plots: unacceptable]**

<br>

<img src="./images/skewness_reduction.gif" alt="skewness_reduction.gif" width="50%" >

- *The Histogram plot corresponding to each features* ***after being transformed*** *to Normal Distribution.*



<br><br>

<br>

### Colaboratory Notebook
The Second way to train and test RFC is to use the `.ipynb` file in the main directory. It is very informative and builds up your intuition of the process of pre-processing and makes you more knowledgeable about the dataset. In addition, you don't even need to clone the repository, because it can be executed by Google Colaboratory Online.
<br><br>


## References:

The following list contains several links to every resource that helped us implement this project.

1.  Kaggle dataset published by [UCI MACHINE LEARNING](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
2.  The project which is developed by [BUDDHINI W](https://www.kaggle.com/code/buddhiniw/breast-cancer-prediction)
3.  Maths-ML developed by [Farzad Yousefi ](https://github.com/F-Yousefi/Maths-ML)
3.  House Price Prediction developed by [Farzad Yousefi ](https://github.com/F-Yousefi/House_Price_Prediction)
4.  Machine Learning course published by [Coursera ](https://www.coursera.org/specializations/machine-learning-introduction)


