# Getting-Started-with-Pytorch
In this notebook I will showcase a convoluted neural network model that achieves 99.6% accuracy on the MNIST Handwritten Digit problem. This model is built using Pytorch Lightning. This package is great for beginners and experts alike as it offers simple yet powerful APIs.

<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*DQSLKyHw2eAkM385mG-29g.png" alt="header.png" width="1000" >

<center>

  MNIST Image from [Wikipedia](https://commons.wikimedia.org/wiki/File:MnistExamples.png)

</center>

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


