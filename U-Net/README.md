# Implementation of U-net for organoid image segmentation, using Keras

The data for training contains 4500 512*512 images to feed a deep learning neural network. Dataset images along with corresponding segmentation masks (ground truth labels). 
* training dataset = 3150 (70%)
* validation dataset = 675 (15%)
* testing dataset = 675 (15%)

The architecture is from [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

