import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds


dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

# Data processing

# U-Net


def double_conv_block(input_image, n_filters):

    # Conv 3x3, ReLU
    conv_relu_one = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(input_image)
    conv_relu_two = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(conv_relu_one)

    return conv_relu_two


def down_sample(input_image, n_filters):
    """
    Encoder
    Two 3x3 convolutions with ReLU activation functions, which perform feature extraction,
    and a 2x2 max-pooling layer for down-sampling.
    """

    conv_relu_two = double_conv_block(input_image, n_filters)
    max_pool = layers.MaxPool2D(2)(conv_relu_two)
    max_pool = layers.Dropout(0.3)(max_pool)

    return conv_relu_two, max_pool


def up_sample(input_image, conv_features, n_filters):
    """
    Decoder
    Comprises up-convolutions (transposed convolutions) that increase the spatial resolution of the feature maps.
    Each step in this path involves an up-convolutional layer followed by concatenation with the corresponding feature maps from the contracting path. This step helps in preserving spatial information and finer details.
    After concatenation, there are two 3x3 convolutions with ReLU activation functions to further process the combined features.
    """
    segmentation_map = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(input_image)
    segmentation_map = layers.concatenate([segmentation_map, conv_features])
    segmentation_map = layers.Dropout(0.3)(segmentation_map)
    segmentation_map = double_conv_block(segmentation_map, n_filters)

    return segmentation_map


def build_unet_model():

    # inputs
    inputs = layers.Input(shape=(128, 128, 3))

    # encoder: contracting path - down sample
    conv_relu_two1, max_pool1 = down_sample(inputs, 64)
    conv_relu_two2, max_pool2 = down_sample(max_pool1, 128)
    conv_relu_two3, max_pool3 = down_sample(max_pool2, 256)
    conv_relu_two4, max_pool4 = down_sample(max_pool3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(max_pool4, 1024)

    # decoder: expanding path - up sample
    up_conv1 = up_sample(bottleneck, max_pool4, 512)
    up_conv2 = up_sample(up_conv1, max_pool3, 256)
    up_conv3 = up_sample(up_conv2, max_pool2, 128)
    up_conv4 = up_sample(up_conv3, max_pool1, 64)

    """
    A 1x1 convolutional layer with a softmax activation function (for multi-class segmentation) or a 
    sigmoid activation function (for binary segmentation) is used to produce the final segmentation map.
    """
    segmentation_map = layers.Conv2D(3, 1, padding="same", activation="softmax")(u9)

    # unet model with Keras Functional API
    u_net = tf.keras.Model(inputs, segmentation_map, name="U-Net")

    return u_net


u_net_model = build_unet_model()
u_net_model.summary()

