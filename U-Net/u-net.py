import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


"""
Dataset images along with  corresponding segmentation masks (ground truth labels) 
indicating the areas of interest (e.g., boundaries, objects, etc.).
"""

# Data processing


def resize_normalize(img_path, target_size=(128, 128)):
    img = Image.open(img_path)
    img = img.resize(target_size)  # Resize to the target size
    img = np.array(img) / 255.0  # Normalize pixel values (assuming 0-255 range)
    return img


def process_dataset(source_folder):
    images = []

    for file_name in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file_name)
        if os.path.isfile(file_path):
            image = resize_normalize(file_path)
            images.append(image)

    images = np.array(images)
    print(images.shape)
    return images


'''
For Example: (544, 128, 128, 3)
544: Number of images in your test dataset.
128 x 128: Height and width of each image after resizing to 128x128 pixels.
3: Represents the three color channels (Red, Green, Blue) in the images, indicating that your images are in RGB format.
'''

# The data for training contains 3625 512*512 images
test_images = process_dataset('dataset/test/')  # 544
valid_images = process_dataset('dataset/valid/')  # 544
train_images = process_dataset('dataset/train/')  # 2537

test_mask_images = process_dataset('dataset/test_mask/')  # 544
valid_mask_images = process_dataset('dataset/valid_mask/')  # 544
train_mask_images = process_dataset('dataset/train_mask/')  # 2537

'''
plt.imshow(test_images[1])
plt.title('Sample Image')
plt.axis('off')
plt.show()
'''


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
    up_conv1 = up_sample(bottleneck, conv_relu_two4, 512)
    up_conv2 = up_sample(up_conv1, conv_relu_two3, 256)
    up_conv3 = up_sample(up_conv2, conv_relu_two2, 128)
    up_conv4 = up_sample(up_conv3, conv_relu_two1, 64)

    """
    A 1x1 convolutional layer with a softmax activation function (for multi-class segmentation) or a 
    sigmoid activation function (for binary segmentation) is used to produce the final segmentation map.
    """
    # The output has three channels corresponding to the 2 classes that the model will classify each pixel for:
    # background, foreground object
    segmentation_map = layers.Conv2D(3, 1, padding="same", activation="sigmoid")(up_conv4)

    # unet model with Keras Functional API
    u_net = tf.keras.Model(inputs, segmentation_map, name="U-Net")

    return u_net


# Prepare U-Net model
unet_model = build_unet_model()
unet_model.summary()

unet_model.compile(optimizer='adam', loss="categorical_crossentropy", metrics="accuracy")
history = unet_model.fit(train_images, train_mask_images, epochs=10, batch_size=32, validation_data=(valid_images, valid_mask_images))


test_loss, test_accuracy = unet_model.evaluate(test_images, test_mask_images)
print(f"Test Accuracy: {test_accuracy}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Prediction
