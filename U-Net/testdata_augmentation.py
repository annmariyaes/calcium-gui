import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# for U-Net data preparation: mask
source_folder = 'dataset/new_test/'
destination_folder = 'dataset/augmented/'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)


files = [f for f in os.listdir(source_folder) if f.endswith('.tif')]


for file_name in files:
    # Construct the full path to the image
    file_path = os.path.join(source_folder, file_name)

    # Read the image using OpenCV
    image = cv2.imread(file_path)

    b, g, r = cv2.split(image)

    # Apply histogram equalization to each channel separately
    eq_b = cv2.equalizeHist(b)
    eq_g = cv2.equalizeHist(g)
    eq_r = cv2.equalizeHist(r)
    equalized_image = cv2.merge((eq_b, eq_g, eq_r))

    cv2.imwrite(destination_folder + 'equalized_' + file_name, equalized_image)

