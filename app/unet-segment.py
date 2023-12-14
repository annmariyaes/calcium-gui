import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model


organoids = [('D:/ann/Experiment/Isoprenaline/Normal 1/', 'D:/ann/Experiment/Isoprenaline/100 nM Isoprenaline 1/', 'D:/ann/Experiment/Isoprenaline/500 nM Isoprenaline 1/', 'D:/ann/Experiment/Isoprenaline/1 uM Isoprenaline 1/'),
    ('D:/ann/Experiment/Isoprenaline/Normal 2/', 'D:/ann/Experiment/Isoprenaline/100 nM Isoprenaline 2/', 'D:/ann/Experiment/Isoprenaline/500 nM Isoprenaline 2/', 'D:/ann/Experiment/Isoprenaline/1 uM Isoprenaline 2/'),
    ('D:/ann/Experiment/Isoprenaline/Normal 3/', 'D:/ann/Experiment/Isoprenaline/100 nM Isoprenaline 3/', 'D:/ann/Experiment/Isoprenaline/500 nM Isoprenaline 3/', 'D:/ann/Experiment/Isoprenaline/1 uM Isoprenaline 3/')]

for i, organoid in enumerate(organoids):
    for folder_path in organoid:

        # Load and preprocess images
        experiment = os.listdir(folder_path)
        files = [f for f in experiment if f.endswith('.tif')]

        images = []
        for file in files:
            image_path = os.path.join(folder_path, file)
            img = Image.open(image_path)

            # Preprocess the image (resize, normalize, convert to array)
            img = img.resize((128, 128))
            img_array = np.array(img) / 255.0
            images.append(img_array)

        images = np.array(images)

        # Load the pre-trained U-Net model
        model = load_model('D:/ann/git2/U-Net/saved_unet.h5')
        predictions = model.predict(images)
        binary_mask = (predictions[0, :, :, 0] > 0.5).astype(np.uint8)
        plt.imshow(binary_mask, cmap='gray')  # Displaying as a grayscale image
        plt.title('U-Net segmentation')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

