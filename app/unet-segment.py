import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from keras.models import load_model
import tensorflow.python.keras.backend as K
from concurrent.futures import ThreadPoolExecutor

def mean_iou(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * K.round(y_pred)))
    union = K.sum(y_true) + K.sum(K.round(y_pred)) - intersection
    iou = intersection / (union + K.epsilon())
    return iou


def dice_coefficient(y_true, y_pred):
    numerator = 2 * K.sum(y_true * y_pred)
    denominator = K.sum(y_true) + K.sum(y_pred)
    dice = numerator / (denominator + K.epsilon())
    return dice


# number of pixels that are classified correctly in the generated segmentation mask
def pixel_wise_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


class Segmentation:
    def __init__(self, organoids):
        self.organoids = organoids



    def generate_intensity_plot(self):
        intensity_plot_paths = []
        colors = ['green', 'purple', 'orange', 'red']  # Define colors for each concentration
        title = self.organoids[0][0].split('/')[-2].split()[0]

        for i, organoid in enumerate(self.organoids):

            num_frames = self.get_nframes(organoid[0])
            time_intervals = np.linspace(0, 10, num_frames)

            mean_pixel_intensity, heart_rate = zip(*self.process_organoids(*organoid))

            # pixel intensity plot
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            for j, intensity in enumerate(mean_pixel_intensity):
                axes[j//2, j%2].plot(time_intervals, intensity, color=colors[j], markersize=1)
                axes[j//2, j%2].set_title(organoid[j].split('/')[-1])
                axes[j//2, j%2].set_xlabel('Time (sec)')
                axes[j//2, j%2].set_ylabel('Mean Intensities (pixels)')

            # Adjust layout to prevent overlap
            plt.tight_layout()
            plot_filename = 'static/uploads/' + title + ' intensity ' + str(i + 1) + '.png'
            plt.savefig(plot_filename)
            plt.close()
            intensity_plot_paths.append(plot_filename)

        return intensity_plot_paths



    def generate_heartrate_plot(self, concentrations):

        heart_rates = []
        title = self.organoids[0][0].split('/')[-2].split()[0]

        for i, organoid in enumerate(self.organoids):
            mean_pixel_intensity, heart_rate = zip(*self.process_organoids(*organoid))
            heart_rates.append(heart_rate)

        # plot of heart rate vs concentration
        for r, heart_rate in enumerate(heart_rates):
            plt.scatter(concentrations, heart_rate, marker='o', label=f'Organoid {r + 1}')
        plt.xlabel('Concentration (nM)')
        plt.ylabel('Heart Rate (Hz)')
        plt.title(title)
        plt.legend()
        plt.xticks(concentrations)
        heartrate_vs_concentration = 'static/uploads/' + title + ' heart rate.png'
        plt.savefig(heartrate_vs_concentration)
        plt.close()

        return heartrate_vs_concentration



    # code optimization
    def process_organoids(self, *concentrations):
        with ThreadPoolExecutor() as executor:
            results = executor.map(self.process_concentration, concentrations)
        return list(results)



    def process_concentration(self, concentration):
        pixel_intensity = [self.unet_segment(frame) for frame in self.frames(concentration)]
        # divides each element by the sum of all the elements in raw.
        # This effectively normalizes each element to be a value between 0 and 1.
        total_intensity = np.sum(pixel_intensity)
        normalized = [(concentration / total_intensity) * 100 for concentration in pixel_intensity]

        return normalized, self.calculate_heart_rate(normalized)



    def get_nframes(self, folder_path):
        """
        :param folder_path: path of frames
        :return: number of frames in the folder
        """
        files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        files = files[100:400]
        return len(files)



    def frames(self, folder_path):
        """
        :param folder_path: path of each frame
        :return frames: all frames in one folder
        """

        files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        files = files[100:400]  # 300 frames

        for file in files:
            image_path = os.path.join(folder_path, file)
            frame = Image.open(image_path)

            # Preprocess the image (resize, normalize, convert to array)
            frame = frame.resize((128, 128))
            frame_array = np.array(frame) / 255.0
            yield frame_array


    def unet_segment(self, frame):
        model = load_model('D:/ann/git2/U-Net/saved_unet.h5', compile=False, custom_objects={'mean_iou': mean_iou, 'dice_coefficient': dice_coefficient, 'pixel_wise_accuracy': pixel_wise_accuracy})

        prediction = model.predict(frame[np.newaxis, ...])
        binary_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)

        contours_frame, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours_frame:
            # To draw all the contours in an image
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 3)

            # Fill the area inside the contour with white
            mask = np.zeros_like(frame)
            cv2.fillPoly(mask, [contour], (255, 255, 255))

            # Access the image pixels with white and create a 1D numpy array then add to list
            pts = np.where(mask == 255)
            mean_intensity = np.mean(frame[pts[0], pts[1]])

            return mean_intensity


