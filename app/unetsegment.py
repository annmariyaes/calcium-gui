import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

from keras.models import load_model
import tensorflow.python.keras.backend as K
from concurrent.futures import ThreadPoolExecutor



class UnetSegmentation:

    def __init__(self, organoids, concentrations):
        self.organoids = organoids
        self.concentrations = concentrations
        self.model = load_model('D:/ann/git2/U-Net/updated_unet.h5', compile=False, custom_objects={
                          'mean_iou': self.mean_iou, 'dice_coefficient': self.dice_coefficient, 'pixel_wise_accuracy': self.pixel_wise_accuracy})


    def mean_iou(self, y_true, y_pred):
        intersection = K.sum(K.abs(y_true * K.round(y_pred)))
        union = K.sum(y_true) + K.sum(K.round(y_pred)) - intersection
        iou = intersection / (union + K.epsilon())
        return iou

    def dice_coefficient(self, y_true, y_pred):
        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true) + K.sum(y_pred)
        dice = numerator / (denominator + K.epsilon())
        return dice

    # number of pixels that are classified correctly in the generated segmentation mask
    def pixel_wise_accuracy(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


    def get_nframes(self, folder_path):
        """
        :param folder_path: path of frames
        :return: number of frames in the folder
        """
        files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        files = files[:300]
        return len(files)


    def frames(self, folder_path):
        """
        :param folder_path: path of each frame
        :return frames: all frames in one folder
        """

        files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        files = files[:300]  # 300 frames

        for file in files:
            image_path = os.path.join(folder_path, file)
            frame = Image.open(image_path)

            # Preprocess the image (resize, normalize, convert to array)
            frame = frame.resize((128, 128))
            yield frame


    def unet_segment(self, frame):

        # Pretrained UNet model
        frame_array = np.array(frame)
        frame_array_normalized = frame_array / 255.0
        prediction = self.model.predict(frame_array_normalized[np.newaxis, ...])
        binary_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)

        contours_frame, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours_frame:
            # To draw all the contours in an image
            cv2.drawContours(frame_array, [contour], -1, (0, 255, 255), 3)

            # Fill the area inside the contour with white
            mask = np.zeros_like(frame_array)
            cv2.fillPoly(mask, [contour], (255, 255, 255))

            # Access the image pixels with white and create a 1D numpy array then add to list
            pts = np.where(mask == 255)
            mean_intensity = np.mean(frame_array[pts[0], pts[1]])

            return mean_intensity

    def process_organoids(self, *organoids):
        results = []
        for organoid in organoids:
            with ThreadPoolExecutor() as executor:
                frames = self.frames(organoid)
                pixel_intensity = list(executor.map(self.unet_segment, frames))
            total_intensity = np.sum(pixel_intensity)
            normalized = [(concentration/total_intensity)*100 for concentration in pixel_intensity]
            results.append((normalized, self.calculate_heart_rate(pixel_intensity)))
        return results


    def display_intensity_plot(self):

        for i, organoid in enumerate(self.organoids):

            num_frames = self.get_nframes(organoid[0])
            stop = num_frames/30
            time_intervals = np.linspace(0, stop, num_frames)
            colors = ['green', 'purple', 'orange', 'red']  # Define colors for each concentration
            title = self.organoids[0][0].split('/')[-3]
            # pixel intensity plot (assuming there will be 4 plots)
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            mean_pixel_intensities, _ = zip(*self.process_organoids(*organoid))

            for j, intensity in enumerate(mean_pixel_intensities):
                axes[j//2, j%2].plot(time_intervals, intensity, color=colors[j], markersize=1)
                axes[j//2, j%2].set_title(organoid[j].split('/')[-2])
                axes[j//2, j%2].set_xlabel('Time (sec)')
                axes[j//2, j%2].set_ylabel('Mean Intensities (pixels)')

            # Adjust layout to prevent overlap
            plt.tight_layout()
            plot_filename = 'static/uploads/' + title + ' intensity ' + str(i+1) + '.png'
            plt.savefig(plot_filename)
            plt.close()


    def calculate_heart_rate(self, intensity):
        """
        :param intensity: mean pixel intensity of all frames in a particular concentration
        :return: frequency of that intensity
        """

        # graph of your waveform doesn't start from zero, it implies that there is a DC offset in your signal
        intensity -= np.mean(intensity)

        # DFT of a signal provides a way to represent that signal in terms of its frequency components
        spectrum = fft(intensity)

        num_samples = len(intensity)  # total number of data points or samples in wave.
        sample_rate = 30  # how many data points are recorded per second
        frequencies = fftfreq(num_samples, 1/sample_rate)

        # dominant frequency in a signal is the frequency component that has the highest amplitude
        # Only consider positive frequencies (since the signal is real)
        positive_freq = frequencies[:num_samples//2]
        magnitude = np.abs(spectrum[:num_samples//2])
        dominant_frequency = positive_freq[np.argmax(magnitude)]

        return dominant_frequency

    def display_heartrate_plot(self):

        heart_rates = []
        title = self.organoids[0][0].split('/')[-3]

        for i, organoid in enumerate(self.organoids):
            _, heart_rate = zip(*self.process_organoids(*organoid))
            heart_rates.append(heart_rate)

        # plot of heart rate vs concentration
        for r, heart_rate in enumerate(heart_rates):
            plt.scatter(self.concentrations, heart_rate, marker='o', label=f'Organoid {r+1}')
        plt.xlabel('Concentration (nM)')
        plt.ylabel('Heart Rate (Hz)')
        plt.title(title)
        plt.legend()
        plt.xticks(self.concentrations)
        heartrate_vs_concentration = 'static/uploads/' + title + ' heart rate.png'
        plt.savefig(heartrate_vs_concentration)
        plt.close()

        return heartrate_vs_concentration


'''
concentrations = ['0', '100', '500', '1000']
organoids = [
    ('D:/ann/Experiment/E4031/Normal 1/', 'D:/ann/Experiment/E4031/100 nM E4031 1/', 'D:/ann/Experiment/E4031/500 nM E4031 1/', 'D:/ann/Experiment/E4031/1 uM E4031 1/'),
    ('D:/ann/Experiment/E4031/Normal 2/', 'D:/ann/Experiment/E4031/100 nM E4031 2/', 'D:/ann/Experiment/E4031/500 nM E4031 2/', 'D:/ann/Experiment/E4031/1 uM E4031 2/'),
    ('D:/ann/Experiment/E4031/Normal 3/', 'D:/ann/Experiment/E4031/100 nM E4031 3/', 'D:/ann/Experiment/E4031/500 nM E4031 3/', 'D:/ann/Experiment/E4031/1 uM E4031 3/')
]

us1 = UnetSegmentation(organoids, concentrations)
us1.display_intensity_plot()
'''