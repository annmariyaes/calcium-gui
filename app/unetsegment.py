import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import argrelextrema, find_peaks

from keras.models import load_model
import tensorflow.python.keras.backend as K
from concurrent.futures import ThreadPoolExecutor



class Unet:

    def __init__(self, organoids, chemical, fps, time, t_range):
        self.organoids = organoids
        self.chemical = chemical
        self.fps = int(fps)
        self.time = int(time)
        self.t_range = t_range
        # C:/Users/annma/PycharmProjects/calcium-gui/U-Net/updated_unet.h5
        self.model = load_model('D:/ann/calcium-gui/U-Net/updated_unet.h5', compile=False, custom_objects={
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


    def frames(self, folder_path):
        """
        :param folder_path: path of each frame
        :return frames: all frames in one folder
        """
        files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        start = self.fps * int(self.t_range[0])
        stop = self.fps * int(self.t_range[1])
        files = files[start:stop]  # 300 frames

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
        intensity_plot_paths = []
        for i, organoid in enumerate(self.organoids):
            num_frames = self.fps * int(self.t_range[1]) - self.fps * int(self.t_range[0])
            time_intervals = np.linspace(int(self.t_range[0]), int(self.t_range[1]), num_frames)
            colors = ['blue', 'purple', 'orange', 'magenta']  # Define colors for each concentration

            mean_pixel_intensities, _ = zip(*self.process_organoids(*organoid))

            # pixel intensity plot (assuming there will be 4 plots)
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            for j, intensity in enumerate(mean_pixel_intensities):
                # Find local maxima and minima
                intensity = np.array(intensity)
                maxima_indices = argrelextrema(intensity, np.greater)
                minima_indices = argrelextrema(intensity, np.less)

                axes[j//2, j%2].plot(time_intervals, intensity, color=colors[j], markersize=1)

                axes[j//2, j%2].scatter(time_intervals[maxima_indices], intensity[maxima_indices], color='red', label='Local Maxima')
                axes[j//2, j%2].scatter(time_intervals[minima_indices], intensity[minima_indices], color='green', label='Local Minima')
                axes[j//2, j%2].set_title(organoid[j].split('/')[-1])
                axes[j//2, j%2].set_xlabel('Time (sec)')
                axes[j//2, j%2].set_ylabel('Mean Intensities (pixels)')

            # Adjust layout to prevent overlap
            plt.tight_layout()
            t = self.t_range[0] + '-' + self.t_range[1]
            plot_filename = 'static/uploads/' + self.chemical + ' intensity t' + t + ' ' + str(i+1) + '.png'
            plt.savefig(plot_filename)
            plt.close()
            intensity_plot_paths.append(plot_filename)
        return intensity_plot_paths


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


    def display_heartrate_plot(self, concentrations):
        heart_rates = []
        for i, organoid in enumerate(self.organoids):
            _, heart_rate = zip(*self.process_organoids(*organoid))
            heart_rates.append(heart_rate)

        # plot of heart rate vs concentration
        for r, heart_rate in enumerate(heart_rates):
            marker_size = 100+r*5
            plt.scatter(concentrations, heart_rate, marker='o', s=marker_size, label=f'Organoid {r+1}')
        plt.xlabel('Concentration (nM)')
        plt.ylabel('Heart Rate (Hz)')
        plt.title(self.chemical)
        plt.legend()
        plt.xticks(concentrations)
        heartrate_vs_concentration = 'static/uploads/' + self.chemical + ' heart rate.png'
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

us1 = Unet(organoids, "Nifedifine", 30, 30, [5,10])
us1.display_intensity_plot()
'''