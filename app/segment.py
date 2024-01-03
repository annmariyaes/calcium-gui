import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')

class Segmentation:
    def __init__(self, organoids, chemical, fps, time, t_range):
        self.organoids = organoids
        self.chemical = chemical
        self.fps = int(fps)
        self.time = int(time)
        self.t_range = t_range


    def display_intensity_plot(self):
        intensity_plot_paths = []
        colors = ['green', 'purple', 'orange', 'red']  # Define colors for each concentration
        # title = self.organoids[0][0].split('/')[-3]
        for i, organoid in enumerate(self.organoids):
            num_frames = self.fps * int(self.t_range[1]) - self.fps * int(self.t_range[0])
            time_intervals = np.linspace(int(self.t_range[0]), int(self.t_range[1]), num_frames)
            # pixel intensity plot (assuming there will be 4 plots)
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            mean_pixel_intensities, _ = zip(*self.process_organoids(*organoid))

            for j, intensity in enumerate(mean_pixel_intensities):
                axes[j//2, j%2].plot(time_intervals, intensity, color=colors[j], markersize=1)
                axes[j//2, j%2].set_title(organoid[j].split('/')[-1])
                axes[j//2, j%2].set_xlabel('Time (sec)')
                axes[j//2, j%2].set_ylabel('Mean Intensities (pixels)')

            # Adjust layout to prevent overlap
            plt.tight_layout()
            plot_filename = 'static/uploads/' + self.chemical + ' intensity ' + str(i+1) + '.png'
            plt.savefig(plot_filename)
            plt.close()
            intensity_plot_paths.append(plot_filename)
        return intensity_plot_paths


    def display_heartrate_plot(self, concentrations):
        heart_rates = []
        # title = self.organoids[0][0].split('/')[-3]
        for i, organoid in enumerate(self.organoids):
            _, heart_rate = zip(*self.process_organoids(*organoid))
            heart_rates.append(heart_rate)

        # plot of heart rate vs concentration
        for r, heart_rate in enumerate(heart_rates):
            plt.scatter(concentrations, heart_rate, marker='o', label=f'Organoid {r+1}')
        plt.xlabel('Concentration (nM)')
        plt.ylabel('Heart Rate (Hz)')
        plt.title(self.chemical)
        plt.legend()
        plt.xticks(concentrations)
        heartrate_vs_concentration = 'static/uploads/' + self.chemical + ' heart rate.png'
        plt.savefig(heartrate_vs_concentration)
        plt.close()
        return heartrate_vs_concentration


    # code optimization
    def process_organoids(self, *organoids):
        with ThreadPoolExecutor() as executor:
            results = executor.map(self.process_organoid, organoids)
        return list(results)

    def process_organoid(self, organoid):
        pixel_intensity = [self.roi_mean_intensity(frame) for frame in self.frames(organoid)]
        # divides each element by the sum of all the elements in raw.
        # This effectively normalizes each element to be a value between 0 and 1.
        total_intensity = np.sum(pixel_intensity)
        normalized = [(concentration/total_intensity)*100 for concentration in pixel_intensity]
        return normalized, self.calculate_heart_rate(normalized)


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
            frame = cv2.imread(image_path)
            yield frame


    # Proper segmentation of ROI
    def roi_mean_intensity(self, frame):
        """
        :param frame: path of that frame
        :return calcium_intensity: mean pixel intensity of ROI in that frame
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        '''
        Gaussian blurring is highly effective in removing Gaussian noise from an image.
        We should specify the width and height of the kernel which should be positive and odd. 
        '''
        blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        '''
        Contrast Limited Adaptive Histogram Equalization (CLAHE)
        CLAHE is a variant of Adaptive histogram equalization which takes care of over-amplification of the contrast. 
        CLAHE operates on small regions in the image, called tiles, rather than the entire image. 
        The neighboring tiles are then combined using bilinear interpolation to remove the artificial boundaries. 
        Usually it is applied on the luminance channel
        '''
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_frame = clahe.apply(blur_frame)

        '''
        Combination of dilation followed by erosion, known as morphological closing.
        Erosion basically strips out the outermost layer of pixels in a structure, 
        where as dilation adds an extra layer of pixels on a structure.
        Used to close small holes or gaps in objects and join objects that are close to each other.
        '''
        kernel = np.ones((5, 5), np.uint8)  # kernel size is very sensitive!!!!
        erode_frame = cv2.erode(cv2.dilate(clahe_frame, kernel, iterations=4), kernel, iterations=5)

        # Region of Interest (ROI) Selection
        '''
        Thresholding is a process of converting a grayscale image into a binary image, 
        where pixels are classified into two groups based on intensity values: 
        those above a certain threshold value and those below.
        cv2.THRESH_BINARY: sets all pixel values above a certain threshold to a maximum value (255) 
        and all others to a minimum value (0). 
        cv2.THRESH_OTSU:  calculates an "optimal" threshold value based on the histogram of the image. 
        '''
        _, threshold_frame = cv2.threshold(erode_frame, 12, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        '''
        Contours are curve joining all the continuous points (along the boundary), having same color or intensity.
        If you pass cv.CHAIN_APPROX_NONE, all the boundary points are stored
        '''
        contours_frame, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours_frame:
            # area in square pixels
            area_pixels = cv2.contourArea(contour)

            if area_pixels > 10000:
                # To draw all the contours in an image
                cv2.drawContours(frame, [contour], -1, (0, 255, 255), 3)
                # Fill the area inside the contour with white
                mask = np.zeros_like(frame)
                segmented_frame = cv2.fillPoly(mask, [contour], (255, 255, 255))
                '''
                cv2.imshow('Segmentation', segmented_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''
                # Access the image pixels with white and create a 1D numpy array then add to list
                pts = np.where(mask == 255)
                mean_intensity = np.mean(frame[pts[0], pts[1]])
                return mean_intensity


    # Fast Fourier Transform (FFT)
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



'''
concentrations = ['0', '100', '500', '1000']
organoids = [
    ('D:/ann/Experiment/Isoprenaline/Normal 1/', 'D:/ann/Experiment/Isoprenaline/100 nM Isoprenaline 1/', 'D:/ann/Experiment/Isoprenaline/500 nM Isoprenaline 1/', 'D:/ann/Experiment/Isoprenaline/1 uM Isoprenaline 1/'),
    ('D:/ann/Experiment/Isoprenaline/Normal 2/', 'D:/ann/Experiment/Isoprenaline/100 nM Isoprenaline 2/', 'D:/ann/Experiment/Isoprenaline/500 nM Isoprenaline 2/', 'D:/ann/Experiment/Isoprenaline/1 uM Isoprenaline 2/'),
    ('D:/ann/Experiment/Isoprenaline/Normal 3/', 'D:/ann/Experiment/Isoprenaline/100 nM Isoprenaline 3/', 'D:/ann/Experiment/Isoprenaline/500 nM Isoprenaline 3/', 'D:/ann/Experiment/Isoprenaline/1 uM Isoprenaline 3/')
]

us1 = Segmentation(organoids, concentrations)
us1.display_heartrate_plot()
'''