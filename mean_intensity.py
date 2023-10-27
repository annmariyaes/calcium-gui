import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import time

start_time = time.time()


time_intervals = np.linspace(0, 10, 450)
colors = ['green', 'purple', 'orange', 'red']  # Define colors for each organoid


def frames(folder_path):
    """
    :param folder_path: path of each frame
    :return frames: all frames in one folder
    """
    # Assuming you have a folder with only TIFF files
    files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    files = files[100:550]

    for file in files:
        image_path = os.path.join(folder_path, file)
        frame = cv2.imread(image_path)
        yield frame


def calcium_mean_intensity(frame):
    """
    :param frame: path of each frame
    :return calcium_intensity: pixel intensity of entire image (not particularly segmented or ROI)
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''
    Gaussian blurring is highly effective in removing Gaussian noise from an image.
    We should specify the width and height of the kernel which should be positive and odd. 
    '''
    blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    '''
    Contrast Limited Adaptive Histogram Equalization (CLAHE)
    CLAHE is a variant of Adaptive histogram equalization which takes care of over-amplification of the contrast. 
    CLAHE operates on small regions in the image, called tiles, rather than the entire image. 
    The neighboring tiles are then combined using bilinear interpolation to remove the artificial boundaries. 
    Usually it is applied on the luminance channel
    '''
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe = clahe.apply(blur)

    '''
    Combination of dilation followed by erosion, known as morphological closing.
    Erosion basically strips out the outermost layer of pixels in a structure, 
    where as dilation adds an extra layer of pixels on a structure.
    Used to close small holes or gaps in objects and join objects that are close to each other.
    '''
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(clahe, kernel, iterations=4)
    erode = cv2.erode(dilate, kernel, iterations=5)

    '''
    Thresholding is a process of converting a grayscale image into a binary image, 
    where pixels are classified into two groups based on intensity values: 
    those above a certain threshold value and those below.
    cv2.THRESH_BINARY: sets all pixel values above a certain threshold to a maximum value (255) and all others to a minimum value (0). 
    cv2.THRESH_OTSU:  calculates an "optimal" threshold value based on the histogram of the image. 
    '''
    _, thresholded_frame = cv2.threshold(erode, 12, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Calculate the mean intensity value within the region of interest (ROI)
    calcium_intensity = np.mean(thresholded_frame)

    return calcium_intensity


# Proper segmentation of ROI
def roi_mean_intensity(frame):
    """
    :param frame: path of each frame
    :return calcium_intensity: pixel intensity of segmented fluorescent colored area or ROI
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

    mean_intensities = []
    for contour in contours_frame:
        # area in square pixels
        area_pixels = cv2.contourArea(contour)

        if area_pixels > 10000:
            # print(area_pixels)

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
            mean_intensities.append(mean_intensity)

            # extract the raw fluorescence within each object per frame, as the sum of all pixels in each object
            fluorescent_cal_trace = np.sum(frame[pts[0], pts[1]])

    return mean_intensity


# Fast Fourier Transform (FFT)
def calculate_heart_rate(intensity):
    """
    :param intensity: pixel intensity of each fluorescent image
    :return: frequency of that intensity
    """

    # graph of your waveform doesn't start from zero, it implies that there is a DC offset in your signal
    intensity -= np.mean(intensity)

    # DFT of a signal provides a way to represent that signal in terms of its frequency components
    spectrum = fft(intensity)

    num_samples = len(intensity)  # total number of data points or samples in wave.
    sample_rate = 30  # how many data points are recorded per second
    frequencies = fftfreq(num_samples, 1 / sample_rate)

    # dominant frequency in a signal is the frequency component that has the highest amplitude
    # Only consider positive frequencies (since the signal is real)
    positive_freq = frequencies[:num_samples // 2]
    magnitude = np.abs(spectrum[:num_samples // 2])
    dominant_frequency = positive_freq[np.argmax(magnitude)]

    return dominant_frequency


def process_organoids(*concentrations):
    """
    :param concentrations: takes the path of all images with respective concentration (0nM, 100nM, 500nM, 1000nM) in chemical
    :return: intensities of fluorescent images and its frequency.
    """
    frequencies, normalized = [], []

    for concentration in concentrations:
        # print(concentration)
        concentration_intensity = [roi_mean_intensity(frame) for frame in frames(concentration)]
        total_intensity = np.sum(concentration_intensity)
        normalize = [(conc / total_intensity) * 100 for conc in concentration_intensity]

        normalized.append(normalize)
        frequencies.append(calculate_heart_rate(normalize))

    return np.array(normalized), np.array(frequencies)


def plot_heart_rates(concentrations, heart_rates):
    """
    :param concentrations: 4 concentration value
    :param heart_rates: frequency value
    :return:
    """

    for i, heart_rate in enumerate(heart_rates):
        plt.scatter(concentrations, heart_rate, marker='o', label=f'Organoid {i+1}')
    plt.xlabel('Concentration (nM)')
    plt.ylabel('Heart Rate (Hz)')
    plt.title('Isoprenaline')
    plt.legend()
    plt.xticks(concentrations)
    plt.savefig('Isoprenaline heart rate.png')


# Isoprenaline
# Nifedifine
# E4031
# BPA
'''
concentrations = ['0', '100', '500', '1000']
organoids = [
    ('D:/ann/Experiment/Isoprenaline/Normal 1/', 'D:/ann/Experiment/Isoprenaline/100 nM Isoprenaline 1/', 'D:/ann/Experiment/Isoprenaline/500 nM Isoprenaline 1/', 'D:/ann/Experiment/Isoprenaline/1 uM Isoprenaline 1/'),
    ('D:/ann/Experiment/Isoprenaline/Normal 2/', 'D:/ann/Experiment/Isoprenaline/100 nM Isoprenaline 2/', 'D:/ann/Experiment/Isoprenaline/500 nM Isoprenaline 2/', 'D:/ann/Experiment/Isoprenaline/1 uM Isoprenaline 2/'),
    ('D:/ann/Experiment/Isoprenaline/Normal 3/', 'D:/ann/Experiment/Isoprenaline/100 nM Isoprenaline 3/', 'D:/ann/Experiment/Isoprenaline/500 nM Isoprenaline 3/', 'D:/ann/Experiment/Isoprenaline/1 uM Isoprenaline 3/')
]
'''
concentrations = ['0', '100', '500', '1000']
organoids = [
    ('D:/ann/Experiment/Nifedifine/Normal 1/', 'D:/ann/Experiment/Nifedifine/100 nM Nifedifine 1/', 'D:/ann/Experiment/Nifedifine/1 uM Nifedifine 1/', 'D:/ann/Experiment/Nifedifine/10 uM Nifedifine 1/'),
    ('D:/ann/Experiment/Nifedifine/Normal 2/', 'D:/ann/Experiment/Nifedifine/100 nM Nifedifine 2/', 'D:/ann/Experiment/Nifedifine/1 uM Nifedifine 2/', 'D:/ann/Experiment/Nifedifine/10 uM Nifedifine 2/'),
    ('D:/ann/Experiment/Nifedifine/Normal 3/', 'D:/ann/Experiment/Nifedifine/100 nM Nifedifine 3/', 'D:/ann/Experiment/Nifedifine/1 uM Nifedifine 3/', 'D:/ann/Experiment/Nifedifine/10 uM Nifedifine 3/')
    ]


heart_rates = []
for i, organoid in enumerate(organoids):

    mean_intensity, heart_rate = process_organoids(*organoid)

    # Plot the mean intensities
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for j, intensity in enumerate(mean_intensity):
        # print(organoid[j].split('/')[-2])
        axes[j // 2, j % 2].plot(time_intervals, intensity, color=colors[j], markersize=1)
        axes[j // 2, j % 2].set_title(organoid[j].split('/')[-2])
        axes[j // 2, j % 2].set_xlabel('Relative time (sec)')
        axes[j // 2, j % 2].set_ylabel('Mean Intensities')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(organoid[0].split('/')[-3] + ' intensity ' + str(i + 1))
    plt.close()

    heart_rates.append(heart_rate)

# print(heart_rates)

plot_heart_rates(concentrations, heart_rates)


end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")


'''
max_value = max(max(normal_intensity), max(hundred_nM_intensity), max(five_hundred_nM_intensity), max(one_um_intensity))

min_value = min(min(normal_intensity), min(hundred_nM_intensity), min(five_hundred_nM_intensity), min(one_um_intensity))
plt.stackplot(time_intervals, normal_normalized, hundred_nM_normalized, five_hundred_nM_normalized, one_um_normalized,
              labels=['Normal', '100 nM Isoprenaline', '500 nM Isoprenaline', '1 um E4031'])


# frequency
calculate_frequency(normal_intensity, "Normal")
calculate_frequency(hundred_nM_intensity, "100 nM")
calculate_frequency(five_hundred_nM_intensity, "500 nM")
calculate_frequency(one_um_intensity, "1 um")


# Plot the mean intensities
plt.plot(time_intervals, normal_intensity, color='green', marker='o', markersize=2, label='Normal')
# plt.plot(time_intervals, hundred_nM_intensity, color='purple', marker='o', markersize=2, label='100 nM Isoprenaline')
# plt.plot(time_intervals, five_hundred_nM_intensity, color='orange', marker='o', markersize=2, label='500 nM Isoprenaline')
plt.plot(time_intervals, one_um_intensity, color='red', marker='o', markersize=2, label='1 um E4031')
plt.xlabel('Relative time (sec)')
plt.ylabel('Mean Intensities')


# Isoprenaline increases the force of contraction of the heart muscle.
plt.title('Effect of Isoprenaline on Cardiomyocyte contraction rate (Experiment 1)')
plt.legend()
plt.savefig('Isoprenaline intensity 1.png')
plt.show()
'''




