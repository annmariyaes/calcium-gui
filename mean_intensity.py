import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.fft import fft, fftfreq


time_intervals = np.linspace(0, 15, 450)


def frames(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    files = files[:450]

    frames = []
    for file in files:
        image_path = os.path.join(folder_path, file)
        frame = cv2.imread(image_path)
        frames.append(frame)

    return frames


def calcium_mean_intensity(frame):
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


def roi_mean_intensity(frame):
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
    dilate_frame = cv2.dilate(clahe_frame, kernel, iterations=4)  # joining broken parts of an object.
    erode_frame = cv2.erode(dilate_frame, kernel, iterations=5)  # for removing small white noises

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
    for i, contour in enumerate(contours_frame):
        # area in square pixels
        area_pixels = cv2.contourArea(contour)
        print()

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


# Baseline < 100 nM Isoprenaline < 500 nM Isoprenaline < 1 µM (or 1000nM) Isoprenaline.

# Assuming you have a folder with only TIFF files
normal = 'D:/ann/Experiment/Isoprenaline/Normal 1/'
hundred_nM = 'D:/ann/Experiment/Isoprenaline/100 nM Isoprenaline 1/'
five_hundred_nM = 'D:/ann/Experiment/Isoprenaline/500 nM Isoprenaline 1/'
one_um = 'D:/ann/Experiment/Isoprenaline/1 um Isoprenaline 1/'

# Extract calcium concentration values from the frames
normal_intensity = [roi_mean_intensity(frame) for frame in frames(normal)]
hundred_nM_intensity = [roi_mean_intensity(frame) for frame in frames(hundred_nM)]
five_hundred_nM_intensity = [roi_mean_intensity(frame) for frame in frames(five_hundred_nM)]
one_um_intensity = [roi_mean_intensity(frame) for frame in frames(one_um)]


'''
float(i)/sum(raw) divides each element by the sum of all the elements in raw. 
This effectively normalizes each element to be a value between 0 and 1. 
'''
normalize1 = [(float(i)/sum(normal_intensity))*100 for i in normal_intensity]
normalize2 = [(float(i)/sum(hundred_nM_intensity))*100 for i in hundred_nM_intensity]
normalize3 = [(float(i)/sum(five_hundred_nM_intensity))*100 for i in five_hundred_nM_intensity]
normalize4 = [(float(i)/sum(one_um_intensity))*100 for i in one_um_intensity]


# Calculate heart rates
heart_rate_normal = calculate_heart_rate(normalize1)
heart_rate_100nM = calculate_heart_rate(normalize2)
heart_rate_500nM = calculate_heart_rate(normalize3)
heart_rate_1um = calculate_heart_rate(normalize4)

relative_heart_rate_100nM = (heart_rate_100nM / heart_rate_normal) * 100
relative_heart_rate_500nM = (heart_rate_500nM / heart_rate_normal) * 100
relative_heart_rate_1um = (heart_rate_1um / heart_rate_normal) * 100

# Create a bar plot
concentrations = [0, 100, 500, 1000]
relative_heart_rates = [heart_rate_normal, heart_rate_100nM, heart_rate_500nM, heart_rate_1um]
print(relative_heart_rates)

plt.scatter(concentrations, relative_heart_rates, color='green')
plt.xlabel('Concentration (nM)')
plt.ylabel('Relative Heart Rate (%)')
plt.title('Relative Heart Rate vs Concentration')
# Set the x-axis ticks to be the specific concentrations
plt.xticks(concentrations)
plt.show()


'''
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot the mean intensities
axes[0, 0].plot(time_intervals, normalize1, color='green', markersize=1)
axes[0, 0].set_title('Normal')
axes[0, 0].set_xlabel('Relative time (sec)')
axes[0, 0].set_ylabel('Mean Intensities')

axes[0, 1].plot(time_intervals, normalize2, color='purple', markersize=1)
axes[0, 1].set_title('100 nM Isoprenaline')
axes[0, 1].set_xlabel('Relative time (sec)')
axes[0, 1].set_ylabel('Mean Intensities')

axes[1, 0].plot(time_intervals, normalize3, color='orange', markersize=1)
axes[1, 0].set_title('500 nM Isoprenaline')
axes[1, 0].set_xlabel('Relative time (sec)')
axes[1, 0].set_ylabel('Mean Intensities')

axes[1, 1].plot(time_intervals, normalize4, color='red',  markersize=1)
axes[1, 1].set_title('1 um Isoprenaline')
axes[1, 1].set_xlabel('Relative time (sec)')
axes[1, 1].set_ylabel('Mean Intensities')


# Isoprenaline increases the force of contraction of the heart muscle.
# plt.title('Effect of Isoprenaline on Cardiomyocyte contraction rate (Experiment 1)')
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.savefig('Isoprenaline intensity 1_normalized.png', bbox_inches='tight', dpi=300)
plt.show()
'''


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


# Isoprenaline
# E4031
