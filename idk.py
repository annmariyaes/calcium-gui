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


# Fourier Transform
def calculate_heart_rate(intensity, label):
    time_intervals = np.linspace(0, 15, 450)
    # Perform FFT on normal_intensity
    fft_result = fft(intensity)

    # Generate frequencies corresponding to FFT result
    num_samples = len(intensity)
    sampling_rate = 1 / (time_intervals[1] - time_intervals[0])
    frequencies = fftfreq(num_samples, 1 / sampling_rate)

    # Find the index corresponding to the dominant frequency
    # by only considering the first half of the FFT result (positive values)
    dominant_freq_index = np.argmax(np.abs(fft_result[:num_samples // 2]))
    print(np.argmax(np.abs(fft_result)))

    # Convert the index to frequency in Hz
    dominant_freq_hz = frequencies[dominant_freq_index]

    # print(f"{label}: {heart_rate_bpm:.3f} BPM")
    return dominant_freq_hz


# Baseline < 100 nM Isoprenaline < 500 nM Isoprenaline < 1 µM (or 1000nM) Isoprenaline.

# Assuming you have a folder with only TIFF files
normal = 'D:/ann/Experiment/Isoprenaline/Normal 1/'
hundred_nM = 'D:/ann/Experiment/Isoprenaline/100 nM Isoprenaline 1/'


# Extract calcium concentration values from the frames
normal_intensity = [roi_mean_intensity(frame) for frame in frames(normal)]
hundred_nM_intensity = [roi_mean_intensity(frame) for frame in frames(hundred_nM)]

normalize1 = [(float(i)/sum(normal_intensity))*100 for i in normal_intensity]
normalize2 = [(float(i)/sum(hundred_nM_intensity))*100 for i in hundred_nM_intensity]

# Calculate heart rates
heart_rate_normal = calculate_heart_rate(normalize1, 'Normal')
heart_rate_100nM = calculate_heart_rate(normalize2, '100 nM Isoprenaline')
print(heart_rate_normal, heart_rate_100nM)


