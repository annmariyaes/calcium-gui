import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def frames(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    files = files[:450]

    frames = []
    for file in files:
        image_path = os.path.join(folder_path, file)
        frame = cv2.imread(image_path)
        frames.append(frame)

    return frames


def surface_area(frame):
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
    cv2.THRESH_BINARY: sets all pixel values above a certain threshold to a maximum value (255) and all others to a minimum value (0). 
    cv2.THRESH_OTSU:  calculates an "optimal" threshold value based on the histogram of the image. 
    '''
    _, threshold_frame = cv2.threshold(erode_frame, 12, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    '''
    Contours are curve joining all the continuous points (along the boundary), having same color or intensity.
    If you pass cv.CHAIN_APPROX_NONE, all the boundary points are stored
    '''
    contours_frame, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i, contour in enumerate(contours_frame):
        # area in square pixels
        area_pixels = cv2.contourArea(contour)

        if area_pixels > 10000:
            # To draw all the contours in an image
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 3)

            # Fill the area inside the contour with white
            mask = np.zeros_like(frame)
            segmented_frame = cv2.fillPoly(mask, [contour], (255, 255, 255))

            # actual area!!!
            actual_area = cv2.countNonZero(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

            # Convert pixels to square centimeters
            '''
            To convert the area from square pixels to square centimeters, you'll need to use the conversion factor, 
            which represents how many centimeters one pixel corresponds to. 
            This factor depends on the resolution or dots per inch (DPI) of the image.
            '''
            dpi = 300  # resolution or dots per inch (DPI)
            area_cm2 = (actual_area / math.sqrt(dpi)) * math.sqrt((1 / 2.54))

            # Access the image pixels with white and create a 1D numpy array then add to list
            pts = np.where(mask == 255)
            mean_intensity = np.mean(frame[pts[0], pts[1]])
            # print(mean_intensity)

            # extract the raw fluorescence within each object per frame, as the sum of all pixels in each object
            fluorescent_cal_trace = np.sum(frame[pts[0], pts[1]])

    return area_cm2


time_intervals = np.linspace(0, 15, 450)

# Baseline > 100 nM Isoprenaline > 500 nM Isoprenaline > 1 µM Isoprenaline.

# Assuming you have a folder with only TIFF files
normal = 'D:/ann/Experiment/E4031/Normal 1/'
hundred_nM = 'D:/ann/Experiment/E4031/100 nM E4031 1/'
five_hundred_nM = 'D:/ann/Experiment/E4031/500 nM E4031 1/'
one_um = 'D:/ann/Experiment/E4031/1 um E4031 1/'

normal_area = [surface_area(frame) for frame in frames(normal)]
hundred_nM_area = [surface_area(frame) for frame in frames(hundred_nM)]
five_hundred_nM_area = [surface_area(frame) for frame in frames(five_hundred_nM)]
one_um_area = [surface_area(frame) for frame in frames(one_um)]

plt.plot(time_intervals, normal_area, color='green', marker='o', markersize=2, label='Normal')
plt.plot(time_intervals, hundred_nM_area, color='purple', marker='o', markersize=2, label='100 nM E4031')
plt.plot(time_intervals, five_hundred_nM_area, color='orange', marker='o', markersize=2, label='500 nM E4031')
plt.plot(time_intervals, one_um_area, color='red', marker='o', markersize=2, label='1 um E4031')
plt.xlabel('Relative time (sec)')
plt.ylabel('Surface area (sq cm)')

# Isoprenaline increases the force of contraction of the heart muscle.
plt.title('Effect of E4031 on Cardiomyocyte surface area (Experiment 1)')
plt.legend()
plt.savefig('E4031 area 1')
plt.show()

# Isoprenaline
# E4031

