import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# for U-Net data preparation: mask
source_folder = 'D:/ann/Experiment/dataset/new/'
destination_folder = 'D:/ann/Experiment/dataset/new mask/'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)


files = [f for f in os.listdir(source_folder) if f.endswith('.tif')]
# files = files[100:150]
# print(len(files))

surface_area = []
mean_intensities = []

for file_name in files:
    # Construct the full path to the image
    file_path = os.path.join(source_folder, file_name)

    # Read the image using OpenCV
    image = cv2.imread(file_path)

    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    kernel = np.ones((5, 5), np.uint8)
    dilate_frame = cv2.dilate(clahe_frame, kernel, iterations=4)
    erode_frame = cv2.erode(dilate_frame, kernel, iterations=5)

    # Region of Interest (ROI) Selection
    '''
    Thresholding is a process of converting a grayscale image into a binary image, 
    where pixels are classified into two groups based on intensity values: 
    those above a certain threshold value and those below.
    cv2.THRESH_BINARY: sets all pixel values above a certain threshold to a maximum value (255) and all others to a minimum value (0). 
    cv2.THRESH_OTSU:  calculates an "optimal" threshold value based on the histogram of the image. 
    '''
    _, threshold_frame = cv2.threshold(erode_frame, 12, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # cv2.imshow('Threshold', threshold_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    '''
    Contours are curve joining all the continuous points (along the boundary), having same color or intensity.
    If you pass cv.CHAIN_APPROX_NONE, all the boundary points are stored
    '''
    contours_frame, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(contours_frame))

    for i, contour in enumerate(contours_frame):
        # area in square pixels
        area_pixels = cv2.contourArea(contour)

        if area_pixels > 10000:
            fluorescent_cal_trace = np.sum(contour)
            # print('fluorescent calcium trace:', fluorescent_cal_trace)
            dpi = 300  # resolution or dots per inch (DPI)

            # print(len(contour))
            # print('Area in pixels:', area_pixels)

            # To draw all the contours in an image
            cv2.drawContours(image, [contour], -1, (0, 255, 255), 3)

            # Fill the area inside the contour with white
            mask = np.zeros_like(image)
            segmented_frame = cv2.fillPoly(mask, [contour], (255, 255, 255))

            # actual area with all white regions!!!
            actual_area = cv2.countNonZero(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
            # print('Area:', actual_area)

            # Convert pixels to square centimeters
            '''
            To convert the area from square pixels to square centimeters, you'll need to use the conversion factor, 
            which represents how many centimeters one pixel corresponds to. 
            This factor depends on the resolution or dots per inch (DPI) of the image.
            '''
            area_cm2 = (actual_area / math.sqrt(dpi)) * math.sqrt((1 / 2.54))

            surface_area.append(area_cm2)
            # print('Area in square centimeters:', area_cm2)

            # Access the image pixels with white and create a 1D numpy array then add to list
            pts = np.where(mask == 255)
            mean_intensity = np.mean(image[pts[0], pts[1]])
            mean_intensities.append(mean_intensity)
            # print(mean_intensity)

            # extract the raw fluorescence within each object per frame, as the sum of all pixels in each object
            fluorescent_cal_trace = np.sum(image[pts[0], pts[1]])
            # print('Fluorescent calcium trace:', fluorescent_cal_trace)

            # destination_file = destination_folder + file_name.split('.')[0] + ' segmented' + '.tif'
            # cv2.imwrite(destination_file, segmented_frame)

            # bad segmentation

            if area_cm2 >= 3000:
                destination_file1 = destination_folder + file_name.split('.')[0] + ' segmented' + '.tif'
                cv2.imwrite(destination_file1, segmented_frame)
                print(area_cm2)





