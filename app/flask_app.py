import os
import cv2
import base64
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, redirect, url_for
# Set Matplotlib backend to Agg
import matplotlib
matplotlib.use('Agg')


def frames(folder_path):
    """
    :param folder_path: path of each frame
    :return frames: all frames in one folder
    """
    # Assuming you have a folder with only TIFF files
    files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    files = files[100:550]  # 450 frames

    for file in files:
        image_path = os.path.join(folder_path, file)
        frame = cv2.imread(image_path)
        yield frame


# Proper segmentation of ROI
def roi_mean_intensity(frame):
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

            return mean_intensity


# Fast Fourier Transform (FFT)
def calculate_heart_rate(intensity):
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
    frequencies = fftfreq(num_samples, 1 / sample_rate)

    # dominant frequency in a signal is the frequency component that has the highest amplitude
    # Only consider positive frequencies (since the signal is real)
    positive_freq = frequencies[:num_samples // 2]
    magnitude = np.abs(spectrum[:num_samples // 2])
    dominant_frequency = positive_freq[np.argmax(magnitude)]

    return dominant_frequency


def process_concentration(concentration):
    pixel_intensity = [roi_mean_intensity(frame) for frame in frames(concentration)]
    # divides each element by the sum of all the elements in raw.
    # This effectively normalizes each element to be a value between 0 and 1.
    total_intensity = np.sum(pixel_intensity)
    normalized = [(concentration / total_intensity) * 100 for concentration in pixel_intensity]
    return normalized, calculate_heart_rate(normalized)


# code optimization
def process_organoids(*concentrations):
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_concentration, concentrations)
    return list(results)


def generate_heartrate_plot(organoids, concentrations):

    heart_rates = []
    intensity_plot_paths = []
    time_intervals = np.linspace(0, 10, 450)
    colors = ['green', 'purple', 'orange', 'red']  # Define colors for each organoid
    title = organoids[0][0].split('/')[-2].split()[0]

    for i, organoid in enumerate(organoids):
        mean_pixel_intensity, heart_rate = zip(*process_organoids(*organoid))

        # pixel intensity plot
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        for j, intensity in enumerate(mean_pixel_intensity):
            axes[j // 2, j % 2].plot(time_intervals, intensity, color=colors[j], markersize=1)
            axes[j // 2, j % 2].set_title(organoid[j].split('/')[-1])
            axes[j // 2, j % 2].set_xlabel('Time (sec)')
            axes[j // 2, j % 2].set_ylabel('Mean Intensities (pixels)')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plot_filename = 'static/uploads/' + title + ' intensity ' + str(i + 1) + '.png'
        plt.savefig(plot_filename)
        plt.close()
        intensity_plot_paths.append(plot_filename)

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

    return intensity_plot_paths, heartrate_vs_concentration


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():

    all_folders = []
    for i in range(1, 4):
        file = request.files.get(f'file{i}')
        if file:
            if file.filename.endswith('.zip'):
                # Create uploads directory if it doesn't exist
                if not os.path.exists(UPLOAD_FOLDER):
                    os.makedirs(UPLOAD_FOLDER)

                # Save the uploaded file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                # Extract the contents of the zip file
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(app.config['UPLOAD_FOLDER'], file.filename[:-4]))

                extracted_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename[:-4])
                # print(extracted_folder_path)

                folders = []
                for fold in os.listdir(extracted_folder_path):
                    folder = os.path.join(extracted_folder_path, fold)
                    folder = folder.replace('\\', '/')
                    folders.append(folder)

                all_folders.append(folders)

            else:
                return "Invalid file format. Please upload a .zip file."

    f1 = request.files['file1'].filename
    f2 = request.files['file2'].filename
    f3 = request.files['file3'].filename
    print(f1, f2, f3)  # not working!!

    # textbox to enter concentration values
    textbox_value = request.form['textbox']
    text = [str(x.strip()) for x in ''.join(textbox_value).split(',')]

    # pixel intensity plot, heart rate vs concentration plot
    plot1, plot2 = generate_heartrate_plot(all_folders, text)

    return render_template('index.html',
                           intensity_plots=plot1,
                           plot_heartrate=plot2,
                           zip1=f1, zip2=f2, zip3=f3,
                           concentrations=textbox_value)


if __name__ == '__main__':
    app.run(debug=True)



