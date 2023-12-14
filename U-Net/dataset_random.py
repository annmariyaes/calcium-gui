import os
import random
import shutil

# for U-Net dataset preparation
# Total 3625 images
# train=2537 (70%), valid=544 (15%), test=544 (15%)


def randomly_pick_images(source_folder, destination_folder, n):
    """
    This script will randomly select n images from the specified source folder and copy them to the destination folder.
    Also, deletes the selected images from the source folder after copying them to the destination folder.

    :param source_folder: source folder containing images
    :param destination_folder: destination folder where selected images will be copied
    :param n: number of files to copy into another folder and delete later
    """

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    files = os.listdir(source_folder)
    selected_files = random.sample(files, n)

    for file_name in selected_files:
        source_file_path = os.path.join(source_folder, file_name)
        destination_file_path = os.path.join(destination_folder, file_name)
        shutil.copyfile(source_file_path, destination_file_path)
        os.remove(source_file_path)
        print(f"Image '{file_name}' copied to '{destination_folder}'")


source_directory = 'D:/ann/Experiment/Nifedifine/temp/'
destination_directory = 'D:/ann/Experiment/dataset/new/'

# train=2537 (70%), valid=544 (15%), test=544 (15%)
number_of_images_to_pick = 0
randomly_pick_images(source_directory, destination_directory, number_of_images_to_pick)
