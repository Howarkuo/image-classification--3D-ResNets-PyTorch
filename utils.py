import os
import numpy as np
import nibabel as nib

def check_and_create_dir(dir_path: str):
    if os.path.exists(dir_path):
        print(f'{dir_path} exists')
    else:
        print(f'{dir_path} does not exist. Creating new one...')
        os.makedirs(dir_path, exist_ok=True)

def standardize(input_array: np.ndarray):
    input_array = np.nan_to_num(input_array, copy=False)
    return (input_array - np.mean(input_array)) / np.std(input_array)

def normalize(input_array: np.ndarray):
    input_array = np.nan_to_num(input_array, copy=False)
    # Min-Max Normalization
    return (input_array - np.min(input_array)) / (np.max(input_array) - np.min(input_array)) * 255.

def read_nib_image(input_path: str):
    image = nib.load(input_path).get_fdata()
    image = normalize(image)
    return image
