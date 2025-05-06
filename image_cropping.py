import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
import math
from PIL import Image
from multiprocessing import Pool, cpu_count

input_dir = 'database/resized_train/resized_train'
new_sz = 1024

extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

files = []
for ext in extensions:
    files.extend(glob.glob(os.path.join(input_dir, ext)))

def process_image_for_cropping(image):
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print('\nNo contours found!')
        return image, 0

    cnt = max(contours, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(cnt)
    x = int(x)
    y = int(y)
    r = int(r)

    if r > 100:
        top = max(0, y - r)
        bottom = min(image.shape[0], y + r)
        left = max(0, x - r)
        right = min(image.shape[1], x + r)

        cropped = output[top:bottom, left:right]

        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            resized = cv2.resize(cropped, (new_sz, new_sz), interpolation=cv2.INTER_AREA)
            return resized, 1
        else:
            return image, 0

    return image, 1

def process_image(file):
    try:
        img = cv2.imread(file)
        if img is not None:
            processed_img, success = process_image_for_cropping(img)
            if success:
                filename = os.path.basename(file)
                output_path = os.path.join('output', filename)
                cv2.imwrite(output_path, processed_img)
        else:
            print(f"Could not read file: {file}")
    except Exception as e:
        print(f"Error while processing {file}: {str(e)}")

def crop_image():
    print("Image cropping stage")
    print("------------------------")
    print(f"Number of files found: {len(files)}")

    if not os.path.exists('output'):
        os.makedirs('output')

    num_processes = cpu_count()
    print(f"Using {num_processes} parallel processes")

    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_image, files), total=len(files), desc="Processing images"))
        pool.close()
        pool.join()
    