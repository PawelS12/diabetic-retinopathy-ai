import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Value
import multiprocessing
from tqdm import tqdm

csv_path = r'database\trainLabels_updated.csv'
data_dir = r'output'
output_dir = r'split'

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

wrongPhotosCounter = Value('i', 0)

def copy_single_image(args):
    row, split_type = args
    image_name = row['image'] + '.jpeg'
    level = str(row['level'])
    src_path = os.path.join(data_dir, level, image_name)
    dst_dir = os.path.join(output_dir, split_type, level)
    dst_path = os.path.join(dst_dir, image_name)
    
    os.makedirs(dst_dir, exist_ok=True)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        with wrongPhotosCounter.get_lock():
            wrongPhotosCounter.value += 1

def copy_images(df, split_type, max_workers=4):
    rows = [(row, split_type) for _, row in df.iterrows()]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(copy_single_image, rows), total=len(rows), desc=f"Processing {split_type} images"))

def split_into_sets():
    print("\nDataset splitting stage")
    print("------------------------")
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    df = pd.read_csv(csv_path)

    train_df, temp_df = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)

    copy_images(train_df, 'train')
    copy_images(val_df, 'val')
    copy_images(test_df, 'test')

    print("Dataset splitting complete!")
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} photos")
    print(f"Test set: {len(test_df)} photos")
    print(f"Photos that do not exist: {wrongPhotosCounter.value}")
