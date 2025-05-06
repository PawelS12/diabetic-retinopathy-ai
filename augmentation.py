import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from multiprocessing import Pool, cpu_count

train_dir = r'split/train'
output_dir = r'split/train_augmented'
csv_path = r'database/trainLabels_updated.csv'

num_augmentations = 3

supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

transform = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    # A.RandomScale(scale_limit=0.1, p=0.4),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
    # A.Resize(height=1024, width=1024, p=1.0)
])

def augment_image(args):
    file_path, output_dir, level = args
    try:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Could not read file: {file_path}")
            return None

        filename, ext = os.path.splitext(os.path.basename(file_path))
        output_level_dir = os.path.join(output_dir, level)
        os.makedirs(output_level_dir, exist_ok=True)

        new_entries = []
        for i in range(num_augmentations):
            augmented = transform(image=img)['image']
            new_filename = f"{filename}_aug_{i}{ext}"
            output_path = os.path.join(output_level_dir, new_filename)
            cv2.imwrite(output_path, augmented)
            new_entries.append({'image': os.path.splitext(new_filename)[0], 'level': int(level)})

        return new_entries
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def augment_train_set():
    print("\nDataset augmentation stage")
    print("------------------------")
    
    image_files = []
    for level in os.listdir(train_dir):
        level_dir = os.path.join(train_dir, level)
        if os.path.isdir(level_dir):
            for file in os.listdir(level_dir):
                if file.lower().endswith(supported_extensions):
                    image_files.append((os.path.join(level_dir, file), output_dir, level))

    print(f"Found {len(image_files)} images in {train_dir}")

    if not image_files:
        print("No images found in training set!")
        return

    num_processes = cpu_count()
    print(f"Using {num_processes} parallel processes")

    new_csv_entries = []
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(augment_image, image_files), total=len(image_files), desc="Augmenting images"))
        pool.close()
        pool.join()

    for result in results:
        if result is not None:
            new_csv_entries.extend(result)

    if new_csv_entries:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=['image', 'level'])
        new_df = pd.DataFrame(new_csv_entries)
        updated_df = pd.concat([df, new_df], ignore_index=True)
        updated_df.to_csv(csv_path, index=False)
        print(f"Updated CSV saved to {csv_path} with {len(updated_df)} entries.")
    else:
        print("No new entries added to CSV.")

    print("Dataset augmentation complete!")
    print(f"Augmented images saved to {output_dir}\n")
