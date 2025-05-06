import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Value
from tqdm import tqdm
import glob
from PIL import Image

csv_path = r'database/trainLabels_updated.csv'
data_dir = r'output'
output_dir = r'split'

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

wrongPhotosCounter = Value('i', 0)

supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

def find_image_path(level_dir, base_name):
    for ext in supported_extensions:
        path = os.path.join(level_dir, base_name + ext)
        if os.path.exists(path):
            return path
    return None

def copy_single_image(args):
    row, split_type = args
    base_name = row['image']
    level = str(row['level'])
    level_dir = os.path.join(data_dir, level)
    src_path = find_image_path(level_dir, base_name)
    
    dst_dir = os.path.join(output_dir, split_type, level)
    os.makedirs(dst_dir, exist_ok=True)

    if src_path:
        ext = os.path.splitext(src_path)[1]
        dst_path = os.path.join(dst_dir, base_name + ext)
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
    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    df = pd.read_csv(csv_path)

    df['patient_id'] = df['image'].apply(lambda x: x.split('_')[0])
    df['eye'] = df['image'].apply(lambda x: x.split('_')[1])

    patient_groups = df.groupby('patient_id')

    new_rows = []
    missing_eye_counter = 0

    for patient_id, group in patient_groups:
        if len(group) == 2:
            new_rows.append(group)
        elif len(group) == 1:
            missing_eye_counter += 1
            existing_row = group.iloc[0]
            existing_eye = existing_row['eye']
            missing_eye = 'right' if existing_eye == 'left' else 'left'
            new_image_name = f"{patient_id}_{missing_eye}"

            level = str(existing_row['level'])
            level_dir = os.path.join(data_dir, level)
            src_path = find_image_path(level_dir, existing_row['image'])

            if src_path:
                ext = os.path.splitext(src_path)[1]
                fake_path = os.path.join(level_dir, new_image_name + ext)
                with Image.open(src_path) as img:
                    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                    flipped.save(fake_path)

                new_row = existing_row.copy()
                new_row['image'] = new_image_name
                new_row['eye'] = missing_eye
                new_rows.append(pd.DataFrame([existing_row, new_row]))
            else:
                new_rows.append(group)

    df_complete = pd.concat(new_rows, ignore_index=True)

    unique_patients = df_complete['patient_id'].unique()

    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=(val_ratio + test_ratio), random_state=42
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=test_ratio / (val_ratio + test_ratio), random_state=42
    )

    train_df = df_complete[df_complete['patient_id'].isin(train_patients)]
    val_df = df_complete[df_complete['patient_id'].isin(val_patients)]
    test_df = df_complete[df_complete['patient_id'].isin(test_patients)]

    copy_images(train_df, 'train')
    copy_images(val_df, 'val')
    copy_images(test_df, 'test')

    df_complete[['image', 'level']].to_csv(csv_path, index=False)

    print("Dataset splitting complete!")
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} photos")
    print(f"Test set: {len(test_df)} photos")
    print(f"Photos that do not exist: {wrongPhotosCounter.value}")
    print(f"Patients with only one eye photo (mirrored): {missing_eye_counter}")
