import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

csv_path = r'database/trainLabels_updated.csv'
source_dir = r'to_model'
output_dir = r'to_model'

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

def find_image_path(base_name):
    for ext in supported_extensions:
        path = os.path.join(source_dir, base_name + ext)
        if os.path.exists(path):
            return path
    return None

def split_images():
    print("\nStarting to split the dataset")
    print("------------------------------------------------------------------")

    df = pd.read_csv(csv_path)

    df['patient_id'] = df['image'].apply(lambda x: x.split('_')[0])

    unique_patients = df['patient_id'].unique()
    train_p, temp_p = train_test_split(unique_patients, test_size=(val_ratio + test_ratio), random_state=42)
    val_p, test_p = train_test_split(temp_p, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    splits = {
        'train': df[df['patient_id'].isin(train_p)],
        'val': df[df['patient_id'].isin(val_p)],
        'test': df[df['patient_id'].isin(test_p)]
    }

    for split_name in splits:
        for level in range(5):
            os.makedirs(os.path.join(output_dir, split_name, str(level)), exist_ok=True)

    total_moved = 0
    for split_name, split_df in splits.items():
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Moving to {split_name}"):
            base_name = row['image']
            level = str(row['level'])
            src_path = find_image_path(base_name)
            if src_path:
                ext = os.path.splitext(src_path)[1]
                dst_path = os.path.join(output_dir, split_name, level, base_name + ext)
                shutil.move(src_path, dst_path)
                total_moved += 1

    print("The division process is complete.")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print(f"Total images moved: {total_moved}")

    moved_images = splits['train']['image'].tolist() + splits['val']['image'].tolist() + splits['test']['image'].tolist()
    df_final = df[df['image'].isin(moved_images)]
    df_final[['image', 'level']].to_csv(csv_path, index=False)
