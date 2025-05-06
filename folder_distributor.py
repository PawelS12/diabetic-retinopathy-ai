import pandas as pd
import os
import shutil
from tqdm import tqdm
import glob

def split_photos():
    print("\nPhoto splitting stage")
    print("------------------------")
    csv_file = 'database/trainLabels_updated.csv'
    source_folder = 'output'
    df = pd.read_csv(csv_file)

    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    for level in range(5):
        folder_name = os.path.join(source_folder, str(level))
        os.makedirs(folder_name, exist_ok=True)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Transferring photos"):
        base_name = row['image']
        level = row['level']

        matched_files = []
        for ext in extensions:
            pattern = os.path.join(source_folder, base_name + ext)
            matched_files.extend(glob.glob(pattern))

        if matched_files:
            for source_path in matched_files:
                extension = os.path.splitext(source_path)[1]
                destination_path = os.path.join(source_folder, str(level), base_name + extension)
                shutil.move(source_path, destination_path)

    print("Photo splitting process complete!")
