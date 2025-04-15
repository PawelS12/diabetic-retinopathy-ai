import pandas as pd
import os
import shutil
from tqdm import tqdm


def split_photos():
    csv_file = 'database/trainLabels.csv'
    source_folder = 'output'
    df = pd.read_csv(csv_file)

    for level in range(5):
        folder_name = os.path.join(source_folder, str(level))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Transferring photos"):
        image_name = f"{row['image']}.jpeg"
        level = row['level']

        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(source_folder, str(level), image_name)

        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
        else:
            print(f"File not found: {image_name}")

    print("Photo splitting process complete!")
