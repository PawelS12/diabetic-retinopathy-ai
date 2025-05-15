import os
import glob
from PIL import Image
import imagehash
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd

output_folder = r"to_model"
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
csv_path = r'database\trainLabels.csv'
updated_csv_path = r'database\trainLabels_updated.csv'

def compute_hash(image_path):
    try:
        with Image.open(image_path) as img:
            img_hash = imagehash.dhash(img)
            return (str(img_hash), image_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def parallel_hash_computation(image_files):
    num_processes = mp.cpu_count()
    print(f"I use {num_processes} processes to calculate hashes.")

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(compute_hash, image_files),
                            total=len(image_files),
                            desc="Calculating image hashes"))
    hash_dict = {}
    for result in results:
        if result is not None:
            img_hash, image_path = result
            if img_hash in hash_dict:
                hash_dict[img_hash].append(image_path)
            else:
                hash_dict[img_hash] = [image_path]

    return hash_dict

def remove_duplicates():
    print("\nDuplicate removal stage")
    print("------------------------")
    image_files = []
    for ext in image_extensions:
        files = glob.glob(os.path.join(output_folder, ext))
        image_files.extend(files)
    print(f"Found {len(image_files)} images in {output_folder}")

    if not image_files:
        print("No images found in output folder!")
        return

    hash_dict = parallel_hash_computation(image_files)

    duplicates_count = 0
    kept_files = []
    for img_hash, file_list in hash_dict.items():
        kept_files.append(file_list[0])
        if len(file_list) > 1:
            duplicates_count += len(file_list) - 1
            for file_to_remove in file_list[1:]:
                # print(f"Removing duplicate: {file_to_remove}")
                os.remove(file_to_remove)

    print(f"Found and removed {duplicates_count} duplicates.")
    print(f"{len(image_files) - duplicates_count} unique images remain.")

    df = pd.read_csv(csv_path)
    kept_filenames = [os.path.splitext(os.path.basename(f))[0] for f in kept_files]
    updated_df = df[df['image'].isin(kept_filenames)]
    updated_df.to_csv(updated_csv_path, index=False)
    print(f"Updated CSV saved to {updated_csv_path} with {len(updated_df)} entries.")
