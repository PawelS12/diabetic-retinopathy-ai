import os
import glob
from PIL import Image
import imagehash
from tqdm import tqdm
from multiprocessing import Pool, Manager
import multiprocessing as mp

output_folder = 'output'
image_extensions = ['*.jpeg', '*.jpg', '*.png']


image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(output_folder, ext)))

print(f"Found {len(image_files)} images in the folder {output_folder}")

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

    with Pool(processes=num_processes) as pool:
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
    hash_dict = parallel_hash_computation(image_files)

    duplicates_count = 0
    for img_hash, file_list in hash_dict.items():
        if len(file_list) > 1:
            duplicates_count += len(file_list) - 1
            for file_to_remove in file_list[1:]:
                print(f"Removing duplicate: {file_to_remove}")
                os.remove(file_to_remove)

    print(f"Found and removed {duplicates_count} duplicates.")
    print(f"{len(image_files) - duplicates_count} unique images remain.")
