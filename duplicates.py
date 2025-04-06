import os
import glob
from PIL import Image
import imagehash
from tqdm import tqdm
from multiprocessing import Pool, Manager
import multiprocessing as mp

# Path to the folder with images and supported extensions
output_folder = 'output'
image_extensions = ['*.jpeg', '*.jpg', '*.png']


image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(output_folder, ext)))

print(f"Found {len(image_files)} images in the folder {output_folder}")


# Function to compute hash for a single image
def compute_hash(image_path):
    try:
        with Image.open(image_path) as img:
            # Compute difference hash (dhash)
            img_hash = imagehash.dhash(img)
            return (str(img_hash), image_path)  # Return hash as string and path
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# Function to compute hashes in parallel
def parallel_hash_computation(image_files):
    # Number of processes - by default, number of CPU cores
    num_processes = mp.cpu_count()
    print(f"Używam {num_processes} procesów do obliczania hashy.")

    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        # Compute hashes in parallel with a progress bar
        results = list(tqdm(pool.imap(compute_hash, image_files),
                            total=len(image_files),
                            desc="Obliczanie hashy obrazów"))

    # Filter out None (errors) and build hash dictionary
    hash_dict = {}
    for result in results:
        if result is not None:
            img_hash, image_path = result
            if img_hash in hash_dict:
                hash_dict[img_hash].append(image_path)
            else:
                hash_dict[img_hash] = [image_path]

    return hash_dict


if __name__ == "__main__":
    # Compute hashes in parallel
    hash_dict = parallel_hash_computation(image_files)

    # Find and remove duplicates
    duplicates_count = 0
    for img_hash, file_list in hash_dict.items():
        if len(file_list) > 1:
            duplicates_count += len(file_list) - 1
            for file_to_remove in file_list[1:]:
                print(f"Removing duplicate: {file_to_remove}")
                os.remove(file_to_remove)

    print(f"Found and removed {duplicates_count} duplicates.")
    print(f"{len(image_files) - duplicates_count} unique images remain.")