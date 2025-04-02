import os
import glob
from PIL import Image
import imagehash
from tqdm import tqdm
from multiprocessing import Pool, Manager
import multiprocessing as mp

# Ścieżka do folderu z obrazami
output_folder = 'E:\\diabetic-retinopathy-ai\\output'
image_extensions = ['*.jpeg', '*.jpg', '*.png']  # Obsługiwane rozszerzenia

# Zbierz wszystkie pliki obrazów
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(output_folder, ext)))

print(f"Znaleziono {len(image_files)} obrazów w folderze {output_folder}")


# Funkcja do obliczania hasha dla pojedynczego obrazu
def compute_hash(image_path):
    try:
        with Image.open(image_path) as img:
            # Oblicz difference hash (dhash)
            img_hash = imagehash.dhash(img)
            return (str(img_hash), image_path)  # Zwracamy hash jako string i ścieżkę
    except Exception as e:
        print(f"Błąd podczas przetwarzania {image_path}: {e}")
        return None


# Funkcja do równoległego obliczania hashy
def parallel_hash_computation(image_files):
    # Liczba procesów - domyślnie liczba rdzeni CPU
    num_processes = mp.cpu_count()
    print(f"Używam {num_processes} procesów do obliczania hashy.")

    # Utwórz pulę procesów
    with Pool(processes=num_processes) as pool:
        # Oblicz hashe równolegle z paskiem postępu
        results = list(tqdm(pool.imap(compute_hash, image_files),
                            total=len(image_files),
                            desc="Obliczanie hashy obrazów"))

    # Filtruj None (błędy) i buduj słownik hashy
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
    # Oblicz hashe równolegle
    hash_dict = parallel_hash_computation(image_files)

    # Znajdź i usuń duplikaty
    duplicates_count = 0
    for img_hash, file_list in hash_dict.items():
        if len(file_list) > 1:  # Jeśli mamy więcej niż jeden plik z tym samym hashem
            duplicates_count += len(file_list) - 1
            # Zachowaj pierwszy plik, usuń resztę
            for file_to_remove in file_list[1:]:
                print(f"Usuwanie duplikatu: {file_to_remove}")
                os.remove(file_to_remove)

    print(f"Znaleziono i usunięto {duplicates_count} duplikatów.")
    print(f"Pozostało {len(image_files) - duplicates_count} unikalnych obrazów.")