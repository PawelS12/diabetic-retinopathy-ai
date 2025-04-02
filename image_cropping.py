import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
import math
from PIL import Image
from multiprocessing import Pool, cpu_count

# Sprawdzenie ścieżki i plików
path = 'E:\\diabetic-retinopathy-ai\\database\\resized_train\\resized_train\\*.jpeg'
files = glob.glob(path)

print(f"Ścieżka wyszukiwania: {path}")
print(f"Liczba znalezionych plików: {len(files)}")
if files:
    print("Pierwsze kilka plików:", files[:5])
else:
    print("Nie znaleziono żadnych plików! Sprawdź ścieżkę lub czy katalog zawiera pliki .jpeg")

new_sz = 1024

def crop_image(image):
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print('no contours!')
        flag = 0
        return image, flag

    cnt = max(contours, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(cnt)
    x = int(x)
    y = int(y)
    r = int(r)
    flag = 1

    if r > 100:
        top = max(0, y - r)
        bottom = min(image.shape[0], y + r)
        left = max(0, x - r)
        right = min(image.shape[1], x + r)

        cropped = output[top:bottom, left:right]

        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            resized = cv2.resize(cropped, (new_sz, new_sz), interpolation=cv2.INTER_AREA)
            return resized, flag
        else:
            return image, 0

    return image, flag

def process_image(file):
    """Funkcja do przetwarzania pojedynczego obrazu, wywoływana w procesie równoległym"""
    try:
        img = cv2.imread(file)
        if img is not None:
            processed_img, success = crop_image(img)
            if success:
                filename = os.path.basename(file)
                output_path = f'output/{filename}'
                cv2.imwrite(output_path, processed_img)
                return f"Przetworzono: {filename}"
            else:
                return f"Pominięto: {file} (nieudane przetwarzanie)"
        else:
            return f"Nie udało się wczytać pliku: {file}"
    except Exception as e:
        return f"Błąd przy przetwarzaniu {file}: {str(e)}"

if __name__ == "__main__":
    # Tworzenie katalogu wyjściowego
    if not os.path.exists('output'):
        os.makedirs('output')

    # Ustalanie liczby procesów (domyślnie liczba rdzeni procesora)
    num_processes = cpu_count()
    print(f"Używam {num_processes} procesów równoległych")

    # Tworzenie puli procesów
    with Pool(processes=num_processes) as pool:
        # Równoległe przetwarzanie plików z paskiem postępu
        results = list(tqdm(pool.imap(process_image, files), total=len(files), desc="Przetwarzanie obrazów"))

    # Wyświetlanie wyników (opcjonalne)
    for result in results:
        print(result)