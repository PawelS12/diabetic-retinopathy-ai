import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms, models

MODEL_PATH = "best_model.pt"
NUM_CLASSES = 5
IMG_SIZE = 260
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

def crop_and_process_image(file_path):
    image = cv2.imread(file_path)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise Exception("Nie wykryto konturów!")

    cnt = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(cnt)
    x, y, r = int(x), int(y), int(r)

    if r > 100:
        top = max(0, y - r)
        bottom = min(image.shape[0], y + r)
        left = max(0, x - r)
        right = min(image.shape[1], x + r)
        cropped = output[top:bottom, left:right]
        image = cv2.resize(cropped, (1024, 1024), interpolation=cv2.INTER_AREA)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # 3 kanały

    pil_image = Image.fromarray(image)
    return transform(pil_image).unsqueeze(0).to(DEVICE), pil_image

def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = torch.argmax(probs, 1).item()
        confidence = probs[0, pred_class].item()
    return pred_class, confidence

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Diagnostyka Retinopatii Cukrzycowej")
        self.geometry("600x700")
        self.configure(bg="white")

        self.label = tk.Label(self, text="Załaduj zdjęcie siatkówki oka", bg="white", font=("Helvetica", 16))
        self.label.pack(pady=20)

        self.img_label = tk.Label(self)
        self.img_label.pack()

        self.result = tk.Label(self, text="", font=("Helvetica", 14), bg="white")
        self.result.pack(pady=20)

        self.button = tk.Button(self, text="Wybierz zdjęcie", command=self.load_image, font=("Helvetica", 12))
        self.button.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        try:
            image_tensor, display_image = crop_and_process_image(file_path)
            pred_class, confidence = predict(image_tensor)
            display_image = display_image.resize((400, 400))
            tk_image = ImageTk.PhotoImage(display_image)
            self.img_label.configure(image=tk_image)
            self.img_label.image = tk_image
            self.result.config(
                text=f"Stopień retinopatii: {pred_class} \nPewność: {confidence*100:.2f}%"
            )
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

if __name__ == "__main__":
    app = App()
    app.mainloop()
