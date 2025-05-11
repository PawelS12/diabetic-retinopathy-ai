import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from model import load_trained_model, DEVICE


IMG_SIZE = 224
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = load_trained_model()
model.eval()

def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
    return class_names[pred_class], confidence

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
    if file_path:
        # Wyświetlenie obrazu
        img = Image.open(file_path).resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Predykcja
        prediction, confidence = predict_image(file_path)
        result_label.config(text=f"Prediction: {prediction} ({confidence:.2f})")


root = tk.Tk()
root.title("Diabetic Retinopathy Classifier")
root.geometry("500x500")
root.configure(bg="white")

btn = tk.Button(root, text="Choose Image", command=open_file, font=("Arial", 12), bg="#4285F4", fg="white")
btn.pack(pady=10)

image_label = tk.Label(root, bg="white")
image_label.pack()

result_label = tk.Label(root, text="Prediction: -", font=("Arial", 14), bg="white")
result_label.pack(pady=10)

root.mainloop()