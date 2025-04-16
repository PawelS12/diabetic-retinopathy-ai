import os
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Ścieżki
train_dir = r'split/train'
augmented_dir = r'split/train_augmented'
val_dir = r'split/val'
csv_path = r'database\trainLabels_updated.csv'
model_save_path = r'models/best_model_resnet18_15epoch.pth'

# Parametry
batch_size = 16
num_epochs = 15
learning_rate = 0.001
num_classes = 5
image_size = 224
freeze_backbone = False  # Pełne trenowanie

# Transformacje danych (bez augmentacji, tylko resize + normalizacja)
train_transform = A.Compose([
    A.Resize(height=image_size, width=image_size, p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=image_size, width=image_size, p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Dataset
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, csv_file=None, train_dir=None, augmented_dir=None, val_dir=None, transform=None, is_val=False):
        self.transform = transform
        self.images = []
        self.labels = []

        if is_val:
            for level in os.listdir(val_dir):
                level_dir = os.path.join(val_dir, level)
                if os.path.isdir(level_dir):
                    for file in os.listdir(level_dir):
                        if file.endswith(('.jpeg', '.jpg', '.png')):
                            self.images.append(os.path.join(level_dir, file))
                            self.labels.append(int(level))
        else:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                image_name = row['image'] + '.jpeg'
                level = row['level']
                train_path = os.path.join(train_dir, str(level), image_name)
                aug_path = os.path.join(augmented_dir, str(level), image_name)
                if os.path.exists(train_path):
                    self.images.append(train_path)
                    self.labels.append(level)
                elif os.path.exists(aug_path):
                    self.images.append(aug_path)
                    self.labels.append(level)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        return img, label

# Funkcja trenowania
def train_model():
    # Ustawienie urządzenia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("No GPU available, running on CPU.")

    # Wczytanie danych
    train_dataset = DiabeticRetinopathyDataset(
        csv_file=csv_path,
        train_dir=train_dir,
        augmented_dir=augmented_dir,
        transform=train_transform,
        is_val=False
    )
    val_dataset = DiabeticRetinopathyDataset(
        val_dir=val_dir,
        transform=val_transform,
        is_val=True
    )

    num_workers = os.cpu_count()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")

    # Inicjalizacja modelu ResNet18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Sprawdzenie urządzenia modelu
    print(f"Model device: {next(model.parameters()).device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Trenowanie
    best_val_acc = 0.0
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            images, labels = images.to(device), labels.to(device)
            if i == 0 and epoch == 0:
                print(f"First batch images device: {images.device}")
                print(f"First batch labels device: {labels.device}")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Walidacja
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with Val Acc: {val_acc:.2f}%")
