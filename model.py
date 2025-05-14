import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,
    MulticlassF1Score, MulticlassConfusionMatrix
)
from torch.amp import autocast, GradScaler
from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image
from collections import Counter
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

# === Config ===
DATA_DIR = "to_model"
RESULTS_DIR = "results"
MODEL_PATH = "best_model.pt"
NUM_CLASSES = 5
IMG_SIZE = 260
BATCH_SIZE = 28
EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "gradcam"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# === Transforms ===
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ], p=0.7),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Datasets and Dataloaders ===
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_test_transform)

labels = [label for _, label in train_dataset.samples]
class_counts = np.bincount(labels)
class_weights = 1. / class_counts
sample_weights = [class_weights[label] for label in labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          prefetch_factor=2, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        prefetch_factor=2, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         prefetch_factor=2, persistent_workers=True)

# === Model ===
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
weights = EfficientNet_B2_Weights.IMAGENET1K_V1
model = efficientnet_b2(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

labels = [label for _, label in train_dataset.samples]
class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(NUM_CLASSES), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler(device="cuda")

# === Training ===
def train():
    best_val_loss = float('inf')
    trigger_times = 0
    patience = 12
    min_delta = 0.01

    history = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()

        train_loss = total_loss / len(train_dataset)
        train_acc = correct / len(train_dataset)

        val_loss, val_acc = evaluate(val_loader)

        history.append({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), MODEL_PATH)
            torch.cuda.synchronize()
            import time; time.sleep(3)
            print("Model saved.")
        else:
            trigger_times += 1
            print(f"Early stopping count: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

        plot_metrics([h['train_loss'] for h in history], [h['val_loss'] for h in history], "Loss", os.path.join(RESULTS_DIR, "loss_plot.png"))
        plot_metrics([h['train_acc'] for h in history], [h['val_acc'] for h in history], "Accuracy", os.path.join(RESULTS_DIR, "accuracy_plot.png"))

def evaluate(loader):
    model.eval()
    loss_sum = 0
    correct = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()

    return loss_sum / len(loader.dataset), correct / len(loader.dataset)

def plot_metrics(train_values, val_values, title, path):
    plt.figure()
    plt.plot(train_values, label='Train')
    plt.plot(val_values, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.title(f'{title} Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def test_and_metrics():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu()
            preds = torch.argmax(probs, 1)

            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())
            y_probs.extend(probs.tolist())

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    y_probs = torch.tensor(y_probs)

    acc = MulticlassAccuracy(num_classes=NUM_CLASSES)(y_pred, y_true)
    prec = MulticlassPrecision(num_classes=NUM_CLASSES, average='none')(y_pred, y_true)
    rec = MulticlassRecall(num_classes=NUM_CLASSES, average='none')(y_pred, y_true)
    f1 = MulticlassF1Score(num_classes=NUM_CLASSES, average='none')(y_pred, y_true)
    cm = MulticlassConfusionMatrix(num_classes=NUM_CLASSES)(y_pred, y_true)

    df_class = pd.DataFrame({
        "Class": list(range(NUM_CLASSES)),
        "Precision": prec.numpy(),
        "Recall": rec.numpy(),
        "F1": f1.numpy()
    })
    df_class.to_csv(os.path.join(RESULTS_DIR, "per_class_metrics.csv"), index=False)

    # Precision per class
    plt.figure()
    plt.bar(range(NUM_CLASSES), prec.numpy(), color="green")
    plt.xticks(range(NUM_CLASSES))
    plt.ylim(0, 1)
    plt.title("Precision per Class")
    plt.xlabel("Class")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_per_class.png"))
    plt.close()

    # Recall per class
    plt.figure()
    plt.bar(range(NUM_CLASSES), rec.numpy(), color="blue")
    plt.xticks(range(NUM_CLASSES))
    plt.ylim(0, 1)
    plt.title("Recall per Class")
    plt.xlabel("Class")
    plt.ylabel("Recall")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "recall_per_class.png"))
    plt.close()

    # F1 Score per class
    plt.figure()
    plt.bar(range(NUM_CLASSES), f1.numpy(), color="orange")
    plt.xticks(range(NUM_CLASSES))
    plt.ylim(0, 1)
    plt.title("F1 Score per Class")
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "f1_score_per_class.png"))
    plt.close()

    # Misclassifications
    mismatches = (y_true != y_pred)
    mismatch_pairs = [(int(t), int(p)) for t, p in zip(y_true[mismatches], y_pred[mismatches])]
    mismatch_counter = Counter(mismatch_pairs)

    mismatch_labels = [f"{t}->{p}" for (t, p), _ in mismatch_counter.items()]
    mismatch_values = list(mismatch_counter.values())

    plt.figure(figsize=(10, 4))
    plt.bar(mismatch_labels, mismatch_values, color="red")
    plt.xticks(rotation=45)
    plt.title("Misclassifications (True → Predicted)")
    plt.xlabel("Class Pair")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "misclassifications.png"))
    plt.close()

    # Class distribution comparison
    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(y_pred.tolist())

    all_classes = list(range(NUM_CLASSES))
    true_vals = [true_counts.get(i, 0) for i in all_classes]
    pred_vals = [pred_counts.get(i, 0) for i in all_classes]

    x = np.arange(NUM_CLASSES)
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, true_vals, width, label='Ground Truth')
    plt.bar(x + width/2, pred_vals, width, label='Prediction')
    plt.xticks(x)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution: Ground Truth vs Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "class_distribution_comparison.png"))
    plt.close()

    df_global = pd.DataFrame({
        "Accuracy": [acc.item()],
        "Macro Precision": [prec.mean().item()],
        "Macro Recall": [rec.mean().item()],
        "Macro F1": [f1.mean().item()]
    })
    df_global.to_csv(os.path.join(RESULTS_DIR, "global_metrics.csv"), index=False)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    entropy = -(y_probs * y_probs.log()).sum(dim=1)
    plt.figure()
    plt.hist(entropy.numpy(), bins=30, color='purple')
    plt.title("Prediction Entropy Histogram")
    plt.xlabel("Entropy")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "prediction_entropy.png"))
    plt.close()

    cam_extractor = GradCAM(model, target_layer="features.7")
    test_iter = iter(DataLoader(test_dataset, batch_size=1, shuffle=True))

    for i in range(5):
        img_tensor, label = next(test_iter)
        input_tensor = img_tensor.to(DEVICE)

        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        activation_map = cam_extractor(pred_class, output)[0].cpu()
        img_pil = to_pil_image(img_tensor[0])
        heatmap = to_pil_image(activation_map, mode='F').resize(img_pil.size)

        result = np.array(img_pil.convert("RGB")).astype(np.float32) / 255
        heat = np.array(heatmap.convert("L")).astype(np.float32) / 255
        heat = np.stack([heat]*3, axis=-1)
        overlay = np.clip(0.6 * result + 0.4 * heat, 0, 1)

        plt.figure()
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"True: {label.item()}, Pred: {pred_class}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "gradcam", f"gradcam_{i+1}.png"))
        plt.close()
