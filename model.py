import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import autocast, GradScaler


RESULTS_DIR = "results_plots"
os.makedirs(RESULTS_DIR, exist_ok=True)


BATCH_SIZE = 32
EPOCHS = 70 # 70
NUM_CLASSES = 5
IMG_SIZE = 224 # 224
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "split"

torch.backends.cudnn.benchmark = True

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_test_transform)

NUM_WORKERS = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

labels = [label for _, label in train_dataset.samples]
class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(NUM_CLASSES), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# print(f"📊 Wagi klas: {class_weights}")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def train():
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    history = []

    scaler = GradScaler(device="cuda")

    # === EARLY STOPPING SETUP ===
    patience = 8
    trigger_times = 0
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        min_delta = 0.01

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / len(train_dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        val_loss, val_acc = evaluate(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "train_acc": epoch_acc,
            "val_acc": val_acc
        })

        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)

        # === EARLY STOPPING LOGIC ===
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("Model saved (new best val_loss)!")
        else:
            trigger_times += 1
            print(f"Early stopping trigger count: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

    plot_metrics(train_losses, val_losses, "Loss", os.path.join(RESULTS_DIR, "loss_plot.png"))
    plot_metrics(train_accuracies, val_accuracies, "Accuracy", os.path.join(RESULTS_DIR, "accuracy_plot.png"))


def evaluate(loader):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            running_loss += loss.item() * inputs.size(0)

    avg_loss = running_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    return avg_loss, acc

def test_and_metrics():
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    y_true = []
    y_pred = []
    y_probs = []

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

    print(f"\n🔍 Accuracy: {acc:.4f}")
    for i in range(NUM_CLASSES):
        print(f"Level {i} → Precision: {prec[i]:.2f}, Recall: {rec[i]:.2f}, F1: {f1[i]:.2f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues", xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    metrics = {"Precision": prec.numpy(), "Recall": rec.numpy(), "F1 Score": f1.numpy()}
    for name, values in metrics.items():
        plt.figure()
        plt.bar(range(NUM_CLASSES), values)
        plt.xticks(range(NUM_CLASSES))
        plt.ylim(0, 1)
        plt.title(f"{name} per Class")
        plt.xlabel("Class")
        plt.ylabel(name)
        plt.tight_layout()
        filename = os.path.join(RESULTS_DIR, f"{name.lower().replace(' ', '_')}_per_class.png")
        plt.savefig(filename)
        plt.close()

    macro_prec = MulticlassPrecision(num_classes=NUM_CLASSES, average='macro')(y_pred, y_true)
    macro_rec = MulticlassRecall(num_classes=NUM_CLASSES, average='macro')(y_pred, y_true)
    macro_f1 = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro')(y_pred, y_true)

    macro_values = [acc.item(), macro_prec.item(), macro_rec.item(), macro_f1.item()]
    labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]

    plt.figure()
    plt.bar(labels, macro_values, color="skyblue")
    plt.ylim(0, 1)
    plt.title("Macro Metrics")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "macro_metrics.png"))
    plt.close()

    entropy = -(y_probs * y_probs.log()).sum(dim=1)
    plt.figure()
    plt.hist(entropy.numpy(), bins=30, color='purple')
    plt.title("Histogram Entropii Predykcji")
    plt.xlabel("Entropia (niepewność)")
    plt.ylabel("Liczba próbek")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "prediction_entropy.png"))
    plt.close()

    mismatches = (y_true != y_pred)
    mismatch_pairs = [(int(t), int(p)) for t, p in zip(y_true[mismatches], y_pred[mismatches])]
    mismatch_counter = Counter(mismatch_pairs)

    labels = [f"{t}→{p}" for (t, p), _ in mismatch_counter.items()]
    values = list(mismatch_counter.values())

    plt.figure(figsize=(10, 4))
    plt.bar(labels, values, color="orange")
    plt.xticks(rotation=45)
    plt.title("Błędne Klasyfikacje (prawda → predykcja)")
    plt.xlabel("Pary klas")
    plt.ylabel("Liczba błędów")
    plt.tight_layout()
    plt.savefig("misclassifications.png")
    plt.close()

    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(y_pred.tolist())

    all_classes = list(range(NUM_CLASSES))
    true_vals = [true_counts.get(i, 0) for i in all_classes]
    pred_vals = [pred_counts.get(i, 0) for i in all_classes]

    x = np.arange(NUM_CLASSES)
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, true_vals, width, label='Prawda')
    plt.bar(x + width/2, pred_vals, width, label='Predykcja')
    plt.xticks(x)
    plt.xlabel("Klasa")
    plt.ylabel("Liczba próbek")
    plt.title("Rozkład klas: Prawdziwe vs. Przewidziane")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "class_distribution_comparison.png"))
    plt.close()

    df_class = pd.DataFrame({
        "Class": list(range(NUM_CLASSES)),
        "Precision": prec.numpy(),
        "Recall": rec.numpy(),
        "F1": f1.numpy()
    })
    df_class.to_csv(os.path.join(RESULTS_DIR, "per_class_metrics.csv"), index=False)

    df_global = pd.DataFrame({
        "Accuracy": [acc.item()],
        "Macro Precision": [macro_prec.item()],
        "Macro Recall": [macro_rec.item()],
        "Macro F1": [macro_f1.item()]
    })
    df_global.to_csv(os.path.join(RESULTS_DIR, "global_metrics.csv"), index=False)

    cam_extractor = GradCAM(model, target_layer="features.7")

    test_iter = iter(DataLoader(test_dataset, batch_size=1, shuffle=True))
    os.makedirs("gradcam_outputs", exist_ok=True)

    model.eval()
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

        overlay = (0.6 * result + 0.4 * heat)
        overlay = np.clip(overlay, 0, 1)

        plt.figure()
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"True: {label}, Pred: {pred_class}")
        plt.tight_layout()
        plt.savefig(f"gradcam_outputs/gradcam_{i+1}.png")
        plt.close()

def plot_metrics(train_values, val_values, metric_name, filename):
    plt.figure()
    plt.plot(train_values, label=f"Train {metric_name}")
    plt.plot(val_values, label=f"Val {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train_model():
    train()
    test_and_metrics()


def load_trained_model(weights_path="best_model.pt"):
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model
