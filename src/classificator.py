"""
Training pipeline for ResNet‑18 full fine‑tuning with MixUp, t‑SNE, UMAP,
and extended evaluation metrics (confusion matrix, ROC, hardest samples).
"""

# --- Importations ---
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode to prevent hanging
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import seaborn as sns

# --- UMAP (optionnel) ---
try:
    import umap
    umap_available = True
except ImportError:
    print("UMAP not installed — skipping UMAP visualization.")
    umap_available = False

import argparse


# --- Create visualizations folder ---
VISUALIZATIONS_DIR = "visualizations"
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
print(f"Visualizations will be saved to: {VISUALIZATIONS_DIR}/")


# --- Seed pour reproductibilité ---
def set_seed(seed=42):
    """
    Fixes all relevant random seeds to ensure reproducible training.

    Parameters
    ----------
    seed : int, optional
        Random seed value used for Python, NumPy and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

parser = argparse.ArgumentParser(description="Train or infer with ResNet18 classifier")
parser.add_argument('--infer', type=str, help='Path to image for inference')
args = parser.parse_args()

# --- Constantes globales ---
H = 256
W = 256
BATCH_SIZE = 32
DATA_DIR = "data/processed"
MODEL_DIR = "models"
MODEL_PATH = "models/best_model_resnet18_finetuned.pth"
NUM_EPOCHS = 20
PATIENCE = 3

# --- Transformations ---
train_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
    ),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Chargement du dataset ---
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"),
    transform=train_transform,
    is_valid_file=lambda p: p.lower().endswith((".jpg", ".jpeg", ".png"))
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "val"),
    transform=test_transform,
    is_valid_file=lambda p: p.lower().endswith((".jpg", ".jpeg", ".png"))
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "test"),
    transform=test_transform,
    is_valid_file=lambda p: p.lower().endswith((".jpg", ".jpeg", ".png"))
)

NUM_CLASSES = len(train_dataset.classes)
print("Catégories détectées :", train_dataset.classes)

# Ensure classes are the same
assert train_dataset.classes == val_dataset.classes == test_dataset.classes

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

print(f"Train: {len(train_dataset)}, "
      f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")


# --- Data Distribution Visualization ---
def plot_data_distribution(datasets, names, classes):
    """
    Plots the class distribution across different datasets.

    Parameters
    ----------
    datasets : list
        List of datasets (train, val, test).
    names : list
        Names for each dataset.
    classes : list
        Class names.
    """
    plt.figure(figsize=(15, 5))

    for i, (dataset, name) in enumerate(zip(datasets, names)):
        class_counts = {}
        for _, label in dataset:
            class_name = classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        plt.subplot(1, 3, i + 1)
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title(f'{name} Set Distribution')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'data_distribution.png'), dpi=300, bbox_inches='tight')


plot_data_distribution(
    [train_dataset, val_dataset, test_dataset],
    ['Train', 'Validation', 'Test'],
    train_dataset.classes
)


# --- Modèle ResNet18 ---
def create_resnet18(num_classes):
    """
    Creates a ResNet‑18 model with full fine‑tuning and a custom classifier head.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    Returns
    -------
    torch.nn.Module
        Modified ResNet‑18 model.
    """
    model = resnet18(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def infer_image(image_path):
    """
    Runs inference on a single image and prints the prediction with confidence and top-3 classes.
    
    Parameters
    ----------
    image_path : str
        Path to the input image.
    """
    import os
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    try:
        from PIL import Image
        
        # Use CPU for inference to avoid CUDA issues
        device = torch.device('cpu')
        
        # Load the trained model
        model = create_resnet18(NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        # Preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = test_transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = torch.argmax(outputs, dim=1).item()
        
        # Get top 3 predictions
        top3_indices = np.argsort(probs)[::-1][:3]
        top3_classes = [train_dataset.classes[i] for i in top3_indices]
        top3_probs = [probs[i] for i in top3_indices]
        
        # Print results
        print(f"Predicted category: {train_dataset.classes[pred_idx]} (confidence: {probs[pred_idx]:.2f})")
        print("Top 3 predictions:")
        for cls, prob in zip(top3_classes, top3_probs):
            print(f"  {cls}: {prob:.2f}")
    except Exception as e:
        print(f"Error during inference: {e}")
        return


# Check if inference mode
if args.infer:
    infer_image(args.infer)
    exit()


# --- Device ---
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
print("Utilisation de l'appareil:", device)

model = create_resnet18(NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-5,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)


# --- MixUp ---
def mixup_data(x, y, alpha=0.4):
    """
    Applies MixUp augmentation to a batch of images and labels.

    Parameters
    ----------
    x : torch.Tensor
        Batch of input images.
    y : torch.Tensor
        Batch of labels.
    alpha : float, optional
        Beta distribution parameter controlling MixUp intensity.

    Returns
    -------
    mixed_x : torch.Tensor
        Mixed images.
    y_a : torch.Tensor
        Original labels.
    y_b : torch.Tensor
        Shuffled labels.
    lam : float
        MixUp interpolation coefficient.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


# --- Suivi des métriques ---
train_losses = []
val_losses = []
val_accuracies = []
learning_rates = []

best_val_loss = float("inf")
best_val_acc = 0.0
patience_counter = 0

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Boucle d'entraînement ---
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(
            images, labels, alpha=0.4
        )

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = (
            lam * criterion(outputs, targets_a)
            + (1 - lam) * criterion(outputs, targets_b)
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    learning_rates.append(optimizer.param_groups[0]['lr'])

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} — "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_acc = val_acc
        patience_counter = 0

        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print("→ Nouveau meilleur modèle sauvegardé")
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

    scheduler.step()

print("Entraînement terminé. Meilleure Val Acc:", best_val_acc)

# --- Graphiques Loss & Accuracy ---
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss (ResNet18 full FT + MixUp)")
plt.legend()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'training_validation_loss.png'), dpi=300, bbox_inches='tight')

plt.figure(figsize=(8, 5))
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy (ResNet18 full FT + MixUp)")
plt.legend()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'validation_accuracy.png'), dpi=300, bbox_inches='tight')

plt.figure(figsize=(8, 5))
plt.plot(learning_rates, label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule (Cosine Annealing)")
plt.legend()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'learning_rate_schedule.png'), dpi=300, bbox_inches='tight')


# --- Évaluation sur le test ---
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

all_preds = []
all_labels = []
all_probs = []


def evaluate_model(model, loader):
    """
    Runs inference on a dataloader and collects predictions, labels and probabilities.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    loader : DataLoader
        Test or validation dataloader.

    Returns
    -------
    preds : np.ndarray
        Predicted class indices.
    labels : np.ndarray
        Ground‑truth labels.
    probs : np.ndarray
        Softmax probabilities for each class.
    """
    preds = []
    labels = []
    probs = []

    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            y = y.to(device)

            outputs = model(images)
            p = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())

    return (
        np.array(preds),
        np.array(labels),
        np.array(probs)
    )


all_preds, all_labels, all_probs = evaluate_model(model, test_loader)


# --- Comprehensive Metrics Summary ---
def plot_metrics_summary(labels, preds, classes):
    """
    Creates a comprehensive metrics summary plot.

    Parameters
    ----------
    labels : np.ndarray
        Ground-truth labels.
    preds : np.ndarray
        Predicted labels.
    classes : list of str
        Class names.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.ylabel('Score')
    plt.title('Model Performance Metrics Summary')
    plt.ylim(0, 1)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Per-class metrics
    precision_per_class = precision_score(labels, preds, average=None)
    recall_per_class = recall_score(labels, preds, average=None)
    f1_per_class = f1_score(labels, preds, average=None)

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
    plt.bar(x, recall_per_class, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)

    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()


plot_metrics_summary(all_labels, all_preds, dataset.classes)


# --- Confusion Matrix ---
def plot_confusion_matrix(cm, classes):
    """
    Displays a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    classes : list of str
        Class names.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (ResNet18 full FT + MixUp)")
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')


cm = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm, dataset.classes)


# --- Correlation Matrix of Prediction Probabilities ---
def plot_prediction_correlation(probs, classes):
    """
    Plots a correlation matrix of prediction probabilities across classes.

    Parameters
    ----------
    probs : np.ndarray
        Prediction probabilities.
    classes : list of str
        Class names.
    """
    corr_matrix = np.corrcoef(probs.T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=classes,
        yticklabels=classes,
        cmap="coolwarm",
        vmin=-1, vmax=1
    )
    plt.title("Correlation Matrix of Prediction Probabilities")
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'prediction_correlation.png'), dpi=300, bbox_inches='tight')


plot_prediction_correlation(all_probs, dataset.classes)


# --- Per-class accuracy ---
def compute_per_class_accuracy(model, loader, num_classes):
    """
    Computes accuracy for each class separately.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    loader : DataLoader
        Test dataloader.
    num_classes : int
        Number of classes.

    Returns
    -------
    np.ndarray
        Per-class accuracy values.
    """
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            matches = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i].item()
                correct[label] += matches[i].item()
                total[label] += 1

    return correct / total


per_class_acc = compute_per_class_accuracy(
    model, test_loader, NUM_CLASSES
)

plt.figure(figsize=(10, 5))
plt.bar(dataset.classes, per_class_acc)
plt.ylabel("Accuracy")
plt.title("Per-class Accuracy (ResNet18 full FT + MixUp)")
plt.xticks(rotation=45)
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')


# --- Prediction Confidence Distribution ---
def plot_confidence_distribution(probs, preds, labels, classes):
    """
    Plots histograms of prediction confidence for correct and incorrect predictions.

    Parameters
    ----------
    probs : np.ndarray
        Softmax probabilities.
    preds : np.ndarray
        Predicted class indices.
    labels : np.ndarray
        Ground-truth labels.
    classes : list of str
        Class names.
    """
    max_probs = np.max(probs, axis=1)
    correct = (preds == labels)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(max_probs[correct], bins=20, alpha=0.7, label="Correct", color="green")
    plt.hist(max_probs[~correct], bins=20, alpha=0.7, label="Incorrect", color="red")
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Count")
    plt.title("Prediction Confidence Distribution")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, cls in enumerate(classes):
        cls_probs = max_probs[labels == i]
        plt.hist(cls_probs, bins=15, alpha=0.5, label=cls)

    plt.xlabel("Prediction Confidence")
    plt.ylabel("Count")
    plt.title("Confidence by True Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')


plot_confidence_distribution(all_probs, all_preds, all_labels, dataset.classes)


# --- ROC curves (One-vs-Rest) ---
def plot_roc_curves(labels, probs, classes):
    """
    Plots ROC curves for each class (one-vs-rest).

    Parameters
    ----------
    labels : np.ndarray
        Ground‑truth labels.
    probs : np.ndarray
        Softmax probabilities.
    classes : list of str
        Class names.
    """
    y_bin = label_binarize(labels, classes=list(range(len(classes))))

    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (ResNet18 full FT + MixUp)")
    plt.legend()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()


plot_roc_curves(all_labels, all_probs, dataset.classes)


# --- Precision-Recall curves (One-vs-Rest) ---
def plot_precision_recall_curves(labels, probs, classes):
    """
    Plots Precision-Recall curves for each class (one-vs-rest).

    Parameters
    ----------
    labels : np.ndarray
        Ground‑truth labels.
    probs : np.ndarray
        Softmax probabilities.
    classes : list of str
        Class names.
    """
    y_bin = label_binarize(labels, classes=list(range(len(classes))))

    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        avg_precision = average_precision_score(y_bin[:, i], probs[:, i])
        plt.plot(recall, precision, label=f"{cls} (AP={avg_precision:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (ResNet18 full FT + MixUp)")
    plt.legend()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()


plot_precision_recall_curves(all_labels, all_probs, dataset.classes)


# --- Hardest samples ---
def compute_hardest_samples(model, loader, classes, top_k=12):
    """
    Identifies the samples with the highest loss.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    loader : DataLoader
        Test dataloader.
    classes : list of str
        Class names.
    top_k : int, optional
        Number of hardest samples to display.

    Returns
    -------
    None
    """
    criterion_nr = nn.CrossEntropyLoss(reduction="none")

    losses = []
    imgs = []
    labels = []
    preds = []

    model.eval()
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            y = y.to(device)

            outputs = model(images)
            batch_losses = criterion_nr(outputs, y)

            losses.extend(batch_losses.cpu().numpy())
            imgs.extend(images.cpu())
            labels.extend(y.cpu().numpy())

            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())

    losses = np.array(losses)
    idx_sorted = np.argsort(losses)[::-1]
    top_k = min(top_k, len(idx_sorted))

    plt.figure(figsize=(12, 10))
    for i in range(top_k):
        idx = idx_sorted[i]
        img = imgs[idx].permute(1, 2, 0).numpy()
        img = (
            img * np.array([0.229, 0.224, 0.225])
            + np.array([0.485, 0.456, 0.406])
        ).clip(0, 1)

        plt.subplot(3, 4, i + 1)
        plt.imshow(img)
        plt.title(
            f"Loss={losses[idx]:.2f}\n"
            f"True={classes[labels[idx]]}\n"
            f"Pred={classes[preds[idx]]}"
        )
        plt.axis("off")

    plt.suptitle("Top Hardest Samples (Highest Loss)")
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'hardest_samples.png'), dpi=300, bbox_inches='tight')
    plt.show()


compute_hardest_samples(model, test_loader, dataset.classes)


# --- Sample Predictions Visualization ---
def visualize_sample_predictions(model, loader, classes, num_samples=12):
    """
    Displays sample predictions with images and confidence scores.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    loader : DataLoader
        Test dataloader.
    classes : list of str
        Class names.
    num_samples : int, optional
        Number of samples to display.
    """
    model.eval()
    images_list = []
    labels_list = []
    preds_list = []
    probs_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            images_list.extend(images.cpu())
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())
            probs_list.extend(probs.cpu().numpy())

            if len(images_list) >= num_samples:
                break

    plt.figure(figsize=(15, 10))
    for i in range(min(num_samples, len(images_list))):
        img = images_list[i].permute(1, 2, 0).numpy()
        img = (
            img * np.array([0.229, 0.224, 0.225])
            + np.array([0.485, 0.456, 0.406])
        ).clip(0, 1)

        plt.subplot(3, 4, i + 1)
        plt.imshow(img)

        true_class = classes[labels_list[i]]
        pred_class = classes[preds_list[i]]
        confidence = probs_list[i][preds_list[i]]

        color = "green" if labels_list[i] == preds_list[i] else "red"
        plt.title(
            f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}",
            color=color,
            fontsize=9
        )
        plt.axis("off")

    plt.suptitle("Sample Predictions with Confidence Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
    plt.show()


visualize_sample_predictions(model, test_loader, dataset.classes)


# --- Embeddings t-SNE & UMAP ---
def extract_features(model, x):
    """
    Extracts convolutional features from ResNet‑18 before the classifier.

    Parameters
    ----------
    model : torch.nn.Module
        Trained ResNet‑18.
    x : torch.Tensor
        Input batch.

    Returns
    -------
    torch.Tensor
        Flattened feature vectors.
    """
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    return torch.flatten(x, 1)


embeddings = []
labels_list = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        feats = extract_features(model, images)

        embeddings.append(feats.cpu().numpy())
        labels_list.extend(labels.numpy())

embeddings = np.concatenate(embeddings, axis=0)
labels_list = np.array(labels_list)


# --- t-SNE ---
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
tsne_emb = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for i, cls in enumerate(dataset.classes):
    idx = labels_list == i
    plt.scatter(tsne_emb[idx, 0], tsne_emb[idx, 1], label=cls, alpha=0.6)

plt.legend()
plt.title("t-SNE Embedding Visualization")
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'tsne_embeddings.png'), dpi=300, bbox_inches='tight')
plt.show()


# --- UMAP ---
if umap_available:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    umap_emb = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(dataset.classes):
        idx = labels_list == i
        plt.scatter(umap_emb[idx, 0], umap_emb[idx, 1], label=cls, alpha=0.6)

    plt.legend()
    plt.title("UMAP Embedding Visualization")
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'umap_embeddings.png'), dpi=300, bbox_inches='tight')
    plt.show()


# --- Summary of saved visualizations ---
print("\n" + "="*60)
print("VISUALIZATION SUMMARY")
print("="*60)
print(f"All visualizations have been saved to: {VISUALIZATIONS_DIR}/")
print("\nSaved files:")
visualization_files = [
    "data_distribution.png",
    "training_validation_loss.png",
    "validation_accuracy.png",
    "learning_rate_schedule.png",
    "metrics_summary.png",
    "per_class_metrics.png",
    "confusion_matrix.png",
    "per_class_accuracy.png",
    "confidence_distribution.png",
    "roc_curves.png",
    "precision_recall_curves.png",
    "hardest_samples.png",
    "sample_predictions.png",
    "tsne_embeddings.png",
    "umap_embeddings.png"
]

for i, filename in enumerate(visualization_files, 1):
    filepath = os.path.join(VISUALIZATIONS_DIR, filename)
    if os.path.exists(filepath):
        print(f"{i:2d}. ✓ {filename}")
    else:
        print(f"{i:2d}. ✗ {filename} (not found)")

print(f"\nTotal visualizations: {len(visualization_files)}")
print("="*60)
