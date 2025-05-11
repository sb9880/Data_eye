import os
import time
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.models as models
from torchvision.models import VGG16_Weights, MobileNet_V3_Large_Weights, DenseNet121_Weights

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


# Set random seed for reproducibility
pl.seed_everything(42)

# Check for available device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# Define the data module class
class EyeDiseaseDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./Eye_Dataset", batch_size=32, img_size=224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_ratio, self.val_ratio, self.test_ratio = 0.7, 0.2, 0.1
        
    def setup(self, stage=None):
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load the full dataset
        self.full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        
        # Split into train, validation, and test sets
        total_size = len(self.full_dataset)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, val_size, test_size]
        )
        
        # Save class names and counts
        self.class_names = self.full_dataset.classes
        self.num_classes = len(self.class_names)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
    
    def get_class_counts(self):
        class_counts = {cls: 0 for cls in self.class_names}
        for _, label in self.full_dataset.samples:
            class_counts[self.class_names[label]] += 1
        return class_counts
    
    def show_random_images(self, num_per_class=1):
        """Display random images from each class"""
        # Create a grid with 5 images per row
        rows = len(self.class_names) // 5 + (len(self.class_names) % 5 > 0)
        fig, axes = plt.subplots(rows, 5, figsize=(20, 5 * rows))
        
        # Flatten axes in case the grid is not fully filled
        axes = axes.flatten()
        
        for idx, cls in enumerate(self.class_names):
            # Get the indices for images of the current class
            class_indices = [i for i, (_, label) in enumerate(self.full_dataset.samples) 
                            if self.class_names[label] == cls]
            
            # Choose a random index for the class
            random_idx = random.choice(class_indices)
            img_path, _ = self.full_dataset.samples[random_idx]
            
            # Open and display the image
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].axis("off")
            axes[idx].set_title(cls)
        
        # Hide any unused axes
        for i in range(len(self.class_names), len(axes)):
            axes[i].axis("off")
        
        plt.tight_layout()
        plt.show()
        
    def visualize_class_distribution(self):
        """Visualize the class distribution"""
        class_counts = self.get_class_counts()
        
        # Generate a list of colors based on the number of classes
        colors = plt.cm.viridis(np.linspace(0, 1, len(class_counts)))
        
        # Plot class distribution with different colors
        plt.figure(figsize=(13, 5))
        plt.bar(class_counts.keys(), class_counts.values(), color=colors)
        plt.xlabel("Class")
        plt.ylabel("Number of Images")
        plt.title("Class Distribution in Dataset")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# Base classification model
class EyeClassifierBase(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # These will be defined in subclasses
        self.backbone = None
        self.classifier = None
        
        # Save hyperparameters to the checkpoint
        self.save_hyperparameters()
        
        # For metrics tracking
        self.train_acc = 0
        self.val_acc = 0
        self.test_acc = 0
        self.val_preds = []
        self.val_labels = []
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.train_acc = acc
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.val_acc = acc
        
        # Store predictions and labels for confusion matrix
        self.val_preds.extend(preds.cpu().numpy())
        self.val_labels.extend(y.cpu().numpy())
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.test_acc = acc
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_validation_epoch_end(self):
        return self.val_acc
    
    def get_metrics(self):
        """Return predictions and labels from validation set for analysis"""
        return self.val_preds, self.val_labels


# MobileNetV3 implementation
class MobileNetModel(EyeClassifierBase):
    def __init__(self, num_classes, learning_rate=0.001, freeze_layers=True):
        super().__init__(num_classes, learning_rate)
        
        # Load MobileNetV3 with pre-trained weights
        self.model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        
        if freeze_layers:
            # Freeze all blocks in model.features except for the last two modules
            total_blocks = len(self.model.features)
            for idx, module in enumerate(self.model.features):
                if idx < total_blocks - 2:
                    for param in module.parameters():
                        param.requires_grad = False
                else:
                    # Unfreeze last two blocks
                    for param in module.parameters():
                        param.requires_grad = True
        
        # Replace classifier with a custom head that includes dropout
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[0].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


# DenseNet121 implementation
class DenseNetModel(EyeClassifierBase):
    def __init__(self, num_classes, learning_rate=0.001, freeze_layers=True):
        super().__init__(num_classes, learning_rate)
        
        # Load DenseNet121 with pre-trained weights
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        
        if freeze_layers:
            # For DenseNet121, model.features is a Sequential of various layers.
            # Unfreeze only the last two modules.
            features = list(self.model.features.children())
            total_blocks = len(features)
            for idx, module in enumerate(features):
                if idx < total_blocks - 2:
                    for param in module.parameters():
                        param.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = True
        
        # Replace classifier with dropout and a linear layer
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.model.classifier.in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


# Utility function for plotting results
def plot_metrics(trainer, model, class_names):
    """Plot and display model metrics"""
    # Extract metrics
    train_losses = [x['loss'].item() for x in trainer.callback_metrics if 'loss' in x]
    val_losses = [x['val_loss'].item() for x in trainer.callback_metrics if 'val_loss' in x]
    train_accs = [x['train_acc'].item() for x in trainer.callback_metrics if 'train_acc' in x]
    val_accs = [x['val_acc'].item() for x in trainer.callback_metrics if 'val_acc' in x]
    
    # Plot accuracy and loss curves
    plt.figure(figsize=(12, 5))
    
    # Accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    
    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Get validation predictions and labels
    val_preds, val_labels = model.get_metrics()
    
    # Plot confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    
    # Plot per-class accuracy
    per_class_accuracy = np.diag(cm) / cm.sum(axis=1)
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, per_class_accuracy, color="skyblue")
    plt.xlabel("Classes")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("\nClassification Report:\n")
    print(classification_report(val_labels, val_preds, target_names=class_names))


def compute_classification_metrics(y_true, y_pred, num_classes, class_names):
    """
    Compute precision, recall, and F1-score for each class
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        class_names: Class names for readability
    """
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    
    # Initialize TP, FP, FN
    true_positive = torch.zeros(num_classes)
    false_positive = torch.zeros(num_classes)
    false_negative = torch.zeros(num_classes)
    
    # Compute TP, FP, FN for each class
    for i in range(num_classes):
        true_positive[i] = ((y_pred == i) & (y_true == i)).sum().item()
        false_positive[i] = ((y_pred == i) & (y_true != i)).sum().item()
        false_negative[i] = ((y_pred != i) & (y_true == i)).sum().item()
    
    # Compute precision, recall, and F1-score
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Print results
    print(f"\nClassification Metrics:\n")
    print(f"{'Class':<15}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
    print("=" * 50)
    for i in range(num_classes):
        print(f"{class_names[i]:<15}{precision[i].item():<12.4f}{recall[i].item():<12.4f}{f1_score[i].item():<12.4f}")
    
    # Compute overall accuracy
    accuracy = (y_pred == y_true).sum().item() / len(y_true)
    print(f"\nOverall Accuracy: {accuracy:.4f}\n")


def main():
    try:
        # Initialize data module
        data_module = EyeDiseaseDataModule()
        data_module.setup()
    
    # Display dataset info
    print(f"Classes: {data_module.class_names}, Total Classes: {data_module.num_classes}")
    data_module.visualize_class_distribution()
    data_module.show_random_images()
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',
        filename='eye-classifier-{epoch:02d}-{val_acc:.4f}',
        save_top_k=1,
        mode='max'
    )
    
    logger = TensorBoardLogger("lightning_logs", name="eye_disease")
    
    # Train MobileNetV3
    print("\n=== Training MobileNetV3 Model ===\n")
    mobilenet_model = MobileNetModel(
        num_classes=data_module.num_classes,
        learning_rate=0.001,
        freeze_layers=True
    )
    
    trainer_mobilenet = pl.Trainer(
        max_epochs=20,
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        # Use more compatible device settings
        accelerator='cpu' if not torch.cuda.is_available() else 'gpu',
        devices=1,
    )
    
    trainer_mobilenet.fit(mobilenet_model, data_module)
    
    # Evaluate MobileNetV3
    print("\n=== MobileNetV3 Evaluation ===\n")
    trainer_mobilenet.test(mobilenet_model, data_module)
    plot_metrics(trainer_mobilenet, mobilenet_model, data_module.class_names)
    compute_classification_metrics(
        mobilenet_model.val_labels, 
        mobilenet_model.val_preds,
        data_module.num_classes,
        data_module.class_names
    )
    
    # Train DenseNet121
    print("\n=== Training DenseNet121 Model ===\n")
    densenet_model = DenseNetModel(
        num_classes=data_module.num_classes,
        learning_rate=0.001,
        freeze_layers=True
    )
    
    trainer_densenet = pl.Trainer(
        max_epochs=20,
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        # Use more compatible device settings
        accelerator='cpu' if not torch.cuda.is_available() else 'gpu',
        devices=1,
    )
    
    trainer_densenet.fit(densenet_model, data_module)
    
    # Evaluate DenseNet121
    print("\n=== DenseNet121 Evaluation ===\n")
    trainer_densenet.test(densenet_model, data_module)
    plot_metrics(trainer_densenet, densenet_model, data_module.class_names)
    compute_classification_metrics(
        densenet_model.val_labels, 
        densenet_model.val_preds,
        data_module.num_classes,
        data_module.class_names
    )
    
    # Compare models
    print("\n=== Model Comparison ===\n")
    print(f"MobileNetV3 Validation Accuracy: {mobilenet_model.val_acc:.4f}")
    print(f"DenseNet121 Validation Accuracy: {densenet_model.val_acc:.4f}")
    
    # Save best model - example using mobilenet
        torch.save(mobilenet_model.state_dict(), "mobilenet_eye_classifier.pt")
        
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser(description='Eye Disease Classification')
    parser.add_argument('--data_dir', type=str, default='./Eye_Dataset', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    return parser.parse_args()
def main():
    args = parse_args()
    data_module = EyeDiseaseDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    # Update other parts to use args
