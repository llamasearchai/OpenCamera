#!/usr/bin/env python3
"""
Scene Classification Model Training for OpenCam Auto Exposure
Author: Nik Jois <nikjois@llamasearch.ai>

This script trains a CNN model to classify camera scenes for intelligent auto exposure.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SceneDataset(Dataset):
    """Dataset class for scene classification"""
    
    def __init__(self, data_dir: str, transform=None, augment=False):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing scene images organized by class
            transform: Image transformations
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        
        # Scene classes
        self.classes = ['indoor', 'outdoor', 'portrait', 'landscape', 'lowlight', 'backlit']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = []
        self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
    
    def _load_samples(self):
        """Load image paths and labels from directory structure"""
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((str(img_path), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SceneClassifier(nn.Module):
    """CNN model for scene classification"""
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        """
        Initialize model.
        
        Args:
            num_classes: Number of scene classes
            pretrained: Whether to use pretrained weights
        """
        super(SceneClassifier, self).__init__()
        
        # Use ResNet18 as backbone
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Extract features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Global average pooling
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Final classification layer
        x = self.backbone.fc(x)
        
        return x

class SceneClassifierTrainer:
    """Training pipeline for scene classifier"""
    
    def __init__(self, config: Dict):
        """Initialize trainer with configuration"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = SceneClassifier(
            num_classes=len(config['classes']),
            pretrained=config['pretrained']
        ).to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def prepare_data(self):
        """Prepare training and validation datasets"""
        # Data transforms
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        full_dataset = SceneDataset(
            data_dir=self.config['data_dir'],
            transform=train_transform
        )
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update validation dataset transform
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def validate(self) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """Full training loop"""
        logger.info("Starting training...")
        best_val_acc = 0.0
        
        for epoch in range(self.config['num_epochs']):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            logger.info(f'Epoch {epoch+1}/{self.config["num_epochs"]} '
                       f'Train Loss: {train_loss:.4f} '
                       f'Val Loss: {val_loss:.4f} '
                       f'Val Acc: {val_acc:.2f}% '
                       f'Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
                logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        logger.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }
        
        save_path = Path(self.config['output_dir']) / filename
        torch.save(checkpoint, save_path)
        logger.info(f'Model saved to {save_path}')
    
    def export_onnx(self, output_path: str):
        """Export model to ONNX format"""
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f'Model exported to ONNX: {output_path}')
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        save_path = Path(self.config['output_dir']) / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Training history plot saved to {save_path}')
        plt.show()

def create_synthetic_dataset(output_dir: str, num_samples_per_class: int = 1000):
    """Create synthetic dataset for training"""
    logger.info(f"Creating synthetic dataset with {num_samples_per_class} samples per class")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    classes = ['indoor', 'outdoor', 'portrait', 'landscape', 'macro', 'lowlight', 'backlit', 'highcontrast']
    
    for class_name in classes:
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i in range(num_samples_per_class):
            # Generate synthetic image based on class
            if class_name == 'lowlight':
                image = np.random.normal(30, 15, (224, 224, 3))
            elif class_name == 'backlit':
                image = np.full((224, 224, 3), 200)
                image[56:168, 56:168] = 50  # Dark subject
            elif class_name == 'highcontrast':
                image = np.random.choice([20, 220], (224, 224, 3))
            else:
                image = np.random.normal(128, 30, (224, 224, 3))
            
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Save image
            filename = class_dir / f"{class_name}_{i:04d}.jpg"
            cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    logger.info(f"Synthetic dataset created in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train scene classification model')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--create-synthetic', action='store_true', help='Create synthetic dataset')
    parser.add_argument('--export-onnx', action='store_true', help='Export model to ONNX')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'data_dir': args.data_dir or 'data/scenes',
        'output_dir': args.output_dir,
        'classes': ['indoor', 'outdoor', 'portrait', 'landscape', 'macro', 'lowlight', 'backlit', 'highcontrast'],
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'lr_step_size': 20,
        'lr_gamma': 0.1,
        'num_workers': 4,
        'pretrained': True,
    }
    
    # Load configuration from file if exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Create output directory
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(Path(config['output_dir']) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create synthetic dataset if requested
    if args.create_synthetic:
        create_synthetic_dataset(config['data_dir'])
    
    # Check if data directory exists
    if not os.path.exists(config['data_dir']):
        logger.error(f"Data directory not found: {config['data_dir']}")
        logger.info("Use --create-synthetic to generate synthetic data")
        return
    
    # Initialize trainer
    trainer = SceneClassifierTrainer(config)
    
    # Prepare data
    trainer.prepare_data()
    
    # Train model
    trainer.train()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Export to ONNX if requested
    if args.export_onnx:
        onnx_path = Path(config['output_dir']) / 'scene_classifier.onnx'
        trainer.export_onnx(str(onnx_path))
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main() 