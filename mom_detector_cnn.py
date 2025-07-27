#!/usr/bin/env python3
"""
GPU-Accelerated CNN Bird Detection Model

Usage:
  # Train GPU-accelerated model
  python mom_detector_cnn.py train --gpu

  # Train with transfer learning (faster, better accuracy)
  python mom_detector_cnn.py train --gpu --transfer

  # Infer on image
  python mom_detector_cnn.py infer --image your_image.jpg
"""
import os
import time
import shutil
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── CNN CONFIG ─────────────────────────────────────────────
DATA_DIR = "data"
SUBDIRS_RAW = ("off_nest", "on_nest")
ROI_PCT = (0.49, 0.25, 0.7, 0.60)
IMG_SIZE = (224, 224)  # Standard CNN input size
MODEL_PATH = "mom_detector_svm.joblib"  # Same filename for compatibility
CNN_MODEL_PATH = "mom_detector_cnn.pth"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Advanced data augmentation
TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

VAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# ──────────────────────────────────────────────────────────────


class BirdNestDataset(Dataset):
    """Custom dataset for bird nest detection."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        label = self.labels[idx]
        return image, label


class BirdDetectorCNN(nn.Module):
    """Custom CNN architecture for bird detection."""
    
    def __init__(self, use_transfer_learning=True):
        super(BirdDetectorCNN, self).__init__()
        
        if use_transfer_learning:
            # Use pre-trained ResNet18 as backbone
            self.backbone = models.resnet18(pretrained=True)
            # Freeze early layers
            for param in list(self.backbone.parameters())[:-10]:
                param.requires_grad = False
            # Replace final layer
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512)
        else:
            # Custom CNN
            self.backbone = nn.Sequential(
                # First conv block
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Second conv block
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Third conv block
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Fourth conv block
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                
                nn.Flatten(),
                nn.Linear(512, 512)
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 2 classes: off_nest, on_nest
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class CNNWrapper:
    """Wrapper to make CNN compatible with scikit-learn interface."""
    
    def __init__(self, model, device, transform):
        self.model = model
        self.device = device
        self.transform = transform
        self.classes_ = np.array([0, 1])  # For sklearn compatibility
    
    def predict(self, X):
        """Predict classes for input images."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for img in X:
                if isinstance(img, np.ndarray):
                    # Convert numpy to torch tensor
                    if len(img.shape) == 2:  # Grayscale
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                else:
                    img_tensor = img.unsqueeze(0).to(self.device)
                
                outputs = self.model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.cpu().item())
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities for input images."""
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for img in X:
                if isinstance(img, np.ndarray):
                    if len(img.shape) == 2:  # Grayscale
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                else:
                    img_tensor = img.unsqueeze(0).to(self.device)
                
                outputs = self.model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                probabilities.append(probs.cpu().numpy()[0])
        
        return np.array(probabilities)


def compute_roi_coords(w, h):
    """Compute ROI coordinates based on image dimensions."""
    x_min = int(ROI_PCT[0] * w)
    y_min = int(ROI_PCT[1] * h)
    x_max = int(ROI_PCT[2] * w)
    y_max = int(ROI_PCT[3] * h)
    return x_min, y_min, x_max, y_max


def ensure_cropped():
    """Ensure cropped ROI images exist for all raw images."""
    for sub in SUBDIRS_RAW:
        raw_folder = os.path.join(DATA_DIR, sub)
        crop_folder = os.path.join(DATA_DIR, f"{sub}_cropped")
        os.makedirs(crop_folder, exist_ok=True)
        
        if not os.path.isdir(raw_folder):
            print(f"Warning: missing {raw_folder}")
            continue
            
        for fname in os.listdir(raw_folder):
            raw_path = os.path.join(raw_folder, fname)
            crop_path = os.path.join(crop_folder, fname)
            
            if os.path.isfile(crop_path):
                continue
                
            img = cv2.imread(raw_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            x_min, y_min, x_max, y_max = compute_roi_coords(w, h)
            
            if not (0 <= x_min < x_max <= w and 0 <= y_min < y_max <= h):
                print(f"Skipping {raw_path}, ROI out of bounds")
                continue
                
            roi_img = img[y_min:y_max, x_min:x_max]
            cv2.imwrite(crop_path, roi_img)
            
        print(f"Cropped {sub} → {crop_folder}")


def load_dataset():
    """Load dataset for CNN training."""
    image_paths, labels = [], []
    
    for label, sub in enumerate(SUBDIRS_RAW):
        crop_folder = os.path.join(DATA_DIR, f"{sub}_cropped")
        if not os.path.isdir(crop_folder):
            print(f"Warning: missing {crop_folder}")
            continue
            
        for fname in os.listdir(crop_folder):
            path = os.path.join(crop_folder, fname)
            if os.path.isfile(path) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(path)
                labels.append(label)
    
    return image_paths, labels


def train_cnn_model(use_gpu=True, use_transfer_learning=True, epochs=EPOCHS):
    """Train CNN model with GPU acceleration."""
    print(f" Training CNN on {'GPU' if use_gpu and torch.cuda.is_available() else 'CPU'}")
    print(f"Device: {DEVICE}")
    
    # Ensure cropped images exist
    print("Ensuring cropped ROI images exist...")
    ensure_cropped()
    
    # Load dataset
    print("Loading dataset...")
    image_paths, labels = load_dataset()
    print(f"Total samples: {len(labels)}")
    
    if len(labels) < 10:
        print("Warning: Very few samples. Consider adding more training data.")
        return None
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = BirdNestDataset(train_paths, train_labels, TRAIN_TRANSFORM)
    val_dataset = BirdNestDataset(val_paths, val_labels, VAL_TRANSFORM)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print(f"Creating CNN model (transfer learning: {use_transfer_learning})...")
    model = BirdDetectorCNN(use_transfer_learning=use_transfer_learning)
    model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    print(f"Training for {epochs} epochs...")
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss, val_correct = 0.0, 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
        
        # Calculate accuracies
        train_acc = 100. * train_correct / len(train_dataset)
        val_acc = 100. * val_correct / len(val_dataset)
        
        scheduler.step(val_loss)
        
        # Print progress
        if epoch % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Acc: {train_acc:6.2f}% | "
                  f"Val Acc: {val_acc:6.2f}% | "
                  f"Time: {elapsed:6.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CNN_MODEL_PATH)
    
    # Load best model
    model.load_state_dict(torch.load(CNN_MODEL_PATH))
    
    print(f"\n Training completed in {time.time() - start_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Final evaluation
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1).cpu().numpy()
            y_true.extend(target.numpy())
            y_pred.extend(pred)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Off Nest", "On Nest"]))
    
    # Save compatible model
    wrapper = CNNWrapper(model, DEVICE, VAL_TRANSFORM)
    joblib.dump(wrapper, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} (compatible with original interface)")
    
    return model


def infer_on_image(image_path):
    """Inference on new image using CNN model."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Cannot find file: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"cv2 failed to load: {image_path}")
    
    # Extract ROI
    h, w = img.shape[:2]
    x_min, y_min, x_max, y_max = compute_roi_coords(w, h)
    
    if w >= x_max and h >= y_max:
        roi = img[y_min:y_max, x_min:x_max]
    else:
        roi = img
    
    # Convert to RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Load model
    if os.path.exists(CNN_MODEL_PATH):
        # Load PyTorch model directly
        model = BirdDetectorCNN()
        model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        # Prepare image
        img_tensor = VAL_TRANSFORM(roi_rgb).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            confidence = torch.max(probs).item()
            prediction = torch.argmax(output, dim=1).item()
        
        print(f"{image_path}: {'ON NEST' if prediction else 'OFF NEST'} (confidence: {confidence:.3f})")
    
    else:
        # Fallback to joblib model
        wrapper = joblib.load(MODEL_PATH)
        prediction = wrapper.predict([roi_rgb])[0]
        try:
            probs = wrapper.predict_proba([roi_rgb])[0]
            confidence = max(probs)
            print(f"{image_path}: {'ON NEST' if prediction else 'OFF NEST'} (confidence: {confidence:.3f})")
        except:
            print(f"{image_path}: {'ON NEST' if prediction else 'OFF NEST'}")


def check_gpu_availability():
    """Check GPU availability and performance."""
    print(" GPU Availability Check:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Current device: {DEVICE}")
        
        # Quick speed test
        print("\n Speed test:")
        x = torch.randn(1000, 1000).to(DEVICE)
        start = time.time()
        for _ in range(100):
            _ = torch.mm(x, x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        print(f"100 matrix multiplications: {elapsed:.3f}s")
    else:
        print("  GPU not available, using CPU")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU-Accelerated Bird Nest Detector")
    parser.add_argument("mode", choices=["train", "infer", "check-gpu"], help="Mode")
    parser.add_argument("--image", help="Path for inference image")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--transfer", action="store_true", help="Use transfer learning")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    
    args = parser.parse_args()
    
    if args.mode == "check-gpu":
        check_gpu_availability()
    elif args.mode == "train":
        train_cnn_model(
            use_gpu=args.gpu,
            use_transfer_learning=args.transfer,
            epochs=args.epochs
        )
    elif args.mode == "infer":
        if not args.image:
            parser.error("--image is required for infer mode")
        infer_on_image(args.image)


if __name__ == "__main__":
    main()