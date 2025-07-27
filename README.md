# Bird Nest Detection System

AI system for detecting whether a bird is on or off the nest using multiple model architectures.

## **RECOMMENDED: GPU-Accelerated CNN**

### Training Commands (CNN):
```bash
# Best option: Transfer learning with GPU (3-6 minutes for 6000 photos)
python mom_detector_cnn.py train --gpu --transfer

# CNN from scratch with GPU (8-15 minutes)
python mom_detector_cnn.py train --gpu

# Quick test training (1-2 minutes, 10 epochs)
python mom_detector_cnn.py train --gpu --transfer --epochs 10

# Custom epoch count
python mom_detector_cnn.py train --gpu --transfer --epochs 25

# CPU fallback (if no GPU)
python mom_detector_cnn.py train
```

### GPU Utilities:
```bash
# Check GPU availability and performance
python mom_detector_cnn.py check-gpu

```

## **Enhanced Feature Model (CPU)**

### Training Commands (Enhanced):
```bash
# Full enhanced model with optimization (20-40 minutes)
python mom_detector_enhanced.py train

# With data augmentation (longer training)
python mom_detector_enhanced.py train --augment

# Fast training without optimization (5-10 minutes)
python mom_detector_enhanced.py train --no-optimize

# Skip hyperparameter optimization
python mom_detector_enhanced.py train --no-optimize --augment
```

## **Original Model (Legacy)**

### Training Commands (Original):
```bash
# Simple HOG+SVM model (2-5 minutes)
python mom_detector.py train
```

## **Inference Commands**

### Universal Interface (Auto-detects model type):
```bash
# Works with any trained model (original, enhanced, or CNN)
python mom_detector.py infer --image your_image.jpg

# Output examples:
# your_image.jpg: ON NEST (confidence: 0.923)  # Enhanced/CNN models
# your_image.jpg: OFF NEST                     # Original model
```

### Model-Specific Inference:
```bash
# Direct CNN inference (fastest)
python mom_detector_cnn.py infer --image your_image.jpg

# Direct enhanced model inference
python mom_detector_enhanced.py infer --image your_image.jpg
```

## **Analysis & Comparison Commands**

```bash

# Test model compatibility
python test_compatibility.py

```

## **Performance Comparison**

| Model | Training Time | Accuracy | GPU Support | Confidence Scores |
|-------|---------------|----------|-------------|-------------------|
| **CNN + Transfer (Recommended)** | 3-6 min | 90-95% | Yes | Yes |
| CNN | 8-15 min | 85-92% | Yes | Yes |
| Enhanced SVM | 20-40 min | 75-85% | No | Yes |
| Original HOG+SVM | 2-5 min | 70-80% | No | No |

## **Memory Requirements**

### For 6000 Training Photos:
- **CNN Training**: ~2-3 GB GPU memory, ~4-6 GB RAM
- **Enhanced Training**: ~4-6 GB RAM
- **Disk Space**: ~2-4 GB for cropped images
- **Model Files**: 50-200 MB

## **Setup & Dependencies**

### Virtual Environment:
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install opencv-python scikit-image scikit-learn joblib numpy

# For GPU acceleration (CNN):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## **Data Structure**

```
data/
├── off_nest/           # Images with no bird on nest
├── on_nest/            # Images with bird on nest
├── off_nest_cropped/   # Auto-generated ROI crops
├── on_nest_cropped/    # Auto-generated ROI crops
└── mc_enhanced/        # Misclassified samples for review
```

## **Configuration**

### ROI (Region of Interest) Settings:
```python
# Configurable in all model files
ROI_PCT = (0.49, 0.25, 0.7, 0.60)  # (x_min%, y_min%, x_max%, y_max%)
```

### Model Output:
- All models save to: `mom_detector_svm.joblib`
- CNN also saves PyTorch format: `mom_detector_cnn.pth`
- Automatic model type detection for inference

## **Quick Start Guide**

1. **Prepare your data**: Add images to `data/off_nest/` and `data/on_nest/`

2. **Train the best model**:
   ```bash
   python mom_detector_cnn.py train --gpu --transfer
   ```

3. **Test inference**:
   ```bash
   python mom_detector.py infer --image test_image.jpg
   ```

4. **Check performance**:
   ```bash
   python model_comparison.py
   ```

## **Troubleshooting**

### GPU Issues:
```bash
# Check GPU status
python mom_detector_cnn.py check-gpu

# If GPU out of memory, reduce batch size in mom_detector_cnn.py:
BATCH_SIZE = 16  # Instead of 32
```

### Training Issues:
- **Few samples warning**: Add more training images
- **Poor accuracy**: Try more epochs or transfer learning
- **Slow training**: Use GPU version or reduce image count

### Compatibility:
- All models use the same `mom_detector_svm.joblib` filename
- Original interface automatically detects model type
- Enhanced features used when available, fallback to basic features

## **Key Features**

### CNN Model:
- Transfer learning from ResNet18
- GPU acceleration
- Data augmentation
- 90-95% accuracy

### Enhanced Model:
- Multi-scale HOG features
- Local Binary Pattern (LBP) texture analysis
- Statistical features (brightness, contrast, edges)
- Ensemble classifier (SVM + Random Forest)
- Confidence scores

### Original Model:
- Simple and fast
- Reliable baseline
- Low resource requirements