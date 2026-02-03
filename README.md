# Bird Nest Detection System
Here is a link to a part of the entire stream that you can still watch. https://www.youtube.com/live/vQA4zKOyqT8?si=DcQPsPyXg9NaCLlF
AI system for detecting whether a bird is on or off the nest using multiple model architectures. You may notice it is under the handle "Liam the Chemist" this is because this project was hosted on my brothers website I created for him, thus his youtube handle felt appropriate.

## Project Summary:
**&nbsp;&nbsp;&nbsp;&nbsp;This repo code was used to train a very specific model, that when coupled with my Raspberry PI Bird Camera, could provide useful insight on the mother and father birds' behavior over the nesting cycle!**

&nbsp;&nbsp;&nbsp;&nbsp;In this project I put together a Raspberry PI to stream video over RMTP to a YouTube live stream.
Additionally, I created an API on my brother's website I created, (**liamthechemist.com**, a private repo) and had
a snapshot be piped from ffmpeg, every 20 seconds, to the PI's file system. Then another service on the device would upload that snapshot to the website's API endpoint as a curl cmd.
From that point a CNN transfer model (located on the web server), trained with the files in this repo, would infer the snapshot.

&nbsp;&nbsp;&nbsp;&nbsp;The website would push the model's results and probability to a monitor.log file full of all mother bird activity.
Additionally, I equipped the website with the ability to save images that the model likely got wrong.
I did this by saving the last 5 snapshots submitted through the API endpoint. The snapshots' file names would be changed to represent the AI's prediction for that image. If at any point amongst those 5 most recent snapshots the middle snapshot was the opposite status of the others, we would save this snapshot to an "outliers" folder. Additionally, if ever the AI's confidence score was lower then 75%, we would save that snapshot to a "possible misclassification" folder. I would later be able to determine if those images were indeed misclassifications and then manually insert these new images into the training dataset, then I would retrain a new model. In the end the model was correct about 99% of the time. However, some problems did make this data collection less then optimal.

### Problems Encountered
&nbsp;&nbsp;&nbsp;&nbsp;Originally, the website was equipped only with the legacy AI model, which can't provide a confidence score.
Thus the inferred results are partially binary and post processing can't smartly edit monitor.log regarding that.

&nbsp;&nbsp;&nbsp;&nbsp;Another major issue is that the stream was 1080p 30fps, which sometimes resulted in thermal throttling or even shutdowns.
This project took place in the summer, tempatures were 80-90 degrees on average during data collection. This resulted in tempatures rising above 80C on the chip, causing throttling and shutdowns. Sometimes if the reboot takes too long, the video stream is ended on youtubes end, and I have to manually start a new youtube stream. This however, didn't affect data collection too much because snapshots would still be getting send directly to the website endpoint for processing. I could've fixed this by putting vent holes in the make-shift camera case I was using; a plastic party cup with 2 layered platforms inside for the camera itself on the bottom and the raspberry pi unit directly above.

   Overall, the data is only about 50 hours worth in total and viewable on the repo.
   However, this project was more proof of concept, and proof that I could rapidly develop a pipeline given a limited time scenario.

In this repository, which was not actively developed on since it really didn't start out as something I thought would be worth documenting, until suddenly the project actually felt real, and data was real. You will find images of the augmented images and real images pulled for training, different script files in python used to train the model, monitor.log containing captured data, however log file changed rapidly because I continued working on another better model while the histogram based model was active.

The models I trained included originally a basic histogram model, this worked not amazingly because of the rapid change of lighting and shape especially after the eggs hatched. Then a CNN model was used and was much better, an additional improvement was made by using a transfer model already containing weights depicting basic understanding of shape/edge detection, this accelerated the models "ability to navigate down the gradient more rapidly". Let me explain what I mean by down the gradient. If you think of a given model's weighted matrix and bias as a function whose output is, in this case either Yes or No. Then during back propogation of those weighted matrixes and biases it's important to think, "If the model was wrong to predict No and instead should have predicted Yes, differentially, what changes to previous parameters would result in a "step" towards Yes the next time this image comes up. This is the essense of ML in general, and is often depicted as gravitating down the gradient of some complex space to settle. However, what I am really getting at here with the transfer model is the fact that it often sets your model up to find a more global minimum rather then the local minimum your randomly weighted model might stumble into.




### Project Hardware:
   - **Computer: Raspberry Pi Zero 2W /w an applied heatsink I bought**
   - **OS: Raspberry PI OS Lite (32bit) Debian Bookworm**
   - **Camera: Raspberry Pi Camera Module v3 - NoIR - Wide (120 deg)**
   - **IR Light: Univivi 90 deg 8-LED Outdoor 850nm IR Light (placed a respectful distance from nest)**

## Using the model trainer:
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
| **CNN + Transfer (Used)** | 3-6 min | 90-95% | Yes | Yes |
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

## **Data File Structure**

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
Note that, although I am talking in the scope of this project (data/off_nest/, etc), simply renaming the code, and adjusting
the settings (ROI should be reconfigured!), you can basically train a model to recognize differences between any two photos.
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
- Confidence scores

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
