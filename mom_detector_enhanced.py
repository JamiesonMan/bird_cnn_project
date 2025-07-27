#!/usr/bin/env python3
"""
Enhanced AI model to detect "on nest" vs "off nest" with improved accuracy.

Usage:
  # Train enhanced model
  python mom_detector_enhanced.py train

  # Infer on a new image
  python mom_detector_enhanced.py infer --image path/to/image.jpg

  # Train with data augmentation
  python mom_detector_enhanced.py train --augment

"""
import os
import cv2
import joblib
import shutil
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage import exposure, filters
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ─── ENHANCED CONFIG ─────────────────────────────────────────
DATA_DIR     = "data"
SUBDIRS_RAW  = ("off_nest", "on_nest")
# ROI percentages (x_min_pct, y_min_pct, x_max_pct, y_max_pct)
ROI_PCT      = (0.49, 0.25, 0.7, 0.60)
IMG_SIZE     = (128, 128)              # increased for better features
MODEL_PATH   = "mom_detector_svm.joblib"
SCALER_PATH  = "feature_scaler.joblib"

# Enhanced HOG parameters for multiple scales
HOG_PARAMS_FINE = dict(
    orientations=12,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
)

HOG_PARAMS_COARSE = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),
    block_norm="L2-Hys"
)

# LBP parameters for texture analysis
LBP_RADIUS = 3
LBP_N_POINTS = 24
LBP_METHOD = 'uniform'

# Data augmentation parameters
AUGMENT_PARAMS = dict(
    rotation_range=15,
    brightness_range=0.3,
    contrast_range=0.2,
    noise_std=0.02
)
# ──────────────────────────────────────────────────────────────


def preprocess_image(image):
    """Enhanced preprocessing with contrast enhancement and noise reduction."""
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return filtered


def extract_statistical_features(image):
    """Extract statistical features from the image."""
    features = []
    
    # Basic statistics
    features.extend([
        np.mean(image),
        np.std(image),
        np.median(image),
        np.min(image),
        np.max(image)
    ])
    
    # Contrast and brightness measures
    features.append(np.std(image) / (np.mean(image) + 1e-7))  # contrast ratio
    
    # Histogram features
    hist = cv2.calcHist([image], [0], None, [32], [0, 256])
    hist = hist.flatten() / (hist.sum() + 1e-7)  # normalize
    features.extend(hist)
    
    # Edge density
    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    features.append(edge_density)
    
    return np.array(features)


def extract_lbp_features(image):
    """Extract Local Binary Pattern features for texture analysis."""
    lbp = local_binary_pattern(image, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    
    # Calculate histogram of LBP
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_N_POINTS + 2, 
                          range=(0, LBP_N_POINTS + 2))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-7)  # normalize
    
    return hist


def extract_enhanced_features(image):
    """Extract comprehensive feature set combining multiple methods."""
    # Preprocess image
    processed = preprocess_image(image)
    
    # Multi-scale HOG features
    hog_fine = hog(processed, visualize=False, feature_vector=True, **HOG_PARAMS_FINE)
    hog_coarse = hog(processed, visualize=False, feature_vector=True, **HOG_PARAMS_COARSE)
    
    # LBP texture features
    lbp_features = extract_lbp_features(processed)
    
    # Statistical features
    stat_features = extract_statistical_features(processed)
    
    # Combine all features
    combined_features = np.concatenate([
        hog_fine,
        hog_coarse, 
        lbp_features,
        stat_features
    ])
    
    return combined_features


def augment_image(image):
    """Apply data augmentation to increase training data diversity."""
    augmented_images = [image]  # Include original
    
    # Rotation
    for angle in [-10, -5, 5, 10]:
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(rotated)
    
    # Brightness adjustment
    for factor in [0.8, 0.9, 1.1, 1.2]:
        bright = np.clip(image * factor, 0, 255).astype(np.uint8)
        augmented_images.append(bright)
    
    # Add noise
    for _ in range(2):
        noise = np.random.normal(0, AUGMENT_PARAMS['noise_std'] * 255, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        augmented_images.append(noisy)
    
    return augmented_images


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
        raw_folder  = os.path.join(DATA_DIR, sub)
        crop_folder = os.path.join(DATA_DIR, f"{sub}_cropped")
        os.makedirs(crop_folder, exist_ok=True)
        if not os.path.isdir(raw_folder):
            print(f"Warning: missing {raw_folder}")
            continue
        for fname in os.listdir(raw_folder):
            raw_path  = os.path.join(raw_folder, fname)
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


def load_dataset(use_augmentation=False):
    """Load dataset with enhanced features and optional augmentation."""
    X, y, paths = [], [], []
    
    for label, sub in enumerate(SUBDIRS_RAW):
        crop_folder = os.path.join(DATA_DIR, f"{sub}_cropped")
        if not os.path.isdir(crop_folder):
            print(f"Warning: missing {crop_folder}")
            continue
            
        for fname in os.listdir(crop_folder):
            path = os.path.join(crop_folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Resize image
            resized = cv2.resize(img, IMG_SIZE)
            
            if use_augmentation:
                # Apply augmentation
                augmented_images = augment_image(resized)
                for aug_img in augmented_images:
                    feats = extract_enhanced_features(aug_img)
                    X.append(feats)
                    y.append(label)
                    paths.append(path)
            else:
                # Original image only
                feats = extract_enhanced_features(resized)
                X.append(feats)
                y.append(label)
                paths.append(path)
    
    return np.array(X), np.array(y), paths


def create_ensemble_model():
    """Create an ensemble model combining multiple classifiers."""
    
    # Individual classifiers with different strengths
    svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, kernel='rbf', class_weight='balanced'))
    ])
    
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    
    # Combine in voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm_clf),
            ('rf', rf_clf)
        ],
        voting='soft'  # Use probability-based voting
    )
    
    return ensemble


def optimize_hyperparameters(X_train, y_train):
    """Optimize hyperparameters using grid search."""
    
    param_grid = {
        'svm__svm__C': [0.1, 1, 10],
        'svm__svm__gamma': ['scale', 'auto', 0.001, 0.01],
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [5, 10, 15]
    }
    
    ensemble = create_ensemble_model()
    
    print("Optimizing hyperparameters...")
    grid_search = GridSearchCV(
        ensemble, 
        param_grid, 
        cv=3, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_


def train_and_save(use_augmentation=False, optimize_params=True):
    """Train enhanced model with multiple improvements."""
    print("Ensuring cropped ROI images exist...")
    ensure_cropped()
    
    print("Loading data & extracting enhanced features...")
    X, y, paths = load_dataset(use_augmentation)
    print(f"{len(y)} samples, {X.shape[1]} features each")
    
    if len(y) < 10:
        print("Warning: Very few samples. Consider adding more training data.")
    
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X, y, paths, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training enhanced ensemble model...")
    
    if optimize_params and len(X_train) > 20:
        # Use hyperparameter optimization for larger datasets
        model = optimize_hyperparameters(X_train, y_train)
    else:
        # Use default ensemble for smaller datasets
        model = create_ensemble_model()
        model.fit(X_train, y_train)
    
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(
        y_test, y_pred,
        target_names=["Off Nest", "On Nest"]
    ))
    
    # Feature importance analysis (for Random Forest component)
    if hasattr(model, 'named_estimators_'):
        try:
            rf_importance = model.named_estimators_['rf'].feature_importances_
            print(f"Top 10 most important features (indices): {np.argsort(rf_importance)[-10:]}")
        except:
            pass
    
    # Prepare misclassification folders
    mc_folder = os.path.join(DATA_DIR, "mc_enhanced")
    pred_off_folder = os.path.join(mc_folder, "pred_off")
    pred_on_folder  = os.path.join(mc_folder, "pred_on")
    os.makedirs(pred_off_folder, exist_ok=True)
    os.makedirs(pred_on_folder, exist_ok=True)
    
    mis = np.where(y_pred != y_test)[0]
    if mis.size:
        print(f"Saving {len(mis)} misclassified samples:")
        for idx in mis:
            pred_label  = y_pred[idx]
            actual_label= y_test[idx]
            src          = p_test[idx]
            if pred_label:  # predicted 'On Nest'
                dst_folder = pred_on_folder
                pred_str = "On Nest"
            else:
                dst_folder = pred_off_folder
                pred_str = "Off Nest"
            actual_str = "On Nest" if actual_label else "Off Nest"
            dst = os.path.join(dst_folder, os.path.basename(src))
            shutil.copy(src, dst)
            print(f" {os.path.basename(src)} → predicted {pred_str}, actual {actual_str}")
    
    print(f"Saving enhanced model → {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    
    return model


def infer_on_image(image_path):
    """Inference on new image using enhanced model."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Cannot find file: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"cv2 failed to load: {image_path}")
    
    h, w = img.shape[:2]
    x_min, y_min, x_max, y_max = compute_roi_coords(w, h)
    
    if w >= x_max and h >= y_max:
        roi = img[y_min:y_max, x_min:x_max]
    else:
        roi = img
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    
    # Extract enhanced features
    feats = extract_enhanced_features(resized).reshape(1, -1)
    
    # Load model and predict
    model = joblib.load(MODEL_PATH)
    prediction = model.predict(feats)[0]
    
    # Get prediction confidence if available
    try:
        probabilities = model.predict_proba(feats)[0]
        confidence = max(probabilities)
        print(f"{image_path}: {'ON NEST' if prediction else 'OFF NEST'} (confidence: {confidence:.3f})")
    except:
        print(f"{image_path}: {'ON NEST' if prediction else 'OFF NEST'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Bird Nest Detector")
    parser.add_argument("mode", choices=["train","infer"], help="Mode: train or infer")
    parser.add_argument("--image", help="Path for inference image")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation during training")
    parser.add_argument("--no-optimize", action="store_true", help="Skip hyperparameter optimization")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_and_save(
            use_augmentation=args.augment,
            optimize_params=not args.no_optimize
        )
    else:
        if not args.image:
            parser.error("--image is required for infer mode")
        infer_on_image(args.image)


if __name__ == "__main__":
    main()