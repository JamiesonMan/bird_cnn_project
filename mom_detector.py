
#!/usr/bin/env python3
"""
Train and run a lightweight HOG+SVM model to detect “on nest” vs “off nest”,
with automatic cropping from full HD to ROI and automatic misclassification sorting.

This version uses a dynamic ROI based on image dimensions, so you can adjust
ROI percentage to encompass the full nest area after hatching.

Usage:
  # Train model (crops, trains, evaluates, and sorts misclassifications)
  python mom_detector.py train

  # Infer on a new image
  python mom_detector.py infer --image path/to/image.jpg

"""
import os
import cv2
import joblib
import shutil
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─── CONFIG ───────────────────────────────────────────────
DATA_DIR     = "data"
SUBDIRS_RAW  = ("off_nest", "on_nest")
# ROI percentages (x_min_pct, y_min_pct, x_max_pct, y_max_pct)
ROI_PCT      = (0.49, 0.25, 0.7, 0.60)
IMG_SIZE     = (64, 64)                # resize after cropping
MODEL_PATH   = "mom_detector_svm.joblib"
HOG_PARAMS   = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
)
# ──────────────────────────────────────────────────────────

def extract_hog_features(image):
    return hog(image, visualize=False, feature_vector=True, **HOG_PARAMS)


def is_enhanced_model(model_path):
    """Check if the saved model is an enhanced model."""
    if not os.path.exists(model_path):
        return False
    try:
        model = joblib.load(model_path)
        # Enhanced model is an ensemble, original is LinearSVC
        return hasattr(model, 'estimators_') or hasattr(model, 'named_estimators_')
    except:
        return False


def extract_enhanced_features_compatible(image):
    """Extract enhanced features if enhanced model is available."""
    try:
        from mom_detector_enhanced import extract_enhanced_features
        return extract_enhanced_features(image)
    except ImportError:
        # Fallback to original HOG features
        return extract_hog_features(image)


def compute_roi_coords(w, h):
    x_min = int(ROI_PCT[0] * w)
    y_min = int(ROI_PCT[1] * h)
    x_max = int(ROI_PCT[2] * w)
    y_max = int(ROI_PCT[3] * h)
    return x_min, y_min, x_max, y_max


def ensure_cropped():
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


def load_dataset():
    X, y, paths = [], [], []
    for label, sub in enumerate(SUBDIRS_RAW):
        crop_folder = os.path.join(DATA_DIR, f"{sub}_cropped")
        if not os.path.isdir(crop_folder):
            print(f"Warning: missing {crop_folder}")
            continue
        for fname in os.listdir(crop_folder):
            path = os.path.join(crop_folder, fname)
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            small = cv2.resize(img, IMG_SIZE)
            feats = extract_hog_features(small)
            X.append(feats)
            y.append(label)
            paths.append(path)
    return np.array(X), np.array(y), paths


def train_and_save():
    print("Ensuring cropped ROI images exist...")
    ensure_cropped()
    print("Loading data & features...")
    X, y, paths = load_dataset()
    print(f"{len(y)} samples, {X.shape[1]} features each")

    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X, y, paths, test_size=0.2, random_state=42, stratify=y
    )

    print("Training LinearSVC...")
    svm = LinearSVC(max_iter=5000, class_weight='balanced')
    svm.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = svm.predict(X_test)
    print(classification_report(
        y_test, y_pred,
        target_names=["Off Nest", "On Nest"]
    ))

    # Prepare misclassification folders
    mc_folder = os.path.join(DATA_DIR, "mc")
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

    print(f"Saving model → {MODEL_PATH}")
    joblib.dump(svm, MODEL_PATH)


def infer_on_image(image_path):
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
    
    # Check if we have an enhanced model
    if is_enhanced_model(MODEL_PATH):
        # Use enhanced processing for enhanced model
        small = cv2.resize(gray, (128, 128))  # Enhanced model uses 128x128
        feats = extract_enhanced_features_compatible(small).reshape(1, -1)
    else:
        # Use original processing for original model
        small = cv2.resize(gray, IMG_SIZE)
        feats = extract_hog_features(small).reshape(1, -1)
    
    model = joblib.load(MODEL_PATH)
    prediction = model.predict(feats)[0]
    
    # Try to get confidence if available (enhanced model)
    try:
        probabilities = model.predict_proba(feats)[0]
        confidence = max(probabilities)
        print(f"{image_path}: {'ON NEST' if prediction else 'OFF NEST'} (confidence: {confidence:.3f})")
    except:
        print(f"{image_path}: {'ON NEST' if prediction else 'OFF NEST'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HOG+SVM Nest Detector w/MC subdirs")
    parser.add_argument("mode", choices=["train","infer"], help="Mode: train or infer")
    parser.add_argument("--image", help="Path for inference image")
    args = parser.parse_args()
    if args.mode == "train":
        train_and_save()
    else:
        if not args.image:
            parser.error("--image is required for infer mode")
        infer_on_image(args.image)

if __name__ == "__main__":
    main()

