from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shutil
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO, settings
import cv2

# Disable YOLO cloud sync
settings.update({'sync': False})

# --- CONFIGURATION ---
IMG_SIZE = (224, 224, 3)
BATCH_SIZE = 16
EPOCHS = 40
BASE_PATH = os.path.abspath(os.path.dirname(__file__))
PKL_FILE = os.path.join(BASE_PATH, "out/plants.pkl")
CACHE_DIR = os.path.join(BASE_PATH, "out/yolo_preprocessed")
YOLO_OUT = os.path.join(BASE_PATH, "out/yolo8")
os.makedirs(YOLO_OUT, exist_ok=True)
NUM_WORKERS = max(1, cpu_count() - 1)


# --- IMAGE PREPROCESSING FUNCTION ---
def preprocess_image(args):
    """Resize and convert images for YOLO classification training."""
    path, out_dir, img_size, class_name = args
    try:
        if not os.path.exists(path):
            return None

        img = cv2.imread(path)
        if img is None:
            return None

        # Convert to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size[0], img_size[1]), interpolation=cv2.INTER_AREA)

        class_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        save_path = os.path.join(class_dir, os.path.basename(path))

        # Skip if already exists
        if os.path.exists(save_path):
            return save_path

        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        return save_path

    except Exception:
        return None


# --- LOAD DATA ---
def load_data():
    if not os.path.exists(PKL_FILE):
        raise FileNotFoundError(f"Dataset pickle not found at: {PKL_FILE}")
    df = pd.read_pickle(PKL_FILE)
    if 'label' not in df.columns or 'rgb_path' not in df.columns:
        raise ValueError("DataFrame must contain 'label' and 'rgb_path' columns.")
    return df


# --- PREPROCESS IMAGES (CACHED) ---
def prepare_images(X, y, label_map, split_name):
    """Preprocess images only if not already cached."""
    print(f"\nPreparing {split_name} images...")

    split_cache = os.path.join(CACHE_DIR, split_name)
    os.makedirs(split_cache, exist_ok=True)

    # Check cache completeness
    expected_classes = [label_map[i] for i in np.unique(y)]
    all_exist = all(
        os.path.exists(os.path.join(split_cache, cls))
        and len(os.listdir(os.path.join(split_cache, cls))) > 0
        for cls in expected_classes
    )

    if all_exist:
        print(f"✅ Found existing preprocessed {split_name} images — skipping reprocessing.")
        processed_paths = []
        for cls in expected_classes:
            cls_dir = os.path.join(split_cache, cls)
            processed_paths.extend([os.path.join(cls_dir, f) for f in os.listdir(cls_dir)])
        return np.array(processed_paths), y

    # Otherwise, preprocess fresh
    args_list = [(path, split_cache, IMG_SIZE, label_map[y[i]]) for i, path in enumerate(X)]
    with Pool(NUM_WORKERS) as pool:
        processed_paths = list(
            tqdm(pool.imap(preprocess_image, args_list), total=len(args_list),
                 desc=f"Processing {split_name} images", unit="img")
        )

    valid_indices = [i for i, p in enumerate(processed_paths) if p is not None]
    X_valid = np.array([processed_paths[i] for i in valid_indices])
    y_valid = y[valid_indices]

    print(f"✅ {len(X_valid)} valid {split_name} images (skipped {len(X) - len(X_valid)}).")
    return X_valid, y_valid


# --- TRAIN YOLO MODEL ---
def train_yolo(X_train, X_val, y_train, y_val, label_encoder):
    """Train YOLO classification model."""
    print("\nStarting YOLO classification training...")
    model_path = os.path.join(YOLO_OUT, "yolo_weed_classifier_1/weights/last.pt")
    pretrained = model_path if os.path.exists(model_path) else "yolov8n-cls.pt"
    model = YOLO(pretrained)

    def count_yolo_folders():
        return len([d for d in os.listdir(YOLO_OUT) if os.path.isdir(os.path.join(YOLO_OUT, d))])

    project_name = f"yolo_weed_classifier_{count_yolo_folders() + 1}"

    model.train(
        data=CACHE_DIR,
        task='classify',
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE[0],
        device='cpu',  # change to '0' if CUDA is available
        patience=5,
        project=YOLO_OUT,
        name=project_name,
        exist_ok=True,
        verbose=True,
        val=True,
        resume=True if os.path.exists(model_path) else False,
    )

    return model


# --- EVALUATION ---
def evaluate_model(model, X_val, y_val, label_encoder):
    print("\nRunning evaluation...")
    results = model.predict(list(X_val), imgsz=IMG_SIZE[0], device='cpu', verbose=False)
    y_pred = [int(r.probs.top1) for r in results]

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_val, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(YOLO_OUT, 'normalized_confusion_matrix.png'))
    plt.close()


# --- PREDICTION DEMO ---
def predict_image(image_path, model, label_encoder):
    if not os.path.exists(image_path):
        print(f"Test #  image not found: {image_path}")
        return
    result = model.predict(image_path, imgsz=IMG_SIZE[0], device='cpu', verbose=False)[0]
    pred_idx = result.probs.top1
    pred_class = label_encoder.inverse_transform([pred_idx])[0]
    print(f"Predicted Class: {pred_class}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_data()

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])
    label_map = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    joblib.dump(label_map, os.path.join(YOLO_OUT, "label_map.joblib"))

    print("Class distribution:")
    pprint(pd.Series(y).value_counts())

    # Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(df['rgb_path'].values, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Prepare datasets
    X_train, y_train = prepare_images(X_train, y_train, label_map, "train")
    X_val, y_val = prepare_images(X_val, y_val, label_map, "val")

    # Train and evaluate
    model = train_yolo(X_train, X_val, y_train, y_val, label_encoder)
    evaluate_model(model, X_val, y_val, label_encoder)

    # Test example
    test_img = os.path.join(BASE_PATH, 'mexican-poppy.jpeg')
    predict_image(test_img, model, label_encoder)

    print(f"\n✅ Done in {(time.time() - start_time) / 60:.2f} minutes.")
