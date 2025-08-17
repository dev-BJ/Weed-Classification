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
import tempfile
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO, settings
settings.update({'sync': False})
# from math import ceil

# Preprocessing function for multiprocessing
def preprocess_paths(args):
    path, temp_dir, img_size, class_dir = args
    if os.path.exists(path):
        try:
            img = Image.open(path).convert('RGB').resize(img_size[:2])
            img_array = np.array(img) / 255.0
            if img_array.shape[-1] != 3:
                print(f"Warning: Image {path} has {img_array.shape[-1]} channels, expected 3")
                return None
            save_path = os.path.join(temp_dir, class_dir, os.path.basename(path))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray((img_array * 255).astype(np.uint8)).save(save_path, quality=95)
            return save_path
        except Exception as e:
            print(f"Error preprocessing {path}: {e}")
            return None
    return None

# --- Step 1: Data Preparation ---
base_path = os.path.abspath(os.path.dirname(__file__))
pkl_file = os.path.join(base_path, "out/weed.pkl")

start_time = time.time()
print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")

# Load data
try:
    df = pd.read_pickle(pkl_file)
except FileNotFoundError:
    print(f"Error: {pkl_file} not found.")
    exit(1)

os.makedirs(os.path.join(base_path, 'out/yolo'), exist_ok=True)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)
label_map = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
print("Class distribution:")
pprint(pd.Series(y).value_counts())
label_map['img_size'] = (224, 224, 3)
joblib.dump(label_map, os.path.join(base_path, "out/yolo/label_map.joblib"))

# Split data
X_img_train, X_img_val, y_train, y_val = train_test_split(
    df['rgb_path'].values, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_img_train)}, Validation samples: {len(X_img_val)}")

# Preprocess images in a temporary directory with YOLO structure
img_size = (224, 224, 3)
with tempfile.TemporaryDirectory(dir=base_path) as temp_dir:
    print(f"Using temporary directory: {temp_dir}")
    print(f"Available disk space: {shutil.disk_usage(base_path).free / (1024**3):.2f} GB")
    
    # Debug: Check first image
    if len(X_img_train) > 0 and os.path.exists(X_img_train[0]):
        test_img = Image.open(X_img_train[0]).convert('RGB')
        test_array = np.array(test_img.resize(img_size[:2]))
        print(f"Sample image shape: {test_array.shape}, Channels: {test_array.shape[-1]}")
        if test_array.shape[-1] != 3:
            print("Error: Images are not RGB. Check dataset.")

    # Create YOLO directory structure
    train_dir = os.path.join(temp_dir, 'train')
    val_dir = os.path.join(temp_dir, 'val')
    preprocessed_paths_train = []
    preprocessed_paths_val = []
    num_workers = 4

    # Preprocess training images
    train_args = [(path, train_dir, img_size, label_map[y_train[i]]) for i, path in enumerate(X_img_train)]
    try:
        with Pool(processes=num_workers) as pool:
            preprocessed_paths_train = list(tqdm(
                pool.imap(preprocess_paths, train_args),
                total=len(train_args),
                desc="Processing training images",
                unit="images",
                leave=False
            ))
    except Exception as e:
        print(f"Multiprocessing failed: {e}")
        print("Falling back to sequential processing.")
        preprocessed_paths_train = []
        for args in tqdm(train_args, desc="Processing training images", unit="images"):
            preprocessed_paths_train.append(preprocess_paths(args))

    # Preprocess validation images
    val_args = [(path, val_dir, img_size, label_map[y_val[i]]) for i, path in enumerate(X_img_val)]
    try:
        with Pool(processes=num_workers) as pool:
            preprocessed_paths_val = list(tqdm(
                pool.imap(preprocess_paths, val_args),
                total=len(val_args),
                desc="Processing validation images",
                unit="images",
                leave=False
            ))
    except Exception as e:
        print(f"Multiprocessing failed: {e}")
        preprocessed_paths_val = []
        for args in tqdm(val_args, desc="Processing validation images", unit="images"):
            preprocessed_paths_val.append(preprocess_paths(args))

    # Filter valid paths
    valid_train_indices = [i for i, path in enumerate(preprocessed_paths_train) if path is not None]
    valid_val_indices = [i for i, path in enumerate(preprocessed_paths_val) if path is not None]
    if len(valid_train_indices) == 0 or len(valid_val_indices) == 0:
        print("Error: No valid images loaded.")
        exit(1)
    X_img_train = np.array([preprocessed_paths_train[i] for i in valid_train_indices])
    y_train = y_train[valid_train_indices]
    X_img_val = np.array([preprocessed_paths_val[i] for i in valid_val_indices])
    y_val = y_val[valid_val_indices]
    print(f"Valid training images: {len(X_img_train)}, Valid validation images: {len(X_img_val)}")
    print(f"Failed training images: {len(preprocessed_paths_train) - len(valid_train_indices)}")
    print(f"Failed validation images: {len(preprocessed_paths_val) - len(valid_val_indices)}")

    # --- Step 2: YOLOv8 Training ---
    batch_size = 16
    steps_per_epoch = min(int(np.ceil(len(X_img_train) / batch_size)), 200)
    validation_steps = min(int(np.ceil(len(X_img_val) / batch_size)), 64)
    print(f"Training samples per epoch: {steps_per_epoch * batch_size}, Actual training set: {len(X_img_train)}")
    print(f"Validation samples per epoch: {validation_steps * batch_size}, Actual validation set: {len(X_img_val)}")

    model = YOLO('yolo11n-cls.pt')  # Load pre-trained classification model
    def count_yolo_folders():
        yolo_path = f'{base_path}/out/yolo'
        path_count = 0
        for i in os.listdir(yolo_path):
            if os.path.isfile(i):
                continue
            else:
                path_count = path_count + 1
        return path_count
    
    model.train(
        data=temp_dir,
        task='classify',
        mode='train',
        epochs=30,
        batch=batch_size,
        imgsz=224,
        device='cpu',
        patience=5,
        project=os.path.join(base_path, 'out/yolo'),
        # name=f'yolo_weed_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        name=f'yolo_weed_classifier_{count_yolo_folders() + 1}',
        save=True,
        exist_ok=True,
        verbose=True,
        val=True,
        # resume=True,
    )

    training_time = (time.time() - start_time) / 60
    print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training time: {training_time:.2f} minutes")

    print("Validate...")
    model.val()
    print("Validation done")

    # --- Step 3: Evaluation ---
    # val_images_paths = list(X_img_val)
    # print(f"Passing {len(val_images_paths)} validation image paths for prediction")

    # if len(val_images_paths) == 0:
    #     print("Error: No validation image paths available for prediction.")
    #     exit(1)

    results = model.predict(list(X_img_val), imgsz=224, device='cpu', verbose=False)
    y_pred = [int(result.probs.top1) for result in results]
    y_true = y_val

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    # Normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(base_path, 'out/tf/normalized_confusion_matrix.png'))
    plt.close()

    # --- Step 4: Inference ---
    def predict_weed(image_path, model, class_indices, size=(224, 224)):
        try:
            result = model.predict(image_path, imgsz=size[0], device='cpu', verbose=False)[0]
            predicted_class = result.probs.top1
            return list(class_indices.values())[predicted_class]
        except Exception as e:
            return f"Error processing image: {str(e)}"

    # Example usage
    image_path = os.path.join(base_path, 'mexican-poppy.jpeg')
    if os.path.exists(image_path):
        predicted_class = predict_weed(image_path, model, label_map)
        print(f"Predicted Class: {predicted_class}")
    else:
        print(f"Test image not found at {image_path}")

# Temporary directory is automatically deleted here
print("Temporary directory deleted.")
