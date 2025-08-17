from pprint import pprint
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil
import joblib
import shutil
import time
import tempfile
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
import sys

def preprocess_paths(path):
    if os.path.exists(path):
            try:
                img = tf.keras.preprocessing.image.load_img(path, target_size=img_size[ : 2], color_mode='rgb')
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                save_path = os.path.join(temp_dir, os.path.basename(path))
                tf.keras.preprocessing.image.save_img(save_path, img_array)
                return save_path
            except Exception as e:
                print(f"Error preprocessing {path}: {e}")
                return None
    else:
        return None

# --- Step 1: Data Preparation ---
base_path = os.path.abspath(os.path.dirname(__file__))
pkl_file = os.path.join(base_path, "out/weed.pkl")

print(f'Starting training at {datetime.now().strftime("%Y/%m/%d_%H:%M:%S")}')

# Load data
try:
    df = pd.read_pickle(pkl_file)
except FileNotFoundError:
    print(f"Error: {pkl_file} not found.")
    exit(1)

os.makedirs(os.path.join(base_path, 'out/tf'), exist_ok=True)

# Preprocess images in a temporary directory
img_size = (224, 224, 3)  # Match MobileNetV3Small input size

with tempfile.TemporaryDirectory(dir=base_path) as temp_dir:
    print(f"Using temporary directory: {temp_dir}")
    print(f"Available disk space: {shutil.disk_usage(base_path).free / (1024**3):.2f} GB")
    preprocessed_paths = []
    num_workers = 4

    with Pool(processes=num_workers) as pool:
            preprocessed_paths = list(tqdm(
                pool.imap(preprocess_paths, df['rgb_path']),
                total=len(df['rgb_path']),
                desc=f"Processing paths",
                unit="images",
                leave=False
            ))
    
    # Filter valid paths
    valid_indices = [i for i, path in enumerate(preprocessed_paths) if path is not None]
    X_images = np.array([preprocessed_paths[i] for i in valid_indices])
    y = df['label'].values[valid_indices]
    if len(X_images) == 0:
        print("Error: No valid images loaded.")
        exit(1)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    label_map = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    print("Class distribution:")
    pprint(pd.Series(y).value_counts())

    label_map['img_size'] = img_size
    joblib.dump(label_map, os.path.join(base_path, "out/tf/label_map.joblib"))

    # Split data
    X_img_train, X_img_val, y_train, y_val = train_test_split(
        X_images, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_img_train)}, Validation samples: {len(X_img_val)}")

    # Convert labels to one-hot
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)

    # Data augmentation (simplified for CPU)
    batch_size = 16  # Reduced for Lenovo T560 (8GB RAM)

    train_datagen = ImageDataGenerator(
        rotation_range=15,  # Reduced for CPU
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator()  # No rescaling (done offline)

    # Create data generators
    train_df = pd.DataFrame({'path': X_img_train, 'label': y_train.astype(str)})
    val_df = pd.DataFrame({'path': X_img_val, 'label': y_val.astype(str)})
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df, x_col='path', y_col='label', target_size=img_size[ : 2],
        batch_size=batch_size, class_mode='categorical', shuffle=True, validate_filenames=True, color_mode='rgb'
    )
    val_generator = val_datagen.flow_from_dataframe(
        val_df, x_col='path', y_col='label', target_size=img_size[ : 2],
        batch_size=batch_size, class_mode='categorical', shuffle=False, validate_filenames=True, color_mode='rgb'
    )

    # Log generator output and steps
    # sample_batch = next(train_generator)
    # print(f"Batch shape: {sample_batch[0].shape}, Labels shape: {sample_batch[1].shape}")
    steps_per_epoch = min(ceil(len(X_img_train) / batch_size), 200)  # ~3,200 images
    validation_steps = min(ceil(len(X_img_val) / batch_size), 64)  # ~1,024 images
    print(f"Training samples per epoch: {steps_per_epoch * batch_size}, Actual training set: {len(X_img_train)}")
    print(f"Validation samples per epoch: {validation_steps * batch_size}, Actual validation set: {len(X_img_val)}")

    # print(sample_batch)

    print("Data generators created.")
    # sys.exit(1)

    # --- Step 2: Model Building ---
    base_model = EfficientNetB0(input_shape=img_size, include_top=False, weights='imagenet')
    base_model.trainable = False
    image_input = layers.Input(shape=img_size, name='image_input')
    base_output = base_model(image_input)
    image_pooling = layers.GlobalAveragePooling2D()(base_output)
    image_dense = layers.Dense(128, activation='relu')(image_pooling)
    image_dropout = layers.Dropout(0.3)(image_dense)
    output = layers.Dense(num_classes, activation='softmax')(image_dropout)

    model = models.Model(inputs=image_input, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- Step 3: Training ---
    start_time = time.time()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=20,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping]
    )
    training_time = (time.time() - start_time) / 60
    print(f"Training completed at {datetime.now().strftime('%I:%M %p %Z, %B %d, %Y')}")
    print(f"Training time: {training_time:.2f} minutes")

    # Save model
    model.save(os.path.join(base_path, f'out/tf/weed_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'))

    # --- Step 4: Evaluation ---
    val_images = []
    for path in X_img_val:
        try:
            img = tf.keras.preprocessing.image.load_img(path, target_size=img_size[ : 2], color_mode='rgb')
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            val_images.append(img_array)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    val_images = np.array(val_images)

    val_predictions = model.predict(val_images)
    y_pred = np.argmax(val_predictions, axis=1)
    y_true = y_val  # No need to argmax since y_val is already encoded

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

    # Training history plot
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(base_path, 'out/tf/training_history.png'))
    plt.close()

    # --- Step 5: Inference ---
    def predict_weed(image_path, model, class_indices, size=(224, 224)):
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=size, color_mode='rgb')
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            return list(class_indices.values())[predicted_class]
        except Exception as e:
            return f"Error processing image: {str(e)}"

    # Example usage
    image_path = os.path.join(base_path, 'mexican-poppy.jpeg')  # Update with actual path
    if os.path.exists(image_path):
        predicted_class = predict_weed(image_path, model, label_map)
        print(f"Predicted Class: {predicted_class}")
    else:
        print(f"Test image not found at {image_path}")

# Temporary directory is automatically deleted here
print("Temporary directory deleted.")
