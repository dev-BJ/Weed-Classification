from pprint import pprint
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from utils import feature_generator, extract_features
from math import ceil

# --- Step 1: Data Preparation ---
base_path = os.path.abspath(os.path.dirname(__file__))
pkl_file = os.path.join(base_path, "out/weed.pkl")

print("Starting training...")

# Load and preprocess data
try:
    df = pd.read_pickle(pkl_file)
    # print(df.head())
except FileNotFoundError:
    print(f"Error: {pkl_file} not found.")
    exit(1)

os.makedirs(os.path.join(base_path, 'out/tf'), exist_ok=True)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)
label_map = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
# pprint(label_map)
print("Class distribution:")
pprint(df['label'].value_counts())
# exit(1)

img_size = (224, 224)  # Match MobileNetV2 input size
label_map['img_size'] = img_size
# Save class indices
joblib.dump(label_map, os.path.join(base_path, "out/tf/label_map.joblib"))

# Extract LBP and HOG features
lbp_features = list(feature_generator(df['gray_path'], img_size=img_size))
valid_lbp = [i for i, (hist, _) in enumerate(lbp_features) if hist is not None]

# Combine features
X_handcrafted = np.array([lbp_features[i][0] for i in valid_lbp])
y = df['label'].values[valid_lbp]
print(f"Handcrafted features shape: {X_handcrafted.shape}, {y.shape}")
# print((X_handcrafted[0:10], y[0:10]))

# exit(1)

# Scale handcrafted features
scaler = StandardScaler()
X_handcrafted_scaled = scaler.fit_transform(X_handcrafted)
joblib.dump(scaler, os.path.join(base_path, "out/tf/scaler.joblib"))

# Load images for MobileNetV2
def load_images(image_paths, target_size=(224, 224)):
    images = []
    for path in image_paths:
        try:
            img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            images.append(None)
    return images

# X_images = load_images([df['image_path'].values[i] for i in tqdm(valid_lbp, desc="Loading images for MobileNetV2", unit="image")])
X_images = np.array([os.path.exists(df['rgb_path'].values[i]) for i in valid_lbp])
valid_image_indices = [i for i, img in enumerate(X_images) if img is not False]
# X_images = np.array([X_images[i] for i in valid_image_indices])

# Actual selection happened here
X_images = np.array([df['rgb_path'].iloc[i] for i in valid_image_indices])
X_handcrafted_scaled = X_handcrafted_scaled[valid_image_indices]
y = y[valid_image_indices]
# pprint(X_images[0:5])
# pprint(X_handcrafted_scaled[0:5])
# pprint(y[0:5])
# exit(1)

if len(X_images) == 0:
    print("Error: No valid images loaded.")
    exit(1)

# Split data
X_img_train, X_img_val, X_hand_train, X_hand_val, y_train, y_val = train_test_split(
    X_images, X_handcrafted_scaled, y, test_size=0.2, random_state=42
)

# Convert labels to one-hot for TensorFlow
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)
# print(y_train_onehot[0:2])
# exit(1)

# Data augmentation
# train_datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

batch_size = 16
# Create a custom generator for images and handcrafted features
def custom_generator(img_data, handcrafted_data, labels, batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    df = pd.DataFrame({
        'path': img_data,
        'label': labels
    })
    df['label'] = df['label'].astype(str)  # Convert to strings for categorical mode
    gen = datagen.flow_from_dataframe(
        df,
        x_col='path',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',  # Automatically one-hot encodes labels
        shuffle=True,
        validate_filenames=True  # Ensure all file paths are valid
    )
    while True:
        img_batch, label_batch = next(gen)
        indices = gen.index_array[:batch_size]
        pprint(len(indices))
        hand_batch = handcrafted_data[indices]
        # print(img_batch.shape, hand_batch.shape, label_batch.shape)
        yield (img_batch, hand_batch), label_batch

train_generator = custom_generator(X_img_train, X_hand_train, y_train, batch_size=batch_size)
sample_batch = next(train_generator)
print("Image batch shape:", sample_batch[0][0].shape)  # Should be (batch_size, 224, 224, 3)
print("Handcrafted batch shape:", sample_batch[0][1].shape)  # Should be (batch_size, feature_dim)
print("Label batch shape:", sample_batch[1].shape)  # Should be (batch_size, num_classes)
val_generator = custom_generator(X_img_val, X_hand_val, y_val, batch_size=batch_size)
# print("Train generator size:", len(train_generator))
# print("Validation generator size:", len(val_generator)
print("Data generators created.")
# exit(1)

# --- Step 2: Model Building ---
# MobileNetV2 for image features
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Input for handcrafted features
handcrafted_input = layers.Input(shape=(X_handcrafted_scaled.shape[1],), name='handcrafted_input')
handcrafted_dense = layers.Dense(64, activation='relu')(handcrafted_input)
handcrafted_dropout = layers.Dropout(0.5)(handcrafted_dense)

# Image processing branch
image_input = layers.Input(shape=(224, 224, 3), name='image_input')
base_output = base_model(image_input)
image_pooling = layers.GlobalAveragePooling2D()(base_output)
image_dense = layers.Dense(128, activation='relu')(image_pooling)
image_dropout = layers.Dropout(0.5)(image_dense)

# Combine features
combined = layers.Concatenate()([image_dropout, handcrafted_dropout])
dense_combined = layers.Dense(128, activation='relu')(combined)
dropout_combined = layers.Dropout(0.5)(dense_combined)
output = layers.Dense(num_classes, activation='softmax')(dropout_combined)

# Build and compile model
model = models.Model(inputs=[image_input, handcrafted_input], outputs=output)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Step 3: Training ---
history = model.fit(
    train_generator,
    steps_per_epoch=ceil(len(X_img_train) / batch_size),
    epochs=5,
    validation_data=val_generator,
    validation_steps=ceil(len(X_img_val) / batch_size)
)

# Save model with timestamp
model.save(os.path.join(base_path, f'out/tf/weed_classifier_hybrid_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'))

# --- Step 4: Evaluation ---
# Predict on validation set
val_predictions = model.predict([X_img_val, X_hand_val])
y_pred = np.argmax(val_predictions, axis=1)
y_true = np.argmax(y_val_onehot, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Plot and save confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(base_path, 'out/tf/confusion_matrix.png'))
plt.close()

# Plot and save training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(base_path, 'out/tf/training_history.png'))
plt.close()

# --- Step 5: Inference ---
# def predict_weed(image_path, model, class_indices, scaler, size=(224, 224)):
#     try:
#         # Load and preprocess image
#         img = tf.keras.preprocessing.image.load_img(image_path, target_size=size)
#         img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # Extract and scale handcrafted features
#         lbp_hist = extract_features(image_path, feature_type='lbp')
#         hog_hist = extract_features(image_path, feature_type='hog')
#         if lbp_hist[0] is None or hog_hist[0] is None:
#             return f"Error extracting features from {image_path}"
#         handcrafted_features = np.concatenate([lbp_hist[0], hog_hist[0]])
#         handcrafted_features_scaled = scaler.transform([handcrafted_features])

#         # Predict
#         prediction = model.predict([img_array, handcrafted_features_scaled])
#         predicted_class = np.argmax(prediction, axis=1)[0]
#         return list(class_indices.values())[predicted_class]
#     except Exception as e:
#         return f"Error processing image: {str(e)}"

# # Example usage
# image_path = '/path/to/test_image.jpg'  # Update with actual path
# with open(os.path.join(base_path, 'out/class_indices.json'), 'r') as f:
#     class_indices = json.load(f)
# scaler = joblib.load(os.path.join(base_path, 'out/scaler.joblib'))
# predicted_class = predict_weed(image_path, model, class_indices, scaler)
# print(f"Predicted Class: {predicted_class}")

# Optional: Visualize LBP histogram for a sample image
sample_image = df['image_path'].iloc[0]
sample_features = extract_features(sample_image)
if sample_features[0] is not None:
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(sample_features[0])), sample_features[0])
    plt.title(f'LBP Histogram for {os.path.basename(sample_image)}')
    plt.xlabel('Bin')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(base_path, 'out/lbp_histogram.png'))
    plt.close()