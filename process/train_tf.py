from pprint import pprint
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil
import joblib
from datetime import datetime

# --- Step 1: Data Preparation ---
base_path = os.path.abspath(os.getcwd())
pkl_file = os.path.join(base_path, "process","out","weed.pkl")

print("Starting training...")

df = None
# Load and preprocess data
try:
    df = pd.read_pickle(pkl_file)
except FileNotFoundError:
    print(f"Error: {pkl_file} not found.")
    exit(1)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)
label_map = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
print("Class distribution:")
pprint(df['label'].value_counts())

img_size = (224, 224)  # Match MobileNetV2 input size
label_map['img_size'] = img_size

# Filter valid image paths
X_images = np.array([df['rgb_path'].iloc[i] for i in range(len(df)) if os.path.exists(df['rgb_path'].iloc[i])])
y = df['label'].values
if len(X_images) == 0:
    print("Error: No valid images loaded.")
    exit(1)

# Split data
X_img_train, X_img_val, y_train, y_val = train_test_split(
    X_images, y, test_size=0.2, random_state=42
)

# Convert labels to one-hot for TensorFlow
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)

# Data augmentation
batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_df = pd.DataFrame({'path': X_img_train, 'label': y_train.astype(str)})
val_df = pd.DataFrame({'path': X_img_val, 'label': y_val.astype(str)})

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    validate_filenames=True
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    validate_filenames=True
)

print("Data generators created.")

# --- Step 2: Model Building ---
# MobileNetV2 for image features
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Model architecture
image_input = layers.Input(shape=(224, 224, 3), name='image_input')
base_output = base_model(image_input)
image_pooling = layers.GlobalAveragePooling2D()(base_output)
image_dense = layers.Dense(128, activation='relu')(image_pooling)
image_dropout = layers.Dropout(0.5)(image_dense)
output = layers.Dense(num_classes, activation='softmax')(image_dropout)

# Build and compile model
model = models.Model(inputs=image_input, outputs=output)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Step 3: Training ---
history = model.fit(
    train_generator,
    steps_per_epoch=ceil(len(X_img_train) / batch_size),
    epochs=15,
    validation_data=val_generator,
    validation_steps=ceil(len(X_img_val) / batch_size)
)

os.makedirs(os.path.join(base_path, 'process/out/tf'), exist_ok=True)
# Save class indices
joblib.dump(label_map, os.path.join(base_path, "process/out/tf", "label_map.joblib"))
# Save model with timestamp
model.save(os.path.join(base_path, f'process/out/tf/weed_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras'))

# --- Step 4: Evaluation ---
# Load validation images for prediction
val_images = []
for path in X_img_val:
    try:
        img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        val_images.append(img_array)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
val_images = np.array(val_images)

# Predict on validation set
val_predictions = model.predict(val_images)
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
plt.show()
plt.savefig(os.path.join(base_path, 'out/tf/confusion_matrix.png'))
plt.close()

# Plot and save training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig(os.path.join(base_path, 'out/tf/training_history.png'))
plt.close()

# --- Step 5: Inference ---
def predict_weed(image_path, model, class_indices, size=(224, 224)):
    try:
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=size)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return list(class_indices.values())[predicted_class]
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Example usage
# image_path = os.path.join(base_path, 'test/control-of-mexican-poppy-in-beans-tanzania-1.jpeg')  # Update with actual path
# predicted_class = predict_weed(image_path, model, label_map)
# print(f"Predicted Class: {predicted_class}")