import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import feature_generator, extract_features
import time

if __name__ == '__main__':
    # Set base path to handle file paths robustly
    base_path = os.path.abspath(os.getcwd())
    pkl_file = os.path.join(base_path, "process/out","weed.pkl")

    print("Starting training...")
    print("")

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
    # print(label_map)

    # Check class distribution
    print("Class distribution:")
    print(df['label'].value_counts())

    # Extract features and filter valid samples
    img_size = (64, 64)

    label_map['img_size'] = img_size
    
    hog = list(feature_generator(df['gray_path'], feature_type='hog', img_size=img_size))
    valid_hog = [i for i, (hist, _) in enumerate(hog) if hist is not None]
    # print("HOG", [hog[i][0] for i in valid_hog if i < 5])
    # print(valid_hog)

    X = np.array([hog[i][0] for i in valid_hog])
    # print(X[X == None])

    y = np.array([df['label'].values[i] for i in valid_hog])
    # print(y[y == None])


    print(f"Training X shape: {X.shape}, Memory: {X.nbytes / 1024 / 1024:.2f} MB")
    print(f"Training y shape: {y.shape}, Memory: {y.nbytes / 1024 / 1024:.2f} MB")
    # exit(1)

    # Check if any valid data remains
    if len(X) == 0:
        print("Error: No valid features extracted. Check image paths and files.")
        exit(1)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X = y = None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # print(X_train_scaled[X_train_scaled == None])
    X_val_scaled = scaler.transform(X_val)
    # print(X_val_scaled[X_val_scaled == None])

    print(f"Training scaled data shape: {X_train_scaled.shape}, Memory: {X_train_scaled.nbytes / 1024 / 1024:.2f} MB")

    print(f"Validation scaled data shape: {X_val_scaled.shape}, Memory: {X_val_scaled.nbytes / 1024 / 1024:.2f} MB")
    X_train = X_val = None
    # time.sleep(20)
    # exit(1)

    # Expanded hyperparameter grid
    param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10],
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['rbf'],
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.3]
    }

    # Define ensemble with SVC probability=True
    ensemble = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier()),
            ('svc', SVC(probability=True)),
            ('xgb', XGBClassifier())
        ],
        voting='soft'
    )

    # Perform grid search with verbose output
    grid_search = RandomizedSearchCV(ensemble, param_grid, n_iter=50, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2, random_state=42)
    # grid_search = GridSearchCV(ensemble, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)

    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    # Train final model
    model = grid_search.best_estimator_
    model.fit(X_train_scaled, y_train)

    # Save model and label encoder
    try:
        os.makedirs(os.path.join(base_path, 'process/out/hog'), exist_ok=True)
        joblib.dump(model, os.path.join(base_path, "process/out/hog", "trained_weed.joblib"))
        joblib.dump(label_map, os.path.join(base_path, "process/out/hog", "label_encoder.joblib"))
        joblib.dump(scaler, os.path.join(base_path, "process/out/hog", "scaler.joblib"))
        print("Model and label encoder saved successfully.")
    except Exception as e:
        print(f"Error saving model or label encoder: {e}")

    # Predict on validation set with tqdm
    y_pred = model.predict([X_val_scaled[i] for i in tqdm(range(len(X_val_scaled)), desc="Predicting")])

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(base_path, 'process/out/hog', 'confusion_matrix.png'))
    plt.show()
