import joblib
from utils import get_image_by_mask, filters, extract_features
import cv2
import os
from tqdm import tqdm
import numpy as np


def predict(image_path, model_path, label_map_path, scaler_path, feature_type):
    # Load model and label encoder
    model = joblib.load(model_path)
    label_map = joblib.load(label_map_path)
    scaler = joblib.load(scaler_path)
    # print(label_encoder)

    # Preprocess image
    image = cv2.imread(image_path)
    # preprocessed_image = get_image_by_mask(image)
    preprocessed_image = filters(image)

    # print(preprocessed_image.shape)

    cv2.imshow('Preprocessed Image', preprocessed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Extract features
    hist, image = extract_features(preprocessed_image, img_size=label_map['img_size'], feature_type=feature_type)
    feature_reshaped = np.array(hist).reshape(1, -1)
    print("Features:", feature_reshaped)
    # print(np.array(hist))
    cv2.imshow("LBP Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    feature_reshaped_scaled = scaler.transform(feature_reshaped)

    # Predict class
    class_index = model.predict(np.array(feature_reshaped_scaled))[0]
    print("Class index", class_index)
    predicted_class = label_map[class_index]

    return predicted_class

if __name__ == '__main__':
    base_path = os.path.abspath(os.path.dirname(__file__))
    test_img_path = os.path.join(base_path, 'test')
    feature_type='hog'
    # print(test_img_path)
    model_path = os.path.join(base_path, f'out/{feature_type}/trained_weed.joblib')
    label_map_path = os.path.join(base_path, f'out/{feature_type}/label_encoder.joblib')
    scaler_path = os.path.join(base_path, f'out/{feature_type}/scaler.joblib')

    for image in os.listdir(test_img_path):
        image_path = os.path.join(test_img_path, image)
        predicted_class = predict(image_path, model_path, label_map_path, scaler_path, feature_type=feature_type)
        print(f'Predicted class: {predicted_class}')