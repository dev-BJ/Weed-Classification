import os
import joblib
from ultralytics import YOLO

base_path = os.path.abspath(os.getcwd())
process_path = os.path.join(base_path, 'process')
label_map = joblib.load(f"{process_path}/out/yolo/label_map.joblib")
model = YOLO(f"{process_path}/out/yolo/yolo_weed_classifier_20250815_184719/weights/best.pt")
def predict_weed(image_path, model, class_indices, size=(224, 224)):
        try:
            result = model.predict(image_path, imgsz=size[0], device='cpu', verbose=False)[0]
            result.save()
            predicted_class = result.probs.top1
            return list(class_indices.values())[predicted_class]
        except Exception as e:
            return f"Error processing image: {str(e)}"

# Example usage
image_path = os.path.join(base_path, 'dataset/test/water-lettuce2.jpg')
if os.path.exists(image_path):
    predicted_class = predict_weed(image_path, model, label_map)
    print(f"Predicted Class: {predicted_class}")
else:
    print(f"Test image not found at {image_path}")
