from fastapi import FastAPI
import numpy as np
import base64
import os
import joblib
from ultralytics import YOLO
import cv2
from pydantic import BaseModel

app = FastAPI()

@app.get("/api/")
async def root():
    return {"message": "Hello World"}


def predict_weed(image, model, class_indices, size=(224, 224)):
        try:
            result = model.predict(image, imgsz=size[0], device='cpu', verbose=False)[0]
            result.save()
            predicted_class = result.probs.top1
            return {'species':list(class_indices.values())[predicted_class], 'confidence':  float(result.probs.top1conf)}
        except Exception as e:
            print('Error:',str(e))
            return {'error': f"Error processing image: {str(e)}"}
        
class ImageClassify(BaseModel):
    image: str
@app.post("/api/classify")
async def classify(data: ImageClassify):

    img_str = data.image
    if img_str.startswith("data:application/octet-stream;base64,"):
        img_str = img_str.split(",")[1]

    binary_data = base64.b64decode(img_str)
    array = np.frombuffer(binary_data, dtype=np.uint8)

    img = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image data"}
    else:
         print("Image decoded successfully")
         os.makedirs("decoded_images", exist_ok=True)
         index = len(os.listdir("decoded_images"))
         cv2.imwrite(f"decoded_images/decoded_image_{index}.jpg", img)

    base_path = os.path.abspath(os.getcwd())
    process_path = os.path.join(base_path, 'process')
    label_map = joblib.load(f"{process_path}/out/yolo/label_map.joblib")
    model = YOLO(f"{process_path}/out/yolo/yolo_weed_classifier_20250815_184719/weights/best.pt")

    prediction = predict_weed(img, model, label_map)
    return prediction
