import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from tqdm import tqdm

def filters(image):
    # Assume BGR input from cv2.imread
    pre_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    pre_image = clahe.apply(pre_image)
    # pre_image = cv2.equalizeHist(pre_image)
    return pre_image

def resize(image, size):
    h, w = image.shape[:2]
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Pad to target size
    padded = np.zeros((target_h, target_w, image.shape[2]) if len(image.shape) == 3 else (target_h, target_w), dtype=image.dtype)
    padded[:new_h, :new_w] = resized
    return resized

def get_image_by_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    lower_brown = np.array([10, 40, 40])
    upper_brown = np.array([20, 255, 255])
    lower_petal = np.array([0, 40, 40])
    upper_petal = np.array([10, 255, 255])
    lower_petal2 = np.array([160, 40, 40])
    upper_petal2 = np.array([180, 255, 255])

    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_brown = cv2.inRange(hsv_image, lower_brown, upper_brown)
    mask_petal1 = cv2.inRange(hsv_image, lower_petal, upper_petal)
    mask_petal2 = cv2.inRange(hsv_image, lower_petal2, upper_petal2)
    mask_petal = cv2.bitwise_or(mask_petal1, mask_petal2)
    combined_mask = cv2.bitwise_or(mask_green, mask_yellow)
    combined_mask = cv2.bitwise_or(combined_mask, mask_brown)
    combined_mask = cv2.bitwise_or(combined_mask, mask_petal)

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    leaf_extracted = cv2.bitwise_and(image, image, mask=combined_mask)
    return leaf_extracted

def extract_features(img, img_size=(512, 512), feature_type='lbp'):
    """Extract LBP features from an image."""
    try:
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            
        gray = cv2.resize(img, img_size)

        hist = image = None

        if feature_type == 'lbp':
            lbp_pattern = local_binary_pattern(gray, P=8, R=3, method='uniform')
            hist, _ = np.histogram(lbp_pattern.ravel(), bins=256, range=[0, 256])
            image = np.uint8(lbp_pattern / lbp_pattern.max() * 255)

        elif feature_type == 'hog':
            hist, image = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)

        return hist, image

    except Exception as e:
        print(f"Error processing {img}: {e}")
        return None, None

# Generator for memory-efficient feature extraction with tqdm
def feature_generator(image_paths, img_size=(1024, 1024), feature_type='lbp'):
    for path in tqdm(image_paths, desc=f'Extracting {feature_type} features', unit="image"):
        hist, image = extract_features(path, img_size, feature_type=feature_type)
        if hist is not None:
            yield hist, image