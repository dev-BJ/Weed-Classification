import cv2
import numpy as np
import os

def filters(image):
    pre_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    pre_image = clahe.apply(pre_image)
    pre_image = cv2.GaussianBlur(pre_image, (5, 5), 0)
    pre_image = cv2.equalizeHist(pre_image)
    # print(pre_image.shape)
    return pre_image

def resize(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def preprocess_positive():
    base_path = os.path.dirname(__file__)
    # print(base_path)
    input_dir = f'{base_path}/train/positive'
    output_dir = f'{base_path}/train/preprocessed_positive'
    annotation_file = f'{base_path}/positives.txt'
    size = (100, 100)

    if os.path.exists(annotation_file):
        os.remove(annotation_file)

    for species_name in os.listdir(input_dir):
        # print(species_name)
        for image_name in os.listdir(os.path.join(input_dir, species_name)):
            if image_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(input_dir, species_name)
                image_path = os.path.join(image_path, image_name)
                # print(image_path)
                image = cv2.imread(image_path)
                preprocessed_image = filters(image)
                if preprocessed_image.shape > size:
                    preprocessed_image = resize(preprocessed_image, size)
                output_path = os.path.join(output_dir, species_name)
                output_path = os.path.join(output_path, image_name)
                # print(output_path)
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, preprocessed_image)

                # cv2.imshow("Image", preprocessed_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Update positives.txt with resized image path
                with open(annotation_file, 'a') as f:
                    f.write(f'{output_path} 1 0 0 {size[0]} {size[1]}\n')
                # break
        # break

def preprocess_negative():
    base_path = os.path.dirname(__file__)
    # print(base_path)
    input_dir = f'{base_path}/train/negative'
    output_dir = f'{base_path}/train/preprocessed_negative'
    annotation_file = f'{base_path}/negatives.txt'
    size = (120, 120)

    if os.path.exists(annotation_file):
        os.remove(annotation_file)

    for species_name in os.listdir(input_dir):
        # print(species_name)
        for image_name in os.listdir(os.path.join(input_dir, species_name)):
            if image_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(input_dir, species_name)
                image_path = os.path.join(image_path, image_name)
                # print(image_path)
                image = cv2.imread(image_path)
                preprocessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if preprocessed_image.shape > size:
                    preprocessed_image = resize(preprocessed_image, size)
                output_path = os.path.join(output_dir, species_name)
                output_path = os.path.join(output_path, image_name)
                # print(output_path)
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, preprocessed_image)

                # cv2.imshow("Image", preprocessed_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Update positives.txt with resized image path
                with open(annotation_file, 'a') as f:
                    f.write(f'{output_path}')
                # break
        # break

if __name__ == "__main__":
    preprocess_positive()
    # preprocess_negative()