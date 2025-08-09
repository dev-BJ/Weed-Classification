from multiprocessing import Pool
import cv2
import numpy as np
import os
import pandas as pd
import pickle as pl
import shutil
from tqdm import tqdm
from utils import get_image_by_mask, filters, resize

def process_image(args):
    image_path, species_name, output_dir, size = args
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        preprocessed_image = filters(image)
        if preprocessed_image.shape[0] > size[0] or preprocessed_image.shape[1] > size[1]:
            preprocessed_image = resize(preprocessed_image, size)
        output_path = os.path.join(output_dir, species_name, os.path.basename(image_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, preprocessed_image)
        return output_path
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def preprocess_positive():
    base_path = os.path.dirname(__file__)
    input_dir = f'{base_path}/train/positive'
    output_dir = f'{base_path}/train/preprocessed_positive'
    annotation_file = f'{base_path}/positives.txt'
    size = (100, 100)

    if os.path.exists(annotation_file):
        os.remove(annotation_file)

    tasks = []
    for species_name in os.listdir(input_dir):
        for image_name in os.listdir(os.path.join(input_dir, species_name)):
            if image_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(input_dir, species_name, image_name)
                tasks.append((image_path, species_name, output_dir, size))

    with Pool() as pool:
        results = pool.map(process_image, tasks)

    with open(annotation_file, 'a') as f:
        for output_path in results:
            if output_path:
                f.write(f'{output_path} 1 0 0 {size[0]} {size[1]}\n')

def preprocess_negative():
    base_path = os.path.dirname(__file__)
    input_dir = f'{base_path}/train/negative'
    output_dir = f'{base_path}/train/preprocessed_negative'
    annotation_file = f'{base_path}/negatives.txt'
    size = (120, 120)

    if os.path.exists(annotation_file):
        os.remove(annotation_file)

    tasks = []
    for species_name in os.listdir(input_dir):
        for image_name in os.listdir(os.path.join(input_dir, species_name)):
            if image_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(input_dir, species_name, image_name)
                tasks.append((image_path, species_name, output_dir, size))

    with Pool() as pool:
        results = pool.map(process_image, tasks)

    with open(annotation_file, 'a') as f:
        for output_path in results:
            if output_path:
                f.write(f'{output_path}\n')

def preprocess_mask():
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_dir = f'{base_path}/train/positive'
    output_dir = f'{base_path}/train/extract'
    weed_pkl_path = f"{base_path}/out"

    gray_image = []
    rgb_image = []
    label = []

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(f'{weed_pkl_path}/weed.pkl'):
        os.remove(f'{weed_pkl_path}/weed.pkl')

    os.makedirs(weed_pkl_path, exist_ok=True)

    for species_name in tqdm(sorted(os.listdir(input_dir)), desc="Processing Images", unit="names", colour='red'):
        for image_name in tqdm(sorted(os.listdir(os.path.join(input_dir, species_name))), desc=f"{species_name}", unit="images", leave=False):
            if image_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(input_dir, species_name, image_name)
                rgb_image.append(image_path)
                try:
                    # print(f'Processing {image_path}')
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Failed to load image: {image_path}")
                        continue
                    # preprocessed_image = get_image_by_mask(image)
                    preprocessed_image = filters(image)
                    # print(preprocessed_image.shape)
                    # cv2.imshow("Fig", preprocessed_image)
                    # cv2.waitKey(0)
                    gray_output_path = os.path.join(output_dir, species_name, image_name)
                    os.makedirs(os.path.dirname(gray_output_path), exist_ok=True)
                    cv2.imwrite(gray_output_path, preprocessed_image)
                    gray_image.append(gray_output_path)
                    label.append(species_name)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
            # break
        # break

    df = pd.DataFrame({'rgb_path': rgb_image, 'gray_path': gray_image ,'label': label})
    df.to_pickle(f'{weed_pkl_path}/weed.pkl')

if __name__ == "__main__":
    preprocess_mask()