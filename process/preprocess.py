import cv2
import numpy as np
import os
import pandas as pd
import pickle as pl
import shutil
from tqdm import tqdm
from utils import get_image_by_mask, filters, resize
from multiprocessing import Pool

def process_image(args):
    """Process a single image and return its paths and label."""
    image_path, species_name, output_dir = args
    try:
        if not image_path.endswith(('.jpg', '.png', '.jpeg')):
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None

        # preprocessed_image = filters(image)
        # gray_output_path = os.path.join(output_dir, species_name, os.path.basename(image_path))
        # os.makedirs(os.path.dirname(gray_output_path), exist_ok=True)
        # cv2.imwrite(gray_output_path, preprocessed_image)

        return {
            'rgb_path': image_path,
            # 'gray_path': gray_output_path,
            'label': species_name
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    
def yield_image_path(args):
    input_dir, species_name, img_limit= args
    for image_name in sorted(os.listdir(os.path.join(input_dir, species_name)))[:img_limit]:
            if image_name.endswith(('.jpg', '.png')):
                yield os.path.join(input_dir, species_name, image_name)
                
def yield_arg_list(input_dir, species_name, output_dir, img_limit=200):
    for image_path in yield_image_path((input_dir, species_name, img_limit)):
        yield (image_path, species_name, output_dir)

def preprocess_mask():
    base_path = os.path.abspath(os.getcwd())
    input_dir = f'{base_path}/dataset/train'
    output_dir = f'{base_path}/dataset/train_gray'
    weed_pkl_path = f"{base_path}/process/out"
    img_limit = 200
    num_workers = os.cpu_count()  # Number of CPU cores for parallel processing
    print("Number of works:", num_workers)

    gray_image = []
    rgb_image = []
    label = []

    # Clean up existing directories and files
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(f'{weed_pkl_path}/weed.pkl'):
        os.remove(f'{weed_pkl_path}/weed.pkl')

    os.makedirs(weed_pkl_path, exist_ok=True)

    # Process each species
    for species_name in tqdm(sorted(os.listdir(input_dir)), desc="Processing Species", unit="species", colour='red'):
        # image_paths = [
        #     os.path.join(input_dir, species_name, image_name)
        #     for image_name in sorted(os.listdir(os.path.join(input_dir, species_name)))[:img_limit]
        #     if image_name.endswith(('.jpg', '.png'))
        # ]

        # Prepare arguments for each image
        # args_list = [(image_path, species_name, output_dir) for image_path in image_paths]
        # args_list = [(image_path, species_name, output_dir) for image_path in yield_image_path((input_dir, species_name, img_limit))]
        args_list = list(yield_arg_list(input_dir, species_name, output_dir, img_limit))

        # Use Pool to parallelize image processing
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_image, args_list),
                total=len(args_list),
                desc=f"Processing {species_name}",
                unit="images",
                leave=False
            ))

        # Collect results
        for result in results:
            if result is not None:
                rgb_image.append(result['rgb_path'])
                # gray_image.append(result['gray_path'])
                label.append(result['label'])

    # Save results to DataFrame
    df = pd.DataFrame(
        {
            'rgb_path': rgb_image,
            # 'gray_path': gray_image,
            'label': label,
        }
    )
    df.to_pickle(f'{weed_pkl_path}/weed.pkl')

if __name__ == "__main__":
    preprocess_mask()