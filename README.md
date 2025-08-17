# WEED Plant Classification
This project is a machine learning-based plant classification system, specifically designed for weed identification.

## Features
- Image Processing: The system preprocesses images to gray scale for LBP and HOG-based feature extraction.
- Model Training: The system trains models using the preprocessed images.
- Image Fetching: The system fetches images from an API using get_images.py and get_images_pool.py.

## Scripts
- get_images.py: Fetches images from an API.
- get_images_pool.py: Fetches images from an API, optimized for faster processing.
- preprocess.py: Preprocesses images to gray scale for LBP and HOG-based feature extraction.
- train_*.py: Trains models using the preprocessed images.

## Getting Started
To get started with the project, follow these steps:

- Clone the repository: git clone [insert repository URL]
- Install dependencies: pip install -r requirements.txt
- Run the image fetching script: python get_images.py or python get_images_pool.py
- Run the preprocessing script: python preprocess.py
- Train the models: python train_*.py

**Note:** 
- *Only train_yolo.py would function properly for now and that is because it's working for what i want at the moment and i'm quite busy for now to correct the others.*
- ### *The adjustments are:*
    - I took removed the gray_path from the weed.pkl in the process/out folder.
    - So you'll need to add a temp directory code to process images in the dataset/train folder to gray, then readjust the X array variable to hold the gray image path.
    - You might want to use Pool, to speed up process especially when you have a large dataset.

## Contributing
Contributions are welcome! If you'd like to contribute to the project, please fork the repository and submit a pull request.

## Contact
- Contact me @ tife2111@gmail.com