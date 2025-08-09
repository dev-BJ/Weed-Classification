import os
import requests
import shutil
import logging
from PIL import Image
from simple_image_download import simple_image_download as simp
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths using pathlib
base_path = os.path.abspath(os.getcwd())
download_dir = os.path.join(base_path, "simple_images")
train_dir = f"{base_path}/train"
seperator = ','

def get_images(query, limit=10, dir_name='positive'):
    """
    Download images using simple_image_download and organize them into train_dir/dir_name/query.
    """
    try:
        response = simp.simple_image_download()
        response.download(query, limit)

        # Create train directory if it doesn't exist
        target_dir = train_dir / dir_name / query
        target_dir.mkdir(parents=True, exist_ok=True)

        # Move images to target directory
        source_dir = download_dir / query
        if source_dir.exists():
            for i in range(1, limit + 1):
                src_file = source_dir / f"{query}_{i}.jpg"
                dest_file = target_dir / f"{query}_{i}.jpg"
                if src_file.exists() and not dest_file.exists():
                    shutil.move(src_file, dest_file)
                    # Validate image
                    try:
                        Image.open(dest_file).verify()
                    except Exception as e:
                        logger.error(f"Invalid image {dest_file}: {e}")
                        dest_file.unlink(missing_ok=True)
            shutil.rmtree(download_dir, ignore_errors=True)
        
        return [str(target_dir / f"{query}_{i}.jpg") for i in range(1, limit + 1) if (target_dir / f"{query}_{i}.jpg").exists()]
    
    except Exception as e:
        logger.error(f"Error downloading images for {query}: {e}")
        return []

def download_inaturalist_images(species_name, max_images=100):
    """
    Download images from iNaturalist API for a given species and save to train_dir/positive/species_name.
    """
    try:
        # Query iNaturalist for taxon ID
        url = f"https://api.inaturalist.org/v1/taxa?q={species_name}&only_id=true"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data['total_results'] == 0:
            logger.warning(f"No taxa found for {species_name}")
            return []

        # Use the first taxon ID for specificity
        taxon_id = data['results'][0]['id']
        logger.info(f"Found taxon ID {taxon_id} for {species_name}")

        # Query observations
        observations_url = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&per_page={max_images}&photos=true"
        data = requests.get(observations_url, timeout=10)
        data.raise_for_status()
        observations = data.json()['results']

        if not observations:
            logger.warning(f"No observations found for {species_name}")
            return []

        # Create directory for species
        species_dir = train_dir / "positive" / species_name
        species_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        for i, obs in enumerate(observations):
            if 'photos' not in obs or not obs['photos']:
                continue

            img_url = obs['photos'][0]['url'].replace('square', 'original')
            dest_file = species_dir / f"{species_name}_{i}.jpg"

            # Skip if file already exists
            if dest_file.exists():
                logger.info(f"Skipping {dest_file.name}, already exists")
                continue

            # Download image
            try:
                img_response = requests.get(img_url, timeout=10)
                img_response.raise_for_status()
                with open(dest_file, 'wb') as f:
                    f.write(img_response.content)

                # Validate image
                try:
                    Image.open(dest_file).verify()
                    downloaded += 1
                    logger.info(f"{downloaded}/{max_images} image(s) downloaded for {species_name}")
                except Exception as e:
                    logger.error(f"Invalid image {dest_file}: {e}")
                    dest_file.unlink(missing_ok=True)
            
            except requests.RequestException as e:
                logger.error(f"Failed to download {img_url}: {e}")
                continue

            # Respect iNaturalist API rate limits (max 100 requests per minute)
            if i % 10 == 0 and i > 0:
                import time
                time.sleep(1)

        logger.info(f"Downloaded {downloaded} images for {species_name}")
        return [str(species_dir / f"{species_name}_{i}.jpg") for i in range(len(observations)) if (species_dir / f"{species_name}_{i}.jpg").exists()]

    except requests.RequestException as e:
        logger.error(f"API error for {species_name}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error for {species_name}: {e}")
        return []

if __name__ == "__main__":
    keywords = [
        'bermuda grass',
        'cogon grass',
        'parthenium weed',
        'siam weed',
        'water hyacinth',
        'cattail',
        'water lettuce',
        'broomrape',
        'dodder',
        'lantana camara',
        'mesquite',
        'mexican poppy',
    ]

    for keyword in keywords:
        logger.info(f"Processing images for {keyword}")
        images = download_inaturalist_images(keyword, max_images=50)
        if images:
            logger.info(f"Downloaded images for {keyword}: {len(images)} files")
        # Uncomment to use get_images as a fallback
        # if not images:
        #     logger.info(f"Falling back to simple_image_download for {keyword}")
        #     images = get_images(keyword, limit=10)
        #     logger.info(f"Downloaded images for {keyword}: {len(images)} files")