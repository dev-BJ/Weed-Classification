import os
import requests
import shutil
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_path = os.path.abspath(os.getcwd())
os.makedirs(f'{base_path}/dataset', exist_ok=True)
train_dir = os.path.join(base_path, 'dataset', 'train')

def download_inaturalist_images(species_name, max_images=100):
    """
    Download images from iNaturalist API for a given species and save to train_dir/positive/species_name.
    """
    try:
        os.makedirs(train_dir, exist_ok=True)

         # Create directory for species
        species_dir = os.path.join(train_dir, species_name)
        os.makedirs(species_dir, exist_ok=True)
        
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

        downloaded = 0
        
        if len(os.listdir(species_dir)) >= len(observations):
            logger.info(f"Skipping {species_name}, already has {len(os.listdir(species_dir))} images")
            return [os.path.join(species_dir, f) for f in os.listdir(species_dir) if os.path.isfile(os.path.join(species_dir, f))]
        else:
            downloaded = len(os.listdir(species_dir))

        for i, obs in enumerate(observations):
            
            if len(os.listdir(species_dir)) >= len(observations):
                break

            if 'photos' not in obs or not obs['photos']:
                continue

            img_url = obs['photos'][0]['url'].replace('square', 'original')
            dest_file = os.path.join(species_dir, f"{obs['id']}.jpg")

            # Skip if file already exists
            if os.path.exists(dest_file):
                logger.info(f"Skipping {dest_file}, already exists")
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
                    logger.info(f"{downloaded}/{len(observations)} image(s) downloaded for {species_name}")
                except Exception as e:
                    logger.error(f"Invalid image {dest_file}: {e}")
                    os.remove(dest_file)
            
            except requests.RequestException as e:
                logger.error(f"Failed to download {img_url}: {e}")
                continue

            # Respect iNaturalist API rate limits (max 100 requests per minute)
            if i % 10 == 0 and i > 0:
                import time
                time.sleep(1)
            
            # if downloaded >= len(observations):
            #     break

        logger.info(f"Downloaded {downloaded} images for {species_name}")
        return [os.path.join(species_dir, f) for f in os.listdir(species_dir) if os.path.isfile(os.path.join(species_dir, f))]

    except requests.RequestException as e:
        logger.error(f"API error for {species_name}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error for {species_name}: {e}")
        return []

if __name__ == "__main__":
    weeds = [
    "African Daisy",
    "African Foxtail",
    "African Olive",
    "Alligatorweed",
    "Bermuda Grass",
    "Bindweed",
    "Black Nightshade",
    "Broomrape",
    "Butterfly Pea",
    "Canada Thistle",
    "Carpet Grass",
    "Cattail",
    "Chamber Bitter",
    "Cogon Grass",
    "Common Chickweed",
    "Common Sowthistle",
    "Crabgrass",
    "Dandelion",
    "Dodder",
    "Foxtail",
    "Goat Weed",
    "Goosegrass",
    "Guinea Grass",
    "Japanese Knotweed",
    "Jungle Rice",
    "Kudzu",
    "Lambsquarters",
    "Lantana camara",
    "Mesquite",
    "Mexican Poppy",
    "Morning Glory",
    "Multiflora Rose",
    "Nutgrass",
    "Parramatta Grass",
    "Parthenium Weed",
    "Pigweed",
    "Plantain",
    "Poison Ivy",
    "Purslane",
    "Quackgrass",
    "Ribwort Plantain",
    "Sensitive Plant",
    "Siam Weed",
    "Spear Grass",
    "Tropical Kudzu",
    "Tridax Daisy",
    "Water Fern",
    "Water Hyacinth",
    "Water Lettuce",
    "White Water Lily",
    "Wireweed",
    "Witchweed"
]

    for weed in weeds:
        logger.info(f"Processing images for {weed}")
        images = download_inaturalist_images(weed, max_images=300)
        if images:
            logger.info(f"Downloaded images for {weed}: {len(images)} files \r\n")
        # Uncomment to use get_images as a fallback
        # if not images:
        #     logger.info(f"Falling back to simple_image_download for {keyword}")
        #     images = get_images(keyword, limit=10)
        #     logger.info(f"Downloaded images for {keyword}: {len(images)} files")