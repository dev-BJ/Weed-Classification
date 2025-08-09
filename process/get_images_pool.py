import os
import requests
import logging
from PIL import Image
from tqdm import tqdm
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from multiprocessing import Pool, Lock, Manager
import functools

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify Pillow import
try:
    from PIL import Image
    logger.info(f"Pillow version: {Image.__version__}")
except ImportError as e:
    logger.error(f"Failed to import Pillow: {e}")
    raise

# Base path relative to script
base_path = os.path.abspath(os.getcwd())
os.makedirs(f'{base_path}/dataset', exist_ok=True)
train_dir = os.path.join(base_path, 'dataset', 'train')

# Shared lock for rate limiting
request_lock = Lock()
request_timestamps = []

def rate_limited_request(url, session, timeout=10):
    """Make an HTTP request while respecting iNaturalist API rate limits (100/min)."""
    global request_timestamps
    with request_lock:
        now = time.time()
        # Remove timestamps older than 60 seconds
        request_timestamps = [t for t in request_timestamps if now - t < 60]
        if len(request_timestamps) >= 100:
            sleep_time = 60 - (now - request_timestamps[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        request_timestamps.append(now)
    
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.HTTPError as e:
        logger.error(f"HTTP error for {url}: {e}")
        return None

def download_image(obs, species_dir, session, downloaded_counter, max_images):
    """Download and validate a single image, used by multiprocessing pool."""
    if downloaded_counter.value >= max_images:
        return None

    if 'photos' not in obs or not obs['photos']:
        return None

    img_url = obs['photos'][0]['url'].replace('square', 'original')
    dest_file = os.path.join(species_dir, f"{obs['id']}.jpg")

    if os.path.exists(dest_file):
        return dest_file

    try:
        img_response = rate_limited_request(img_url, session)
        if not img_response:
            return None

        with open(dest_file, 'wb') as f:
            f.write(img_response.content)

        try:
            img = Image.open(dest_file)
            img.verify()
            img.close()  # Explicitly close to free memory
            img = Image.open(dest_file)  # Reopen for size check
            width, height = img.size
            img.close()
            if width < 100 or height < 100:
                logger.warning(f"Image {dest_file} too small, removing")
                os.remove(dest_file)
                return None
            downloaded_counter.value += 1  # Safe increment, no lock needed
            return dest_file
        except Image.UnidentifiedImageError as e:
            logger.error(f"Invalid image {dest_file}: {e}")
            os.remove(dest_file)
            return None

    except Exception as e:
        logger.error(f"Failed to download {img_url}: {e}")
        return None

def initialize_pool(lock, timestamps):
    """Initialize each process in the pool with shared lock and timestamps."""
    global request_lock, request_timestamps
    request_lock = lock
    request_timestamps = timestamps

def download_inaturalist_images(species_name, max_images=100):
    """
    Download images from iNaturalist API for a given species using multiprocessing.
    """
    try:
        os.makedirs(train_dir, exist_ok=True)
        species_dir = os.path.join(train_dir, species_name)
        os.makedirs(species_dir, exist_ok=True)
        
        # Set up requests session with retries
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429])
        session.mount("https://", HTTPAdapter(max_retries=retries))

        # Query iNaturalist for taxon ID
        url = f"https://api.inaturalist.org/v1/taxa?q={species_name}&only_id=true"
        response = rate_limited_request(url, session)
        if not response:
            return []

        data = response.json()
        if data['total_results'] == 0:
            logger.warning(f"No taxa found for {species_name}")
            return []

        taxon_id = data['results'][0]['id']
        logger.info(f"Found taxon ID {taxon_id} for {species_name}")

        # Query observations
        observations_url = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&per_page={max_images}&photos=true"
        response = rate_limited_request(observations_url, session)
        if not response:
            return []

        observations = response.json()['results']
        if not observations:
            logger.warning(f"No observations found for {species_name}")
            return []

        downloaded = len(os.listdir(species_dir))
        if downloaded >= max_images:
            logger.info(f"Skipping {species_name}, already has {downloaded} images")
            return [os.path.join(species_dir, f) for f in os.listdir(species_dir) if os.path.isfile(os.path.join(species_dir, f))]

        # Set up multiprocessing
        with Manager() as manager:
            downloaded_counter = manager.Value('i', downloaded)
            timestamps = manager.list()
            with Pool(processes=4, initializer=initialize_pool, initargs=(request_lock, timestamps)) as pool:
                download_func = functools.partial(
                    download_image,
                    species_dir=species_dir,
                    session=session,
                    downloaded_counter=downloaded_counter,
                    max_images=max_images
                )
                results = list(tqdm(
                    pool.imap(download_func, observations),
                    total=min(max_images, len(observations)),
                    desc=f"Downloading images for {species_name}"
                ))

        valid_files = [f for f in results if f and os.path.isfile(f)]
        logger.info(f"Downloaded {len(valid_files)} images for {species_name}")
        return valid_files

    except Exception as e:
        logger.error(f"Unexpected error for {species_name}: {e}")
        return []

if __name__ == "__main__":
    weeds = [
        "African Daisy", "African Foxtail", "African Olive", "Alligatorweed", "Bermuda Grass",
        "Bindweed", "Black Nightshade", "Broomrape", "Butterfly Pea", "Canada Thistle",
        "Carpet Grass", "Cattail", "Chamber Bitter", "Cogon Grass", "Common Chickweed",
        "Common Sowthistle", "Crabgrass", "Dandelion", "Dodder", "Foxtail", "Goat Weed",
        "Goosegrass", "Guinea Grass", "Japanese Knotweed", "Jungle Rice", "Kudzu",
        "Lambsquarters", "Lantana camara", "Mesquite", "Mexican Poppy", "Morning Glory",
        "Multiflora Rose", "Nutgrass", "Parramatta Grass", "Parthenium Weed", "Pigweed",
        "Plantain", "Poison Ivy", "Purslane", "Quackgrass", "Ribwort Plantain",
        "Sensitive Plant", "Siam Weed", "Spear Grass", "Tropical Kudzu", "Tridax Daisy",
        "Water Fern", "Water Hyacinth", "Water Lettuce", "White Water Lily", "Wireweed",
        "Witchweed"
    ]

    max_images = 200

    for weed in weeds:
        logger.info(f"Processing images for {weed}")
        images = download_inaturalist_images(weed, max_images=max_images)
        if max_images < len(images):
            weeds.append(weed)
        if images:
            logger.info(f"Downloaded images for {weed}: {len(images)} files \r\n")

    logger.info(f"Completed processing. Total weeds processed: {len(weeds)}")