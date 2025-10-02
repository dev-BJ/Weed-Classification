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

def download_inaturalist_images(species_data, max_images=100):
    """
    Download images from iNaturalist API for a given species using multiprocessing.
    """
    try:
        os.makedirs(train_dir, exist_ok=True)
        species_dir = os.path.join(train_dir, species_data[0])
        os.makedirs(species_dir, exist_ok=True)
        
        # Set up requests session with retries
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429])
        session.mount("https://", HTTPAdapter(max_retries=retries))

        # Query iNaturalist for taxon ID
        url = f"https://api.inaturalist.org/v1/taxa?q={species_data[1]}&only_id=true"
        response = rate_limited_request(url, session)
        if not response:
            return []

        data = response.json()
        if data['total_results'] == 0:
            logger.warning(f"No taxa found for {species_data[0]}")
            return []

        taxon_id = data['results'][0]['id']
        logger.info(f"Found taxon ID {taxon_id} for {species_data[0]}")

        page = 1

        if max_images > 200:
            page = 2

        # Query observations
        observations_url = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&page={page}&per_page={max_images}&photos=true"
        response = rate_limited_request(observations_url, session)
        if not response:
            return []

        observations = response.json()['results']
        if not observations:
            logger.warning(f"No observations found for {species_data[0]}")
            return []

        downloaded = len(os.listdir(species_dir))
        if downloaded >= max_images:
            logger.info(f"Skipping {species_data[0]}, already has {downloaded} images")
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
                    desc=f"Downloading images for {species_data[0]}"
                ))

        valid_files = [f for f in results if f and os.path.isfile(f)]
        logger.info(f"Downloaded {len(valid_files)} images for {species_data[0]}")
        return valid_files

    except Exception as e:
        logger.error(f"Unexpected error for {species_data[0]}: {e}")
        return []

if __name__ == "__main__":
    # Define the common names list
    all_nigeria_plants = [
        "African Breadfruit",
        "African Walnut",
        "African Yam Bean",
        "Acha",
        "Amaranth",
        "Avocado",
        "Bambara Groundnut",
        "Banana",
        "Plaintain (Banana)",
        "African Baobab",
        "Bitter Kola",
        "Calabash",
        "Carrot",
        "Cashew",
        "Castor Bean",
        "Chili Pepper",
        "Cinnamon",
        "Clove",
        "Coconut",
        "Cocoyam",
        "Cotton",
        "Cowpea",
        "Cucumber",
        "Egusi Melon",
        "Flaxseed",
        "Fonio",
        "Garden Egg",
        "Ginger",
        "Groundnut",
        "Guava",
        "Gum Arabic",
        "Hibiscus",
        "Hog Plum",
        "Kenaf",
        "Kola Nut",
        "Lettuce",
        "Locust Bean",
        "Mango",
        "Mangosteen",
        "Melon",
        "Mushroom",
        "Nutmeg",
        "Ogbono",
        "Oil Palm",
        "Okra",
        "Onion",
        "Orange",
        "Pawpaw",
        "Pearl Millet",
        "Pepper",
        "Peppercorn",
        "Pigeon Pea",
        "Pineapple",
        "Potatoes",
        "Rice",
        "Sesame Seed",
        "Shea Butter Nut",
        "Sorrel",
        "Soursop",
        "Soybean",
        "Spinach",
        "Squash",
        "Star Apple",
        "Sugarcane",
        "Sweet Potato",
        "Tamarind",
        "Tangerine",
        "Taro",
        "Tea",
        "Tobacco",
        "Tomato",
        "Watermelon",
        "Wheat",
        # Old list (weeds and common plants) commented out for later use
        "African Daisy",
        "African Foxtail",
        "African Olive",
        "African Spinach",
        "African White Mahogany",
        "Alligatorweed",
        "Bermuda Grass",
        "Bindweed",
        "Bitterleaf",
        "Black Nightshade",
        "Broomrape",
        "Butterfly Pea",
        "Canada Thistle",
        "Carpet Grass",
        "Cassava",
        "Cattail",
        "Celosia",
        "Chamber Bitter",
        "Climbing Ivy",
        "Coconut",
        "Cogon Grass",
        "Cola Nut",
        "Common Bean",
        "Common Chickweed",
        "Common Sowthistle",
        "Common Wild Fig",
        "Corn",
        "Cordyline",
        "Croton",
        "Crown of Thorns",
        "Crabgrass",
        "Dandelion",
        "Date Palm",
        "Dieffenbachia",
        "Dodder",
        "Dumb Cane",
        "Earleaf Acacia",
        "Foxtail",
        "Goat Weed",
        "Goosegrass",
        "Guinea Grass",
        "Japanese Knotweed",
        "Jute Leaf",
        "Jungle Rice",
        "Kudzu",
        "Lambsquarters",
        "Lantana camara",
        "Mesquite",
        "Mexican Poppy",
        "Morning Glory",
        "Moringa",
        "Mother In-Law’s Tongue",
        "Multiflora Rose",
        "Mussaenda",
        "Neem Tree",
        "Nganda Coffee",
        "Nutgrass",
        "Orchids",
        "Parramatta Grass",
        "Parthenium Weed",
        "Pigweed",
        "Plantain",
        "Poison Ivy",
        "Purslane",
        "Purple Heart",
        "Quackgrass",
        "Ribwort Plantain",
        "Scent Leaf",
        "Sensitive Plant",
        "Siam Weed",
        "Spear Grass",
        "Spider Plant",
        "Sunflowers",
        "Tropical Kudzu",
        "Tridax Daisy",
        "Ube",
        "Water Fern",
        "Water Hyacinth",
        "Water Lettuce",
        "Waterleaf",
        "White Water Lily",
        "Wireweed",
        "Witchweed",
        "Yam",
        "Yellow Bush"
    ]

    # Define the botanical names list
    botanical_names = [
        "Treculia africana",  # African Breadfruit
        "Tetracarpidium conophorum",  # African Walnut
        "Sphenostylis stenocarpa",  # African Yam Bean
        "Digitaria iburua",  # Acha
        "Amaranthus spp.",  # Amaranth
        "Persea americana",  # Avocado
        "Vigna subterranea",  # Bambara Groundnut
        "Musa spp",  # Banana
        "Musa × paradisiaca",   #Plaintain(Banana)
        "Adansonia digitata",  # African Baobab
        "Garcinia kola",  # Bitter Kola
        "Lagenaria siceraria",  # Calabash
        "Daucus carota",  # Carrot
        "Anacardium occidentale",  # Cashew
        "Ricinus communis",  # Castor Bean
        "Capsicum frutescens",  # Chili Pepper
        "Cinnamomum verum",  # Cinnamon
        "Syzygium aromaticum",  # Clove
        "Cocos nucifera", # Coconut
        "Colocasia esculenta",  # Cocoyam Xanthosoma spp.
        "Gossypium hirsutum",  # Cotton
        "Vigna unguiculata",  # Cowpea
        "Cucumis sativus",  # Cucumber
        "Citrullus lanatus var. colocynthis",  # Egusi Melon
        "Linum usitatissimum",  # Flaxseed
        "Digitaria exilis",  # Fonio
        "Solanum aethiopicum",  # Garden Egg
        "Zingiber officinale",  # Ginger
        "Arachis hypogaea",  # Groundnut
        "Psidium guajava",  # Guava
        "Senegalia senegal",  # Gum Arabic
        "Hibiscus sabdariffa",  # Hibiscus
        "Spondias mombin",  # Hog Plum
        "Hibiscus cannabinus",  # Kenaf
        "Cola acuminata",  # Kola Nut
        "Lactuca sativa",  # Lettuce
        "Parkia biglobosa",  # Locust Bean
        "Mangifera indica",  # Mango
        "Garcinia mangostana",  # Mangosteen
        "Cucumis melo",  # Melon
        "Agaricus bisporus",  # Mushroom
        "Myristica fragrans",  # Nutmeg
        "Irvingia gabonensis",  # Ogbono
        "Elaeis guineensis",  # Oil Palm
        "Abelmoschus esculentus",  # Okra
        "Allium cepa",  # Onion
        "Citrus sinensis",  # Orange
        "Carica papaya",  # Pawpaw
        "Pennisetum glaucum",  # Pearl Millet
        "Capsicum annuum",  # Pepper
        "Piper nigrum",  # Peppercorn
        "Cajanus cajan",  # Pigeon Pea
        "Ananas comosus",  # Pineapple
        "Solanum tuberosum",  # Potatoes
        "Oryza sativa",  # Rice
        "Sesamum indicum",  # Sesame Seed
        "Vitellaria paradoxa",  # Shea Butter Nut
        "Rumex acetosa",  # Sorrel
        "Annona muricata",  # Soursop
        "Glycine max",  # Soybean
        "Spinacia oleracea",  # Spinach
        "Cucurbita spp.",  # Squash
        "Chrysophyllum albidum",  # Star Apple
        "Saccharum officinarum",  # Sugarcane
        "Ipomoea batatas",  # Sweet Potato
        "Tamarindus indica",  # Tamarind
        "Citrus reticulata",  # Tangerine
        "Colocasia esculenta",  # Taro
        "Camellia sinensis",  # Tea
        "Nicotiana tabacum",  # Tobacco
        "Solanum lycopersicum",  # Tomato
        "Citrullus lanatus",  # Watermelon
        "Triticum aestivum",  # Wheat
        # Old list (weeds and common plants) commented out for later use
        "Senecio pterophorus",  # African Daisy
        "Cenchrus biflorus",  # African Foxtail
        "Olea europaea subsp. cuspidata",  # African Olive
        "Amaranthus cruentus",  # African Spinach
        "Turraeanthus africana",  # African White Mahogany
        "Alternanthera philoxeroides",  # Alligatorweed
        "Cynodon dactylon",  # Bermuda Grass
        "Convolvulus arvensis",  # Bindweed
        "Vernonia amygdalina",  # Bitterleaf
        "Solanum nigrum",  # Black Nightshade
        "Orobanche spp.",  # Broomrape
        "Centrosema pubescens",  # Butterfly Pea
        "Cirsium arvense",  # Canada Thistle
        "Axonopus compressus",  # Carpet Grass
        "Manihot esculenta",  # Cassava
        "Typha spp.",  # Cattail
        "Celosia argentea",  # Celosia
        "Phyllanthus urinaria",  # Chamber Bitter
        "Hedera helix",  # Climbing Ivy
        "Cocos nucifera",  # Coconut
        "Imperata cylindrica",  # Cogon Grass
        "Cola acuminata",  # Cola Nut
        "Phaseolus vulgaris",  # Common Bean
        "Stellaria media",  # Common Chickweed
        "Sonchus oleraceus",  # Common Sowthistle
        "Ficus thonningii",  # Common Wild Fig
        "Zea mays",  # Corn
        "Cordyline fruticosa",  # Cordyline
        "Croton hirtus",  # Croton
        "Euphorbia milii",  # Crown of Thorns
        "Digitaria spp.",  # Crabgrass
        "Taraxacum officinale",  # Dandelion
        "Phoenix dactylifera",  # Date Palm
        "Dieffenbachia spp.",  # Dieffenbachia
        "Cuscuta spp.",  # Dodder
        "Dieffenbachia seguine",  # Dumb Cane
        "Acacia auriculiformis",  # Earleaf Acacia
        "Setaria spp.",  # Foxtail
        "Ageratum conyzoides",  # Goat Weed
        "Eleusine indica",  # Goosegrass
        "Megathyrsus maximus",  # Guinea Grass
        "Reynoutria japonica",  # Japanese Knotweed
        "Corchorus olitorius",  # Jute Leaf
        "Echinochloa colona",  # Jungle Rice
        "Pueraria montana",  # Kudzu
        "Chenopodium album",  # Lambsquarters
        "Lantana camara",  # Lantana camara
        "Prosopis juliflora",  # Mesquite
        "Argemone mexicana",  # Mexican Poppy
        "Ipomoea spp.",  # Morning Glory
        "Moringa oleifera",  # Moringa
        "Sansevieria trifasciata",  # Mother In-Law’s Tongue
        "Rosa multiflora",  # Multiflora Rose
        "Mussaenda frondosa",  # Mussaenda
        "Azadirachta indica",  # Neem Tree
        "Coffea canephora",  # Nganda Coffee
        "Cyperus esculentus",  # Nutgrass
        "Orchidaceae spp.",  # Orchids
        "Sporobolus africanus",  # Parramatta Grass
        "Parthenium hysterophorus",  # Parthenium Weed
        "Amaranthus spp.",  # Pigweed
        "Plantago major",  # Plantain
        "Toxicodendron radicans",  # Poison Ivy
        "Portulaca oleracea",  # Purslane
        "Tradescantia pallida",  # Purple Heart
        "Elymus repens",  # Quackgrass
        "Plantago lanceolata",  # Ribwort Plantain
        "Ocimum gratissimum",  # Scent Leaf
        "Mimosa pudica",  # Sensitive Plant
        "Chromolaena odorata",  # Siam Weed
        "Imperata cylindrica",  # Spear Grass
        "Chlorophytum comosum",  # Spider Plant
        "Helianthus annuus",  # Sunflowers
        "Pueraria phaseoloides",  # Tropical Kudzu
        "Tridax procumbens",  # Tridax Daisy
        "Dacryodes edulis",  # Ube
        "Salvinia auriculata",  # Water Fern
        "Eichhornia crassipes",  # Water Hyacinth
        "Pistia stratiotes",  # Water Lettuce
        "Talinum fruticosum",  # Waterleaf
        "Nymphaea lotus",  # White Water Lily
        "Sida acuta",  # Wireweed
        "Striga hermonthica",  # Witchweed
        "Dioscorea alata",  # Yam
        "Duranta erecta"  # Yellow Bush
    ]

    # Zip the two lists into a list of tuples
    zipped_plants = list(zip(all_nigeria_plants, botanical_names))
    print("Plant Size:", len(zipped_plants))

    # Optionally, print the zipped list to verify
    # for common_name, botanical_name in zipped_plants:
    #     print(f"{common_name}: {botanical_name}")

    for weed in zipped_plants:
        max_images = 200
        if weed[0] == "Cattail":  # Example of skipping specific plants
            max_images = 300
        logger.info(f"Processing images for {weed[0]}")
        if len(os.listdir(os.path.join(train_dir, weed[0]))) >= max_images:
            logger.info(f"Skipping {weed[0]}, directory already exists. \r\n")
            continue
        images = download_inaturalist_images(weed, max_images=max_images)
        if max_images < len(images):
            # zipped_plants.append(weed)
            pass
        if images:
            logger.info(f"Downloaded images for {weed[0]}: {len(images)} files \r\n")

    logger.info(f"Completed processing. Total weeds processed: {len(zipped_plants)}")
