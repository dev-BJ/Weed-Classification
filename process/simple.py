import shutil
import os
import logging
from simple_image_download import simple_image_download as simp


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) 
def download_images(query, limit=200, dir_name='positive'):
    """
    Download images using simple_image_download and save them into train_dir/dir_name/query.
    """
   
    try:
        response = simp.simple_image_download()
        response.download(query, limit)
            

        # # Create train directory if it doesn't exist
        # target_dir = train_dir / dir_name / query
        # target_dir.mkdir(parents=True, exist_ok=True)

        # # Move images to target directory
        # source_dir = download_dir / query
        # if source_dir.exists():
        #     for i in range(1, limit + 1):
        #         src_file = source_dir / f"{query}_{i}.jpg"
        #         dest_file = target_dir / f"{query}_{i}.jpg"
        #         if src_file.exists() and not dest_file.exists():
        #             shutil.move(str(src_file), str(dest_file))
        #     shutil.rmtree(str(source_dir), ignore_errors=True)
        
        # return [str(target_dir / f"{query}_{i}.jpg") for i in range(1, limit + 1) if (target_dir / f"{query}_{i}.jpg").exists()]
    
    except Exception as e:
        logger.error(f"Error downloading images for {query}: {e}")
        return []

if __name__ == "__main__":
    plants = [
        ("Ube","Dacryodes edulis")
    ]
    for plant in plants:
        print(f"Downloading images for {plant[0]}...")
        download_images(plant[1], limit=200, dir_name='positive')

    # Example usage
    # download_images("Dacryodes edulis", limit=200, dir_name='positive')