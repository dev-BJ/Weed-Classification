from turtle import st
from simple_image_download import simple_image_download as simp
import os
import requests
import shutil

base_path = os.path.dirname(__file__)
download_dir = f'{base_path}/simple_images'
train_dir = f'{base_path}/train'
seperator = ','

def get_images(query, limit=10, dir='positive'):
    try:
        response = simp.simple_image_download()
        response.download(query, limit)

    # print(base_path)

    # Check if the directory exists before trying to rename it
        os.makedirs(train_dir, exist_ok=True)
        os.rename(f'{download_dir}/{query}/', f'{train_dir}/{dir}/{query}/')
        os.rmdir(download_dir)
    except Exception as e:
        print(f"Error downloading images: {str(e)}")
        return []
   
    # Assuming the images are saved in a folder named after the query
    return [f"{query}/{query}_{i}.jpg" for i in range(1, limit + 1)]

def download_inaturalist_images(species_name, max_images=100):
         try:
            url = f"https://api.inaturalist.org/v1/taxa?q={species_name}&only_id=true"
            response = requests.get(url).json()
            #  print(response)
            if response['total_results'] > 0:
                #   taxon_id = response['results'][0]['id']
                taxon_id = [response['results'][i]['id'] for i in range(len(response['results']))]
                #   print(len(taxon_id))
                taxon_id = seperator.join(str(x) for x in taxon_id)
                #   print(taxon_id)
                observations_url = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&per_page={max_images}&photos=true"
                data = requests.get(observations_url).json()
                #   print("Observations:", data['results'])
                # print("Observations count:", len(data['results']))
                
                if len(data['results']) > 0:
                    if os.path.exists(f"{train_dir}/positive/{species_name}"):
                        shutil.rmtree(f"{train_dir}/positive/{species_name}")
                    os.makedirs(f"{train_dir}/positive/{species_name}", exist_ok=True)

                    for i, obs in enumerate(data['results']):
                        # print(obs['photos'])
                        if obs['photos']:
                            img_url = obs['photos'][0]['url'].replace('square', 'original')
                            img_data = requests.get(img_url)
                            if img_data.ok:
                                with open(f"{train_dir}/positive/{species_name}/{species_name}_{i}.jpg", 'wb') as f:
                                    f.write(img_data.content)
                            else:
                                continue
                        print(f"{i+1}/{len(data['results'])} image(s) downloaded for {species_name}")

            else:
                print(f"No observations found for {species_name}")

         except Exception as e:
            print(f"Error Occured: {str(e)}")
            return []

if __name__ == "__main__":
    # Example usage
    keywords = [
        # 'pig weed',
        # 'witch weed',
        # 'black jack',
        # 'gallant soldier',
        # 'fat hen',
        # 'purple nutsedge',
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
    #     images = get_images("moon", limit=6)
    #     print("Downloaded images:", images)
        print(f"Looking up image(s) for {keyword}")
        download_inaturalist_images(keyword, 50)