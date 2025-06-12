import requests
import os
import logging

def import_raw_data(raw_data_relative_path, filenames, bucket_folder_url):
    '''Import filenames from bucket_folder_url into raw_data_relative_path.'''
    # Créer le dossier si nécessaire
    os.makedirs(raw_data_relative_path, exist_ok=True)

    # Télécharger tous les fichiers
    for filename in filenames:
        input_file = os.path.join(bucket_folder_url, filename)
        output_file = os.path.join(raw_data_relative_path, filename)

        print(f"[INFO] Downloading {input_file} as {os.path.basename(output_file)}")
        try:
            response = requests.get(input_file)
            if response.status_code == 200:
                with open(output_file, "wb") as f:
                    f.write(response.content)  # Écrit le contenu réel du fichier
                print(f"[INFO] File {output_file} created successfully.")
            else:
                print(f"[ERROR] Failed to download {input_file}. Status code: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] An error occurred while downloading {input_file}: {e}")


def main(raw_data_relative_path="./data/raw",
         filenames=["genome-scores.csv", "genome-tags.csv", "links.csv",
                   "movies.csv", "ratings.csv", "README.txt", "tags.csv"],
         bucket_folder_url="https://mlops-project-db.s3.eu-west-1.amazonaws.com/movie_recommandation/"):
    """Download data from AWS S3 to ./data/raw."""
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logger = logging.getLogger(__name__)
    logger.info('Raw data set created successfully.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()