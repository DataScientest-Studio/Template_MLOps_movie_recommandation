import requests
import os

from src.utils.logger import logger


def import_raw_data(
    dataset_dir: str, filenames: list[str], bucket_object_url: str
) -> str:
    """Import filenames from bucket_folder_url in raw_data_relative_path

    Args:
        dataset_dir (str): local dataset dir
        filenames (list[str]): files
        bucket_folder_url (str): bucket
    """

    # Create dir for raw data
    dir_raw = os.path.join(dataset_dir, "raw")
    os.makedirs(dir_raw, exist_ok=True)

    # Download all the files
    for filename in filenames:
        object_url = os.path.join(bucket_object_url, filename)
        output_file = os.path.join(dir_raw, filename)

        if os.path.exists(output_file):
            logger.debug(f"File '{output_file}'already present.")
        else:
            logger.debug(f"Downloading '{object_url}' in '{output_file}' ...")
            response = requests.get(object_url)
            if response.status_code == 200:
                content = response.text
                text_file = open(output_file, "wb")
                text_file.write(content.encode("utf-8"))
                text_file.close()
            else:
                logger.warning(
                    f"Error accessing the object {object_url}:", response.status_code
                )
    return dir_raw
