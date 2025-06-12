import pytest
import os
from unittest.mock import patch
from src.data.import_raw_data import import_raw_data, main

def test_import_raw_data(tmp_path, requests_mock):
    bucket_folder_url = "https://mock-bucket.s3.amazonaws.com/"
    filenames = ["file1.txt", "file2.txt"]
    raw_data_relative_path = tmp_path / "data" / "raw"
    os.makedirs(raw_data_relative_path)

    for filename in filenames:
        file_url = os.path.join(bucket_folder_url, filename)
        requests_mock.get(file_url, text=f"Content of {filename}")

    import_raw_data(raw_data_relative_path=str(raw_data_relative_path), 
                    filenames=filenames, 
                    bucket_folder_url=bucket_folder_url)

    for filename in filenames:
        file_path = raw_data_relative_path / filename
        assert file_path.exists()
        with open(file_path, "r") as f:
            content = f.read()
            assert content == f"Content of {filename}"

@patch("builtins.input", side_effect=["y"])
def test_import_raw_data_file_exists(mock_input, tmp_path, requests_mock):
    bucket_folder_url = "https://mock-bucket.s3.amazonaws.com/"
    filenames = ["file1.txt"]
    raw_data_relative_path = tmp_path / "data" / "raw"
    os.makedirs(raw_data_relative_path)

    existing_file = raw_data_relative_path / filenames[0]
    with open(existing_file, "w") as f:
        f.write("Existing content")

    file_url = os.path.join(bucket_folder_url, filenames[0])
    requests_mock.get(file_url, text="New content")

    import_raw_data(raw_data_relative_path=str(raw_data_relative_path), 
                    filenames=filenames, 
                    bucket_folder_url=bucket_folder_url)

    with open(existing_file, "r") as f:
        content = f.read()
        assert content == "New content"