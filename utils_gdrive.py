import os
import gdown

def download_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Downloading {output_path} from Google Drive...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"File {output_path} already exists. Skipping download.")

def download_multiple_files(file_map):
    """
    file_map: dict of {local_path: google_drive_file_id}
    """
    for local_path, file_id in file_map.items():
        download_from_gdrive(file_id, local_path)