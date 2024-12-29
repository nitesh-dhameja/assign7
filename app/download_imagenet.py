import os
import boto3
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import logging
import sys
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('download_progress.log')
    ]
)

class ImageNetDownloader:
    def __init__(self, hf_token, aws_access_key_id, aws_secret_access_key, bucket_name):
        """
        Initialize the downloader with necessary credentials
        """
        self.hf_token = hf_token
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        self.bucket_name = bucket_name
        self.temp_dir = "temp_download"
        os.makedirs(self.temp_dir, exist_ok=True)

    def download_and_upload(self):
        """
        Download ImageNet dataset and upload to S3
        """
        try:
            # Files to download from Hugging Face
            files = [
                "ILSVRC2012_img_train.tar",
                "ILSVRC2012_img_val.tar",
                "ILSVRC2012_devkit_t12.tar.gz"
            ]

            for file in files:
                logging.info(f"Starting download of {file}")
                
                # Download file from Hugging Face
                local_path = os.path.join(self.temp_dir, file)
                hf_hub_download(
                    repo_id="ILSVRC/imagenet-1k",
                    filename=f"data/{file}",
                    token=self.hf_token,
                    local_dir=self.temp_dir,
                    local_dir_use_symlinks=False
                )

                # Upload to S3
                logging.info(f"Uploading {file} to S3")
                self.upload_to_s3(local_path, file)

                # Remove local file after upload
                os.remove(local_path)
                logging.info(f"Removed local copy of {file}")

            logging.info("Download and upload completed successfully")

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise

        finally:
            # Cleanup
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logging.info("Cleaned up temporary directory")

    def upload_to_s3(self, file_path, s3_key):
        """
        Upload a file to S3 with progress tracking
        """
        file_size = os.path.getsize(file_path)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {s3_key}") as pbar:
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                f"imagenet/{s3_key}",
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )

if __name__ == "__main__":
    # Get credentials from environment variables
    hf_token = os.getenv("HF_TOKEN")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("S3_BUCKET_NAME")

    # Validate credentials
    if not all([hf_token, aws_access_key_id, aws_secret_access_key, bucket_name]):
        logging.error("Missing required environment variables")
        sys.exit(1)

    # Initialize and run downloader
    downloader = ImageNetDownloader(
        hf_token=hf_token,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        bucket_name=bucket_name
    )
    
    downloader.download_and_upload() 