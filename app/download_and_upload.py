import os
from dotenv import load_dotenv
import boto3
from huggingface_hub import hf_hub_download, login, list_repo_files, HfApi
import logging
from tqdm import tqdm
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_file_url(repo_id, filename, token):
    """
    Get the direct download URL for a file
    """
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.head(url, headers=headers, allow_redirects=True)
    return r.url

def stream_to_s3(url, bucket_name, s3_key, headers=None, chunk_size=8192):
    """
    Stream file from URL directly to S3 without storing locally
    """
    s3_client = boto3.client('s3')
    
    # Get file size for progress bar
    response = requests.head(url, headers=headers)
    total_size = int(response.headers.get('content-length', 0))
    
    # Set up progress bar
    progress = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    # Create multipart upload
    mpu = s3_client.create_multipart_upload(Bucket=bucket_name, Key=s3_key)
    
    try:
        parts = []
        part_number = 1
        
        # Stream the file
        response = requests.get(url, headers=headers, stream=True)
        buffer = bytearray()
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                buffer.extend(chunk)
                progress.update(len(chunk))
                
                # If buffer size is greater than 5MB, upload it as a part
                if len(buffer) >= 5 * 1024 * 1024:
                    part = s3_client.upload_part(
                        Bucket=bucket_name,
                        Key=s3_key,
                        PartNumber=part_number,
                        UploadId=mpu['UploadId'],
                        Body=buffer
                    )
                    parts.append({
                        'PartNumber': part_number,
                        'ETag': part['ETag']
                    })
                    part_number += 1
                    buffer = bytearray()
        
        # Upload any remaining data
        if buffer:
            part = s3_client.upload_part(
                Bucket=bucket_name,
                Key=s3_key,
                PartNumber=part_number,
                UploadId=mpu['UploadId'],
                Body=buffer
            )
            parts.append({
                'PartNumber': part_number,
                'ETag': part['ETag']
            })
        
        # Complete multipart upload
        s3_client.complete_multipart_upload(
            Bucket=bucket_name,
            Key=s3_key,
            UploadId=mpu['UploadId'],
            MultipartUpload={'Parts': parts}
        )
        
        progress.close()
        logging.info(f"Successfully uploaded to s3://{bucket_name}/{s3_key}")
        
    except Exception as e:
        progress.close()
        s3_client.abort_multipart_upload(
            Bucket=bucket_name,
            Key=s3_key,
            UploadId=mpu['UploadId']
        )
        raise e

def download_and_stream_to_s3():
    """
    Download ImageNet dataset from Hugging Face and stream directly to S3
    """
    logging.info("Starting download and streaming to S3...")
    
    # Get environment variables
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    bucket_name = os.getenv("S3_BUCKET_NAME")
    
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable not set")
    
    # Login to Hugging Face
    login(token=hf_token)
    
    # Define the files we want to download (training and validation data)
    files_to_download = [
        "data/train_images_0.tar.gz",
        "data/train_images_1.tar.gz",
        "data/train_images_2.tar.gz",
        "data/train_images_3.tar.gz",
        "data/train_images_4.tar.gz",
        "data/val_images.tar.gz",
        "data/test_images.tar.gz"
    ]
    
    # Headers for requests
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Process each file
    for file_path in files_to_download:
        s3_key = f"imagenet/{os.path.basename(file_path)}"
        logging.info(f"Processing {file_path}...")
        
        try:
            # Get direct download URL
            url = get_file_url("ILSVRC/imagenet-1k", file_path, hf_token)
            
            # Stream to S3
            stream_to_s3(url, bucket_name, s3_key, headers=headers)
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            raise

def main():
    # Load environment variables
    load_dotenv()
    
    try:
        # Download and stream to S3
        download_and_stream_to_s3()
        
        logging.info("Successfully completed streaming process")
    except Exception as e:
        logging.error(f"Error in process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 