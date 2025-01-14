import os
import torch
from torch.utils.data import Dataset
import boto3
from PIL import Image
import io
import logging
import time
import pyarrow.parquet as pq
import pyarrow as pa
from botocore.exceptions import ClientError
from tqdm import tqdm

class S3ImageNetDataset(Dataset):
    def __init__(self, bucket_name, transform=None, is_train=True, max_retries=3, retry_delay=1):
        """
        Dataset for loading ImageNet from S3 with streaming support
        """
        self.bucket_name = bucket_name
        self.transform = transform
        self.is_train = is_train
        self.s3_client = boto3.client('s3')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Determine the directory (train or validation)
        self.data_dir = 'imagenet/train' if is_train else 'imagenet/validation'
        logging.info(f"Loading dataset from s3://{bucket_name}/{self.data_dir}")
        
        # List all available directories
        self.discover_structure()
        
    def verify_arrow_file(self, file_key):
        """
        Verify the integrity of an Arrow file
        """
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            return True
        except ClientError as e:
            logging.error(f"Error verifying Arrow file {file_key}: {str(e)}")
            return False

    def get_object_with_retry(self, file_key, start_byte=None, end_byte=None):
        """
        Get S3 object with retry logic
        """
        for attempt in range(self.max_retries):
            try:
                range_header = {}
                if start_byte is not None and end_byte is not None:
                    range_header['Range'] = f'bytes={start_byte}-{end_byte}'
                
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=file_key,
                    **range_header
                )
                return response
            except ClientError as e:
                if attempt == self.max_retries - 1:
                    raise
                logging.warning(f"Retry {attempt + 1}/{self.max_retries} for {file_key}")
                time.sleep(self.retry_delay * (attempt + 1))
        
        raise Exception(f"Failed to get object after {self.max_retries} retries")

    def discover_structure(self):
        """
        Discover the dataset structure in S3 without loading data into memory
        """
        try:
            # List contents of the data directory
            paginator = self.s3_client.get_paginator('list_objects_v2')
            prefix = f"{self.data_dir}/"
            
            # Get all files (both .tar.gz and .arrow files)
            self.data_files = []
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith(('.tar.gz', '.arrow')):
                        self.data_files.append(obj['Key'])
            
            logging.info(f"Found {len(self.data_files)} data files")
            
            # Load metadata and sample data from each file
            self.file_sizes = []  # Store size of each file
            self.cumulative_sizes = [0]  # For indexing into correct file
            total_samples = 0
            all_labels = set()
            
            for data_file in tqdm(self.data_files, desc="Loading dataset structure"):
                try:
                    # Get file metadata using head_object
                    head = self.s3_client.head_object(
                        Bucket=self.bucket_name,
                        Key=data_file
                    )
                    file_size = head['ContentLength']
                    
                    # Get the object
                    response = self.s3_client.get_object(
                        Bucket=self.bucket_name,
                        Key=data_file
                    )
                    
                    if data_file.endswith('.arrow'):
                        # Process Arrow file
                        stream = pa.ipc.open_stream(response['Body'])
                        batch = next(stream)
                        if batch is not None:
                            if 'label' in batch.schema.names:
                                labels = batch['label'].to_numpy()
                                all_labels.update(labels)
                            num_records = len(batch)
                    else:
                        # Process tar.gz file
                        # Each tar.gz file contains images for one class
                        class_id = int(data_file.split('/')[-1].split('_')[1].split('.')[0])
                        all_labels.add(class_id)
                        # Estimate number of images (assuming average image size)
                        num_records = file_size // (150 * 1024)  # Assuming average image size of 150KB
                    
                    # Store file information
                    self.file_sizes.append(num_records)
                    total_samples += num_records
                    self.cumulative_sizes.append(total_samples)
                    
                    logging.info(f"Processed {data_file}: found {num_records} records")
                    
                except Exception as e:
                    logging.error(f"Error processing file {data_file}: {str(e)}")
                    continue
            
            if not all_labels:
                raise ValueError("No valid labels found in the dataset")
            
            if not self.file_sizes:
                raise ValueError("No valid data files could be processed")
            
            # Create label mapping
            self.classes = sorted(all_labels)
            self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
            logging.info(f"Found {total_samples} samples with {len(self.class_to_idx)} classes")
            
        except Exception as e:
            logging.error(f"Error discovering dataset structure: {str(e)}")
            raise

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # Find which file contains this index
        file_idx = next(i for i, size in enumerate(self.cumulative_sizes[1:], 1) 
                       if idx < size) - 1
        local_idx = idx - self.cumulative_sizes[file_idx]
        data_file = self.data_files[file_idx]
        
        for attempt in range(self.max_retries):
            try:
                # Get the file with retry logic
                response = self.get_object_with_retry(data_file)
                
                if data_file.endswith('.arrow'):
                    # Process Arrow file
                    stream = pa.ipc.open_stream(response['Body'])
                    current_idx = 0
                    for batch in stream:
                        if batch is not None:
                            batch_size = len(batch)
                            if current_idx + batch_size > local_idx:
                                record_idx = local_idx - current_idx
                                image_data = batch['image'][record_idx]['bytes'].as_buffer()
                                label = batch['label'][record_idx].as_py()
                                break
                            current_idx += batch_size
                else:
                    # Process tar.gz file
                    import tarfile
                    import io
                    tar = tarfile.open(fileobj=io.BytesIO(response['Body'].read()))
                    image_files = [f for f in tar.getmembers() if f.name.endswith(('.JPEG', '.jpg', '.jpeg'))]
                    if local_idx >= len(image_files):
                        raise ValueError(f"Index {local_idx} out of range for {data_file}")
                    image_file = image_files[local_idx]
                    image_data = tar.extractfile(image_file).read()
                    # Extract class ID from file path
                    label = int(data_file.split('/')[-1].split('_')[1].split('.')[0])
                
                # Convert to PIL Image
                try:
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                except Exception as e:
                    raise ValueError(f"Failed to decode image: {str(e)}")
                
                # Apply transforms
                if self.transform:
                    try:
                        image = self.transform(image)
                    except Exception as e:
                        raise ValueError(f"Transform failed: {str(e)}")
                
                return image, self.class_to_idx[label]
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logging.error(f"Error loading record at index {idx} from file {data_file}: {str(e)}")
                    raise
                logging.warning(f"Retry {attempt + 1}/{self.max_retries} for index {idx}")
                time.sleep(self.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"Failed to load record after {self.max_retries} retries") 