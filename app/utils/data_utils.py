import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image

def setup_imagenet(data_dir='path/to/imagenet'):
    """
    Verifies and returns paths to existing ImageNet dataset
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"ImageNet dataset not found in {data_dir}. "
            f"Expected train data in {train_dir} and validation data in {val_dir}"
        )

    logging.info(f"Using ImageNet dataset from {data_dir}")
    return str(train_dir), str(val_dir)

def create_dummy_image(path, size=(224, 224)):
    """
    Creates a dummy RGB image for testing
    """
    # Create a random RGB image
    img_array = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(path)

def download_sample_data(data_dir='data/imagenet-sample', num_classes=10, samples_per_class=10):
    """
    Creates a small subset of dummy data for testing
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    # Create directories if they don't exist
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy data for testing
    for split_dir in [train_dir, val_dir]:
        for class_id in range(num_classes):
            class_dir = split_dir / f'class_{class_id}'
            class_dir.mkdir(exist_ok=True)
            
            # Create dummy image files with actual content
            for i in range(samples_per_class):
                dummy_img = class_dir / f'img_{i}.jpg'
                if not dummy_img.exists():
                    create_dummy_image(dummy_img)

    logging.info(f"Created sample dataset at {data_dir}")
    return str(train_dir), str(val_dir)