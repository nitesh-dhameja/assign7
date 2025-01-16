import os
import torch
import json
from transformers import AutoConfig
from huggingface_hub import HfApi
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_checkpoint(checkpoint_path):
    """Load model checkpoint safely"""
    try:
        # Load checkpoint with weights_only=True for security
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        logging.info(f"Successfully loaded checkpoint")
        return state_dict
    except Exception as e:
        logging.error(f"Error loading checkpoint: {str(e)}")
        raise

def upload_to_huggingface(checkpoint_path, repo_id):
    """Upload model to HuggingFace Hub"""
    # Load checkpoint
    state_dict = load_model_checkpoint(checkpoint_path)
    
    # Create config
    config = {
        "model_type": "resnet",  # Specify model type
        "architectures": ["ResNet50"],
        "hidden_size": 2048,
        "num_hidden_layers": 50,
        "num_attention_heads": 1,
        "num_channels": 3,
        "image_size": 224,
        "patch_size": 16,
        "num_labels": 1000,
        "id2label": {str(i): f"LABEL_{i}" for i in range(1000)},
        "label2id": {f"LABEL_{i}": str(i) for i in range(1000)}
    }
    
    # Save config and model
    os.makedirs("./temp_model", exist_ok=True)
    
    # Save config
    with open("./temp_model/config.json", "w") as f:
        json.dump(config, f)
    
    # Save model state
    torch.save(state_dict, "./temp_model/pytorch_model.bin")
    
    # Create model card
    model_card = """---
language: en
tags:
- image-classification
- pytorch
- resnet
- imagenet
datasets:
- imagenet-1k
metrics:
- accuracy
---

# ResNet50 ImageNet Classifier

This model is a ResNet50 architecture trained on the ImageNet dataset for image classification.

## Model Description

- **Model Type:** ResNet50
- **Task:** Image Classification
- **Training Data:** ImageNet (ILSVRC2012)
- **Number of Parameters:** ~23M
- **Input:** RGB images of size 224x224

## Usage

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

# Load model and processor
model = AutoModelForImageClassification.from_pretrained("jatingocodeo/ImageNet")
processor = AutoImageProcessor.from_pretrained("jatingocodeo/ImageNet")

# Prepare image
image = Image.open("path/to/image.jpg")
inputs = processor(image, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
```

## Training

The model was trained on the ImageNet dataset with the following configuration:
- Optimizer: AdamW
- Learning Rate: 0.003 with cosine scheduling
- Batch Size: 256
- Data Augmentation: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomAffine, RandomPerspective
"""
    
    # Save model card
    with open("./temp_model/README.md", "w") as f:
        f.write(model_card)
    
    # Upload to Hub
    logging.info(f"Uploading model to {repo_id}...")
    api = HfApi()
    api.upload_folder(
        folder_path="./temp_model",
        repo_id=repo_id,
        repo_type="model"
    )
    
    logging.info(f"Model uploaded successfully to {repo_id}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    repo_id = os.getenv("HF_REPO_ID", "jatingocodeo/ImageNet")
    
    upload_to_huggingface(checkpoint_path, repo_id) 