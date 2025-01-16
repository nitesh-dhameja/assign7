import os
import torch
import torch.nn as nn
from huggingface_hub import HfApi, login
import torchvision.models as models
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_checkpoint(checkpoint_path):
    """
    Load the trained model from checkpoint
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize model
    model = models.resnet50(weights=None)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint

def upload_to_huggingface(model, checkpoint, repo_id):
    """
    Upload model to Hugging Face Hub
    """
    try:
        # Save model config and metadata
        config = {
            'architecture': 'ResNet50',
            'num_classes': 1000,
            'best_accuracy': checkpoint['best_acc'],
            'training_epochs': checkpoint['epoch'] + 1,
            'framework': 'PyTorch',
            'task': 'image-classification'
        }
        
        # Create model card content
        model_card = f"""
# ResNet50 ImageNet Classifier

## Model Description
This model is a ResNet50 architecture trained on ImageNet dataset.

## Performance
- Best Validation Accuracy: {checkpoint['best_acc']:.2f}%
- Training Epochs: {checkpoint['epoch'] + 1}

## Training Details
- Framework: PyTorch
- Task: Image Classification
- Dataset: ImageNet
- Input Size: 224x224
- Number of Classes: 1000

## Usage
```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# Load model and processor
model = AutoModelForImageClassification.from_pretrained("{repo_id}")
processor = AutoImageProcessor.from_pretrained("{repo_id}")

# Prepare image
image = Image.open("path/to/image.jpg")
inputs = processor(image, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
```
"""
        
        # Save model in the format expected by Hugging Face
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, 'model.pth')
        
        # Initialize Hugging Face API
        api = HfApi()
        
        # Upload files
        logging.info(f"Uploading files to {repo_id}...")
        api.upload_file(
            path_or_fileobj='model.pth',
            path_in_repo='pytorch_model.bin',
            repo_id=repo_id,
            repo_type='model'
        )
        
        # Create/update model card
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo='README.md',
            repo_id=repo_id,
            repo_type='model'
        )
        
        logging.info(f"Successfully uploaded model to {repo_id}")
        
    except Exception as e:
        logging.error(f"Error uploading to Hugging Face: {str(e)}")
        raise

def main():
    # Load environment variables
    load_dotenv()
    
    # Get Hugging Face token
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
    
    # Login to Hugging Face
    login(token=hf_token)
    
    # Load model checkpoint
    checkpoint_path = 'checkpoints/best_model.pth'
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    model, checkpoint = load_model_checkpoint(checkpoint_path)
    
    # Your Hugging Face repository ID (username/repo-name)
    repo_id = os.getenv('HF_REPO_ID')
    if not repo_id:
        raise ValueError("HF_REPO_ID not found in environment variables")
    
    # Upload to Hugging Face
    upload_to_huggingface(model, checkpoint, repo_id)

if __name__ == '__main__':
    main() 