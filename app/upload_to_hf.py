import os
import torch
import json
from transformers import AutoConfig
from huggingface_hub import HfApi

def load_model_checkpoint(checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint

def upload_to_huggingface(checkpoint_path, repo_id):
    """Upload model to HuggingFace Hub"""
    # Load checkpoint
    checkpoint = load_model_checkpoint(checkpoint_path)
    
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
    
    # Save config
    os.makedirs("./temp_model", exist_ok=True)
    with open("./temp_model/config.json", "w") as f:
        json.dump(config, f)
    
    # Save model state
    torch.save(checkpoint["model_state_dict"], "./temp_model/pytorch_model.bin")
    
    # Upload to Hub
    api = HfApi()
    api.upload_folder(
        folder_path="./temp_model",
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"Model uploaded successfully to {repo_id}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    repo_id = os.getenv("HF_REPO_ID", "jatingocodeo/ImageNet")
    
    upload_to_huggingface(checkpoint_path, repo_id) 