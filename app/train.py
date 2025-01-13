import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import logging
import time
import math
from tqdm import tqdm
from dotenv import load_dotenv
from s3_dataset import S3ImageNetDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

def create_data_loaders(batch_size, num_workers=4):
    """
    Create data loaders for training and validation
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get S3 bucket info from environment
    bucket_name = os.getenv("S3_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable is not set")
    logging.info(f"Using S3 bucket: {bucket_name}")

    # Create datasets
    train_dataset = S3ImageNetDataset(bucket_name, transform=train_transform, is_train=True)
    val_dataset = S3ImageNetDataset(bucket_name, transform=val_transform, is_train=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger validation batch size
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader, len(train_dataset.classes)

def create_model(num_classes):
    """
    Create and initialize the ResNet50 model
    """
    model = models.resnet50(weights=None)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # Modify final layer for ImageNet
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.normal_(model.fc.weight, mean=0, std=0.01)
    nn.init.constant_(model.fc.bias, 0)
    
    return model

def setup_training(model, learning_rate=0.1):
    """
    Set up training components (optimizer, criterion, etc.)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    scaler = GradScaler()
    
    return criterion, optimizer, scaler

def save_training_log(log_file, epoch, train_loss, train_acc, val_loss, val_acc, current_lr=None, is_header=False):
    """
    Save training logs to a markdown file
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    if is_header:
        with open(log_file, "w") as f:
            f.write("# ImageNet Training Log\n\n")
            f.write("## Training Configuration\n")
            f.write("- Model: ResNet50 (from scratch)\n")
            f.write("- Target: 70% Top-1 Accuracy\n")
            f.write("- Dataset: ImageNet\n\n")
            f.write("## Training Progress\n\n")
            f.write("| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR | Target Met |\n")
            f.write("|-------|------------|-----------|----------|---------|----|-----------|\n")
    else:
        target_met = "✓" if val_acc >= 70.0 else "✗"
        lr_str = f"{current_lr:.6f}" if current_lr is not None else "N/A"
        
        with open(log_file, "a") as f:
            f.write(f"| {epoch:5d} | {train_loss:.4f} | {train_acc:.2f}% | {val_loss:.4f} | {val_acc:.2f}% | {lr_str} | {target_met} |\n")

def save_training_summary(log_file, best_val_acc, total_time):
    """
    Save training summary to the markdown file
    """
    with open(log_file, "a") as f:
        f.write("\n## Training Summary\n\n")
        f.write(f"- Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"- Target Accuracy (70.0%) {'Achieved' if best_val_acc >= 70.0 else 'Not Achieved'}\n")
        f.write(f"- Total Training Time: {total_time:.2f} seconds\n")
        
        # Add hardware info
        f.write("\n### Hardware Information\n")
        if torch.cuda.is_available():
            f.write(f"- GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"- CUDA Version: {torch.version.cuda}\n")
        else:
            f.write("- Running on CPU\n")

def train_model(num_epochs=100, batch_size=256, learning_rate=0.1):
    # Track start time
    start_time = time.time()
    
    # Create timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(os.getcwd(), 'logs', f"training_log_{timestamp}.md")
    
    # Initialize log file with header
    save_training_log(log_file, 0, 0, 0, 0, 0, is_header=True)
    
    # Rest of your training configuration...
    # ... (keep existing code) ...
    
    for epoch in range(num_epochs):
        # ... (keep existing training loop code) ...
        
        # After calculating metrics for the epoch
        current_lr = optimizer.param_groups[0]['lr']
        save_training_log(
            log_file,
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr
        )
        
        # ... (keep rest of the epoch code) ...
    
    # Save final summary
    total_time = time.time() - start_time
    save_training_summary(log_file, best_val_acc, total_time)
    
    return train_losses, val_losses, train_accs, val_accs

def main():
    # Load environment variables
    load_dotenv()
    
    # Training configuration
    config = {
        'batch_size': 256,
        'epochs': 100,
        'learning_rate': 0.1,
        'num_workers': 4
    }
    
    # Set up CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, num_classes = create_data_loaders(
        config['batch_size'],
        config['num_workers']
    )
    
    # Create and initialize model
    model = create_model(num_classes).to(device)
    criterion, optimizer, scaler = setup_training(model, config['learning_rate'])
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        num_epochs=config['epochs'],
        learning_rate=config['learning_rate']
    )

if __name__ == "__main__":
    main() 