import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
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
        logging.FileHandler('training.log', mode='w')
    ]
)

def create_data_loaders(batch_size, num_workers=8):
    """
    Create data loaders for training and validation
    """
    # Data augmentation for training (ImageNet standard augmentation)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2
        ),
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

    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    logging.info(f"Number of classes: {len(train_dataset.classes)}")

    # Create data loaders with optimized settings
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
    Create and initialize ResNet50 model
    """
    model = models.resnet50(weights=None)  # Training from scratch
    
    # Initialize weights properly
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
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scaler, device, num_epochs=90, learning_rate=0.1):
    """
    Train the model with mixed precision and gradient accumulation
    """
    # Track start time
    start_time = time.time()
    
    # Create timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.md")
    
    # Initialize log file
    with open(log_file, "w", encoding='utf-8') as f:
        f.write("# ImageNet Training Log\n\n")
        f.write("## Configuration\n")
        f.write(f"- Model: ResNet50 (from scratch)\n")
        f.write(f"- Epochs: {num_epochs}\n")
        f.write(f"- Base Learning Rate: {learning_rate}\n")
        f.write(f"- Batch Size: {train_loader.batch_size}\n")
        f.write(f"- Device: {device}\n\n")
        f.write("## Training Progress\n\n")
        f.write("| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |\n")
        f.write("|-------|------------|-----------|----------|---------|----|\n")
    
    # Training configuration
    warmup_epochs = 5
    best_acc = 0.0
    accumulation_steps = 4  # Gradient accumulation steps
    
    # Training loop
    for epoch in range(num_epochs):
        # Learning rate schedule with warmup
        if epoch < warmup_epochs:
            current_lr = learning_rate * (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            current_lr = learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Mixed precision forward pass
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # Update metrics
            train_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'acc': f'{acc:.2f}%',
                'lr': current_lr
            })
            
            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validation'):
                inputs, targets = inputs.to(device), targets.to(device)
                
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Save logs
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(f"| {epoch+1:3d} | {train_loss:.4f} | {train_acc:.2f}% | {val_loss:.4f} | {val_acc:.2f}% | {current_lr:.6f} |\n")
        
        # Log to console
        logging.info(
            f'Epoch {epoch+1}/{num_epochs} - '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
            f'LR: {current_lr:.6f}'
        )
        
        # Save checkpoint if best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(checkpoint, f'checkpoint_{timestamp}.pth')
            logging.info(f'Saved checkpoint with accuracy: {best_acc:.2f}%')
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}_{timestamp}.pth')
    
    # Save final summary
    training_time = time.time() - start_time
    with open(log_file, "a", encoding='utf-8') as f:
        f.write("\n## Training Summary\n\n")
        f.write(f"- Best Validation Accuracy: {best_acc:.2f}%\n")
        f.write(f"- Total Training Time: {training_time/3600:.2f} hours\n")
        f.write(f"- Average Time per Epoch: {training_time/num_epochs/60:.2f} minutes\n")
        if torch.cuda.is_available():
            f.write(f"- GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"- CUDA Version: {torch.version.cuda}\n")
    
    return best_acc

def main():
    # Load environment variables
    load_dotenv()
    
    # Training configuration
    config = {
        'batch_size': 64,  # Reduced batch size for full dataset
        'epochs': 90,      # Standard ImageNet training epochs
        'learning_rate': 0.1,
        'num_workers': 8
    }
    
    # Set up CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
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
    
    # Create model
    model = create_model(num_classes).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    
    # Mixed precision training
    scaler = GradScaler('cuda')
    
    # Train model
    best_acc = train_model(
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
    
    logging.info(f"Training completed. Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()