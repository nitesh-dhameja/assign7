import os
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from s3_dataset import S3ImageNetDataset
import torchvision.models as models
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def save_training_log(log_file, epoch, train_loss, train_acc, val_loss, val_acc, lr, is_header=False):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    if is_header:
        header = "| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Learning Rate |\n"
        header += "|-------|------------|-----------|----------|----------|---------------|\n"
        with open(log_file, 'w') as f:
            f.write(header)
    
    with open(log_file, 'a') as f:
        f.write(f"| {epoch:5d} | {train_loss:.4f} | {train_acc:.2f}% | {val_loss:.4f} | {val_acc:.2f}% | {lr:.6f} |\n")

def save_training_summary(log_file, best_acc, train_losses, train_accs, val_losses, val_accs):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write("# Training Summary\n\n")
        f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final Training Accuracy: {train_accs[-1]:.2f}%\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {val_accs[-1]:.2f}%\n\n")
        
        f.write("## Training Progress\n\n")
        f.write("```\n")
        for epoch, (tl, ta, vl, va) in enumerate(zip(train_losses, train_accs, val_losses, val_accs)):
            f.write(f"Epoch {epoch:3d}: Train Loss={tl:.4f}, Train Acc={ta:.2f}%, Val Loss={vl:.4f}, Val Acc={va:.2f}%\n")
        f.write("```\n")

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        try:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': running_loss/(batch_idx+1),
                'Acc': 100.*correct/total,
                'LR': scheduler.get_last_lr()[0]
            })
            
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': running_loss/(batch_idx+1),
                    'Acc': 100.*correct/total
                })
                
            except Exception as e:
                logging.error(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    return running_loss/len(val_loader), 100.*correct/total

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100):
    best_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        try:
            # Training phase
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, epoch
            )
            
            # Validation phase
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Store metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Save checkpoint if validation accuracy improves
            if val_acc > best_acc:
                logging.info(f'Saving checkpoint... Validation Accuracy: {val_acc}%')
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'acc': val_acc,
                    'epoch': epoch,
                }
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                torch.save(state, './checkpoints/best_model.pth')
                best_acc = val_acc
            
            # Log metrics
            logging.info(
                f'Epoch {epoch}: Train Loss: {train_loss:.3f} | '
                f'Train Acc: {train_acc:.3f}% | Val Loss: {val_loss:.3f} | '
                f'Val Acc: {val_acc:.3f}% | LR: {scheduler.get_last_lr()[0]:.6f}'
            )
            
            # Save training log
            save_training_log(
                log_file='logs/training_log.md',
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=scheduler.get_last_lr()[0],
                is_header=(epoch == 0)
            )
            
        except Exception as e:
            logging.error(f"Error in epoch {epoch}: {str(e)}")
            continue
    
    # Save final training summary
    save_training_summary(
        log_file='logs/training_summary.md',
        best_acc=best_acc,
        train_losses=train_losses,
        train_accs=train_accs,
        val_losses=val_losses,
        val_accs=val_accs
    )
    
    return train_losses, train_accs, val_losses, val_accs

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Set up data transforms with stronger augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2
        ),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    bucket_name = os.getenv('S3_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable not set")

    logging.info("Creating datasets...")
    train_dataset = S3ImageNetDataset(bucket_name=bucket_name, transform=train_transform, is_train=True)
    val_dataset = S3ImageNetDataset(bucket_name=bucket_name, transform=val_transform, is_train=False)

    # Create data loaders with larger batch size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Create model with proper initialization
    logging.info("Creating model...")
    model = models.resnet50(weights=None)  # Training from scratch
    
    # Initialize the weights properly
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model = model.to(device)

    # Set up training with better hyperparameters
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,  # Higher initial learning rate
        weight_decay=0.05,  # Stronger weight decay
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    num_epochs = 100
    warmup_epochs = 5
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=warmup_epochs/num_epochs,  # Warmup period
        div_factor=10,  # Initial lr = max_lr/10
        final_div_factor=100,  # Final lr = max_lr/1000
        anneal_strategy='cos'
    )

    # Train model
    logging.info("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs
    )

if __name__ == '__main__':
    main()

