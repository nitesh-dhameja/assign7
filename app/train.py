import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models.resnet_model import ResNet50
from tqdm import tqdm
import logging
import urllib.request
import zipfile
from torchsummary import summary  # Import the summary function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to download the ImageNet dataset if it doesn't exist
def download_imagenet_data(data_dir='path/to/imagenet'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logging.info("Downloading ImageNet dataset...")
        # URL for the ImageNet dataset (this is a placeholder; replace with the actual URL)
        url = 'https://www.kaggle.com/api/v1/datasets/download/ifigotin/imagenetmini-1000'
        zip_path = os.path.join(data_dir, 'archive.zip')

        # Download the dataset
        urllib.request.urlretrieve(url, zip_path)

        # Extract the dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        logging.info("ImageNet dataset downloaded and extracted.")
    else:
        logging.info("ImageNet dataset already exists. Skipping download.")

def train_model(num_epochs=100, batch_size=32, learning_rate=0.001):
    # Check and download the ImageNet dataset
    download_imagenet_data()

    # Data augmentation and normalization for training
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the ImageNet dataset (replace with your dataset path)
    train_dataset = datasets.ImageFolder(root='path/to/imagenet/imagenet-mini/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = ResNet50(num_classes=1000)  # Create the model
    logging.info("Model Summary:")  # Log the model summary
    summary(model, (3, 224, 224))  # Assuming input size is (3, 224, 224)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels = inputs, labels  # Move data to GPU, removed cuda

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Log the epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        # Test the model after each epoch
        test_model(model)

    # Save the trained model
    torch.save(model.state_dict(), 'resnet50_imagenet_model.pth')

def test_model(model):
    # Load the model
    model.eval()  # Set the model to evaluation mode

    # Load test data (replace with your dataset path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(root='path/to/imagenet/imagenet-mini/val', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs, labels  # Move data to GPU, removed cuda
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test accuracy after epoch: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    train_model() 