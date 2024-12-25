# ResNet50 from Scratch for ImageNet Classification

This project implements a ResNet50 model from scratch using PyTorch to classify images from the ImageNet dataset. The model is trained and evaluated on the dataset, with functionality to download the dataset if it is not already present.

## Requirements

- Python 3.8 or higher
- PyTorch
- torchvision
- tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/resnet-imagenet.git
   cd resnet-imagenet
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The project uses the ImageNet dataset, which consists of 1,000 classes. You can download the dataset using the provided functionality in the code. The dataset will be downloaded to the specified directory if it does not already exist.

## Training the Model

To start training the model, run the following command:
