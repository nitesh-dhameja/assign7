# ImageNet Classification with ResNet50

This project implements an ImageNet classifier using ResNet50 architecture, trained from scratch. The model is deployed on Hugging Face Hub and accessible through a Gradio interface.

## Project Structure
```
assign9/
├── app/
│   ├── train.py           # Training script
│   ├── app.py            # Gradio interface
│   ├── s3_dataset.py     # Custom dataset for S3
│   └── upload_to_hf.py   # HuggingFace upload script
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
└── README.md
```

## Model Details

- **Architecture**: ResNet50
- **Dataset**: ImageNet (ILSVRC2012)
- **Input Size**: 224x224
- **Number of Classes**: 1000
- **Training Infrastructure**: AWS EC2

## Training Configuration

- **Optimizer**: AdamW
- **Learning Rate**: 0.003 with cosine scheduling
- **Batch Size**: 256
- **Data Augmentation**:
  - RandomResizedCrop
  - RandomHorizontalFlip
  - ColorJitter
  - RandomAffine
  - RandomPerspective
  - Normalization

## Results

The model achieves competitive performance on the ImageNet validation set:
- Training Accuracy: ~55%
- Validation Accuracy: ~45%

## Deployment

### Model
- **HuggingFace Repository**: [jatingocodeo/ImageNet](https://huggingface.co/jatingocodeo/ImageNet)
- **Model Format**: PyTorch

### Demo
- **HuggingFace Space**: [ImageNet Classifier](https://huggingface.co/spaces/jatingocodeo/imagenet-classifier)
- **Interface**: Gradio web application
- **Features**:
  - Image upload
  - Top-5 class predictions
  - Confidence scores
  - Example images

## Usage

### Local Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export S3_BUCKET_NAME="your-bucket"
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="your-region"
export HUGGINGFACE_TOKEN="your-token"
```

3. Train the model:
```bash
cd app
python train.py
```

4. Upload to HuggingFace:
```bash
python upload_to_hf.py
```

### Using the Model

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

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

## Web Interface

The model is accessible through a user-friendly web interface:
1. Visit [ImageNet Classifier](https://huggingface.co/spaces/jatingocodeo/imagenet-classifier)
2. Upload an image or use provided examples
3. Get instant predictions with confidence scores

## References

- [ImageNet Dataset](https://www.image-net.org/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Gradio Documentation](https://gradio.app/docs/)
