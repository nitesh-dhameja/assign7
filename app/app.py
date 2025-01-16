import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import json
import torchvision.models as models
from transformers import AutoImageProcessor

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(LABELS_URL)
labels = json.loads(response.text)

def load_model():
    """
    Load model and processor from Hugging Face Hub
    """
    model_id = "jatingocodeo/ImageNet"
    
    # Initialize ResNet50 model
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1000)  # 1000 ImageNet classes
    
    # Load model weights
    checkpoint = torch.hub.load_state_dict_from_url(
        f"https://huggingface.co/{model_id}/resolve/main/pytorch_model.bin",
        map_location="cpu"
    )
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Create processor
    processor = AutoImageProcessor.from_pretrained(model_id)
    return model, processor

def predict(image):
    """
    Make prediction on input image
    """
    if image is None:
        return None
    
    try:
        # Load model and processor (with caching)
        model, processor = load_model()
        
        # Process image
        inputs = processor(image, return_tensors="pt")
        
        # Get predictions
        with torch.no_grad():
            outputs = model(inputs.pixel_values)
            
        # Get probabilities and classes
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, k=5)
        
        # Format results
        results = {}
        for prob, idx in zip(top_probs, top_indices):
            label = labels[idx.item()]
            confidence = prob.item() * 100
            results[label] = confidence
        
        return results
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Create Gradio interface
title = "ImageNet Classifier"
description = """
## ResNet50 ImageNet Classifier

This model classifies images into 1000 ImageNet categories. Upload an image or use one of the example images to get predictions.

### Instructions:
1. Upload an image using the input box below
2. The model will predict the top 5 classes for the image
3. Results show class names and confidence scores

### Model Details:
- Architecture: ResNet50
- Dataset: ImageNet
- Input Size: 224x224
- Number of Classes: 1000
"""

# Example images
examples = [
    "examples/dog.jpg",
    "examples/cat.jpg",
    "examples/bird.jpg",
    "examples/car.jpg",
    "examples/flower.jpg"
]

# Create the interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title=title,
    description=description,
    examples=examples,
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

# Launch the app
iface.launch() 