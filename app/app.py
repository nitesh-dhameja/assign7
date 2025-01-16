import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoImageProcessor, AutoModelForImageClassification
import json

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(LABELS_URL)
labels = json.loads(response.text)

def load_model():
    """
    Load model and processor from Hugging Face Hub
    """
    model_id = "YOUR_MODEL_REPO_ID"  # Replace with your model repo ID
    model = AutoModelForImageClassification.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    return model, processor

def predict(image):
    """
    Make prediction on input image
    """
    # Load model and processor (with caching)
    model, processor = load_model()
    model.eval()
    
    # Process image
    inputs = processor(image, return_tensors="pt")
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Get probabilities and classes
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    top_probs, top_indices = torch.topk(probs, k=5)
    
    # Format results
    results = []
    for prob, idx in zip(top_probs, top_indices):
        label = labels[idx.item()]
        confidence = prob.item() * 100
        results.append((label, confidence))
    
    return {label: conf for label, conf in results}

# Create Gradio interface
title = "ImageNet Classifier"
description = """
This model classifies images into 1000 ImageNet categories.
Upload an image or provide an image URL to get predictions.
"""

examples = [
    ["examples/dog.jpg"],
    ["examples/cat.jpg"],
    ["examples/bird.jpg"]
]

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title=title,
    description=description,
    examples=examples,
    theme=gr.themes.Default()
).launch() 