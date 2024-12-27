import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models.resnet_model import ResNet50Model

def test_model_accuracy(model_path='resnet50_imagenet_model.pth'):
    # Load the model
    model = ResNet50Model(num_classes=1000)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Load test data (replace with your dataset path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(root='/opt/dlami/nvme/path/to/imagenet/imagenet-mini/val', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test accuracy: {accuracy:.2f}%')
    return accuracy

def test_case_1():
    assert test_model_accuracy() > 70, "Test case 1 failed: Accuracy is below 70%"

def test_case_2():
    assert test_model_accuracy() > 75, "Test case 2 failed: Accuracy is below 75%"

def test_case_3():
    assert test_model_accuracy() > 80, "Test case 3 failed: Accuracy is below 80%"

if __name__ == "__main__":
    test_case_1()
    test_case_2()
    test_case_3() 