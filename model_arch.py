import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

IMAGE_SIZE = (224, 224)

class PneumoniaModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = resnet18(weights='DEFAULT') 
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])