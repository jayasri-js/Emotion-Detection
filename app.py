import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision import models

device = "cpu"


model = models.resnet50()
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False
)
model.fc = torch.nn.Linear(model.fc.in_features, 8)

state_dict = torch.load("model.pth", map_location=device)
model.load_state_dict(state_dict)

model.eval()

classes = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5])
])

def predict(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img)
        probs = F.softmax(out, dim=1)[0]
    return {classes[i]: float(probs[i]) for i in range(len(classes))}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=8),
    title="Emotion Detection"
).launch()