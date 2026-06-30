import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ── Define the same CNN architecture used in training ──
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ── Load trained model ──
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load('src/cnn_cifar10_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# ── Image preprocessing (must match training) ──
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ── UI ──
st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️")
st.title("🖼️ CIFAR-10 Image Classifier")
st.write("Upload an image and the CNN will predict one of 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_prob, top_class = torch.max(probs, 0)

    st.subheader(f"Prediction: **{classes[top_class.item()]}**")
    st.write(f"Confidence: {top_prob.item()*100:.2f}%")

    st.bar_chart({classes[i]: probs[i].item() for i in range(10)})