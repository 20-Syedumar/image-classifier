# 🖼️ CIFAR-10 Image Classifier (CNN + Streamlit)

A Convolutional Neural Network built from scratch in PyTorch to classify images into 10 categories, with a live Streamlit web app for interactive predictions.

---

## 📊 Results

**Overall Test Accuracy: 75.65%**

| Class | Accuracy |
|---|---|
| Truck | 88.70% |
| Car | 84.80% |
| Plane | 84.70% |
| Ship | 83.60% |
| Frog | 80.60% |
| Deer | 75.40% |
| Horse | 73.10% |
| Dog | 62.80% |
| Bird | 64.30% |
| Cat | 58.50% |

> Vehicles (truck, car, plane, ship) were classified far more reliably than animals — animals share more visual similarity (fur, poses, body shapes), making them harder to distinguish at CIFAR-10's 32x32 resolution.

---

## 🖥️ Live Demo

A Streamlit app (`app.py`) lets you upload any image and get a real-time prediction with confidence scores across all 10 classes.

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
image-classifier/
├── data/                          ← CIFAR-10 dataset (auto-downloaded, gitignored)
├── notebooks/
│   └── image_classifier.ipynb     ← Training notebook (EDA, model, training, evaluation)
├── src/
│   └── cnn_cifar10_model.pth      ← Saved trained model weights
├── app.py                         ← Streamlit frontend
├── .gitignore
└── README.md
```

---

## 🧠 Model Architecture

A custom CNN with 3 convolutional blocks followed by 3 fully connected layers:

```
Conv2d(3 → 32) → ReLU → MaxPool
Conv2d(32 → 64) → ReLU → MaxPool
Conv2d(64 → 128) → ReLU → MaxPool
Flatten
Linear(2048 → 512) → ReLU → Dropout(0.5)
Linear(512 → 64) → ReLU → Dropout(0.5)
Linear(64 → 10)
```

**Total parameters:** 1,175,818

---

## 🔧 Tech Stack

- **Python 3.12**
- **PyTorch & Torchvision** — model building and training
- **Matplotlib & NumPy** — visualization
- **Streamlit** — interactive web app
- **Pillow** — image preprocessing for the app

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/20-Syedumar/image-classifier.git
cd image-classifier
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**
```bash
pip install torch torchvision matplotlib numpy jupyter ipykernel streamlit pillow
```

**4. Run the notebook (optional — to retrain)**
- Open `notebooks/image_classifier.ipynb` in VS Code
- Run all cells top to bottom (CIFAR-10 will auto-download, ~170MB)

**5. Run the web app**
```bash
streamlit run app.py
```
Opens automatically at `http://localhost:8501`

---

## 📌 Key Steps

### 1. Data Loading
- CIFAR-10 dataset: 50,000 training images, 10,000 test images, 10 classes
- Normalized pixel values to [-1, 1] range for stable training

### 2. Model Training
- 10 epochs, Adam optimizer (lr=0.001), CrossEntropyLoss
- Batch size: 32
- Training loss decreased steadily from 1.63 → 0.58 across epochs (no overfitting observed)

### 3. Evaluation
- Achieved 75.65% test accuracy on unseen data
- Per-class breakdown revealed vehicles outperform animals — a known pattern in CIFAR-10 due to higher intra-class visual variation among animal categories

### 4. Deployment
- Built a Streamlit app that loads the saved model and serves real-time predictions on uploaded images
- Displays prediction confidence as a bar chart across all 10 classes

---

## 💡 Key Learnings

- CNNs learn hierarchical visual features — early layers detect edges/colors, deeper layers detect shapes/objects
- Dropout (0.5) helped prevent overfitting despite ~1.17M trainable parameters
- Confusable class pairs (e.g. car ↔ truck, cat ↔ dog) reveal where additional data or architecture improvements would help most
- Wrapping a trained PyTorch model in Streamlit turns a notebook experiment into a usable product in under 50 lines of code

---

## 📈 Future Improvements

- [ ] Use transfer learning (ResNet18/MobileNet pretrained on ImageNet) to push accuracy above 90%
- [ ] Add data augmentation (random flips/crops) to improve animal class accuracy
- [ ] Add a confusion matrix visualization
- [ ] Deploy the Streamlit app publicly via Streamlit Community Cloud
