import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms

# --- Model & Analysis functions (unchanged) ---

@st.cache_resource
def load_ai_model():
    weight_path = 'ai_detector_weights.pth'
    if not os.path.exists(weight_path):
        return None
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def analyze_forensics(image_pil, quality=90):
    temp_file = "temp_resave.jpg"
    image_pil.convert('RGB').save(temp_file, 'JPEG', quality=quality)
    resaved_pil = Image.open(temp_file)
    ela_img = ImageChops.difference(image_pil.convert('RGB'), resaved_pil)
    ela_np = np.array(ela_img.convert('L'))
    std_val = np.std(ela_np)
    mean_val = np.mean(ela_np)
    if std_val < 1.8:
        diagnosis = "SYNTHETIC / TOO CLEAN"
        score = 30
    elif std_val > 5.5:
        diagnosis = "LOCALIZED ANOMALY (INPAINTING)"
        score = max(10, int(100 - (std_val * 8)))
    else:
        diagnosis = "NATURAL VARIANCE"
        score = 95
    if os.path.exists(temp_file):
        os.remove(temp_file)
    return score, diagnosis, mean_val, std_val

# --- Streamlit UI ---

st.title("🔍 AI vs Real Image Detector")

ai_model = load_ai_model()

if ai_model is None:
    st.error("⚠️ Model weights not found. Place `ai_detector_weights.pth` in the same folder.")
else:
    st.success("✅ Model loaded successfully.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and ai_model:
    contents = uploaded_file.read()
    image_pil = Image.open(io.BytesIO(contents))
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        # Forensic analysis
        ela_score, diagnosis, m_val, s_val = analyze_forensics(image_pil)
        structural_prob = 100 - ela_score

        # Neural network analysis
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image_pil.convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            outputs = ai_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            ai_prob = probs[1].item() * 100

    st.subheader("📊 Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔬 Forensic Analysis (ELA)")
        st.metric("Anomaly Probability", f"{structural_prob}%")
        st.write(f"**Diagnosis:** {diagnosis}")
        st.write(f"**Noise Variance:** {round(s_val, 4)}")

    with col2:
        st.markdown("### 🧠 Neural Network")
        st.metric("AI Probability", f"{ai_prob:.2f}%")
        if ai_prob > 50:
            st.error("🤖 LIKELY AI GENERATED")
        else:
            st.success("📷 LIKELY REAL IMAGE")
