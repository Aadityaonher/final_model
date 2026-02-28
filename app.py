import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops
import os
import torch
import torch.nn as nn
from torchvision import models, transforms

# cnn setup
@st.cache_resource
def load_ai_model():
    weight_path = 'ai_detector_weights.pth'
    if not os.path.exists(weight_path):
        return None
        
    try:
       
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_ai_probability(image_pil, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image_pil.convert('RGB')).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        ai_prob = probabilities[1].item() * 100
        real_prob = probabilities[0].item() * 100
        
    return ai_prob, real_prob

#Track A: Frequency Analysis (Global DCT)
def compute_dct(gray_img):
    f_img = np.float32(gray_img) / 255.0
    dct = cv2.dct(f_img)
    return np.log(np.abs(dct) + 1)

#Track B: Heatmap ELA & AI-Aware Scoring 
def analyze_forensics(image_pil, quality=90):
    temp_file = "temp_resave.jpg"
    image_pil.convert('RGB').save(temp_file, 'JPEG', quality=quality)
    resaved_pil = Image.open(temp_file)
    
    ela_img = ImageChops.difference(image_pil.convert('RGB'), resaved_pil)
    ela_np = np.array(ela_img.convert('L'))
    
    mean_val = np.mean(ela_np)
    std_val = np.std(ela_np)
    
    if std_val < 1.8:
        diagnosis = "SYNTHETIC / TOO CLEAN"
        score = 30 
        color = "inverse"
    elif std_val > 5.5:
        diagnosis = "LOCALIZED ANOMALY (INPAINTING)"
        score = max(10, int(100 - (std_val * 8))) 
        color = "inverse"
    else:
        diagnosis = "NATURAL VARIANCE"
        score = 95
        color = "normal"
        
    normalized_ela = cv2.normalize(ela_np, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(normalized_ela, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    return heatmap, score, diagnosis, color, mean_val, std_val

#UI Layout
st.set_page_config(page_title="Forensic Suite Pro", layout="wide")
st.title("🛡️ Ultimate Deepfake Forensic Suite")
st.markdown("Combining structural variance math with a custom-trained ResNet-50 Neural Network.")

# model loading
ai_model = load_ai_model()

uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, 1)
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    
    heatmap, ela_score, diagnosis, delta_color, m_val, s_val = analyze_forensics(image_pil)
    dct_map = compute_dct(cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY))

    
    col1, col2 = st.columns(2)
    with col1:
        track_b_ai_prob = 100 - ela_score 
        st.metric(label="Structural Anomaly Probability (Track B)", value=f"{track_b_ai_prob}%", delta=diagnosis, delta_color=delta_color)
    with col2:
        if ai_model:
            ai_prob, real_prob = predict_ai_probability(image_pil, ai_model)
            ai_status = "LIKELY AI GENERATED" if ai_prob > 50 else "LIKELY AUTHENTIC"
            st.metric(label="Neural Network AI Probability (Track C)", value=f"{ai_prob:.2f}%", delta=ai_status, delta_color="inverse" if ai_prob > 50 else "normal")
        else:
            st.warning("⚠️ Track C Offline: 'ai_detector_weights.pth' not found. Finish training to activate the neural network.")

# visual tabs
    tab1, tab2 = st.tabs(["🔍 Visual Analysis", "📊 Statistical Data"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Frequency Track (DCT)")
            st.image(dct_map, caption="Global Artifact Pattern", clamp=True, use_container_width=True)
        with c2:
            st.subheader("Variance Heatmap (ELA)")
            st.image(heatmap, caption="Red = High Variance (Possible Inpainting)", use_container_width=True)

    with tab2:
        st.subheader("Forensic Metadata Extract")
        st.write(f"**Mean Noise Level:** {m_val:.4f}")
        st.write(f"**Noise Variance:** {s_val:.4f}")
        st.write(f"**Resolution:** {image_cv.shape[1]}x{image_cv.shape[0]}")

else:
    st.info("Upload a forensic sample to begin.")