import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tomato Leaf Ninja", page_icon="🍅")

# --- 1. DOWNLOAD & LOAD MODEL ---
@st.cache_resource
def load_model():
  
    model_url = "https://github.com/SamanFatima7/Tomato-Leaf-Ninja/releases/download/v1.0/tomato_leaf_ninja.h5" 
    model_path = "model.h5"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI Model... please wait."):
            r = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(r.content)
    
    return tf.keras.models.load_model(model_path, , compile=False)

model = load_model()

# --- 2. CLASS NAMES ---
class_names = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 
    'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot', 
    'Yellow_Leaf_Curl_Virus', 'Mosaic_virus', 'Healthy'
]

# --- 3. UI DESIGN ---
st.title("🍅 Tomato Leaf Ninja")
st.markdown("### 🥋 Slicing through Diseases with 94% Accuracy")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Leaf", use_container_width=True)
    
    # Preprocessing
    img_resized = img.resize((256, 256)) 
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array)
        result = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.success(f"**Diagnosis:** {result.replace('_', ' ')}")
    st.write(f"**Confidence Score:** {confidence:.2f}%")
