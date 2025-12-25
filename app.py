import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Tomato Leaf Ninja Pro",
    page_icon="🍅",
    layout="centered"
)

# --- MODEL LOADING (With Cache) ---
@st.cache_resource
def load_master_model():
    # 1. REPLACE THIS URL with your new GitHub Release link for the MASTER.h5 file
    url = "https://github.com/SamanFatima7/Tomato-Leaf-Ninja/releases/download/v1.0/tomato_leaf_ninja_MASTER.h5" 
    path = "tomato_leaf_ninja_MASTER.h5"
    
    if not os.path.exists(path):
        with st.spinner("Downloading the Master AI Model... Please wait."):
            response = requests.get(url)
            with open(path, "wb") as f:
                f.write(response.content)
    
    # Load model without compiling (faster and safer for Streamlit)
    return tf.keras.models.load_model(path, compile=False)

# Load the model
try:
    model = load_master_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- CLASS LABELS ---
# These must match the order in your Kaggle training exactly
class_names = [
    'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 
    'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 
    'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Healthy'
]

# --- USER INTERFACE ---
st.title("🍅 Tomato Leaf Ninja Pro")
st.markdown("### 🥋 High-Precision Disease Diagnosis")
st.write("Upload a photo of a tomato leaf to get an instant AI diagnosis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    # 2. PREPROCESSING (The EfficientNet Way)
    # Resize to 224x224
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 3. PREDICTION
    with st.spinner("Analyzing..."):
        predictions = model.predict(img_array)
        prediction_index = np.argmax(predictions)
        result = class_names[prediction_index]
        confidence = np.max(predictions) * 100

    # 4. RESULTS DISPLAY
    st.divider()
    if confidence > 50:
        st.success(f"**Diagnosis:** {result}")
        st.info(f"**Confidence Score:** {confidence:.2f}%")
        
        # Simple recommendation logic for your FYP
        if result == 'Healthy':
            st.balloons()
            st.write("✅ Your plant looks great! Keep up the good work.")
        else:
            st.warning(f"🚩 Action Required: This leaf shows signs of {result}. Consult the management guide.")
    else:
        st.warning("⚠️ The AI is unsure. Please provide a clearer, closer photo of the leaf.")

st.sidebar.info("Developed by Saman Fatima | BSCS 2022-2026")
