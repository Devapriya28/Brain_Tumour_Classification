# 🧠 Brain Tumor Classification — Streamlit (h5 version)
# Modern UI + Local Background Image

import os
import json
import base64
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------
# 🔧 CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
)

# --------- 🌈 GLOBAL CUSTOM CSS (FONT, CARDS, ETC) ----------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: "Poppins", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Only text color, background we set separately */
[data-testid="stAppViewContainer"] {
    color: #f9fafb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.9);
}

/* Remove default header background */
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

/* Title styling */
.app-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(120deg, #38bdf8, #a855f7, #f97316);
    -webkit-background-clip: text;
    color: transparent;
    margin-bottom: 0.3rem;
}

.app-subtitle {
    font-size: 0.98rem;
    color: #e5e7eb;
}

/* Glassmorphism card */
.glass-card {
    background: rgba(15, 23, 42, 0.8);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    border: 1px solid rgba(148, 163, 184, 0.35);
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.7);
}

/* Section titles */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

/* Prediction badge */
.pred-label {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #ecfdf3;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
}

.confidence-text {
    font-size: 0.95rem;
    color: #e5e7eb;
}

/* Download button */
.stDownloadButton button {
    border-radius: 999px;
    padding: 0.45rem 1.2rem;
    font-weight: 500;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border-radius: 16px;
}

/* Expander style */
.streamlit-expanderHeader {
    font-weight: 600;
}

/* Layout padding */
.block-container {
    padding-top: 1.5rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 🌄 Local file → background
def set_local_background(image_path: str):
    """Set app background from a local image file."""
    if not os.path.exists(image_path):
        st.error(f"❌ Background image not found: {image_path}")
        return

    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    ext = image_path.split(".")[-1].lower()
    if ext == "png":
        mime = "image/png"
    else:
        mime = "image/jpeg"

    bg_css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:{mime};base64,{encoded}") no-repeat center center fixed;
        background-size: cover;
    }}
    [data-testid="stAppViewContainer"] > .main {{
        background-size: cover !important;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)


# -----------------------------
# 🔧 CONSTANTS  (👉 EDIT THESE)
# -----------------------------
MODEL_PATH = r"D:\DATA SCIENCE\CODE\git\project_6\InceptionV3_best.h5"  # your h5 model path

# 👉 CHANGE THIS to your local background image path
BG_IMAGE_PATH = r"D:\DATA SCIENCE\CODE\git\project_6\mri_bg.jpg"

IMG_SIZE = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# ✅ Apply local background
set_local_background(BG_IMAGE_PATH)

# -----------------------------
# 🏷 HEADER
# -----------------------------
with st.container():
    st.markdown('<div class="app-title">🧠 Brain Tumor Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        'Upload a brain MRI image to predict the tumor type using a deep learning model.'
        '</div>',
        unsafe_allow_html=True
    )
    st.write("")

# -----------------------------
# 🧩 LOAD TRAINED MODEL
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_keras_model(path: str):
    return load_model(path)

# -----------------------------
# 📊 SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model Status")
    try:
        assert os.path.exists(MODEL_PATH), f"Model not found at:\n{MODEL_PATH}"
        model = load_keras_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        st.caption("Model: `InceptionV3_best.h5`")
    except Exception as e:
        st.error(f"❌ Model load failed:\n{e}")
        st.stop()

    st.markdown("---")
    st.markdown("### 🧾 Info")
    st.caption(
        "This app uses a CNN (InceptionV3) trained on brain MRI images to classify:\n"
        "- Glioma\n- Meningioma\n- No Tumor\n- Pituitary"
    )

# -----------------------------
# 🔧 HELPERS
# -----------------------------
def preprocess_image(pil_img: Image.Image, target_hw=(224, 224)) -> np.ndarray:
    """PIL -> float32 [1,H,W,C] normalized to [0,1]."""
    arr = np.array(pil_img.convert("RGB"))
    arr = tf.image.resize(arr, target_hw)
    arr = arr / 255.0
    arr = np.expand_dims(arr, 0)  # [1,H,W,C]
    return arr

# -----------------------------
# 📤 IMAGE UPLOAD + PREVIEW
# -----------------------------
left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload MRI Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an MRI Image (JPG/PNG/JPEG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="mri_uploader"
    )

    if not uploaded_file:
        st.info("👆 Upload a brain MRI image to begin prediction.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"❌ Could not read the image file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    st.image(image, caption="🩺 Uploaded MRI Image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# 🔮 PREDICT
# -----------------------------
img_batch = preprocess_image(image, IMG_SIZE)

try:
    preds = model.predict(img_batch)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

probs = preds[0]
pred_class = int(np.argmax(probs))
pred_label = CLASS_NAMES[pred_class]
confidence = float(probs[pred_class])

# -----------------------------
# 🎯 DISPLAY PREDICTION RESULTS
# -----------------------------
with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Prediction Results</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="pred-label">'
        f'<span>Predicted Tumor Type:</span> {pred_label.capitalize()}'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p class="confidence-text">Model confidence: <b>{confidence*100:.2f}%</b></p>',
        unsafe_allow_html=True
    )

    # Download result as JSON
    result = {
        "predicted_label": pred_label,
        "confidence": confidence,
        "per_class": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
    }

    st.download_button(
        "⬇️ Download prediction as JSON",
        data=json.dumps(result, indent=2),
        file_name="prediction_result.json",
        mime="application/json",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# 📈 CONFIDENCE BAR CHART
# -----------------------------
st.write("")
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔍 Model Confidence per Class</div>', unsafe_allow_html=True)


conf_data = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
st.bar_chart(conf_data, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# 📚 ABOUT SECTION
# -----------------------------
with st.expander("ℹ️ About this App"):
    st.write("""
This application uses a **deep learning model** based on **InceptionV3** trained on brain MRI images
to classify the tumor type into one of four categories:

- Glioma  
- Meningioma  
- No Tumor  
- Pituitary Tumor  

**Tech Stack:** TensorFlow · Keras · Streamlit  
**Model File:** `InceptionV3_best.h5`
""")
