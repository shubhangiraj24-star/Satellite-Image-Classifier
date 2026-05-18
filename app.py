import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Satellite AI Classifier",
    page_icon="🛰️",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_ai_model():

    model = load_model(
        "best_satellite_model.keras",
        compile=False
    )

    return model

model = load_ai_model()

# =========================
# CLASS LABELS
# =========================
classes = [
    "Sea/Lake",
    "Agriculture",
    "Urban Area",
    "Crop Land",
    "Forest",
    "Highway",
    "Industrial Area",
    "Pasture",
    "River",
    "Vegetation",
    "Water Body"
]

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:white;
}

.subtitle{
    text-align:center;
    color:#aaaaaa;
    margin-bottom:25px;
}

.card{
    background:#1e293b;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 0px 15px rgba(0,255,150,0.15);
}

.prediction{
    font-size:34px;
    font-weight:bold;
    color:#00ff99;
}

.confidence{
    font-size:22px;
    color:white;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🛰️ Satellite AI")

st.sidebar.write(
    "EfficientNetB0 + Grad-CAM"
)

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("## Supported Classes")

for c in classes:
    st.sidebar.write(f"• {c}")

# =========================
# TITLE
# =========================
st.markdown(
    '<div class="main-title">🛰️ Satellite Image Classification AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Explainable AI using EfficientNetB0 + Grad-CAM</div>',
    unsafe_allow_html=True
)

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# MAIN
# =========================
if uploaded_file is not None:

    # =========================
    # LOAD IMAGE
    # =========================
    image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)

    original_img = image_np.copy()

    # =========================
    # DISPLAY IMAGE
    # =========================
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # =========================
    # PREPROCESS IMAGE
    # =========================
    img = cv2.resize(
        image_np,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    img = img.astype("float32")

    img = preprocess_input(img)

    img_array = np.expand_dims(img, axis=0)

    # =========================
    # PREDICTION
    # =========================
    with st.spinner("Analyzing Satellite Image..."):

        predictions = model.predict(img_array)

    pred_index = np.argmax(predictions[0])

    predicted_class = classes[pred_index]

    confidence = float(np.max(predictions[0]) * 100)

    # =========================
    # RESULT CARD
    # =========================
    st.markdown("## Prediction Result")

    st.markdown(f"""
    <div class="card">

        <div class="prediction">
            {predicted_class}
        </div>

        <div class="confidence">
            Confidence: {confidence:.2f}%
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.progress(confidence / 100)

    # =========================
    # TOP 3 PREDICTIONS
    # =========================
    st.markdown("## Top Predictions")

    top3_idx = predictions[0].argsort()[-3:][::-1]

    for i in top3_idx:

        st.write(f"### {classes[i]}")

        st.progress(float(predictions[0][i]))

        st.write(f"{predictions[0][i] * 100:.2f}%")

    # =========================
    # GRAD-CAM
    # =========================
    st.markdown("## Grad-CAM Visualization")

    base_model = model.layers[0]

    last_conv_layer = base_model.get_layer("top_conv")

    feature_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    with tf.GradientTape() as tape:

        conv_outputs = feature_model(img_array)

        tape.watch(conv_outputs)

        x = base_model.get_layer("top_bn")(conv_outputs)

        x = base_model.get_layer("top_activation")(x)

        for layer in model.layers[1:]:

            x = layer(x)

        preds = x

        class_channel = preds[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    heatmap /= np.max(heatmap)


    # =========================
    # RESIZE HEATMAP
    # =========================
    heatmap = cv2.resize(
        heatmap,
        (
            original_img.shape[1],
            original_img.shape[0]
        )
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # =========================
    # OVERLAY
    # =========================
    superimposed_img = cv2.addWeighted(
        original_img,
        0.6,
        heatmap,
        0.4,
        0
    )

    # =========================
    # DISPLAY GRADCAM
    # =========================
    col1, col2 = st.columns(2)

    with col1:

        st.image(
            heatmap,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    with col2:

        st.image(
            superimposed_img,
            caption="AI Attention Map",
            use_container_width=True
        )

    # =========================
    # HOW IT WORKS
    # =========================
    st.markdown("## How It Works")

    st.markdown("""
    1. Upload satellite image  
    2. EfficientNetB0 extracts deep features  
    3. AI predicts land category  
    4. Grad-CAM highlights important regions  
    5. Confidence scores are displayed  
    """)

else:

    st.info("Upload a satellite image to begin.")