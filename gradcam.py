import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from PIL import Image

# =========================
# IMAGE PATH
# =========================
image_path = "test.jpg.jpg"

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
# LOAD MODEL
# =========================
model = load_model(
    "best_satellite_model.keras",
    compile=False
)

print("\nModel Loaded Successfully!")

# =========================
# LOAD IMAGE
# =========================
img = Image.open(image_path).convert("RGB")

img = np.array(img)

original_img = img.copy()

# =========================
# PREPROCESS IMAGE
# =========================
img = cv2.resize(
    img,
    (224, 224),
    interpolation=cv2.INTER_CUBIC
)

img = img.astype("float32")

img = preprocess_input(img)

img_array = np.expand_dims(img, axis=0)

# =========================
# PREDICTION
# =========================
predictions = model.predict(img_array)

pred_index = np.argmax(predictions[0])

predicted_class = classes[pred_index]

confidence = float(np.max(predictions[0]) * 100)

print(f"\nPrediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

# =========================
# GET BASE MODEL
# =========================
base_model = model.layers[0]

# Last conv layer
last_conv_layer = base_model.get_layer("top_conv")

# =========================
# CREATE FEATURE MODEL
# =========================
feature_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=last_conv_layer.output
)

# =========================
# GRAD-CAM
# =========================
with tf.GradientTape() as tape:

    # Get convolution outputs
    conv_outputs = feature_model(img_array)

    tape.watch(conv_outputs)

    # Pass through remaining EfficientNet layers
    x = base_model.get_layer("top_bn")(conv_outputs)
    x = base_model.get_layer("top_activation")(x)

    # Pass through custom classifier head
    for layer in model.layers[1:]:
        x = layer(x)

    predictions = x

    pred_index = tf.argmax(predictions[0])

    loss = predictions[:, pred_index]

# =========================
# COMPUTE GRADIENTS
# =========================
grads = tape.gradient(loss, conv_outputs)

# Remove batch dimension
conv_outputs = conv_outputs[0]

# Mean gradients
pooled_grads = tf.reduce_mean(
    grads[0],
    axis=(0, 1)
)

# Convert to numpy
conv_outputs = conv_outputs.numpy()

pooled_grads = pooled_grads.numpy()

# =========================
# GENERATE HEATMAP
# =========================
for i in range(pooled_grads.shape[-1]):

    conv_outputs[:, :, i] *= pooled_grads[i]

heatmap = np.mean(
    conv_outputs,
    axis=-1
)

# ReLU
heatmap = np.maximum(heatmap, 0)

# Normalize
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

# Convert to color
heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

# Convert BGR -> RGB
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
# DISPLAY RESULTS
# =========================
plt.figure(figsize=(16, 6))

# Original
plt.subplot(1, 3, 1)
plt.imshow(original_img)
plt.title("Original Image")
plt.axis("off")

# Heatmap
plt.subplot(1, 3, 2)
plt.imshow(heatmap)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

# Overlay
plt.subplot(1, 3, 3)
plt.imshow(superimposed_img)
plt.title(f"{predicted_class} ({confidence:.2f}%)")
plt.axis("off")

plt.tight_layout()

plt.show()