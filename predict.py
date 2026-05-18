import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================
# LOAD SAVED MODEL
# =========================

model = tf.keras.models.load_model("satellite_model.keras")

# =========================
# CLASS NAMES
# =========================

categories = [
    "agriculture",
    "forest",
    "vegetation",
    "highway",
    "industrial",
    "pasture",
    "crops",
    "city",
    "river",
    "water"
]

# =========================
# LOAD TEST IMAGE
# =========================

img_path = "test.jpg.jpg"

img = cv2.imread(img_path)

if img is None:
    print("❌ test.jpg.jpg not found")

else:

    original = img.copy()

    # Resize image
    img = cv2.resize(img, (64, 64))

    # Normalize
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    print("🌍 Prediction:", categories[predicted_class])

    print(f"🔥 Confidence: {confidence:.2f}%")

    # Show image
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))

    plt.title(
        f"{categories[predicted_class]} ({confidence:.2f}%)"
    )

    plt.axis("off")

    plt.show()