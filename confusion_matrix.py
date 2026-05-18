import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.metrics import confusion_matrix, classification_report

# =========================
# SETTINGS
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32

# =========================
# DATASET PATH
# =========================
dataset_path = "dataset"

# =========================
# CLASS NAMES
# =========================
classes = [
    "SeaLake",
    "Agriculture",
    "Urban",
    "Crop",
    "Forest",
    "Highway",
    "Industrial",
    "Pasture",
    "River",
    "Vegetation",
    "Water"
]

# =========================
# LOAD MODEL
# =========================
model = load_model("best_satellite_model.keras")

print("\nModel Loaded Successfully!")

# =========================
# VALIDATION DATA
# =========================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# =========================
# PREDICTIONS
# =========================
print("\nGenerating Predictions...\n")

predictions = model.predict(val_data)

# Predicted labels
y_pred = np.argmax(predictions, axis=1)

# True labels
y_true = val_data.classes

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=classes
))

# =========================
# PLOT CONFUSION MATRIX
# =========================
plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes
)

plt.title("Satellite Image Classification Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.tight_layout()

plt.show()