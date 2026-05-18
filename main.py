import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================

IMG_SIZE = 64
EPOCHS = 15
BATCH_SIZE = 32

# =========================
# DATASET PATH
# =========================

dataset_path = "dataset"

# =========================
# CLASS NAMES
# =========================

categories = sorted(os.listdir(dataset_path))

print("📂 Classes Found:", categories)

# =========================
# LOAD DATASET
# =========================

data = []
labels = []

print("🚀 Loading dataset...")

for category in categories:

    path = os.path.join(dataset_path, category)

    class_num = categories.index(category)

    for img_name in os.listdir(path):

        try:
            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Normalize image
            img = img.astype("float32") / 255.0

            data.append(img)
            labels.append(class_num)

        except Exception as e:
            print("Error:", e)

# Convert to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

print("✅ Dataset Loaded")
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("✅ Train/Test Split Done")

# =========================
# DATA AUGMENTATION
# =========================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# =========================
# BETTER CNN MODEL
# =========================

model = tf.keras.models.Sequential([

    tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    data_augmentation,

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        len(categories),
        activation='softmax'
    )
])

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# EARLY STOPPING
# =========================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =========================
# TRAIN MODEL
# =========================

print("🚀 Training Started...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

# =========================
# SAVE MODEL
# =========================

model.save("satellite_model.keras")

print("✅ Model Saved")

# =========================
# MODEL EVALUATION
# =========================

loss, accuracy = model.evaluate(X_test, y_test)

print(f"🎯 Test Accuracy: {accuracy * 100:.2f}%")

# =========================
# ACCURACY GRAPH
# =========================

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train", "Validation"])

plt.show()

# =========================
# TEST IMAGE PREDICTION
# =========================

img_path = "test.jpg"

img = cv2.imread(img_path)

if img is None:

    print("❌ test.jpg not found")

else:

    original = img.copy()

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    print("🌍 Prediction:", categories[predicted_class])

    print(f"🔥 Confidence: {confidence:.2f}%")

    # SHOW IMAGE

    plt.imshow(
        cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    )

    plt.title(
        f"{categories[predicted_class]} ({confidence:.2f}%)"
    )

    plt.axis("off")

    plt.show()