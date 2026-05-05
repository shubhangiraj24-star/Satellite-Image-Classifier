import cv2
import os
import numpy as np
import tensorflow as tf

# 🔍 DEBUG: check files in folder
print("Files in main folder:", os.listdir())

data = []
labels = []

categories = ["city", "forest"]

# 🔹 Load dataset
for category in categories:
    path = os.path.join("dataset", category)
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        
        img = cv2.imread(img_path)
        
        if img is None:
            print("❌ Skipped:", img_name)
            continue
        
        img = cv2.resize(img, (128, 128))
        
        data.append(img)
        labels.append(category)

# 🔹 Convert to arrays
data = np.array(data)
data = data / 255.0   # normalization

label_map = {"city": 0, "forest": 1}
labels = np.array([label_map[label] for label in labels])

print("Data shape:", data.shape)
print("Labels:", labels)

# 🔹 Build model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# 🔹 Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("🚀 Starting training...")

# 🔹 Train
model.fit(data, labels, epochs=5)

# 🔥 PREDICTION PART

img = cv2.imread("test.jpg.jpg")

if img is None:
    print("❌ Test image not found")
else:
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    prediction = model.predict(img)

    if np.argmax(prediction) == 0:
        print("🌆 Prediction: CITY")
    else:
        print("🌳 Prediction: FOREST")