import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# =========================
# IMAGE SETTINGS
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# =========================
# DATASET PATH
# =========================
dataset_path = "dataset"

# =========================
# DATA GENERATOR
# =========================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,

    rotation_range=30,
    zoom_range=0.2,

    horizontal_flip=True,
    vertical_flip=True,

    brightness_range=[0.8, 1.2]
)

# =========================
# TRAIN DATA
# =========================
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# =========================
# VALIDATION DATA
# =========================
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# =========================
# PRINT CLASS INFO
# =========================
print("\nDetected Classes:")
print(train_data.class_indices)

print("\nNumber of Classes:")
print(train_data.num_classes)

# =========================
# LOAD EFFICIENTNETB0
# =========================
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# =========================
# FREEZE MOST LAYERS
# =========================
for layer in base_model.layers[:-20]:
    layer.trainable = False

# =========================
# BUILD MODEL
# =========================
model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    BatchNormalization(),

    Dense(256, activation='relu'),
    Dropout(0.4),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(train_data.num_classes, activation='softmax')
])

# =========================
# MODEL SUMMARY
# =========================
model.summary()

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# CALLBACKS
# =========================

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Reduce learning rate automatically
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# Save best model
checkpoint = ModelCheckpoint(
    "best_satellite_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# =========================
# TRAIN MODEL
# =========================
print("\nTraining Started...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# =========================
# SAVE FINAL MODEL
# =========================
model.save("satellite_efficientnet.keras")

print("\nModel Saved Successfully!")
print("Saved as: satellite_efficientnet.keras")

# =========================
# FINAL ACCURACY
# =========================
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

# =========================
# ACCURACY GRAPH
# =========================
plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train", "Validation"])

plt.grid(True)

plt.show()

# =========================
# LOSS GRAPH
# =========================
plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["Train", "Validation"])

plt.grid(True)

plt.show()