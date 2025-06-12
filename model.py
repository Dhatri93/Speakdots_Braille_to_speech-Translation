from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re



# Define the local dataset path
dataset_path = Path(r"C:/Users/jayad/Downloads/Braille Dataset/Braille Dataset/Braille Dataset")

# Check if dataset is correctly extracted
if dataset_path.exists():
    print(" Braille Dataset Directory Exists:", dataset_path)

    # List files inside dataset
    files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.jpeg")) + list(dataset_path.glob("*.png"))
    print(f"Number of Files: {len(files)}")  # Check total number of images
    print(" Sample Files:", [f.name for f in files[:10]])  # Display first 10 image files

    if len(files) == 0:
        raise ValueError(" ERROR: The dataset folder is empty!")
else:
    raise ValueError(" ERROR: Braille Dataset directory does NOT exist!")

# Define image size
IMG_SIZE = 64  # Resize all images to 64x64 pixels


# Function to load images and labels
def load_data():
    images = []
    labels = []

    for file in files:
        try:
            # Load image in grayscale
            img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f" Warning: Unable to read image {file.name}. Skipping...")
                continue
            
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
            images.append(img)

            # Extract label from filename (e.g., "a1.JPG12whs.jpg" → "a")
            match = re.search(r"[a-zA-Z]", file.name)
            if match:
                label = match.group(0).lower()
                labels.append(label)
            else:
                print(f" Warning: No valid label found in {file.name}. Skipping...")

        except Exception as e:
            print(f" ERROR: Failed to process image {file.name}: {e}")

    return np.array(images), np.array(labels)


# Load dataset
X, y = load_data()

# Check dataset shape
if len(X) == 0:
    raise ValueError(" ERROR: No valid images were loaded!")

print("Dataset Loaded Successfully!")
print(f" Image Data Shape: {X.shape}")
print(f" Label Data Shape: {y.shape}")

# Normalize images (convert pixel values from 0-255 to 0-1)
X = X / 255.0

# Reshape images for CNN input (adding single channel for grayscale)
X = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 1)

# Encode labels to numerical format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)  # Number of unique classes

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(" Data Preprocessing Complete!")
print(f"Training Data Shape: {X_train.shape}, Labels: {y_train.shape}")
print(f"Testing Data Shape: {X_test.shape}, Labels: {y_test.shape}")

# Define CNN Model Architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(" CNN Model Created Successfully!")

# Train CNN Model
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Save Model
model_path = "braille_model.h5"
model.save(model_path)
print(" Model Saved:", model_path)

print(" Model Training Complete & Saved Successfully!")
