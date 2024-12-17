import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
from PIL import Image
from transformers import AutoModelForImageClassification
from TS_Recognition import *
import tensorflow as tf
import torch
import random

image_size = (224, 224)  # ViT input size
test_dir = '/Users/habibaalaa/Plant-Classification-And-Disease-Recognition/Val/images'  # Treat val as test
batch_size = 64

final_model = "/Users/habibaalaa/Plant-Classification-And-Disease-Recognition/final_model"

# Function to load data
import os
import numpy as np
from PIL import Image

def load_data(data_dir, image_size=(224, 224)):
    images = []
    labels = []
    
    # Sort directories to ensure consistent processing order
    for label in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            main_label = label.split("___")[0]  # Extract main label
            
            # Sort files within each directory
            for img_name in sorted(os.listdir(class_dir)):
                # Skip non-image files like .DS_Store
                if img_name.startswith('.') or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(image_size)
                        images.append(np.array(img))
                        labels.append(main_label)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)

# Load test data
test_images, test_labels = load_data(test_dir)

# Check if any images were loaded
print(f"Number of images loaded: {len(test_images)}")

# If no images are loaded, print a warning and exit
if len(test_images) == 0:
    print("No images were loaded. Check the directory and try again.")
    exit()

# Normalize images to [0, 1]
test_images = test_images / 255.0

# Encode labels
label_encoder = LabelEncoder()
test_labels_encoded = label_encoder.fit_transform(test_labels)

# Retrieve mappings
id2label = {idx: label for idx, label in enumerate(label_encoder.classes_)}
label2id = {label: idx for idx, label in id2label.items()}

test_images = np.array(test_images)  

test_images_tensor = torch.tensor(test_images).permute(0, 3, 1, 2).float()  

# Load the ViT model
model = AutoModelForImageClassification.from_pretrained(final_model)

device = torch.device('cpu')
model = model.to(device)
test_images_tensor = test_images_tensor.to(device)

model.eval()  
with torch.no_grad():  
    outputs = model(test_images_tensor)
    logits = outputs.logits 
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()  

# Calculate accuracy
accuracy = accuracy_score(test_labels_encoded, predictions)
print(f"Accuracy: {accuracy:.2f}")

for idx, label in id2label.items():
    print(f"Class: {label}, ID: {idx}")

print(predictions)