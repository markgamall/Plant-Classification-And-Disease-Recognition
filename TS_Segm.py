import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
from PIL import Image
from transformers import AutoModelForImageClassification
import tensorflow as tf

def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation="relu", 
                               kernel_initializer="he_normal", padding="same")(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation="relu", 
                               kernel_initializer="he_normal", padding="same")(x)
    return x

def upsample_block(inputs, conv_prev, num_filters):
    up = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(inputs)
    concat = tf.keras.layers.concatenate([up, conv_prev])
    conv = conv_block(concat, num_filters)
    return conv


# Function to load images and masks, resizing them to the target dimensions
def load_images_from_dir(image_dir, mask_dir):
    images = []
    masks = []
    
    image_subfolders = os.listdir(image_dir)
    
    for subfolder in image_subfolders:
        subfolder_image_path = os.path.join(image_dir, subfolder)
        subfolder_mask_path = os.path.join(mask_dir, subfolder)

        if os.path.isdir(subfolder_image_path):
            image_files = os.listdir(subfolder_image_path)

            for image_file in image_files:
                image_path = os.path.join(subfolder_image_path, image_file)
                mask_file = find_mask_for_image(image_file, subfolder_mask_path)
                
                if mask_file:
                    mask_path = os.path.join(subfolder_mask_path, mask_file)
                    
                    try:
                        # Load and resize images and masks
                        image = Image.open(image_path).resize((IMG_WIDTH, IMG_HEIGHT))
                        mask = Image.open(mask_path).resize((IMG_WIDTH, IMG_HEIGHT))
                        
                        # Convert to numpy arrays
                        image = np.array(image)
                        mask = np.array(mask)
                        
                        # Append to lists
                        images.append(image)
                        masks.append(mask)
                    except Exception as e:
                        print(f"Error loading image or mask {image_file}: {e}")
                else:
                    print(f"No matching mask found for image {image_file}. Skipping.")
    
    return np.array(images), np.array(masks)

def find_mask_for_image(image_file, mask_dir):
    image_name_without_extension = os.path.splitext(image_file)[0]
    mask_files = os.listdir(mask_dir)
    for mask_file in mask_files:
        if image_name_without_extension in mask_file:
            return mask_file
    return None

custom_objects = {"custom_function": conv_block,"second":upsample_block}

# Load the saved model
loaded_model = tf.keras.models.load_model("/kaggle/input/u-net/other/default/1/model.h5",custom_objects=custom_objects)

val_image_dir = "/kaggle/input/cv25-project-dataset/Project Data/Project Data/Val/images"
val_mask_dir = "/kaggle/input/segmentation/Segmented Data/Val"

val_images, val_masks = load_images_from_dir(val_image_dir, val_mask_dir)
val_images = val_images / 255.0
val_masks = (val_masks > 0).astype(np.float32)

# Evaluate
loss, accuracy = model.evaluate(val_images, val_masks)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")