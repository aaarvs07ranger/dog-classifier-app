import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# -------------------------
# 1. Load your trained model
# -------------------------
model = tf.keras.models.load_model('dog_breed_predictor_model.keras')

# -------------------------
# 2. Rebuild class indices from your dataset folder
# -------------------------
train_dir = 'archive/images/Images'   # same folder used during training

temp_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
temp_generator = temp_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# class_indices: {"breed_name": class_number}
class_indices = temp_generator.class_indices

# Reverse mapping: class_number → breed_name
idx_to_class = {v: k for k, v in class_indices.items()}

# -------------------------
# 3. ResNet50 preprocessing
# -------------------------
preprocess = tf.keras.applications.resnet50.preprocess_input

def predict_breed(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess(img_array)   # VERY IMPORTANT for ResNet50

    predictions = model.predict(img_array)
    breed_index = np.argmax(predictions)
    
    return idx_to_class[breed_index]

def clean_label(label):
    # Remove the WordNet ID
    name = label.split("-", 1)[1]
    # Replace underscores with spaces
    name = name.replace("_", " ")
    # Capitalize each word
    return name.title()

# Example usage
test_img = input("Enter path to image: ")
print("Predicted breed:", clean_label(predict_breed(test_img)))
