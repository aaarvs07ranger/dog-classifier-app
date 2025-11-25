import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'archive/images/Images'

datagen = ImageDataGenerator()
generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=1,
    shuffle=False
)

with open("class_indices.json", "w") as f:
    json.dump(generator.class_indices, f)

print("class_indices.json created!")
