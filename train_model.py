import tensorflow as tf
from tensorflow.keras import layers, models
import json

# ---------------------------------------
# 1. Enable MIXED PRECISION (faster on T4)
# ---------------------------------------
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ---------------------------------------
# 2. Load Dataset using FAST tf.data API
# ---------------------------------------
train_dir = "archive/images/Images"

batch_size = 32
img_size = (224, 224)

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# Get class names
class_names = train_ds.class_names
num_classes = len(class_names)

# Save class indices
class_indices = {i: name for i, name in enumerate(class_names)}
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f)

# ---------------------------------------------------------
# 3. Add important pipeline speedups: prefetch
# ---------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ---------------------------------------------------------
# 4. Data augmentation (on GPU, very fast)
# ---------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1)
])

# ---------------------------------------------------------
# 5. Build ResNet50 model
# ---------------------------------------------------------
def build_resnet50_model(input_shape=(224, 224, 3), num_classes=120):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)

    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes, activation="softmax", dtype="float32"
    )(x)

    model = models.Model(inputs, outputs)
    return model

model = build_resnet50_model(num_classes=num_classes)

# ---------------------------------------------------------
# 6. Compile (Adam + mixed precision)
# ---------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------------------------
# 7. Train
# ---------------------------------------------------------
history = model.fit(
    train_ds,
    epochs=30,
    validation_data=val_ds
)

# ---------------------------------------------------------
# 8. Save Keras model
# ---------------------------------------------------------
model.save("dog_breed_predictor_model.keras")
print("Model saved successfully to the directory")

# TRAIN THE MODEL IN GOOGLE COLAB/SOME CLOUD GPU SERVICE FOR MUCH FASTER TRAINING TIMES