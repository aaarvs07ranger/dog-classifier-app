from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import json
import uvicorn

app = FastAPI()

# --------------------------------------------------
# CORS (set * or restrict to your frontend domain)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load Model + Class Index Mapping
# --------------------------------------------------
model = tf.keras.models.load_model("dog_breed_predictor_model.keras")

with open("class_indices.json") as f:
    class_indices = json.load(f)

# Reverse mapping: index → label
idx_to_class = {v: k for k, v in class_indices.items()}

# ResNet preprocessing
preprocess = tf.keras.applications.resnet50.preprocess_input

# --------------------------------------------------
# Helper: Clean label into nice text
# --------------------------------------------------
def clean_label(label: str) -> str:
    name = label.split("-", 1)[1]          # remove WordNet ID
    name = name.replace("_", " ")          # fix underscores
    return name.title()                    # capitalize nicely

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Load + resize image
    img = Image.open(BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))

    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # IMPORTANT: ResNet preprocessing
    img_array = preprocess(img_array)

    # Predict
    predictions = model.predict(img_array)
    breed_index = int(np.argmax(predictions))
    predicted_label = idx_to_class[breed_index]

    # Clean it up for readable frontend display
    pretty_name = clean_label(predicted_label)

    return {
        "breed_raw": predicted_label,
        "breed": pretty_name,
        "confidence": float(np.max(predictions))
    }

# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
