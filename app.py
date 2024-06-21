from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",  # Add this line with the port where your HTML is served
    "http://127.0.0.1:5500",  # Add the origin of your HTML page
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model paths
MODEL_PATHS = {
    "PlantDisease": "Plant_Disease.h5",
    "PlantQuality": "Apple_quality.h5",
    "OrangeDisease": "Orange_disease.h5",
    "OrangeQuality": "Orrange_Quality.h5",
}

# Load models
MODELS = {name: tf.keras.models.load_model(path) for name, path in MODEL_PATHS.items()}

# Define class names for each model
CLASS_NAMES = {
    "PlantDisease": ["Apple_scab", "Black_rot", "Cedar_apple_rust"," healthy"],
    "PlantQuality": ["Bad Quality_Fruits", "Good Quality_Fruits", "Mixed Qualit_Fruits"],
    "OrangeDisease": ["blackspot", "canker", "fresh","grenning"],
    "OrangeQuality": ["Good Quality", "Orange Rotten" ],
    # Add class names for your additional models
}       


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.get("/load_model/{model_name}")
async def load_model(model_name: str):
    # Check if the requested model_name exists
    if model_name in MODELS:
        model_config = {"url": model_name}
        return model_config
    else:
        return {"error": f"Model {model_name} not found"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    # Check if the requested model_name exists
    if model_name not in MODELS:
        return {"error": f"Model {model_name} not found"}

    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    model = MODELS[model_name]
    class_names = CLASS_NAMES[model_name]

    predictions = model.predict(img_batch)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
