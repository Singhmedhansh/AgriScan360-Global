from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

MODEL_PATH = "outputs/model-final-20251129-185521"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return JSONResponse({
        "class": predicted_class,
        "confidence": confidence
    })

@app.get("/")
def root():
    return {"message": "Potato Disease Classifier API Running!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
