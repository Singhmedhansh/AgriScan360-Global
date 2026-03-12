from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from src.gradcam import generate_gradcam, overlay_heatmap

BASE_DIR = Path(__file__).resolve().parent.parent
WEBAPP_DIR = BASE_DIR / "webapp"
STATIC_DIR = WEBAPP_DIR / "static"
TEMPLATES_DIR = WEBAPP_DIR / "templates"
MODEL_PATH = BASE_DIR / "outputs" / "potato_model.keras"
GRADCAM_OUTPUT_PATH = STATIC_DIR / "gradcam_result.png"

class_names = [
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
]

app = FastAPI()

STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Serve static assets from /static.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def preprocess_image(image_path: Path) -> np.ndarray:
    """Load an image file and return a normalized model-ready batch tensor."""
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(224, 224),
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    temp_path: Path | None = None
    try:
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(await file.read())

        image_batch = preprocess_image(temp_path)
        preds = model.predict(image_batch)
        heatmap = generate_gradcam(model, image_batch)

        original_image = np.array(Image.open(temp_path).convert("RGB"))
        overlay_image = overlay_heatmap(original_image, heatmap)
        Image.fromarray(overlay_image).save(GRADCAM_OUTPUT_PATH)

        class_idx = int(np.argmax(preds))
        confidence = float(preds[0][class_idx])
        disease = class_names[class_idx]

        print("Raw prediction:", preds)
        print("Predicted class:", disease)

        return JSONResponse(
            {
                "disease_name": disease,
                "confidence_score": round(confidence * 100, 2),
                "severity_pct": 0,
                "heatmap_url": "/static/gradcam_result.png",
            }
        )
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
