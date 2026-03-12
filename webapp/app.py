from pathlib import Path
import tempfile

import cv2
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
SEGMENTED_OUTPUT_PATH = STATIC_DIR / "segmented_leaf.png"
USE_SEGMENTATION = False

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


def segment_leaf(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image for segmentation: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    leaf = img[y : y + h, x : x + w]

    return leaf


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

        if USE_SEGMENTATION:
            leaf_img = segment_leaf(temp_path)
            cv2.imwrite(str(SEGMENTED_OUTPUT_PATH), leaf_img)
        else:
            leaf_img = cv2.imread(str(temp_path))
            if leaf_img is None:
                raise ValueError(f"Could not read uploaded image: {temp_path}")

        leaf_img = cv2.resize(leaf_img, (224, 224))
        leaf_img = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2RGB)

        # Keep pixel scale in [0, 255] to match EfficientNet preprocessing in training.
        image_batch = leaf_img.astype(np.float32)
        image_batch = np.expand_dims(image_batch, axis=0)

        preds = model.predict(image_batch)
        heatmap = generate_gradcam(model, image_batch)

        original_image = leaf_img
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
