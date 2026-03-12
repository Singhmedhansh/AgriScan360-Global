from datetime import datetime
from pathlib import Path
import random
import tempfile
import time

import cv2
import numpy as np
import pytz
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
SEVERITY_HEATMAP_THRESHOLD = 0.5
HOST_COMMON_NAME = "Potato"
HOST_SCIENTIFIC_NAME = "Solanum tuberosum"

DISEASE_INFO = {
    "Potato Early Blight": {
        "pathogen": "Alternaria solani",
        "description": "Early blight is a fungal disease of potato that produces concentric target-like lesions on leaves.",
    },
    "Potato Late Blight": {
        "pathogen": "Phytophthora infestans",
        "description": "Late blight is a destructive oomycete disease causing rapidly expanding water-soaked lesions.",
    },
    "Potato Healthy": {
        "pathogen": "None detected",
        "description": "Leaf tissue appears healthy with no visible disease symptoms.",
    },
}

TREATMENT_PROTOCOLS = {
    "Potato Late Blight": {
        "immediate": [
            "Remove and destroy infected foliage",
            "Isolate affected plants from healthy stock",
            "Improve canopy airflow to reduce humidity",
        ],
        "chemical": [
            {"fungicide": "Mancozeb 75 WP", "rate": "2.5 kg/ha", "interval": "7 days", "phi": "5 days"},
            {"fungicide": "Chlorothalonil 720 SC", "rate": "1.5 L/ha", "interval": "10 days", "phi": "7 days"},
            {"fungicide": "Azoxystrobin 23 SC", "rate": "1.0 L/ha", "interval": "14 days", "phi": "3 days"},
        ],
        "prevention": [
            "Practice crop rotation with non-solanaceous crops",
            "Use drip irrigation instead of overhead watering",
            "Plant resistant potato varieties",
            "Improve soil drainage",
        ],
    },
    "Potato Early Blight": {
        "immediate": [
            "Remove severely infected leaves",
            "Apply protective fungicide spray",
            "Reduce plant stress with balanced fertilization",
        ],
        "chemical": [
            {"fungicide": "Chlorothalonil", "rate": "1.5 L/ha", "interval": "7 days", "phi": "5 days"},
            {"fungicide": "Azoxystrobin", "rate": "1.0 L/ha", "interval": "10 days", "phi": "3 days"},
        ],
        "prevention": [
            "Maintain proper plant nutrition",
            "Use certified disease-free seed tubers",
            "Practice crop rotation",
        ],
    },
    "Potato Healthy": {
        "immediate": [
            "No disease detected",
            "Continue monitoring crop health",
        ],
        "chemical": [],
        "prevention": [
            "Maintain proper irrigation practices",
            "Ensure adequate sunlight exposure",
            "Inspect crops regularly",
        ],
    },
}

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


def compute_severity_pct(heatmap: np.ndarray) -> float:
    heatmap_arr = np.asarray(heatmap, dtype=np.float32)
    max_val = float(np.max(heatmap_arr))
    if max_val <= 0:
        return 0.0

    heatmap_norm = heatmap_arr / max_val
    infection_mask = heatmap_norm > SEVERITY_HEATMAP_THRESHOLD
    infected_pixels = float(np.sum(infection_mask))
    total_pixels = float(heatmap_norm.size)
    if total_pixels <= 0:
        return 0.0

    return round((infected_pixels / total_pixels) * 100.0, 2)


def classify_severity_stage(severity_pct: float) -> str:
    if severity_pct < 5:
        return "Trace"
    if severity_pct < 20:
        return "Mild"
    if severity_pct < 40:
        return "Moderate"
    if severity_pct < 70:
        return "Severe"
    return "Critical"


def classify_confidence_level(confidence_score: float) -> str:
    if confidence_score < 0.40:
        return "Low"
    if confidence_score < 0.60:
        return "Medium"
    if confidence_score < 0.80:
        return "High"
    return "Very High"


def compute_environmental_data() -> tuple[int, int, int, float]:
    temperature = random.randint(18, 32)
    humidity = random.randint(55, 90)
    wind_speed = random.randint(5, 20)
    sunlight_hours = round(random.uniform(4.0, 9.0), 1)
    return temperature, humidity, wind_speed, sunlight_hours


def compute_risk_score(severity_pct: float, temperature: float, humidity: float) -> int:
    risk_score = 0

    if severity_pct > 60:
        risk_score += 4
    elif severity_pct > 30:
        risk_score += 3
    elif severity_pct > 10:
        risk_score += 2

    if humidity > 80:
        risk_score += 3
    elif humidity > 60:
        risk_score += 2

    if 10 <= temperature <= 24:
        risk_score += 2

    return min(risk_score, 10)


def classify_risk_level(risk_score: int) -> str:
    if risk_score <= 3:
        return "LOW"
    if risk_score <= 6:
        return "MEDIUM"
    return "HIGH"


def compute_spread_probability(risk_score: int) -> int:
    return min(max(risk_score * 10, 0), 100)


def estimate_yield_impact(severity_pct: float) -> str:
    if severity_pct < 10:
        return "0-5%"
    if severity_pct < 30:
        return "5-15%"
    if severity_pct < 60:
        return "15-35%"
    return "35-60%"


def compute_treatment_urgency(risk_score: int) -> str:
    if risk_score < 3:
        return "Low"
    if risk_score <= 6:
        return "Medium"
    return "High"


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
        start_time = time.time()
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
        severity_pct = compute_severity_pct(heatmap)
        severity_stage = classify_severity_stage(severity_pct)

        original_image = leaf_img
        overlay_image = overlay_heatmap(original_image, heatmap)
        Image.fromarray(overlay_image).save(GRADCAM_OUTPUT_PATH)

        class_idx = int(np.argmax(preds))
        confidence = float(preds[0][class_idx])
        confidence_level = classify_confidence_level(confidence)
        disease = class_names[class_idx]
        is_healthy = disease == "Potato Healthy"
        disease_info = DISEASE_INFO.get(disease, DISEASE_INFO["Potato Healthy"])
        treatment_plan = TREATMENT_PROTOCOLS.get(disease, TREATMENT_PROTOCOLS["Potato Healthy"])

        temperature, humidity, wind_speed, sunlight_hours = compute_environmental_data()

        if is_healthy:
            severity_pct = 0
            severity_stage = "None"
            risk_score = 0
            risk_level = "None"
            spread_probability = 0
            treatment_urgency = "None"
            lesion_count = 0
        else:
            risk_score = compute_risk_score(severity_pct, temperature, humidity)
            risk_level = classify_risk_level(risk_score)
            spread_probability = compute_spread_probability(risk_score)
            treatment_urgency = compute_treatment_urgency(risk_score)
            lesion_count = int(severity_pct * 0.5)

        yield_impact = estimate_yield_impact(severity_pct)
        processing_time = round(time.time() - start_time, 2)
        tz = pytz.timezone("Asia/Kolkata")
        timestamp = datetime.now(tz).strftime("%Y-%m-%d • %H:%M %Z")

        print("Raw prediction:", preds)
        print("Predicted class:", disease)

        return JSONResponse(
            {
                "disease_name": disease,
                "confidence_score": round(confidence * 100, 2),
                "confidence_level": confidence_level,
                "severity_pct": severity_pct,
                "severity_stage": severity_stage,
                "timestamp": timestamp,
                "processing_time": processing_time,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "spread_probability": spread_probability,
                "yield_impact": yield_impact,
                "treatment_urgency": treatment_urgency,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "sunlight_hours": sunlight_hours,
                "temperature_c": temperature,
                "humidity_pct": humidity,
                "wind_speed_kmh": wind_speed,
                "host_common": HOST_COMMON_NAME,
                "host_scientific": HOST_SCIENTIFIC_NAME,
                "pathogen": disease_info["pathogen"],
                "description": disease_info["description"],
                "lesion_count": lesion_count,
                "treatment_plan": treatment_plan,
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
