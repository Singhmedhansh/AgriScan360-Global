from datetime import datetime
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import pytz
import requests
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from src.constants import CLASS_NAMES, NUM_CLASSES
from src.gradcam import generate_gradcam, overlay_heatmap
from src.segmentation import segment_leaf as segment_leaf_with_mask

BASE_DIR = Path(__file__).resolve().parent.parent
WEBAPP_DIR = BASE_DIR / "webapp"
STATIC_DIR = WEBAPP_DIR / "static"
TEMPLATES_DIR = WEBAPP_DIR / "templates"
MODEL_PATH = BASE_DIR / "outputs" / "potato_model_v2.keras"
GRADCAM_OUTPUT_PATH = STATIC_DIR / "gradcam_result.png"
SEGMENTED_OUTPUT_PATH = STATIC_DIR / "segmented_leaf.png"
USE_SEGMENTATION = False
SEVERITY_HEATMAP_THRESHOLD = 0.5
HOST_COMMON_NAME = "Potato"
HOST_SCIENTIFIC_NAME = "Solanum tuberosum"
OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
LEAF_PIXEL_MIN_RATIO = 0.03
LEAF_PIXEL_MIN_ABSOLUTE = 2500
INVALID_IMAGE_MESSAGE = "Please upload a clear potato leaf image."

DISEASE_INFO = {
    # ===== v3: re-enable when field-domain blight data is sourced =====
    # "Early_Blight": {
    #     "pathogen": "Alternaria solani",
    #     "description": "Early blight is a fungal disease of potato that produces concentric target-like lesions on leaves.",
    # },
    # "Late_Blight": {
    #     "pathogen": "Phytophthora infestans",
    #     "description": "Late blight is a destructive oomycete disease causing rapidly expanding water-soaked lesions.",
    # },
    # ===================================================================
    "Healthy": {
        "pathogen": "None detected",
        "description": "Leaf tissue appears healthy with no visible disease symptoms.",
    },
    "Fungi": {
        "pathogen": "Fungal pathogen group",
        "description": "Foliar symptoms consistent with a fungal infection. Specific species identification requires laboratory confirmation.",
    },
    "Bacteria": {
        "pathogen": "Bacterial pathogen group",
        "description": "Foliar symptoms consistent with bacterial infection. Specific species identification requires laboratory confirmation.",
    },
    "Pest": {
        "pathogen": "Insect pest damage",
        "description": "Foliar damage consistent with insect feeding or oviposition. Specific pest identification requires visual inspection of the plant.",
    },
    "Virus": {
        "pathogen": "Viral pathogen group",
        "description": "Foliar symptoms consistent with viral infection. Specific virus identification requires laboratory confirmation (e.g., ELISA or PCR).",
    },
}

TREATMENT_PROTOCOLS = {
    # ===== v3: re-enable when field-domain blight data is sourced =====
    # "Late_Blight": {
    #     "immediate": [
    #         "Remove and destroy infected foliage",
    #         "Isolate affected plants from healthy stock",
    #         "Improve canopy airflow to reduce humidity",
    #     ],
    #     "chemical": [
    #         {"fungicide": "Mancozeb 75 WP", "rate": "2.5 kg/ha", "interval": "7 days", "phi": "5 days"},
    #         {"fungicide": "Chlorothalonil 720 SC", "rate": "1.5 L/ha", "interval": "10 days", "phi": "7 days"},
    #         {"fungicide": "Azoxystrobin 23 SC", "rate": "1.0 L/ha", "interval": "14 days", "phi": "3 days"},
    #     ],
    #     "prevention": [
    #         "Practice crop rotation with non-solanaceous crops",
    #         "Use drip irrigation instead of overhead watering",
    #         "Plant resistant potato varieties",
    #         "Improve soil drainage",
    #     ],
    # },
    # "Early_Blight": {
    #     "immediate": [
    #         "Remove severely infected leaves",
    #         "Apply protective fungicide spray",
    #         "Reduce plant stress with balanced fertilization",
    #     ],
    #     "chemical": [
    #         {"fungicide": "Chlorothalonil", "rate": "1.5 L/ha", "interval": "7 days", "phi": "5 days"},
    #         {"fungicide": "Azoxystrobin", "rate": "1.0 L/ha", "interval": "10 days", "phi": "3 days"},
    #     ],
    #     "prevention": [
    #         "Maintain proper plant nutrition",
    #         "Use certified disease-free seed tubers",
    #         "Practice crop rotation",
    #     ],
    # },
    # ===================================================================
    "Healthy": {
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
    "Fungi": {
        "fungicides": ["Carbendazim 1g/L", "Propiconazole 1ml/L"],
        "interval": "Every 7 days",
        "phi": "14 days",
        "immediate": ["Remove and destroy infected plant material", "Improve air circulation between plants"],
        "prevention": ["Avoid overhead irrigation", "Practice crop rotation", "Use certified disease-free seed"]
    },
    "Bacteria": {
        "fungicides": ["Copper oxychloride 3g/L", "Streptomycin sulfate 0.5g/L"],
        "interval": "Every 10 days",
        "phi": "21 days",
        "immediate": ["Remove infected plants immediately", "Avoid working in field when wet"],
        "prevention": ["Use disease-free certified seed", "Avoid waterlogging", "Disinfect tools between rows"]
    },
    "Pest": {
        "fungicides": ["Imidacloprid 0.5ml/L", "Spinosad 1ml/L"],
        "interval": "Inspect weekly, spray only when threshold exceeded",
        "phi": "7 days",
        "immediate": ["Manual removal of visible pests", "Install yellow sticky traps"],
        "prevention": ["Intercrop with marigold", "Avoid excess nitrogen fertilization"]
    },
    "Virus": {
        "fungicides": ["No curative chemical — remove infected plants", "Imidacloprid 0.5ml/L for aphid vector control"],
        "interval": "Monitor weekly",
        "phi": "N/A",
        "immediate": ["Uproot and destroy infected plants immediately to prevent spread"],
        "prevention": ["Use certified virus-free seed", "Control aphids early in season", "Remove weed hosts around field"]
    },
}

app = FastAPI()

STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Serve static assets from /static.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if not os.getenv("OPENWEATHER_API_KEY", "").strip():
    print("Warning: OPENWEATHER_API_KEY is missing. Weather-aware metrics will fall back to base risk.")


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


def get_weather(lat: float, lon: float) -> dict[str, float | str]:
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not configured.")

    response = requests.get(
        OPENWEATHER_API_URL,
        params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
        timeout=8,
    )
    response.raise_for_status()

    payload = response.json()
    temperature = float(payload["main"]["temp"])
    humidity = float(payload["main"]["humidity"])
    wind_speed_ms = float(payload["wind"]["speed"])
    wind_speed_kmh = round(wind_speed_ms * 3.6, 1)

    sunrise = payload.get("sys", {}).get("sunrise")
    sunset = payload.get("sys", {}).get("sunset")
    sunlight_hours: float | None = None
    if isinstance(sunrise, (int, float)) and isinstance(sunset, (int, float)) and sunset > sunrise:
        sunlight_hours = round((float(sunset) - float(sunrise)) / 3600.0, 1)

    return {
        "temperature": round(temperature, 1),
        "humidity": round(humidity, 1),
        "wind_speed": wind_speed_kmh,
        "sunlight_hours": sunlight_hours,
        "source": "openweathermap",
        "location": payload.get("name", ""),
    }


def compute_risk_score(severity_pct: float, temperature: float, humidity: float, disease_class: str | None = None) -> int:
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

    # Late_Blight / Phytophthora window (cool + wet)
    if 10 <= temperature <= 24:
        risk_score += 2

    # Class-specific weather windows
    if disease_class == "Bacteria" and 25 <= temperature <= 35 and humidity > 80:
        risk_score += 2
    elif disease_class == "Virus" and 20 <= temperature <= 30:
        risk_score += 2
    elif disease_class == "Pest" and 20 <= temperature <= 35:
        risk_score += 2
    elif disease_class == "Fungi" and 20 <= temperature <= 28 and humidity > 75:
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


def compute_base_risk_score(severity_pct: float, confidence_score: float) -> int:
    risk_score = 0

    if severity_pct > 60:
        risk_score += 5
    elif severity_pct > 30:
        risk_score += 4
    elif severity_pct > 10:
        risk_score += 3
    elif severity_pct > 3:
        risk_score += 2
    else:
        risk_score += 1

    if confidence_score >= 0.80:
        risk_score += 3
    elif confidence_score >= 0.60:
        risk_score += 2
    elif confidence_score >= 0.40:
        risk_score += 1

    return min(risk_score, 10)


model_path_keras = MODEL_PATH
model_path_h5 = Path(str(MODEL_PATH).replace('.keras', '.h5'))

if not model_path_keras.exists() and not model_path_h5.exists():
    print(
        f"Warning: model file not found at {model_path_keras} or {model_path_h5}. "
        "The /predict endpoint will return 503 until the v2 model is trained."
    )
    model = None
else:
    model = None
    if model_path_keras.exists():
        try:
            model = tf.keras.models.load_model(model_path_keras, compile=False)
            print(f"Loaded model from {model_path_keras}")
        except (ValueError, OSError) as e:
            print(f"Failed to load .keras format: {e}")
            print(f"Falling back to HDF5: {model_path_h5}")
    if model is None:
        if not model_path_h5.exists():
            raise RuntimeError(
                f"Could not load {model_path_keras} and HDF5 fallback {model_path_h5} is missing."
            )
        model = tf.keras.models.load_model(model_path_h5, compile=False)
        print(f"Loaded model from {model_path_h5}")
    assert model.output_shape[-1] == NUM_CLASSES, (
        f"Model head mismatch: expected {NUM_CLASSES} classes "
        f"(from src.constants.CLASS_NAMES), got {model.output_shape[-1]}. "
        f"Did you load the wrong .keras file?"
    )

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/weather")
async def weather(lat: float = Query(...), lon: float = Query(...)):
    try:
        weather_data = get_weather(lat, lon)
        return JSONResponse(weather_data)
    except Exception as exc:
        print("Weather provider error:", exc)
        raise HTTPException(status_code=502, detail="Unable to fetch weather data.") from exc


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Train the v2 model and place it at {MODEL_PATH}.",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    temp_path: Path | None = None
    try:
        start_time = time.time()
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(await file.read())

        uploaded_img = cv2.imread(str(temp_path))
        if uploaded_img is None:
            raise ValueError(f"Could not read uploaded image: {temp_path}")

        segmented_leaf, leaf_mask = segment_leaf_with_mask(uploaded_img)
        leaf_pixels = int(np.sum(leaf_mask > 0))
        min_leaf_pixels = max(LEAF_PIXEL_MIN_ABSOLUTE, int(leaf_mask.size * LEAF_PIXEL_MIN_RATIO))

        quality_warning: str | None = None
        use_segmented_leaf = USE_SEGMENTATION and leaf_pixels >= min_leaf_pixels
        if leaf_pixels < min_leaf_pixels:
            # Segmentation can under-detect non-uniform green leaves; continue with original image.
            quality_warning = "Leaf segmentation confidence is low; using original image for diagnosis."

        cv2.imwrite(str(SEGMENTED_OUTPUT_PATH), segmented_leaf)
        leaf_img = segmented_leaf if use_segmented_leaf else uploaded_img

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

        max_prob = float(np.max(preds[0]))
        if max_prob < 0.5 and quality_warning is None:
            quality_warning = "Model confidence is low for this image; results may be less reliable."

        class_idx = int(np.argmax(preds))
        confidence = float(preds[0][class_idx])
        confidence_level = classify_confidence_level(confidence)
        disease = CLASS_NAMES[class_idx]
        is_healthy = disease == "Healthy"
        disease_info = DISEASE_INFO.get(disease, DISEASE_INFO["Healthy"])
        treatment_plan = TREATMENT_PROTOCOLS.get(disease, TREATMENT_PROTOCOLS["Healthy"])

        weather_available = True
        try:
            weather_data = get_weather(lat, lon)
        except Exception as exc:
            print("Weather lookup fallback in /predict:", exc)
            weather_available = False
            weather_data = {}

        if weather_available:
            temperature: float | str = float(weather_data["temperature"])
            humidity: float | str = float(weather_data["humidity"])
            wind_speed: float | str = float(weather_data["wind_speed"])
            sunlight_hours: float | str | None = weather_data.get("sunlight_hours")
        else:
            temperature = "N/A"
            humidity = "N/A"
            wind_speed = "N/A"
            sunlight_hours = "N/A"

        if is_healthy:
            severity_pct = 0
            severity_stage = "None"
            risk_score = 0
            risk_level = "None"
            spread_probability = 0
            treatment_urgency = "None"
            lesion_count = 0
        else:
            if weather_available:
                risk_score = compute_risk_score(severity_pct, float(temperature), float(humidity), disease)
            else:
                risk_score = compute_base_risk_score(severity_pct, confidence)
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

        response_message_parts: list[str] = []
        if quality_warning:
            response_message_parts.append(quality_warning)
        if not weather_available:
            response_message_parts.append(
                "Diagnosis complete. Weather data unavailable, showing base risk metrics."
            )
        response_message = " ".join(response_message_parts) if response_message_parts else None

        return JSONResponse(
            {
                "disease_name": disease,
                "confidence_score": round(confidence * 100, 2),
                "status": "ok",
                "message": response_message,
                "weather_available": weather_available,
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
                "leaf_pixels": leaf_pixels,
                "min_leaf_pixels": min_leaf_pixels,
                "heatmap_url": "/static/gradcam_result.png",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
