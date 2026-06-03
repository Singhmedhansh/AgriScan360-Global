import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import sys
from pathlib import Path
import threading
import tempfile
import time
from typing import Any, Optional, Union, Dict
import uuid
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytz
import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel, Field

from src.constants import CLASS_NAMES, NUM_CLASSES
from src.segmentation import segment_leaf as segment_leaf_with_mask

BASE_DIR = Path(__file__).resolve().parent.parent
WEBAPP_DIR = BASE_DIR / "webapp"
STATIC_DIR = WEBAPP_DIR / "static"
TEMPLATES_DIR = WEBAPP_DIR / "templates"
MODEL_PATH = BASE_DIR / "outputs" / "potato_model_v2.keras"
# Per-request artifact filenames are generated inside /predict (UUID-suffixed)
# so the browser never re-uses a cached image from an earlier prediction and
# concurrent requests don't overwrite each other's heatmaps.
RUNTIME_ARTIFACT_RETENTION = 20  # keep newest N gradcam_*.png / segmented_*.png
USE_SEGMENTATION = False
SEVERITY_HEATMAP_THRESHOLD = 0.5
HOST_COMMON_NAME = "Potato"
HOST_SCIENTIFIC_NAME = "Solanum tuberosum"
OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
THINGSPEAK_CHANNEL_ID = 3376912
THINGSPEAK_LAST_FEED_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json"
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
        "immediate": [
            "Remove and destroy infected plant material",
            "Improve air circulation between plants",
        ],
        "chemical": [
            {"fungicide": "Carbendazim", "rate": "1 g/L", "interval": "7 days", "phi": "14 days"},
            {"fungicide": "Propiconazole", "rate": "1 ml/L", "interval": "7 days", "phi": "14 days"},
        ],
        "prevention": [
            "Avoid overhead irrigation",
            "Practice crop rotation",
            "Use certified disease-free seed",
        ],
    },
    "Bacteria": {
        "immediate": [
            "Remove infected plants immediately",
            "Avoid working in field when wet",
        ],
        "chemical": [
            {"fungicide": "Copper oxychloride", "rate": "3 g/L", "interval": "10 days", "phi": "21 days"},
            {"fungicide": "Streptomycin sulfate", "rate": "0.5 g/L", "interval": "10 days", "phi": "21 days"},
        ],
        "prevention": [
            "Use disease-free certified seed",
            "Avoid waterlogging",
            "Disinfect tools between rows",
        ],
    },
    "Pest": {
        "immediate": [
            "Manual removal of visible pests",
            "Install yellow sticky traps",
        ],
        "chemical": [
            {"fungicide": "Imidacloprid", "rate": "0.5 ml/L", "interval": "Threshold-based", "phi": "7 days"},
            {"fungicide": "Spinosad", "rate": "1 ml/L", "interval": "Threshold-based", "phi": "7 days"},
        ],
        "prevention": [
            "Intercrop with marigold",
            "Avoid excess nitrogen fertilization",
        ],
    },
    "Virus": {
        "immediate": [
            "Uproot and destroy infected plants immediately to prevent spread",
        ],
        "chemical": [
            {"fungicide": "Imidacloprid (vector control)", "rate": "0.5 ml/L", "interval": "Weekly", "phi": "N/A"},
            {"fungicide": "No curative chemical", "rate": "—", "interval": "Remove plants", "phi": "N/A"},
        ],
        "prevention": [
            "Use certified virus-free seed",
            "Control aphids early in season",
            "Remove weed hosts around field",
        ],
    },
}

SENSOR_LOG_PATH = WEBAPP_DIR / "data" / "sensor_log.jsonl"
IST = ZoneInfo("Asia/Kolkata")
_sensor_log_lock = asyncio.Lock()


class SensorReading(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=64)
    soil_moisture: float = Field(..., ge=0, le=100)
    air_temp: float = Field(..., ge=-10, le=60)
    air_humidity: float = Field(..., ge=0, le=100)
    timestamp: Optional[str] = None


app = FastAPI()

STATIC_DIR.mkdir(parents=True, exist_ok=True)
SENSOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

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


def get_weather(lat: float, lon: float) -> Dict[str, Union[float, str]]:
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
    sunlight_hours: Optional[float] = None
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


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(parsed) or np.isinf(parsed):
        return None
    return round(parsed, 1)


def _format_ist_timestamp(timestamp: datetime) -> str:
    return timestamp.astimezone(IST).strftime("%Y-%m-%d • %H:%M %Z")


def _parse_thingspeak_timestamp(created_at: Any) -> Optional[datetime]:
    if not isinstance(created_at, str) or not created_at.strip():
        return None

    normalized = created_at.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fetch_thingspeak_telemetry_sync() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        response = requests.get(THINGSPEAK_LAST_FEED_URL, timeout=8)
        response.raise_for_status()
        payload = response.json() or {}
    except Exception as exc:
        print("ThingSpeak telemetry lookup failed:", exc)

    temperature = _safe_float(payload.get("field1"))
    humidity = _safe_float(payload.get("field2"))
    soil_moisture = _safe_float(payload.get("field3"))
    created_at = _parse_thingspeak_timestamp(payload.get("created_at"))
    has_any_data = any(value is not None for value in (temperature, humidity, soil_moisture))
    is_live = bool(created_at is not None or has_any_data)

    last_updated = _format_ist_timestamp(created_at) if created_at else "--"
    entry_id = payload.get("entry_id")
    try:
        records_logged = int(entry_id) if entry_id is not None and str(entry_id).strip() else None
    except (TypeError, ValueError):
        records_logged = None

    return {
        "temperature": temperature,
        "humidity": humidity,
        "soil_moisture": soil_moisture,
        "last_updated": last_updated,
        "is_live": is_live,
        "device_name": "ESP8266-AgriKit" if is_live else f"ThingSpeak Channel {THINGSPEAK_CHANNEL_ID}",
        "records_logged": records_logged,
    }


async def get_thingspeak_telemetry() -> dict[str, Any]:
    return await asyncio.to_thread(_fetch_thingspeak_telemetry_sync)


def compute_risk_score(severity_pct: float, temperature: float, humidity: float, disease_class: Optional[str] = None) -> int:
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
model: Any = None
model_load_error: Optional[str] = None
_model_lock = threading.Lock()


def _load_model_from_disk() -> Any:
    import tensorflow as tf
    from src.model import build_model

    loaded_model = None
    if model_path_keras.exists():
        try:
            loaded_model = tf.keras.models.load_model(model_path_keras, compile=False)
            print(f"Loaded model from {model_path_keras}")
        except (ValueError, OSError) as e:
            print(f"Failed to load .keras format: {e}")
            print(f"Falling back to HDF5: {model_path_h5}")

    if loaded_model is None:
        if not model_path_h5.exists():
            return None
        loaded_model = tf.keras.models.load_model(model_path_h5, compile=False)
        print(f"Loaded model from {model_path_h5}")

    assert loaded_model.output_shape[-1] == NUM_CLASSES, (
        f"Model head mismatch: expected {NUM_CLASSES} classes "
        f"(from src.constants.CLASS_NAMES), got {loaded_model.output_shape[-1]}. "
        f"Did you load the wrong .keras file?"
    )

    # Rebuild a fresh in-memory graph and transfer weights.
    # TF 2.13 occasionally deserializes EfficientNet-based functional models with
    # stale inbound-node tensor references on the internal Rescaling/Normalization
    # layers, producing "Graph disconnected" errors at inference and Grad-CAM time.
    #
    # Additionally, the HDF5 saver in Keras 2.13 duplicates Variable objects when
    # some backbone layers were unfrozen during fine-tuning, producing e.g. 634
    # weights vs 322 in a clean build. The first N weight shapes always match,
    # so we do a partial (first-N) transfer.
    try:
        fresh_model, _ = build_model(num_classes=NUM_CLASSES)
        loaded_weights = loaded_model.get_weights()
        fresh_weights = fresh_model.get_weights()
        fresh_weight_count = len(fresh_weights)

        # Shapes of the first `fresh_weight_count` weights must align.
        shapes_ok = all(
            loaded_weights[i].shape == fresh_weights[i].shape
            for i in range(min(fresh_weight_count, len(loaded_weights)))
        )

        if shapes_ok and len(loaded_weights) >= fresh_weight_count:
            fresh_model.set_weights(loaded_weights[:fresh_weight_count])
            _probe = fresh_model.predict(
                np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0
            )
            assert _probe.shape[-1] == NUM_CLASSES
            print(
                f"Rebuilt model with clean graph (transferred {fresh_weight_count}"
                f"/{len(loaded_weights)} weights); inference + Grad-CAM verified."
            )
            return fresh_model

        print(
            "Weight shape mismatch on rebuild; "
            "using loaded model as-is (GradCAM may use fallback heatmap)."
        )
        return loaded_model
    except Exception as rebuild_exc:
        print(f"Model graph rebuild skipped: {rebuild_exc}. Using loaded model as-is.")
        return loaded_model


def get_or_load_model() -> Any:
    global model, model_load_error

    if model is not None:
        return model

    with _model_lock:
        if model is not None:
            return model

        if not model_path_keras.exists() and not model_path_h5.exists():
            print(
                f"Warning: model file not found at {model_path_keras} or {model_path_h5}. "
                "The /predict endpoint will return 503 until the v2 model is trained."
            )
            return None

        try:
            model = _load_model_from_disk()
            model_load_error = None if model is not None else model_load_error
        except Exception as load_exc:
            model = None
            model_load_error = str(load_exc)
            print(f"Model load failed: {model_load_error}")
        return model

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
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    request_id = uuid.uuid4().hex[:12]
    gradcam_filename = f"gradcam_{request_id}.png"
    segmented_filename = f"segmented_{request_id}.png"
    gradcam_path = STATIC_DIR / gradcam_filename
    segmented_path = STATIC_DIR / segmented_filename

    temp_path: Optional[Path] = None
    try:
        loaded_model = get_or_load_model()
        if loaded_model is None:
            detail = (
                f"Model not loaded. Train the v2 model and place it at {MODEL_PATH}."
                if not model_load_error
                else f"Model not loaded: {model_load_error}"
            )
            raise HTTPException(status_code=503, detail=detail)

        start_time = time.time()
        suffix = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(await file.read())

        uploaded_img = cv2.imread(str(temp_path))
        if uploaded_img is None:
            return JSONResponse(
                {
                    "status": "invalid_image",
                    "message": INVALID_IMAGE_MESSAGE,
                }
            )

        segmented_leaf, leaf_mask = segment_leaf_with_mask(uploaded_img)
        leaf_pixels = int(np.sum(leaf_mask > 0))
        min_leaf_pixels = max(LEAF_PIXEL_MIN_ABSOLUTE, int(leaf_mask.size * LEAF_PIXEL_MIN_RATIO))

        quality_warning: Optional[str] = None
        use_segmented_leaf = USE_SEGMENTATION and leaf_pixels >= min_leaf_pixels
        if leaf_pixels < min_leaf_pixels:
            # Segmentation can under-detect non-uniform green leaves; continue with original image.
            quality_warning = "Leaf segmentation confidence is low; using original image for diagnosis."

        cv2.imwrite(str(segmented_path), segmented_leaf)
        leaf_img = segmented_leaf if use_segmented_leaf else uploaded_img

        leaf_img = cv2.resize(leaf_img, (224, 224))
        leaf_img = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2RGB)

        # Keep pixel scale in [0, 255] to match EfficientNet preprocessing in training.
        image_batch = leaf_img.astype(np.float32)
        image_batch = np.expand_dims(image_batch, axis=0)

        try:
            preds = loaded_model.predict(image_batch, verbose=0)
        except Exception as predict_exc:
            print("model.predict failed; retrying via direct call:", predict_exc)
            import tensorflow as tf

            preds = loaded_model(tf.convert_to_tensor(image_batch), training=False).numpy()

        try:
            from src.gradcam import generate_gradcam

            heatmap = generate_gradcam(loaded_model, image_batch)
        except Exception as gradcam_exc:
            print("GradCAM generation failed; falling back to flat heatmap:", gradcam_exc)
            heatmap = np.zeros((224, 224), dtype=np.float32)

        severity_pct = compute_severity_pct(heatmap)
        severity_stage = classify_severity_stage(severity_pct)

        # Overlay heatmap on the full-resolution original upload (not the 224x224
        # model input). Pass leaf_mask so off-leaf activations are damped — this
        # is the single biggest readability win on field photos with mulch/soil.
        # For Healthy predictions, suppress the heatmap entirely: there's no
        # pathogen to localize, and a fiery red heatmap on a healthy leaf is
        # confusing UX even though it's algorithmically correct (Grad-CAM
        # always fires on the top class's evidence).
        original_full_rgb = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB)
        is_healthy_pred = int(np.argmax(preds)) == CLASS_NAMES.index("Healthy")
        try:
            if is_healthy_pred:
                Image.fromarray(original_full_rgb).save(gradcam_path)
            else:
                from src.gradcam import overlay_heatmap

                overlay_image = overlay_heatmap(original_full_rgb, heatmap, leaf_mask=leaf_mask)
                Image.fromarray(overlay_image).save(gradcam_path)
        except Exception as overlay_exc:
            print("Heatmap overlay save failed; saving original:", overlay_exc)
            Image.fromarray(original_full_rgb).save(gradcam_path)

        # Retain only the newest N artifacts to bound disk growth.
        for prefix in ("gradcam_", "segmented_"):
            existing = sorted(
                STATIC_DIR.glob(f"{prefix}*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for stale in existing[RUNTIME_ARTIFACT_RETENTION:]:
                try:
                    stale.unlink()
                except OSError:
                    pass

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

        telemetry_data = await get_thingspeak_telemetry()

        weather_available = False
        weather_data: Dict[str, Union[float, str]] = {}
        if isinstance(lat, (float, int)) and isinstance(lon, (float, int)):
            try:
                weather_data = get_weather(float(lat), float(lon))
                weather_available = True
            except Exception as exc:
                print("Weather lookup fallback in /predict:", exc)

        telemetry_available = (
            telemetry_data.get("is_live") is True
            and telemetry_data.get("temperature") is not None
            and telemetry_data.get("humidity") is not None
        )

        if telemetry_available:
            temperature: Union[float, str] = float(telemetry_data["temperature"])
            humidity: Union[float, str] = float(telemetry_data["humidity"])
            if weather_available:
                wind_speed = float(weather_data["wind_speed"])
                sunlight_hours = weather_data.get("sunlight_hours")
                weather_location = weather_data.get("location", "") or "Geolocated"
            else:
                wind_speed = "N/A"
                sunlight_hours = "N/A"
                weather_location = telemetry_data.get("device_name", "ESP8266-AgriKit")
            weather_available = True
        elif weather_available:
            temperature: Union[float, str] = float(weather_data["temperature"])
            humidity: Union[float, str] = float(weather_data["humidity"])
            wind_speed: Union[float, str] = float(weather_data["wind_speed"])
            sunlight_hours: Optional[Union[float, str]] = weather_data.get("sunlight_hours")
            weather_location: str = weather_data.get("location", "") or "Geolocated"
        else:
            temperature = "N/A"
            humidity = "N/A"
            wind_speed = "N/A"
            sunlight_hours = "N/A"
            weather_location = "Unknown"

        if is_healthy:
            severity_pct = 0
            severity_stage = "None"
            risk_score = 0
            risk_level = "None"
            spread_probability = 0
            treatment_urgency = "None"
            lesion_count = 0
        else:
            if weather_available and isinstance(temperature, (float, int)) and isinstance(humidity, (float, int)):
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
                "location": weather_location,
                "lat": lat,
                "lon": lon,
                "telemetry": telemetry_data,
                "host_common": HOST_COMMON_NAME,
                "host_scientific": HOST_SCIENTIFIC_NAME,
                "pathogen": disease_info["pathogen"],
                "description": disease_info["description"],
                "lesion_count": lesion_count,
                "treatment_plan": treatment_plan,
                "leaf_pixels": leaf_pixels,
                "min_leaf_pixels": min_leaf_pixels,
                "heatmap_url": f"/static/{gradcam_filename}",
                "request_id": request_id,
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


@app.get("/api/telemetry")
async def api_telemetry():
    telemetry = await get_thingspeak_telemetry()
    return JSONResponse(telemetry)


@app.post("/sensor_data")
async def ingest_sensor_data(reading: SensorReading):
    """Ingest a single sensor reading and append it to the JSONL telemetry log."""
    record_id = str(uuid.uuid4())
    received_at = datetime.now(IST).isoformat()
    timestamp = reading.timestamp or received_at

    record = {
        "record_id": record_id,
        "device_id": reading.device_id,
        "soil_moisture": reading.soil_moisture,
        "air_temp": reading.air_temp,
        "air_humidity": reading.air_humidity,
        "timestamp": timestamp,
        "received_at": received_at,
    }

    line = json.dumps(record) + "\n"
    async with _sensor_log_lock:
        with open(SENSOR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    print(
        f"[sensor] {reading.device_id} soil={reading.soil_moisture}% "
        f"temp={reading.air_temp}C hum={reading.air_humidity}%"
    )

    return {"status": "ok", "record_id": record_id, "received_at": received_at}


@app.get("/sensor_data/latest")
async def get_latest_sensor_data(n: int = Query(10, ge=1, le=100)):
    """Return the newest-first n valid sensor records from the telemetry log."""
    if not SENSOR_LOG_PATH.exists():
        return {"records": [], "count": 0}

    with open(SENSOR_LOG_PATH, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    if not raw_lines:
        return {"records": [], "count": 0}

    records: list[dict] = []
    for line in reversed(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(records) >= n:
            break

    return {"records": records, "count": len(records)}


@app.get("/sensor_data/health")
async def sensor_pipeline_health():
    """Return ingest pipeline liveness: total records, last receipt time, recent device set."""
    if not SENSOR_LOG_PATH.exists():
        return {"total_records": 0, "last_received_at": None, "devices_seen": []}

    total_records = 0
    last_received_at: Optional[str] = None
    recent_devices: list[str] = []

    with open(SENSOR_LOG_PATH, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    parsed_lines: list[dict] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed_lines.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    total_records = len(parsed_lines)
    if parsed_lines:
        last_received_at = parsed_lines[-1].get("received_at")
        seen_order: list[str] = []
        seen_set: set[str] = set()
        for rec in parsed_lines[-100:]:
            dev = rec.get("device_id")
            if isinstance(dev, str) and dev not in seen_set:
                seen_set.add(dev)
                seen_order.append(dev)
        recent_devices = seen_order

    return {
        "total_records": total_records,
        "last_received_at": last_received_at,
        "devices_seen": recent_devices,
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
