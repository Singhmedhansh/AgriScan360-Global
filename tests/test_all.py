"""
AgriScan360 — Comprehensive Test Suite
Tests: imports, constants, segmentation, model build, gradcam, dataset,
       webapp helpers, FastAPI endpoints, frontend template.
"""
import os, sys, json, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

SAMPLE_IMG = ROOT / "samples" / "early_blight.JPG"
RESULTS = []

import io, locale
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def record(name, passed, detail=""):
    RESULTS.append({"test": name, "passed": passed, "detail": detail})
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ─────────────────────────────────────────────
#  1. IMPORT TESTS
# ─────────────────────────────────────────────
section("1. IMPORT TESTS")

for mod_name in [
    "tensorflow", "keras", "numpy", "cv2", "PIL",
    "fastapi", "uvicorn", "requests", "pytz", "jinja2",
    "sklearn", "matplotlib", "seaborn",
]:
    try:
        __import__(mod_name)
        record(f"import {mod_name}", True)
    except Exception as e:
        record(f"import {mod_name}", False, str(e))

# ─────────────────────────────────────────────
#  2. SRC MODULE IMPORTS
# ─────────────────────────────────────────────
section("2. SRC MODULE IMPORTS")

for mod in ["src.constants", "src.dataset", "src.model", "src.segmentation", "src.gradcam", "src.predict", "src.evaluate"]:
    try:
        __import__(mod)
        record(f"import {mod}", True)
    except Exception as e:
        record(f"import {mod}", False, str(e))

# ─────────────────────────────────────────────
#  3. CONSTANTS VALIDATION
# ─────────────────────────────────────────────
section("3. CONSTANTS VALIDATION")

from src.constants import CLASS_NAMES, NUM_CLASSES

try:
    assert isinstance(CLASS_NAMES, list), "CLASS_NAMES not a list"
    assert len(CLASS_NAMES) == 5, f"Expected 5 classes, got {len(CLASS_NAMES)}"
    assert CLASS_NAMES == ["Healthy", "Fungi", "Bacteria", "Pest", "Virus"]
    assert NUM_CLASSES == 5
    record("CLASS_NAMES correct", True, str(CLASS_NAMES))
except AssertionError as e:
    record("CLASS_NAMES correct", False, str(e))

# ─────────────────────────────────────────────
#  4. SEGMENTATION TESTS
# ─────────────────────────────────────────────
section("4. SEGMENTATION TESTS")

import cv2
import numpy as np
from src.segmentation import segment_leaf

# 4a. Empty image
try:
    segment_leaf(np.array([]))
    record("segment_leaf rejects empty", False, "no error raised")
except ValueError:
    record("segment_leaf rejects empty", True)
except Exception as e:
    record("segment_leaf rejects empty", False, str(e))

# 4b. Solid green image (should find contour)
try:
    green = np.zeros((200, 200, 3), dtype=np.uint8)
    green[:] = (0, 128, 0)  # BGR green
    seg, mask = segment_leaf(green)
    assert seg.shape == green.shape, "shape mismatch"
    assert mask.shape == (200, 200), "mask shape"
    assert np.sum(mask > 0) > 0, "mask empty"
    record("segment_leaf green image", True, f"leaf_pixels={np.sum(mask>0)}")
except Exception as e:
    record("segment_leaf green image", False, str(e))

# 4c. Solid red (no green — should fallback)
try:
    red = np.zeros((200, 200, 3), dtype=np.uint8)
    red[:] = (0, 0, 200)
    seg, mask = segment_leaf(red)
    assert np.all(mask == 255), "fallback should produce full mask"
    record("segment_leaf fallback on non-green", True)
except Exception as e:
    record("segment_leaf fallback on non-green", False, str(e))

# 4d. Real sample image
try:
    assert SAMPLE_IMG.exists(), f"sample not found: {SAMPLE_IMG}"
    img = cv2.imread(str(SAMPLE_IMG))
    assert img is not None, "cv2 failed to read sample"
    seg, mask = segment_leaf(img)
    leaf_px = int(np.sum(mask > 0))
    record("segment_leaf sample image", True, f"leaf_pixels={leaf_px}, shape={seg.shape}")
except Exception as e:
    record("segment_leaf sample image", False, str(e))

# ─────────────────────────────────────────────
#  5. MODEL BUILD TEST
# ─────────────────────────────────────────────
section("5. MODEL BUILD TEST")

from src.model import build_model

try:
    model, base_model = build_model(num_classes=5, base="EfficientNetB0", lr=4e-6)
    assert model.output_shape[-1] == 5
    assert not base_model.trainable
    record("build_model(5 classes)", True, f"output_shape={model.output_shape}")
except Exception as e:
    record("build_model(5 classes)", False, str(e))

# Bad base
try:
    build_model(base="ResNet50")
    record("build_model rejects bad base", False, "no error")
except ValueError:
    record("build_model rejects bad base", True)

# ─────────────────────────────────────────────
#  6. INFERENCE + GRAD-CAM TEST (synthetic)
# ─────────────────────────────────────────────
section("6. INFERENCE + GRAD-CAM (synthetic input)")

import tensorflow as tf
from src.gradcam import generate_gradcam, overlay_heatmap, _find_last_conv_layer

try:
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32) * 255
    preds = model.predict(dummy, verbose=0)
    assert preds.shape == (1, 5), f"preds shape {preds.shape}"
    assert abs(float(np.sum(preds[0])) - 1.0) < 0.01, "softmax doesn't sum to 1"
    record("model.predict synthetic", True, f"preds={np.round(preds[0],3)}")
except Exception as e:
    record("model.predict synthetic", False, str(e))

try:
    heatmap = generate_gradcam(model, dummy)
    assert heatmap.ndim == 2
    assert heatmap.shape[0] == 224
    record("generate_gradcam", True, f"heatmap shape={heatmap.shape}")
except Exception as e:
    record("generate_gradcam", False, str(e))

try:
    overlay = overlay_heatmap(dummy[0].astype(np.uint8), heatmap)
    assert overlay.shape == (224, 224, 3)
    record("overlay_heatmap", True, f"shape={overlay.shape}")
except Exception as e:
    record("overlay_heatmap", False, str(e))

try:
    layer = _find_last_conv_layer(model)
    assert isinstance(layer, tf.keras.layers.Conv2D)
    record("_find_last_conv_layer", True, f"layer={layer.name}")
except Exception as e:
    record("_find_last_conv_layer", False, str(e))

# ─────────────────────────────────────────────
#  7. INFERENCE ON SAMPLE IMAGE
# ─────────────────────────────────────────────
section("7. INFERENCE ON SAMPLE IMAGE")

try:
    img = cv2.imread(str(SAMPLE_IMG))
    img_rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
    batch = img_rgb.astype(np.float32)[np.newaxis, ...]
    preds = model.predict(batch, verbose=0)
    idx = int(np.argmax(preds[0]))
    conf = float(preds[0][idx])
    record("predict sample image", True, f"class={CLASS_NAMES[idx]}, conf={conf:.4f}")
    hm = generate_gradcam(model, batch)
    record("gradcam on sample", True, f"heatmap range=[{hm.min():.3f}, {hm.max():.3f}]")
except Exception as e:
    record("predict sample image", False, str(e))

# ─────────────────────────────────────────────
#  8. DATASET PREPROCESSING
# ─────────────────────────────────────────────
section("8. DATASET PREPROCESSING")

from src.dataset import preprocess_image

try:
    processed = preprocess_image(str(SAMPLE_IMG), img_size=224)
    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == tf.float32
    assert float(np.max(processed)) <= 255.1
    record("preprocess_image", True, f"shape={processed.shape}, max={float(np.max(processed)):.1f}")
except Exception as e:
    record("preprocess_image", False, str(e))

# ─────────────────────────────────────────────
#  9. WEBAPP HELPER FUNCTIONS
# ─────────────────────────────────────────────
section("9. WEBAPP HELPER FUNCTIONS")

# Import helpers directly from app module namespace
sys.path.insert(0, str(ROOT / "webapp"))

# We test the pure functions by importing them
from webapp.app import (
    compute_severity_pct, classify_severity_stage, classify_confidence_level,
    compute_risk_score, classify_risk_level, compute_spread_probability,
    estimate_yield_impact, compute_treatment_urgency, compute_base_risk_score,
    DISEASE_INFO, TREATMENT_PROTOCOLS, CLASS_NAMES as APP_CLASS_NAMES,
)

# severity pct
try:
    hm = np.zeros((10, 10), dtype=np.float32)
    assert compute_severity_pct(hm) == 0.0
    hm_full = np.ones((10, 10), dtype=np.float32)
    pct = compute_severity_pct(hm_full)
    assert pct == 100.0, f"full heatmap should be 100%, got {pct}"
    record("compute_severity_pct", True)
except Exception as e:
    record("compute_severity_pct", False, str(e))

# severity stage
try:
    assert classify_severity_stage(2) == "Trace"
    assert classify_severity_stage(10) == "Mild"
    assert classify_severity_stage(30) == "Moderate"
    assert classify_severity_stage(50) == "Severe"
    assert classify_severity_stage(80) == "Critical"
    record("classify_severity_stage", True)
except Exception as e:
    record("classify_severity_stage", False, str(e))

# confidence level
try:
    assert classify_confidence_level(0.3) == "Low"
    assert classify_confidence_level(0.5) == "Medium"
    assert classify_confidence_level(0.7) == "High"
    assert classify_confidence_level(0.9) == "Very High"
    record("classify_confidence_level", True)
except Exception as e:
    record("classify_confidence_level", False, str(e))

# risk score
try:
    r = compute_risk_score(50, 22, 85, "Fungi")
    assert 0 <= r <= 10
    record("compute_risk_score", True, f"score={r}")
except Exception as e:
    record("compute_risk_score", False, str(e))

# risk level
try:
    assert classify_risk_level(2) == "LOW"
    assert classify_risk_level(5) == "MEDIUM"
    assert classify_risk_level(8) == "HIGH"
    record("classify_risk_level", True)
except Exception as e:
    record("classify_risk_level", False, str(e))

# spread probability
try:
    assert compute_spread_probability(5) == 50
    assert compute_spread_probability(0) == 0
    assert compute_spread_probability(10) == 100
    record("compute_spread_probability", True)
except Exception as e:
    record("compute_spread_probability", False, str(e))

# yield impact
try:
    assert estimate_yield_impact(5) == "0-5%"
    assert estimate_yield_impact(20) == "5-15%"
    assert estimate_yield_impact(45) == "15-35%"
    assert estimate_yield_impact(70) == "35-60%"
    record("estimate_yield_impact", True)
except Exception as e:
    record("estimate_yield_impact", False, str(e))

# treatment urgency
try:
    assert compute_treatment_urgency(1) == "Low"
    assert compute_treatment_urgency(5) == "Medium"
    assert compute_treatment_urgency(9) == "High"
    record("compute_treatment_urgency", True)
except Exception as e:
    record("compute_treatment_urgency", False, str(e))

# base risk score
try:
    r = compute_base_risk_score(50, 0.85)
    assert 0 <= r <= 10
    record("compute_base_risk_score", True, f"score={r}")
except Exception as e:
    record("compute_base_risk_score", False, str(e))

# DISEASE_INFO completeness
try:
    for cls in CLASS_NAMES:
        assert cls in DISEASE_INFO, f"{cls} missing from DISEASE_INFO"
        assert "pathogen" in DISEASE_INFO[cls]
        assert "description" in DISEASE_INFO[cls]
    record("DISEASE_INFO completeness", True, f"keys={list(DISEASE_INFO.keys())}")
except Exception as e:
    record("DISEASE_INFO completeness", False, str(e))

# TREATMENT_PROTOCOLS completeness
try:
    for cls in CLASS_NAMES:
        assert cls in TREATMENT_PROTOCOLS, f"{cls} missing from TREATMENT_PROTOCOLS"
    record("TREATMENT_PROTOCOLS completeness", True)
except Exception as e:
    record("TREATMENT_PROTOCOLS completeness", False, str(e))

# ─────────────────────────────────────────────
#  10. FASTAPI APP TESTS (TestClient)
# ─────────────────────────────────────────────
section("10. FASTAPI ENDPOINT TESTS")

try:
    from fastapi.testclient import TestClient
    from webapp.app import app as fastapi_app
    client = TestClient(fastapi_app)

    # GET /
    r = client.get("/")
    record("GET / status", r.status_code == 200, f"status={r.status_code}")
    record("GET / has HTML", "AgriScan" in r.text, f"len={len(r.text)}")

    # GET /weather (no API key → 502)
    r = client.get("/weather?lat=12.9&lon=77.5")
    record("GET /weather (no key → 502)", r.status_code == 502, f"status={r.status_code}")

    # POST /predict without file
    r = client.post("/predict", data={"lat": "12.9", "lon": "77.5"})
    record("POST /predict no file → 422", r.status_code == 422, f"status={r.status_code}")

    # POST /predict with sample image (model is freshly built, not trained, but tests the full pipeline)
    with open(SAMPLE_IMG, "rb") as f:
        r = client.post(
            "/predict",
            files={"file": ("early_blight.JPG", f, "image/jpeg")},
            data={"lat": "12.97", "lon": "77.59"},
        )
    if r.status_code == 200:
        data = r.json()
        record("POST /predict 200", True, f"disease={data.get('disease_name')}, conf={data.get('confidence_score')}")
        # Validate response schema
        required = ["disease_name", "confidence_score", "severity_pct", "severity_stage",
                     "risk_score", "risk_level", "treatment_plan", "heatmap_url"]
        missing = [k for k in required if k not in data]
        record("response schema complete", len(missing) == 0, f"missing={missing}" if missing else "all keys present")
    elif r.status_code == 503:
        record("POST /predict 503 (no model)", True, "model not loaded — expected")
    else:
        record("POST /predict", False, f"status={r.status_code}, body={r.text[:200]}")
except Exception as e:
    record("FastAPI test client", False, traceback.format_exc()[-300:])

# ─────────────────────────────────────────────
#  11. TEMPLATE INTEGRITY
# ─────────────────────────────────────────────
section("11. TEMPLATE INTEGRITY")

try:
    tmpl = (ROOT / "webapp" / "templates" / "index.html").read_text(encoding="utf-8")
    checks = {
        "has <html>": "<html" in tmpl,
        "has tailwindcss": "tailwindcss" in tmpl,
        "has lucide icons": "lucide" in tmpl,
        "has upload zone": 'id="uploadZone"' in tmpl,
        "has runDiagnosis()": "runDiagnosis()" in tmpl,
        "has diagnosisPanel": 'id="diagnosisPanel"' in tmpl,
        "has gradcamImage": 'id="gradcamImage"' in tmpl,
        "has metricsPanel": 'id="metricsPanel"' in tmpl,
        "has pipeline-status": 'id="pipeline-status"' in tmpl,
        "has fetch /predict": "fetch('/predict'" in tmpl or 'fetch("/predict"' in tmpl,
    }
    for name, ok in checks.items():
        record(f"template: {name}", ok)
except Exception as e:
    record("template integrity", False, str(e))

# ─────────────────────────────────────────────
#  12. FILE STRUCTURE
# ─────────────────────────────────────────────
section("12. FILE STRUCTURE")

expected_files = [
    "requirements.txt", "README.md", ".gitignore",
    "src/__init__.py", "src/constants.py", "src/model.py",
    "src/dataset.py", "src/segmentation.py", "src/gradcam.py",
    "src/train.py", "src/evaluate.py", "src/predict.py",
    "webapp/__init__.py", "webapp/app.py",
    "webapp/templates/index.html", "samples/early_blight.JPG",
]
for f in expected_files:
    exists = (ROOT / f).exists()
    record(f"file exists: {f}", exists)

# ─────────────────────────────────────────────
#  13. MODEL ARTIFACTS CHECK
# ─────────────────────────────────────────────
section("13. MODEL ARTIFACTS")

keras_exists = (ROOT / "outputs" / "potato_model_v2.keras").exists()
h5_exists = (ROOT / "outputs" / "potato_model_v2.h5").exists()
record("potato_model_v2.keras present", keras_exists, "MISSING — needs training" if not keras_exists else "")
record("potato_model_v2.h5 present", h5_exists, "MISSING — needs training" if not h5_exists else "")

api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
record("OPENWEATHER_API_KEY set", bool(api_key), "NOT SET" if not api_key else "configured")

# ─────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────
section("SUMMARY")

total = len(RESULTS)
passed = sum(1 for r in RESULTS if r["passed"])
failed = sum(1 for r in RESULTS if not r["passed"])

print(f"\n  Total: {total}  |  Passed: {passed}  |  Failed: {failed}")
print(f"  Pass rate: {passed/total*100:.1f}%\n")

if failed:
    print("  FAILED TESTS:")
    for r in RESULTS:
        if not r["passed"]:
            print(f"    [FAIL] {r['test']}: {r['detail']}")

print()
