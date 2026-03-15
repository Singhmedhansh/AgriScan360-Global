# Potato_disease: Intelligent Diagnostic & Risk Advisory System

An AI-powered precision agriculture platform built to help potato farmers in India detect foliar disease early, interpret model decisions, and act quickly with weather-aware risk advisory.

This project combines computer vision, explainable AI, and geospatial weather context to reduce crop-loss risk from **Early Blight** and **Late Blight**, while also identifying healthy plants.

## Academic Context

This work is developed as part of the **B.Tech AI/ML curriculum at RV College of Engineering (RVCE)** for EL (Engineering Liaison), Semester 2.

## Why This Matters

Potato disease progression is strongly influenced by local weather conditions (humidity, temperature, airflow). A generic image classifier is often not enough for field decisions. This system extends disease classification into a practical advisory engine by coupling:

- Leaf-level disease detection
- Explainability via Grad-CAM
- Real-time weather context via geolocation + OpenWeather API
- Structured class-specific action plans for intervention

## Technical Stack

- **Language**: Python 3.13
- **Backend/API**: FastAPI + Uvicorn (`webapp/app.py`)
- **ML Framework**: TensorFlow / Keras
- **CNN Backbone**: EfficientNetB0 (`src/model.py`)
- **Computer Vision**: OpenCV (HSV segmentation, image pre/post-processing)
- **Explainable AI**: Grad-CAM (`src/gradcam.py`)
- **Frontend**: Jinja2 template + Tailwind CSS dashboard (`webapp/templates/index.html`)
- **External Data**: OpenWeather API (weather-aware inference context)

## System Pipeline

The diagnosis flow is designed as a practical field pipeline:

1. **HSV-Based Leaf Segmentation** (`src/segmentation.py`)
2. **CNN Inference (EfficientNetB0-based model)** (`src/model.py` + loaded model in `webapp/app.py`)
3. **Grad-CAM Visualization for XAI** (`src/gradcam.py`)
4. **Risk and Advisory Enrichment** (weather + severity + treatment protocol logic in `webapp/app.py`)

### 1) HSV Leaf Segmentation

- Converts uploaded BGR image to HSV color space.
- Applies green-range thresholding (`lower_green` / `upper_green`) to isolate leaf tissue.
- Uses morphological open/close operations to denoise the mask.
- Extracts largest contour as leaf ROI.
- If segmentation fails (no contour), falls back to full image mask.

This gives robust preprocessing under non-lab image conditions and attempts to isolate the biologically relevant region.

### 2) CNN Inference

- Input resized to `224 x 224` and converted to RGB.
- Pixel scale intentionally kept in `[0, 255]` to match training preprocessing.
- Model outputs class probabilities for:
	- `Potato Early Blight`
	- `Potato Healthy`
	- `Potato Late Blight`

Architecture notes (`src/model.py`):

- EfficientNetB0 (`include_top=False`, ImageNet pretrained)
- Feature head: GAP -> BatchNorm -> Dense(256) -> Dropout(0.4) -> Dense(128) -> Dropout(0.3) -> Softmax
- Two-stage training pattern in `src/train.py`:
	- initial frozen-backbone training
	- controlled fine-tuning of final backbone layers

### 3) Grad-CAM Explainability

- Locates last convolution layer automatically (supports nested backbone structure).
- Computes class-specific gradients via `tf.GradientTape`.
- Produces normalized heatmap and resizes to input resolution.
- Overlays heatmap on original image for interpretability.

The output is returned in API response as `heatmap_url` and rendered in dashboard for trust-building and clinical-style diagnosis review.

## `/predict` Endpoint Deep-Dive

Primary API: `POST /predict`

### Request Inputs

- `file`: leaf image (`multipart/form-data`)
- `lat`: user latitude (from browser geolocation)
- `lon`: user longitude (from browser geolocation)

### Internal Decision Flow

1. Reads uploaded image to temporary path.
2. Runs segmentation and computes leaf-pixel quality gates.
3. Falls back to original image if segmentation confidence is low.
4. Runs model prediction and selects top class + confidence.
5. Generates Grad-CAM heatmap and computes severity percentage.
6. Fetches weather using OpenWeather (`temperature`, `humidity`, `wind_speed`, `sunlight_hours`).
7. Computes risk analytics (if non-healthy class):
	 - `risk_score` (0-10)
	 - `risk_level` (LOW / MEDIUM / HIGH)
	 - `spread_probability`
	 - `yield_impact`
	 - `treatment_urgency`
8. Maps predicted class to pathogen metadata and class-specific treatment protocols.
9. Returns structured diagnosis payload for dashboard rendering.

### Weather-Aware Risk Metrics

Risk is not based on class label alone. The service combines:

- Visual severity from Grad-CAM-activated regions
- Real-time humidity and temperature
- Derived spread probability and expected yield impact

This enables field-relevant advisories instead of static classification.

### Class-Specific Action Plans

The API returns `treatment_plan` with sections:

- `immediate`: immediate containment actions
- `chemical`: fungicide table with rate / interval / PHI
- `prevention`: long-term agronomic prevention strategy

For `Potato Healthy`, severity/risk outputs are neutralized and treatment sections adapt accordingly.

## UI/UX Highlights

Frontend: `webapp/templates/index.html`

- Tailwind CSS powered diagnostics dashboard with responsive layout.
- Real-time pipeline status steps (segmentation -> classifier -> Grad-CAM -> risk -> report).
- Dynamic rendering of treatment protocols from API payload:
	- immediate action checklist
	- chemical treatment table
	- prevention cards
- Class-aware and severity-aware badges (diagnosis severity, urgency, risk visibility).
- Grad-CAM panel with side-by-side original specimen and heatmap.
- Weather context panel (temperature, humidity, wind, sunlight) tied to user geolocation.

## Project Structure

```text
.
|-- data/
|-- outputs/
|-- samples/
|-- src/
|   |-- segmentation.py
|   |-- model.py
|   |-- gradcam.py
|   |-- train.py
|   `-- predict.py
|-- webapp/
|   |-- app.py
|   |-- templates/index.html
|   `-- static/
|-- requirements.txt
`-- README.md
```

## Installation (Windows)

### 1. Clone and enter project

```powershell
git clone <your-repo-url>
cd Potato_disease
```

### 2. Create and activate virtual environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure OpenWeather API key

```powershell
setx OPENWEATHER_API_KEY "your_openweather_api_key_here"
```

Close and reopen terminal after `setx`, then verify:

```powershell
echo $env:OPENWEATHER_API_KEY
```

### 5. Run the FastAPI app

```powershell
python webapp\app.py
```

Open: `http://127.0.0.1:8000`

## API Summary

- `GET /`: Dashboard UI
- `GET /weather?lat=..&lon=..`: Weather data fetch
- `POST /predict`: Full diagnosis + XAI + weather-aware advisory

## Model Artifacts

- Trained model path: `outputs/potato_model.keras`
- Grad-CAM output image: `webapp/static/gradcam_result.png`
- Segmented leaf preview: `webapp/static/segmented_leaf.png`

## High-Level Architecture Diagram

```mermaid
graph TD
    %% Core nodes
    User[User or Farmer]
    Frontend[Tailwind Dashboard\nwebapp/templates/index.html]
    Backend[FastAPI Backend\nwebapp/app.py]
    RiskEngine[Risk Advisory and Action Plan]

    %% ML pipeline
    subgraph ML_Pipeline [ML Inference and XAI]
        direction TB
        Segmentation[HSV Leaf Segmentation\nsrc/segmentation.py]
        Model[EfficientNetB0 Model\noutputs/potato_model.keras]
        GradCAM[Grad-CAM XAI\nsrc/gradcam.py]
    end

    %% External context
    subgraph External_API [Geospatial Context]
        OpenWeather[OpenWeather API\nExternal Service]
    end

    %% End-to-end flow
    User -- "1. Upload image and geolocation" --> Frontend
    Frontend -- "2. POST /predict" --> Backend
    Backend -- "3a. Preprocess image" --> Segmentation
    Segmentation -- "ROI and mask" --> Model
    Model -- "Classification logits" --> GradCAM
    GradCAM -- "4a. Diagnosis and heatmap" --> Backend
    Backend -- "3b. Fetch local weather" --> OpenWeather
    OpenWeather -- "4b. Weather data" --> Backend
    Backend -- "5. Enrich diagnosis" --> RiskEngine
    RiskEngine -- "6. Return JSON response" --> Frontend
    Frontend -- "7. Render report" --> User

    %% Styling
    classDef internal fill:#f4ecff,stroke:#333,stroke-width:1px;
    classDef external fill:#e6f0ff,stroke:#1f4e8c,stroke-width:1px,stroke-dasharray: 5 5;
    classDef model fill:#fff6cc,stroke:#6b5900,stroke-width:1px;

    class Backend,Frontend,RiskEngine internal;
    class OpenWeather external;
    class Model model;
```

## Production Readiness Notes

- Explicit invalid-image and low-confidence handling.
- Segmentation quality checks with fallback behavior.
- Timezone-aware diagnosis timestamping (`Asia/Kolkata`).
- Consistent, structured JSON payload for frontend rendering.
- Explainability artifact generation for decision transparency.

## Contributing

Contributions are welcome and encouraged.

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`.
3. Make changes with clear commits.
4. Add or update tests/documentation where relevant.
5. Open a Pull Request with:
	 - problem statement
	 - design summary
	 - screenshots/API samples (if UI/API changes)

Recommended contribution areas:

- Model calibration and confidence reliability
- Additional potato disease classes and localization
- Offline weather fallback strategy
- Evaluation dashboards and experiment tracking
- CI/CD, containerization, and deployment hardening

## Disclaimer

This system provides decision support and not a substitute for expert agronomist consultation. Chemical recommendations should be validated against local agricultural regulations and extension guidelines.

## Acknowledgment

Developed for academic and practical impact in Indian agriculture under the RVCE AI/ML learning track.

Developed by Medhansh Pratap Singh
Github Handle - Singhmedhansh