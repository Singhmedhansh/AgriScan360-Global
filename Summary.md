# AgriScan360-Global — Full Project Report (v2.2.2)

> A medium-length, plain-English breakdown of what this project is, how it works, what algorithms it runs, what data it learned from, why we trained it the way we did, and what it's for.

---

## 0. Context (Why this report exists)

The user asked for an end-to-end, plain-English report on the current project covering: algorithms (with flowcharts), dataset + pre/post-processing, model & training technique, broader research context, advantages of our chosen technique, and the model's purpose & use cases. This document collects all of that in one place using only information found in the repo (`d:\AgriScan360-Global`) plus public research as background context for point #4.

---

## 1. What the project IS, in one paragraph

**AgriScan360-Global** is a precision-agriculture web application that lets a farmer (or extension worker) **photograph a plant leaf, upload it through a web dashboard, and instantly receive**:

1. A **disease diagnosis** classified into one of **5 broad pathogen groups** (Healthy, Fungi, Bacteria, Pest, Virus).
2. A **confidence score** for that diagnosis.
3. A **visual heatmap (Grad-CAM)** showing exactly which parts of the leaf the AI looked at when making the decision — so farmers can trust *and verify* the call.
4. A **severity %** (how much of the leaf is affected).
5. A **weather-aware risk score** (0–10) computed by combining the AI's output with live temperature, humidity and wind data from the OpenWeather API for the user's GPS coordinates.
6. A **treatment plan** — immediate actions, recommended chemicals, and prevention steps — tailored to the predicted pathogen group.

The system is built as a **FastAPI backend** + **Tailwind/Jinja2 dashboard**, written in Python 3.11 on top of TensorFlow 2.13 / Keras 2.13.1.

**Built for:** farmers, agritech extension workers, and the RVCE B.Tech AI/ML engineering-liaison curriculum.
**Potential use cases:** field-side disease screening on a phone or tablet, early-warning alerts in farm cooperatives, decision support before spraying, training material for agricultural students, and a foundation for an offline rural-deployment app.

---

## 2. The Algorithms Used (with Flowcharts)

The pipeline chains together **five distinct algorithms**, each handling a different stage:

| Stage | Algorithm | Purpose | Code |
|---|---|---|---|
| 1 | **HSV color thresholding + morphological ops** | Isolate the leaf from the background | [src/segmentation.py](src/segmentation.py) |
| 2 | **EfficientNetB0 CNN** (transfer-learned) | Classify into 5 pathogen groups | [src/model.py](src/model.py) |
| 3 | **Softmax + argmax** | Convert raw network outputs into a class label + confidence | [webapp/app.py:415-431](webapp/app.py#L415-L431) |
| 4 | **Grad-CAM** | Generate a visual "where the AI looked" heatmap | [src/gradcam.py](src/gradcam.py) |
| 5 | **Rule-based weather-aware risk engine** | Combine severity + class + weather → risk score & treatment plan | [webapp/app.py:164-320](webapp/app.py#L164-L320) |

### 2.1 Top-level flow (end-to-end pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER (farmer / browser)                       │
│           uploads leaf image  +  shares GPS lat/lon                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │  POST /predict  (multipart)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FastAPI backend  (webapp/app.py)                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
   ┌─────────────────────┐   STAGE 1: PREPROCESSING
   │ HSV leaf            │   • BGR → HSV
   │ segmentation        │   • green mask  (H 20-95, S 25-255, V 20-255)
   │ (segmentation.py)   │   • morphology (open 5x5, close 9x9)
   └──────────┬──────────┘   • largest-contour leaf cut-out
              │              • quality check  (≥ 2500 px or 3% of image)
              ▼
   ┌─────────────────────┐   STAGE 2: NEURAL CLASSIFIER
   │ resize 224x224      │
   │ BGR → RGB           │
   │ float32 batch       │
   │ EfficientNetB0 CNN  │   ← ImageNet-pretrained, fine-tuned
   │ + custom 5-class    │
   │ classification head │
   └──────────┬──────────┘
              │  raw logits  shape (1, 5)
              ▼
   ┌─────────────────────┐   STAGE 3: POSTPROCESSING
   │ softmax → probs     │
   │ argmax → class idx  │   → "Fungi" / "Bacteria" / etc.
   │ max prob → confidence│  → low-confidence flag if < 0.5
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐   STAGE 4: EXPLAINABILITY
   │ Grad-CAM            │   • last Conv2D layer of EfficientNetB0
   │ (gradcam.py)        │   • d(class score)/d(feature map)
   │                     │   • weighted activation map
   │                     │   • upsample to 224x224
   │                     │   • JET colormap, 70% original + 30% heat
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐   STAGE 5: SEVERITY + WEATHER + ADVISORY
   │ severity %          │   threshold heatmap > 0.5  → infected pixel ratio
   │ OpenWeather API     │   fetch temp / humidity / wind / sun for GPS
   │ rule-based risk     │   class-specific weather windows
   │ engine              │   risk_score 0-10  →  LOW / MEDIUM / HIGH
   │ treatment lookup    │   per-class chemicals + immediate actions
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │ JSON response       │   disease, confidence, severity_pct, risk_score,
   │ + saved heatmap PNG │   risk_level, spread_probability, yield_impact,
   │ rendered on         │   urgency, treatment plan, weather, timestamp
   │ Tailwind dashboard  │
   └─────────────────────┘
```

### 2.2 EfficientNetB0 + custom head — the brain of the system

```
INPUT  224 × 224 × 3 RGB leaf
   │
   ▼
[ EfficientNetB0 backbone ]   ← ImageNet weights
   • MBConv blocks (mobile inverted bottleneck)
   • Squeeze-and-Excitation
   • Swish activations
   • internal Normalization layer (handles [0,255] → [-1,1])
   │
   ▼
[ GlobalAveragePooling2D ]    ← collapses spatial map to a vector
   │
   ▼
[ BatchNormalization ]
   │
   ▼
[ Dense(256, ReLU) ] → [ Dropout(0.4) ]
   │
   ▼
[ Dense(128, ReLU) ] → [ Dropout(0.3) ]
   │
   ▼
[ Dense(5, Softmax) ]         ← Healthy, Fungi, Bacteria, Pest, Virus
   │
   ▼
OUTPUT  vector of 5 probabilities (sum = 1)
```

Source: [src/model.py:8-37](src/model.py#L8-L37)

### 2.3 Two-stage training flow (the "training technique" diagram)

```
            ┌─────────────────────────────────────────────┐
            │            DATA  (plant_dataset_staging)     │
            │   /_train  + /_val   structured by class     │
            └─────────────────┬───────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────────────┐
            │  Augmentation (train only, on-the-fly)       │
            │   flip · rotate ±20° · zoom 15% ·            │
            │   translate 10% · brightness ±20%            │
            └─────────────────┬───────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  STAGE 1 — FROZEN BACKBONE  (20 epochs)                  │
   │  EfficientNetB0 weights  =  FROZEN                       │
   │  Custom head (256→128→5)  =  TRAINABLE                   │
   │  Adam, lr = 3e-4, categorical_crossentropy               │
   │  class_weight='balanced'  ← compensates imbalance        │
   │  EarlyStopping(val_loss, patience=5, restore_best)       │
   └──────────────────────────┬───────────────────────────────┘
                              │  best weights restored
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  STAGE 2 — CONTROLLED FINE-TUNE  (10 epochs)             │
   │  Last 20 layers of EfficientNetB0  =  UNFROZEN           │
   │  All BatchNorm layers  =  STILL FROZEN                   │
   │  Adam, lr = 1e-5  ← much smaller, prevents catastrophic  │
   │                     forgetting of ImageNet features      │
   │  Custom ModelCheckpoint by val_accuracy                  │
   └──────────────────────────┬───────────────────────────────┘
                              ▼
            ┌─────────────────────────────────────────────┐
            │  Save:  potato_model_v2.keras  (preferred)  │
            │     +   potato_model_v2.h5     (fallback)   │
            └─────────────────────────────────────────────┘
```

Source: [src/train.py:40-151](src/train.py#L40-L151)

### 2.4 Grad-CAM (explainability) flow

```
predicted class index k
        │
        ▼
get last Conv2D layer of EfficientNetB0
        │
        ▼
forward pass → feature maps  A   (shape H'×W'×C)
class score y_k for input image
        │
        ▼
gradients  ∂y_k / ∂A          ← what made the score high?
global-average-pool → α_c     ← per-channel importance weight
weighted sum  Σ_c α_c · A_c   → coarse heatmap
ReLU(heatmap)                 → keep only positive evidence
normalize to [0, 1]
upsample bilinearly to 224×224
JET colormap → blend (0.7 leaf + 0.3 heat)
        │
        ▼
saved as static/gradcam_result.png
```

Source: [src/gradcam.py:18-51](src/gradcam.py#L18-L51)

---

## 3. The Training Dataset, Preprocessing, and Postprocessing

### 3.1 The dataset (what the model learned from)

- **Layout:** two flat folders — `plant_dataset_staging/_train` and `plant_dataset_staging/_val` — each containing one subfolder per class. TensorFlow's `image_dataset_from_directory()` infers labels from folder names.
- **5 classes** (canonical, single source of truth in [src/constants.py:4](src/constants.py#L4)):
  `Healthy`, `Fungi`, `Bacteria`, `Pest`, `Virus`.
- **Approximate distribution** (v2 staging, ~1,300 images total):
  - Fungi ≈ 743 (largest)
  - Healthy ≈ 199 (smallest)
  - Bacteria, Pest, Virus ≈ 100–200 each
- **Origin:** Field-domain plant leaf images aggregated and re-grouped from public corpora (PlantVillage / PlantDoc lineage from v1) into pathogen *groups* rather than specific diseases. The class imbalance is handled with `sklearn.utils.compute_class_weight('balanced', …)` — the loss for under-represented classes is upweighted automatically ([src/train.py:59-64](src/train.py#L59-L64)).
- **Validation set:** 397 images, used for early stopping & checkpointing.
- **Why pathogen groups, not species?** v1 used potato-specific labels (Early_Blight, Late_Blight) trained on lab-domain photos with grey concrete backgrounds. When the model met real field images it learned a "shortcut" — predicting blight whenever it saw concrete instead of looking at the leaf. v2 dropped those classes and switched to broader pathogen groups until field-domain blight data is sourced for v3.

### 3.2 Preprocessing (what happens *before* the network sees an image)

#### Training-time preprocessing — [src/dataset.py:8-49](src/dataset.py#L8-L49)
1. **Read & resize** to 224×224 with `image_dataset_from_directory(image_size=(224,224), label_mode='categorical')`.
2. **Cast to float32**, *but keep pixels in [0, 255]* — EfficientNetB0 has its own internal Normalization layer, so manual rescaling would double-normalize.
3. **Augmentation pipeline (train only)** — applied on-the-fly each epoch:
   - `RandomFlip(horizontal)`
   - `RandomRotation(0.055)`  ≈ ±20°
   - `RandomZoom(0.15)`
   - `RandomTranslation(0.1, 0.1)`
   - `RandomBrightness(0.2, value_range=[0,255])`
4. **Batch (32) + prefetch(AUTOTUNE)** for GPU pipelining.

#### Inference-time preprocessing — [webapp/app.py:391-413](webapp/app.py#L391-L413)
1. `cv2.imread()` the uploaded file.
2. **HSV leaf segmentation** ([src/segmentation.py:5-32](src/segmentation.py#L5-L32)):
   BGR→HSV → green mask → open 5×5, close 9×9 → largest contour → cropped leaf + binary mask.
3. **Quality gate:** if leaf-pixel count < max(2500, 3% of image) → fall back to using the full image and flag a warning.
4. **Resize** to 224×224.
5. **BGR → RGB** (OpenCV reads BGR, the model was trained on RGB).
6. **Float32** + `np.expand_dims(axis=0)` → batch of 1.

### 3.3 Postprocessing (what happens *after* the network produces numbers)

[webapp/app.py:415-471](webapp/app.py#L415-L471)
1. `model.predict(batch)` → `(1, 5)` softmax vector.
2. `np.argmax` → class index, mapped through `CLASS_NAMES` to a human label.
3. The probability at that index becomes the **confidence score**.
4. **Confidence buckets:** Low <40%, Medium 40–60%, High 60–80%, Very High ≥80%. If max prob < 0.5 → `quality_warning` flag is set so the dashboard can hint at retaking the photo.
5. **Severity %** is derived from the **Grad-CAM heatmap**, not from the classifier directly: pixels with heatmap intensity > 0.5 are counted as "infected", divided by leaf area → percentage. Then bucketed: Trace <5%, Mild 5–20%, Moderate 20–40%, Severe 40–70%, Critical 70%+.
6. **Weather enrichment:** GPS lat/lon → OpenWeather API → temperature, humidity, wind, sunlight.
7. **Risk engine** (rule-based, not learned):
   - Class-specific weather windows boost the score (e.g. fungal risk peaks at 20–28 °C + humidity > 75%; bacterial at 25–35 °C + > 80% RH; viral at 20–30 °C; pest at 20–35 °C).
   - Final `risk_score` clamped to 0–10 → `LOW` (0-3) / `MEDIUM` (4-6) / `HIGH` (7-10).
   - `spread_probability = risk_score × 10`; `yield_impact` from severity buckets; `treatment_urgency` from risk score.
8. **Treatment lookup** — a static dictionary keyed by class returns chemicals, dosages, intervals, and cultural-control steps ([webapp/app.py:39-149](webapp/app.py#L39-L149)).
9. **Response:** a single JSON payload with ~25 fields plus a saved `gradcam_result.png` URL, rendered into the Tailwind dashboard.

---

## 4. Model Information & Training Technique

| Item | Value |
|---|---|
| **Framework** | TensorFlow 2.13.0 / Keras 2.13.1 |
| **Backbone** | EfficientNetB0, ImageNet-pretrained |
| **Input** | 224 × 224 × 3 RGB, raw [0, 255] pixel range |
| **Output** | 5-class softmax (Healthy, Fungi, Bacteria, Pest, Virus) |
| **Trainable parameters in head** | GAP → BN → Dense(256, ReLU) → Drop(0.4) → Dense(128, ReLU) → Drop(0.3) → Dense(5, Softmax) |
| **Optimizer** | Adam |
| **Loss** | Categorical crossentropy (one-hot labels) |
| **Stage-1 LR / epochs** | 3e-4 / 20, backbone frozen |
| **Stage-2 LR / epochs** | 1e-5 / 10, last 20 layers unfrozen, BN still frozen |
| **Batch size** | 32 |
| **Class imbalance** | `sklearn.compute_class_weight('balanced', …)` |
| **Regularisation** | Dropout 0.4 / 0.3, EarlyStopping (patience 5, restore best) |
| **Checkpointing** | Custom callback by `val_accuracy` (Keras 2.13 ModelCheckpoint serialization bug workaround) |
| **Save format** | `.keras` preferred, `.h5` fallback |
| **Reported performance** | Macro F1 = **0.74** on 397 val images |
| **Per-class F1** | Bacteria 0.91 · Fungi 0.74 · Healthy 0.71 · Pest 0.66 · Virus 0.66 |
| **Known weakness** | Virus recall 51% — visual symptoms overlap with other groups; lab confirmation recommended |

**Training technique in one line:** *Two-stage transfer learning — feature extraction first (frozen backbone + new head), then small-LR fine-tuning of the last 20 backbone layers — with class-weighted loss and live data augmentation.*

Source files: [src/train.py](src/train.py), [src/model.py](src/model.py), [src/dataset.py](src/dataset.py), [src/evaluate.py](src/evaluate.py).

---

## 5. Broader Research Context (web findings)

This is the part the user asked me to "search the entire web" for — public research that contextualises our model and learning style.

### 5.1 EfficientNetB0 in plant-disease classification (2025 literature)
- **Springer / Discover Computing 2025** — EfficientNetB0 fine-tuned through transfer learning on a 38-class, ~87,000-image plant-disease corpus reached **>99% accuracy** alongside DenseNet121 and InceptionResNetV2.
- **Nature Scientific Reports 2025** — fine-tuned EfficientNet-B0 on apple leaves hit **99.69% / 99.78%** test accuracy on the APV / PV datasets respectively.
- **Tomato leaf disease (IJIST 2025)** — EfficientNetB0 + transfer learning + augmentation reached **88.4%** accuracy across 6 disease categories.
- **Lightweight model benchmark (Springer 2026)** — EfficientNetB0 has the **lowest memory consumption and FLOPs** of the leading CNNs, making it the go-to choice for edge / mobile deployment in resource-constrained farms.
- **Plain takeaway for our project:** we picked an architecture that is the de-facto 2025 state-of-the-art for this exact problem class — it is small enough to run on a laptop or a future mobile build, and transfer learning from ImageNet is the standard recipe. Our 0.74 macro F1 is below the 99% headline numbers because (a) we use far less data — ~1,300 images vs 87,000 — and (b) we deliberately classify *broad pathogen groups across many crops*, which is harder than single-crop multi-class.

### 5.2 Two-stage transfer learning (our exact training technique)
- **TensorFlow official tutorial** + multiple 2025 reviews agree on the same recipe we use: **freeze backbone → train head to convergence → unfreeze top layers → fine-tune at much lower LR**, keeping BatchNorm frozen.
- Why everyone does it this way:
  - If you fine-tune from epoch 0, large random gradients from an untrained head **destroy the pretrained ImageNet features** before they can be used.
  - Feature-extraction stage requires **only one forward pass per image** through the frozen backbone → fastest, cheapest, lowest-memory.
  - Frozen backbones **reduce overfitting** when the new dataset is small — exactly our situation (~1,300 images).
  - Unfreezing only the last layers + tiny LR lets the model **adapt high-level features to plant pathology** without catastrophic forgetting.

### 5.3 Grad-CAM in plant-disease XAI
- Grad-CAM is the **most widely used** explainability method in agricultural deep learning (Nature 2025, Springer 2026, BMC Plant Biology 2025). Recent papers have used it on ResNet152 for corn leaf disease, multi-scale attention nets, and lightweight CNNs.
- The agricultural rationale matches ours exactly: *"users can validate whether the AI is indeed focusing on the diseased patch of the leaf and not irrelevant background areas, which boosts trust and helps make better decisions."*
- A documented caveat (arxiv 2504.10527) — Grad-CAM is **post-hoc and qualitative**: it shows correlation with the prediction, not causation. We surface this honestly in the README's known-limitations section.

**Sources:**
- [Crop disease detection using EfficientNetB0 — Springer Discover Computing](https://link.springer.com/article/10.1007/s10791-025-09881-y)
- [Fine-tuned EfficientNet-B0 for apple leaves — Nature Scientific Reports](https://www.nature.com/articles/s41598-025-04479-2)
- [Tomato leaf disease with EfficientNetB0 — IJIST](https://journal.50sea.com/index.php/IJIST/article/view/1574)
- [Lightweight deep-learning benchmarks — Springer Discover IoT](https://link.springer.com/article/10.1007/s43926-026-00310-0)
- [Transfer learning and fine-tuning — TensorFlow official guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Freezing layers in deep learning — Exxact](https://www.exxactcorp.com/blog/deep-learning/guide-to-freezing-layers-in-ai-models)
- [Grad-CAM for corn leaf disease — BMC Plant Biology](https://link.springer.com/article/10.1186/s12870-025-06386-0)
- [Lightweight explainable CNN for plant disease — Nature Scientific Reports](https://www.nature.com/articles/s41598-025-94083-1)
- [XAI techniques for plant disease — arxiv 2504.10527](https://arxiv.org/pdf/2504.10527)

---

## 6. Why our training technique is a good choice (advantages over alternatives)

| Alternative | What it does | Why ours is better for this project |
|---|---|---|
| **Train from scratch** (random init) | Learn every kernel from our ~1,300 images | Would catastrophically overfit; ImageNet already gave us excellent edge / texture / shape detectors for free. |
| **Pure feature extraction** (freeze backbone forever, train head only) | Fast, cheap, very robust on tiny datasets | Backbone never adapts to *plant* texture — leaves have characteristic veining, lesion patterns, chlorosis cues that ImageNet (cats, cars, dogs) doesn't emphasise. We'd cap accuracy too low. |
| **Full fine-tune from epoch 0** | Update every layer with the same LR | Random head gradients destroy pretrained features; needs much more data and compute; high risk of catastrophic forgetting and unstable BN statistics. |
| **Heavier backbone** (ResNet152, ViT-Large) | Higher headline accuracy | 5–10× more parameters, slow on a laptop, infeasible on mobile/edge — directly conflicts with our "field-deployable" goal. |
| **Single-stage fine-tune at one LR** | Simpler to implement | Either too high (destroys backbone) or too low (head doesn't converge). Our two-stage schedule decouples those concerns. |
| **No class weighting** | Default loss | Our dataset is 7× imbalanced (Fungi 743 vs Healthy 199); without weighting the model would just predict "Fungi" most of the time. `compute_class_weight('balanced', …)` directly fixes this in the loss. |
| **No augmentation** | Faster epochs | Tiny dataset → severe overfit. Our flip / rotate / zoom / translate / brightness pipeline is essentially "free extra data". |
| **No segmentation step** | Feed raw image | Cluttered backgrounds (soil, hands, tools) cause shortcut learning — exactly the v1 bug. HSV segmentation forces the network to look at the leaf. |

**Net advantages of our chosen recipe:**
1. **Small-data friendly** — works with 1,300 images where from-scratch CNNs would need 100k+.
2. **Fast to train** — Stage-1 epochs are cheap (only the head's ~200k params update); Stage-2 only updates the top 20 layers.
3. **Stable** — BatchNorm frozen during fine-tune avoids the well-known "BN statistic drift" that ruins fine-tuned models.
4. **Mobile/edge ready** — EfficientNetB0 is the lightest of the high-accuracy CNN family, so future deployment to a phone is realistic.
5. **Honest about uncertainty** — confidence buckets, low-confidence flag, Grad-CAM overlay, and a "lab confirmation recommended" note for Virus all surface model limitations to the user, instead of hiding them.
6. **Explainable by design** — Grad-CAM lets a farmer literally *see* whether the AI looked at the lesion or at a soil shadow. That trust step is what makes the system actually usable in the field.

---

## 7. What the model DOES, what it's BUILT FOR, and use cases (consolidated)

**What it does (in one sentence):** Given a leaf photograph and the user's GPS coordinates, the system tells the user *which broad type of pathogen is most likely affecting the plant, how confident it is, where on the leaf it sees evidence, how serious it looks, how risky the local weather is for spread, and what to do about it.*

**What it is built for:**
- A **field-side decision-support tool** for farmers and agronomy extension workers — fast, visual, weather-aware.
- A **teaching artefact** for the RVCE B.Tech AI/ML programme — every stage (segmentation, CNN, Grad-CAM, rule engine, REST API, frontend) is a separately readable module.
- A **research baseline** — the modular design (constants → dataset → model → train → evaluate → predict → webapp) is easy to swap pieces in or out (e.g. plug in a ViT or a federated-learning client).

**Concrete use cases:**
1. **Pre-spray check** — before mixing chemicals, snap a leaf and confirm whether it's fungal, bacterial, viral, pest, or healthy, so the right product is used.
2. **Cooperative-level early warning** — if multiple uploads in the same village + matching weather window all flag HIGH risk, the cooperative can issue an alert.
3. **Insurance and advisory loops** — a structured JSON record per scan (timestamped, geo-located, with confidence + heatmap) is auditable and can feed crop-insurance claims or government advisory dashboards.
4. **Student lab / hackathon base** — easy to clone, swap class lists, add a new crop, or replace the backbone with a newer model for coursework.
5. **Future v3 extensions** named in the README: restoring Early/Late Blight as field-trained classes, offline weather fallback, mobile build, federated learning across farms.

---

## 8. Verification (how to confirm this report against the repo)

- **Architecture:** open [src/model.py](src/model.py) — confirm `EfficientNetB0(weights='imagenet', include_top=False)` + the head described in §2.2.
- **Two-stage training:** [src/train.py](src/train.py) — confirm `base_model.trainable = False` then later `base_model.trainable = True` with `for layer in base_model.layers[:-20]: layer.trainable = False` and the lower fine-tune LR.
- **Class list:** [src/constants.py](src/constants.py) — confirm `CLASS_NAMES = ["Healthy","Fungi","Bacteria","Pest","Virus"]`.
- **Preprocessing:** [src/dataset.py](src/dataset.py), [src/segmentation.py](src/segmentation.py).
- **Postprocessing & risk engine:** [webapp/app.py:164-471](webapp/app.py#L164-L471).
- **Grad-CAM:** [src/gradcam.py](src/gradcam.py).
- **Run end-to-end:**
  - Train: `python -m src.train --data_dir plant_dataset_staging --img_size 224 --epochs 20`
  - Serve: `python -m webapp.app` → http://127.0.0.1:8000
  - CLI predict: `python -m src.predict --model outputs/potato_model_v2.keras --classes class_names.json --image samples/early_blight.JPG`
  - Reported metrics regenerated by: `python -m src.evaluate` → `outputs/eval/confusion_matrix.png`.

---

## 9. TL;DR

AgriScan360-Global is a **5-class plant-pathogen-group classifier** built on **EfficientNetB0 with two-stage transfer learning**, wrapped in a **FastAPI dashboard** that combines the AI's prediction with **Grad-CAM explainability**, **leaf-level severity estimation**, and a **rule-based weather-aware risk engine** to give farmers an actionable, trustable, field-side disease diagnosis. It is small, modular, explainable, and deliberately optimised for low-data and low-compute deployment — at the cost of being a *broad pathogen-group* classifier rather than a specific-disease classifier (a trade made on purpose to avoid the v1 background-shortcut bug).
