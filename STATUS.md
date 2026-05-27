# AgriScan360 — Environment Status (2026-05-10)

Diagnostic-only report. No fixes applied.

## Summary

The app **runs successfully** when launched via `python -m uvicorn webapp.app:app` from the repo root. The README's documented launch command `python webapp/app.py` **fails** with a `ModuleNotFoundError`. One non-fatal model-loading issue is logged at startup but is gracefully recovered (matches a known limitation documented in [README.md:271](README.md#L271)).

## 1. Virtual environment — OK

- `.venv` exists at [.venv/](.venv/)
- Python: **3.11.9** (matches the v2 requirement in [README.md:227](README.md#L227))
- All packages from [requirements.txt](requirements.txt) appear installed (tensorflow 2.13.0, keras 2.13.1, numpy 1.24.3, protobuf 3.20.3, fastapi 0.103.2, uvicorn 0.23.2, opencv-python 4.8.1.78, pillow 10.0.1, pytz 2023.3, requests 2.31.0, scikit-learn 1.3.2, matplotlib 3.7.2, seaborn 0.12.2, python-multipart 0.0.6, jinja2 3.1.2).

## 2. Model artifacts — OK

Both v2 model files are present in [outputs/](outputs/):

| File | Size | Status |
|------|------|--------|
| `outputs/potato_model_v2.keras` | ~31 MB | Present, but **fails to load** at runtime (see §4) |
| `outputs/potato_model_v2.h5` | ~46 MB | Present, **loads successfully** as fallback |
| `outputs/potato_model.keras` | ~31 MB | v1 artifact, not used by v2 app |

## 3. OPENWEATHER_API_KEY — OK

Environment variable is set (32-char value, redacted). `/weather?lat=..&lon=..` returns HTTP 200 against live OpenWeather API.

## 4. App run — BROKEN (as documented), WORKS via uvicorn

### 4a. `python webapp/app.py` — FAILS

```
Traceback (most recent call last):
  File "C:\Users\singh\Downloads\AgriScan360\webapp\app.py", line 20, in <module>
    from src.constants import CLASS_NAMES, NUM_CLASSES
ModuleNotFoundError: No module named 'src'
```

**Root cause**: When Python is launched with a script path (`python webapp/app.py`), `sys.path[0]` is set to `webapp/`, not the repo root. The `from src.constants import ...` line at [webapp/app.py:20](webapp/app.py#L20) cannot resolve `src` because it is a sibling of `webapp/`, not a child. The README in [README.md:208](README.md#L208) and [README.md:249](README.md#L249) instructs users to run exactly this command, so the documented quick-start path is broken.

The `if __name__ == "__main__"` block at [webapp/app.py:659-660](webapp/app.py#L659-L660) calls `uvicorn.run("app:app", ...)` which would also break for the same reason if it were ever reached.

### 4b. `python -m uvicorn webapp.app:app` from repo root — WORKS

Full startup log:

```
Failed to load .keras format: Layer 'normalization' expected 3 variables, but received 0 variables during loading.
  Expected: ['normalization/mean:0', 'normalization/variance:0', 'normalization/count:0']
Falling back to HDF5: outputs/potato_model_v2.h5
Loaded model from outputs/potato_model_v2.h5
Rebuilt model with clean graph (transferred 322/634 weights); inference + Grad-CAM verified.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8002
```

Smoke tests:
- `GET /` → **HTTP 200** (dashboard renders)
- `GET /weather?lat=12.97&lon=77.59` → **HTTP 200** (Bangalore coords; live weather returned)
- `POST /predict` not exercised (would require an uploaded leaf image).

### 4c. `.keras` load failure — known limitation, gracefully handled

The `Layer 'normalization' expected 3 variables` error on the `.keras` file is the **known Keras 2.13 EfficientNet bug** documented in [README.md:271](README.md#L271). The fallback path at [webapp/app.py:374-380](webapp/app.py#L374-L380) loads the `.h5` artifact instead, and the rebuild-and-transfer-weights logic at [webapp/app.py:387-424](webapp/app.py#L387-L424) reconstructs a clean graph (transferring 322/634 weights — the duplicated weights from H5 fine-tuning are dropped, also matching the comment at [webapp/app.py:393-395](webapp/app.py#L393-L395)). This is **expected behavior**, not a regression.

## What's broken / missing

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 1 | `python webapp/app.py` fails with `ModuleNotFoundError: No module named 'src'` — the documented launch command does not work because the repo root is not on `sys.path`. | **High** (blocks the README's quick-start) | [webapp/app.py:20](webapp/app.py#L20), [README.md:208](README.md#L208) |
| 2 | The `__main__` guard at the bottom of `app.py` would also fail (uses `app:app` which assumes cwd is `webapp/`, but then `from src...` would still fail). | Same root cause as #1 | [webapp/app.py:659-660](webapp/app.py#L659-L660) |

## What's NOT broken

- venv, dependencies, Python version
- Model artifacts (both `.keras` and `.h5` present)
- `OPENWEATHER_API_KEY` set and working
- App functionality once launched correctly via `python -m uvicorn webapp.app:app` from the repo root
- `.keras` load failure → `.h5` fallback (documented, working as designed)
