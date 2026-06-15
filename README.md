# Smart Parking System

> YOLOv8-OBB powered parking occupancy detection with a FastAPI backend and React dashboard.

Smart Parking System is a full-stack computer vision project that detects parking spaces from parking-lot images and classifies each detected space as **available** or **occupied**. It includes image upload, batch detection, annotated output previews, model evaluation dashboards, and live API runtime metrics.

---

## Overview

| Area | Details |
| --- | --- |
| Model | YOLOv8m-OBB |
| Dataset | PKLot |
| Backend | FastAPI inference API |
| Frontend | React + Vite dashboard |
| Current mAP50 | 99.48% |
| Current mAP50-95 | 99.47% |

---

## Features

- Upload JPEG, PNG, BMP, or WEBP parking-lot images
- Detect rotated parking spaces with YOLOv8-OBB
- Classify every detected space as available or occupied
- View total spaces, available count, occupied count, occupancy percentage, and inference time
- Display annotated output images with per-space overlays
- Run batch detection across multiple images
- Adjust confidence, IoU threshold, image size, and enhanced scan settings from the UI
- Review model evaluation metrics, confusion matrix, per-class scores, and runtime API metrics
- Keep detection results visible when switching between Detection and Evaluation pages

---

## Model Performance

Evaluation summary for the current checkpoint:

| Metric | Score |
| --- | --- |
| Precision | 99.90% |
| Recall | 99.90% |
| mAP50 | 99.48% |
| mAP50-95 | 99.47% |
| Test Images | 1,863 |
| Instances | 107,508 |

---

## Tech Stack

| Layer | Tools |
| --- | --- |
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| ML / Vision | Ultralytics YOLOv8-OBB, PyTorch, OpenCV, NumPy, Shapely |
| Frontend | React 18, Vite, Axios, Recharts, Tailwind CSS |
| Training | Modal cloud GPU scripts |

---

## Project Structure

```text
.
|-- backend/
|   |-- api/
|   |   `-- routes.py
|   |-- core/
|   |   `-- config.py
|   |-- models/
|   |   `-- schemas.py
|   |-- services/
|   |   `-- inference.py
|   |-- utils/
|   |   `-- image_utils.py
|   `-- main.py
|-- frontend/
|   |-- src/
|   |   |-- components/
|   |   |-- pages/
|   |   |-- services/
|   |   |-- App.jsx
|   |   |-- index.css
|   |   `-- main.jsx
|   |-- .env.example
|   |-- package.json
|   |-- package-lock.json
|   `-- vite.config.js
|-- models/
|   `-- .gitkeep
|-- training/
|   |-- train_modal.py
|   `-- test_modal.py
|-- .env.example
|-- .gitignore
|-- README.md
`-- requirements.txt
```

---

## Model Checkpoint

The trained model checkpoint must be placed locally at:

```text
models/best.pt
```

The checkpoint is not committed to GitHub because it is a large binary file. GitHub blocks normal pushes of files over 100 MB, and this project ignores model checkpoint formats such as `.pt`, `.pth`, `.onnx`, and `.engine`.

If `models/best.pt` is missing, the backend will still start, but prediction requests will return:

```text
503 Model not loaded
```

Recommended storage options for the checkpoint:

- GitHub Release asset
- Cloud storage such as S3, GCS, or Azure Blob
- Git LFS if model versioning is required

---

## Requirements

| Dependency | Version |
| --- | --- |
| Python | 3.10 or newer |
| Node.js | 18 or newer |
| npm | Bundled with Node.js |
| Model file | `models/best.pt` |

GPU is optional. With `DEVICE=auto`, the backend uses CUDA when available, MPS on supported Apple hardware, and CPU otherwise.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/BasuPatil09/Smart-Parking-System.git
cd Smart-Parking-System
```

### 2. Backend Setup

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Create a backend `.env` file:

```bash
copy .env.example .env
```

Backend environment variables:

```env
CHECKPOINT_PATH=models/best.pt
DEVICE=auto
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
HOST=0.0.0.0
PORT=8000
EXPOSE_API_DOCS=true
```

Start the backend:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Backend URLs:

| Service | URL |
| --- | --- |
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

Set `EXPOSE_API_DOCS=false` before deployment if public API docs should be disabled.

### 3. Frontend Setup

Install frontend dependencies:

```bash
cd frontend
npm install
```

Create a frontend `.env` file:

```bash
copy .env.example .env
```

Frontend environment variables:

```env
VITE_API_BASE_URL=/api
```

Start the frontend:

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

The Vite dev server proxies `/api` requests to the backend at `http://localhost:8000`.

---

## API Reference

### `GET /health`

Returns backend liveness status.

### `GET /status`

Returns model load status, active device, and server uptime.

### `GET /metrics`

Returns runtime metrics such as request count, average inference time, and last request timestamp.

### `GET /model-evaluation`

Returns stored model evaluation metrics displayed on the Evaluation page.

### `POST /predict-image`

Accepts a multipart image upload and returns occupancy predictions plus a base64-encoded annotated JPEG.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `image` | file | required | Parking-lot image |
| `conf` | float | `0.15` | Detection confidence threshold |
| `iou` | float | `0.55` | NMS IoU threshold |
| `imgsz` | int | `1600` | YOLO inference image size |
| `max_det` | int | `1000` | Maximum detections returned |
| `augment` | bool | `true` | Enhanced tiled scan for dense lots |

Example:

```bash
curl -X POST "http://localhost:8000/predict-image?conf=0.15&iou=0.55&imgsz=1600&max_det=1000&augment=true" ^
  -F "image=@parking_lot.jpg"
```

Response fields:

```json
{
  "total_spots": 28,
  "available": 2,
  "occupied": 26,
  "occupancy_pct": 92.9,
  "inference_ms": 573.6,
  "spots": [],
  "annotated_image_b64": "<base64-encoded-jpeg>"
}
```

---

## Frontend Build

Build the production frontend:

```bash
cd frontend
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

The compiled output is written to:

```text
frontend/dist/
```

---

## Training

Training and test evaluation scripts are stored in:

```text
training/train_modal.py
training/test_modal.py
```

Run training:

```bash
modal run training/train_modal.py
```

Run test evaluation:

```bash
modal run training/test_modal.py
```

After training, copy the generated checkpoint to:

```text
models/best.pt
```

---

## GitHub Notes

The repository intentionally ignores generated files, local environments, datasets, logs, and model binaries.

Ignored examples:

```text
venv/
__pycache__/
frontend/node_modules/
frontend/dist/
.env
frontend/.env
data/
reports/
logs/
runs/
*.pt
*.pth
*.onnx
*.engine
```

Committed examples:

```text
.env.example
frontend/.env.example
requirements.txt
frontend/package.json
frontend/package-lock.json
models/.gitkeep
source code
README.md
```

Do not commit real secrets, local `.env` files, datasets, generated outputs, or trained checkpoints.

---

## Troubleshooting

### `503 Model not loaded`

Place the checkpoint at:

```text
models/best.pt
```

Then restart the backend.

### Frontend cannot reach backend

Make sure the backend is running on port `8000` and `frontend/.env` contains:

```env
VITE_API_BASE_URL=/api
```

### PowerShell blocks npm

Use:

```powershell
npm.cmd run dev
npm.cmd run build
```

### Large model file cannot be pushed

Keep `best.pt` local, upload it as a GitHub Release asset, store it in cloud storage, or use Git LFS.

### Slow inference on CPU

Large images can be slower on CPU. Use a CUDA-enabled PyTorch install and keep `DEVICE=auto` to run on GPU when available.

---

## License

Add a license before making this repository public.
