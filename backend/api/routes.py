import asyncio
import time
from datetime import datetime, timezone
from typing import Optional
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from backend.models.schemas import ErrorResponse, HealthResponse, MetricsResponse, PredictionResponse, StatusResponse
from backend.utils.image_utils import decode_image_bytes, is_accepted_image_mime
router = APIRouter()
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MODEL_EVALUATION = {'current': {'name': 'YOLOv8m-OBB PKLot 1280', 'trained_on': '2026-05-19', 'checkpoint': 'best.pt', 'imgsz': 1280, 'split': 'held-out test', 'images': 1863, 'instances': 107508, 'precision': 0.999, 'recall': 0.999, 'map50': 0.994825, 'map50_95': 0.99473, 'classes': [{'name': 'available', 'images': 1625, 'instances': 55672, 'precision': 0.999, 'recall': 0.999, 'map50': 0.995, 'map50_95': 0.995}, {'name': 'occupied', 'images': 1461, 'instances': 51836, 'precision': 0.999, 'recall': 0.999, 'map50': 0.995, 'map50_95': 0.995}], 'confusion_matrix': {'labels': ['available', 'occupied'], 'matrix': [[0.999, 0.001], [0.001, 0.999]], 'note': 'Normalized summary derived from the test precision/recall logs.'}, 'sample_predictions': [{'label': 'Dense lot', 'spots': 124, 'occupancy_pct': 66.9, 'mode': 'Enhanced scan'}, {'label': 'Original validation', 'spots': 105812, 'occupancy_pct': None, 'mode': 'Modal val'}, {'label': 'Held-out test', 'spots': 107508, 'occupancy_pct': None, 'mode': 'Modal test'}]}, 'previous': {'name': 'YOLOv8s-OBB PKLot 640', 'trained_on': 'Notebook run, Ultralytics 8.4.44', 'checkpoint': 'replaced best.pt', 'imgsz': 640, 'split': 'held-out test', 'images': 1863, 'instances': 107508, 'precision': 0.9993299365099826, 'recall': 0.9992273125846922, 'map50': 0.9948195176948023, 'map50_95': 0.9934793546955522, 'classes': [{'name': 'available', 'images': 1625, 'instances': 55672, 'precision': 0.999, 'recall': 0.999, 'map50': 0.995, 'map50_95': 0.994}, {'name': 'occupied', 'images': 1461, 'instances': 51836, 'precision': 0.999, 'recall': 0.999, 'map50': 0.995, 'map50_95': 0.993}]}}
_request_count: int = 0
_total_infer_ms: float = 0.0
_last_request_at: Optional[str] = None

def _record_request(inference_ms: float) -> None:
    global _request_count, _total_infer_ms, _last_request_at
    _request_count += 1
    _total_infer_ms += inference_ms
    _last_request_at = datetime.now(timezone.utc).isoformat()

def _avg_inference_ms() -> float:
    if _request_count == 0:
        return 0.0
    return round(_total_infer_ms / _request_count, 2)

def _get_service(request: Request):
    service = getattr(request.app.state, 'inference_service', None)
    if service is None:
        raise HTTPException(status_code=503, detail='Model not loaded. Place best.pt in models/ and restart the server.')
    return service

@router.post('/predict-image', response_model=PredictionResponse, summary='Detect parking spot occupancy from an image', description='Upload a parking lot JPEG/PNG. YOLOv8-OBB detects all spots and classifies each as AVAILABLE or OCCUPIED.', responses={200: {'description': 'Detection successful'}, 422: {'model': ErrorResponse, 'description': 'Invalid or empty image'}, 500: {'description': 'Inference error'}, 503: {'description': 'Model not loaded'}})
async def predict_image(request: Request, image: UploadFile=File(..., description='Parking lot image'), conf: float=0.15, iou: float=0.55, imgsz: int=1600, max_det: int=1000, augment: bool=True):
    if not is_accepted_image_mime(image.content_type or ''):
        raise HTTPException(status_code=422, detail=f"Unsupported image type '{image.content_type}'. Accepted: image/jpeg, image/png, image/bmp, image/webp.")
    content_length = request.headers.get('content-length')
    if content_length:
        try:
            if int(content_length) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail='Image too large. Maximum size is 30 MB.')
        except ValueError:
            raise HTTPException(status_code=400, detail='Invalid content-length header.') from None
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail='Uploaded image is empty.')
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail='Image too large. Maximum size is 30 MB.')
    if not 320 <= imgsz <= 1920:
        raise HTTPException(status_code=422, detail='imgsz must be between 320 and 1920.')
    if not 1 <= max_det <= 2000:
        raise HTTPException(status_code=422, detail='max_det must be between 1 and 2000.')
    try:
        image_bgr: np.ndarray = decode_image_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    service = _get_service(request)
    try:
        result: PredictionResponse = await asyncio.to_thread(service.predict, image_bgr, conf, iou, imgsz, max_det, augment)
    except Exception as exc:
        print(f'[predict-image] {exc}')
        raise HTTPException(status_code=500, detail='Inference failed.') from exc
    _record_request(result.inference_ms)
    return result

@router.get('/status', response_model=StatusResponse, summary='Model and server status')
async def get_status(request: Request):
    service = getattr(request.app.state, 'inference_service', None)
    start_time = getattr(request.app.state, 'start_time', time.time())
    model_loaded = service is not None
    return StatusResponse(model_loaded=model_loaded, device=service.device_name if model_loaded else 'none', uptime_seconds=round(time.time() - start_time, 2))

@router.get('/metrics', response_model=MetricsResponse, summary='Request statistics')
async def get_metrics():
    return MetricsResponse(requests_served=_request_count, avg_inference_ms=_avg_inference_ms(), last_request_at=_last_request_at)

@router.get('/model-evaluation', summary='Model evaluation summary')
async def get_model_evaluation():
    return MODEL_EVALUATION

@router.get('/health', response_model=HealthResponse, summary='Liveness probe')
async def health_check():
    return HealthResponse(status='ok')
