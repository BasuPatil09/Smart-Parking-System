import time
from typing import List, Optional
import cv2
import numpy as np
try:
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None
from backend.models.schemas import BoundingBox, PredictionResponse, SpotPrediction
from backend.utils.image_utils import encode_image_to_base64
_CLASS_NAMES = {0: 'available', 1: 'occupied'}
_COLOUR_AVAILABLE = (0, 210, 0)
_COLOUR_OCCUPIED = (0, 0, 210)
_COLOUR_TEXT = (255, 255, 255)
DEFAULT_CONF = 0.15
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 1600
DEFAULT_MAX_DET = 1000
ENHANCED_CONF_CAP = 0.15
ENHANCED_CONF_FLOOR = 0.1
ENHANCED_IOU_FLOOR = 0.45
TILE_SIZE = 768
TILE_OVERLAP = 192

def _poly_iou(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if Polygon is not None:
        poly_a = Polygon(points_a.astype(float).tolist())
        poly_b = Polygon(points_b.astype(float).tolist())
        if not poly_a.is_valid or not poly_b.is_valid:
            return 0.0
        inter = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area
        return float(inter / union) if union > 0 else 0.0
    contour_a = points_a.astype(np.float32)
    contour_b = points_b.astype(np.float32)
    area_a = abs(cv2.contourArea(contour_a))
    area_b = abs(cv2.contourArea(contour_b))
    inter, _ = cv2.intersectConvexConvex(contour_a, contour_b)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def _dedupe_detections(detections: list[dict], iou_threshold: float=0.45) -> list[dict]:
    kept: list[dict] = []
    for det in sorted(detections, key=lambda item: item['confidence'], reverse=True):
        duplicate = any((_poly_iou(det['obb_points'], prev['obb_points']) >= iou_threshold for prev in kept))
        if not duplicate:
            kept.append(det)
    return sorted(kept, key=lambda item: (round(float(item['center_y']) / 24), float(item['center_x'])))

def _tile_slices(height: int, width: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    stride = tile_size - overlap
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    if not xs or xs[-1] != max(width - tile_size, 0):
        xs.append(max(width - tile_size, 0))
    if not ys or ys[-1] != max(height - tile_size, 0):
        ys.append(max(height - tile_size, 0))
    return [(x, y, min(x + tile_size, width), min(y + tile_size, height)) for y in ys for x in xs]

def _draw_obb_overlay(image_bgr: np.ndarray, spots_data: list) -> np.ndarray:
    annotated = image_bgr.copy()
    for spot in spots_data:
        colour = _COLOUR_AVAILABLE if spot['status'] == 'available' else _COLOUR_OCCUPIED
        corners = spot['obb_points'].reshape((-1, 1, 2))
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [spot['obb_points']], colour)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
        cv2.polylines(annotated, [corners], True, colour, 2, cv2.LINE_AA)
        pts = spot['obb_points']
        tx = int(pts[:, 0].min())
        ty = max(int(pts[:, 1].min()) - 4, 14)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.38
        thick = 1
        label_id = f"#{spot['id']}"
        label_conf = f"{spot['confidence']:.0%}"
        for i, text in enumerate((label_id, label_conf)):
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            text_y = ty + i * (th + 4)
            cv2.rectangle(annotated, (tx - 1, text_y - th - 2), (tx + tw + 1, text_y + 2), (0, 0, 0), cv2.FILLED)
            cv2.putText(annotated, text, (tx, text_y), font, scale, _COLOUR_TEXT if i == 0 else colour, thick, cv2.LINE_AA)
    total = len(spots_data)
    available = sum((1 for s in spots_data if s['status'] == 'available'))
    occupied = total - available
    lines = [(f'Total    : {total}', _COLOUR_TEXT), (f'Free     : {available}', _COLOUR_AVAILABLE), (f'Occupied : {occupied}', _COLOUR_OCCUPIED)]
    font, scale, thick = (cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pad, line_h = (6, 22)
    max_w = max((cv2.getTextSize(t, font, scale, thick)[0][0] for t, _ in lines))
    h_img, w_img = annotated.shape[:2]
    bx0, by0 = (w_img - max_w - pad * 2 - 8, 8)
    bx1, by1 = (w_img - 8, by0 + line_h * len(lines) + pad * 2)
    ov2 = annotated.copy()
    cv2.rectangle(ov2, (bx0, by0), (bx1, by1), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(ov2, 0.65, annotated, 0.35, 0, annotated)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(annotated, text, (bx0 + pad, by0 + pad + (i + 1) * line_h - 4), font, scale, colour, thick, cv2.LINE_AA)
    return annotated

class InferenceService:

    def __init__(self, model, device_name: str) -> None:
        self._model = model
        self._device_name = device_name

    def _collect_detections(self, image_bgr: np.ndarray, conf: float, iou: float, imgsz: int, max_det: int, x_offset: int=0, y_offset: int=0) -> list[dict]:
        results = self._model.predict(source=image_bgr, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, augment=False, verbose=False, device=self._device_name)
        result = results[0]
        detections: list[dict] = []
        if result.obb is None or len(result.obb) == 0:
            return detections
        xywhr = result.obb.xywhr.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()
        cls_ids = result.obb.cls.cpu().numpy().astype(int)
        obb_points = result.obb.xyxyxyxy.cpu().numpy().astype(np.int32)
        for i in range(len(xywhr)):
            cx, cy, w, h, angle_rad = xywhr[i]
            pts = obb_points[i].copy()
            pts[:, 0] += x_offset
            pts[:, 1] += y_offset
            detections.append({'center_x': float(cx) + x_offset, 'center_y': float(cy) + y_offset, 'width': float(w), 'height': float(h), 'angle': float(np.degrees(angle_rad)), 'confidence': float(confs[i]), 'status': _CLASS_NAMES.get(int(cls_ids[i]), 'available'), 'obb_points': pts})
        return detections

    def predict(self, image_bgr: np.ndarray, conf: float=DEFAULT_CONF, iou: float=DEFAULT_IOU, imgsz: int=DEFAULT_IMGSZ, max_det: int=DEFAULT_MAX_DET, augment: bool=False) -> PredictionResponse:
        t_start = time.perf_counter()
        if augment:
            effective_conf = min(max(conf, ENHANCED_CONF_FLOOR), ENHANCED_CONF_CAP)
            effective_iou = max(iou, ENHANCED_IOU_FLOOR)
        else:
            effective_conf = conf
            effective_iou = iou
        t_infer_start = time.perf_counter()
        h_img, w_img = image_bgr.shape[:2]
        detections = self._collect_detections(image_bgr=image_bgr, conf=effective_conf, iou=effective_iou, imgsz=imgsz, max_det=max_det)
        if augment:
            for x0, y0, x1, y1 in _tile_slices(h_img, w_img, TILE_SIZE, TILE_OVERLAP):
                tile = image_bgr[y0:y1, x0:x1]
                detections.extend(self._collect_detections(image_bgr=tile, conf=effective_conf, iou=effective_iou, imgsz=imgsz, max_det=max_det, x_offset=x0, y_offset=y0))
            detections = _dedupe_detections(detections)
        t_infer_end = time.perf_counter()
        inference_ms = (t_infer_end - t_infer_start) * 1000.0
        spot_predictions: List[SpotPrediction] = []
        spots_data_for_overlay: list = []
        n_available = n_occupied = 0
        for i, det in enumerate(detections, start=1):
            status = det['status']
            confidence = det['confidence']
            if status == 'occupied':
                n_occupied += 1
            else:
                n_available += 1
            bbox = BoundingBox(center_x=round(det['center_x'], 2), center_y=round(det['center_y'], 2), width=round(det['width'], 2), height=round(det['height'], 2), angle=round(det['angle'], 2))
            spot_predictions.append(SpotPrediction(id=i, status=status, confidence=round(confidence, 4), bbox=bbox))
            spots_data_for_overlay.append({'id': i, 'status': status, 'confidence': confidence, 'obb_points': det['obb_points']})
        total_spots = len(spot_predictions)
        occupancy_pct = n_occupied / total_spots * 100.0 if total_spots > 0 else 0.0
        annotated_bgr = _draw_obb_overlay(image_bgr, spots_data_for_overlay)
        b64_image = encode_image_to_base64(annotated_bgr)
        response = PredictionResponse(total_spots=total_spots, available=n_available, occupied=n_occupied, occupancy_pct=round(occupancy_pct, 2), inference_ms=round(inference_ms, 2), spots=spot_predictions, annotated_image_b64=b64_image)
        t_total = (time.perf_counter() - t_start) * 1000.0
        print(f'[inference] spots={total_spots} | available={n_available} | occupied={n_occupied} | occupancy={occupancy_pct:.1f}% | conf={effective_conf:.2f} | iou={effective_iou:.2f} | imgsz={imgsz} | max_det={max_det} | augment={augment} | infer={inference_ms:.1f}ms | total={t_total:.1f}ms')
        return response

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def model_is_loaded(self) -> bool:
        return self._model is not None

def build_inference_service(checkpoint_path: str, device: str='auto') -> InferenceService:
    import os
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError('ultralytics is not installed. Run: pip install ultralytics')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'YOLO checkpoint not found: {checkpoint_path}\nPlace best.pt in the models/ directory.')
    if device == 'auto':
        import torch
        if torch.cuda.is_available():
            device_str = 'cuda'
        elif torch.backends.mps.is_available():
            device_str = 'mps'
        else:
            device_str = 'cpu'
    else:
        device_str = device
    print(f'[inference] Loading YOLO checkpoint: {checkpoint_path}')
    print(f'[inference] Device: {device_str}')
    model = YOLO(checkpoint_path)
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    model.predict(source=dummy, verbose=False, device=device_str)
    service = InferenceService(model=model, device_name=device_str)
    print('[inference] InferenceService ready.')
    return service
