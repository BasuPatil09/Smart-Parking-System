from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())
    center_x: float = Field(..., description='X coordinate of the spot centre in pixels.', examples=[319.0])
    center_y: float = Field(..., description='Y coordinate of the spot centre in pixels.', examples=[107.0])
    width: float = Field(..., gt=0, description='Width of the rotated rectangle in pixels.', examples=[55.0])
    height: float = Field(..., gt=0, description='Height of the rotated rectangle in pixels.', examples=[28.0])
    angle: float = Field(..., ge=-180.0, le=180.0, description='Rotation angle in degrees.', examples=[-53.0])

class SpotPrediction(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())
    id: int = Field(..., ge=1, description='Spot identifier assigned from detection order.', examples=[1])
    status: Literal['available', 'occupied'] = Field(..., description='Predicted occupancy status.', examples=['available'])
    confidence: float = Field(..., ge=0.0, le=1.0, description='Predicted class confidence.', examples=[0.94])
    bbox: BoundingBox = Field(..., description='Rotated-rectangle geometry of the spot.')

    @field_validator('confidence')
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)

class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    total_spots: int = Field(..., ge=0, description='Total number of parking spots detected.', examples=[42])
    available: int = Field(..., ge=0, description='Number of spots predicted as available.', examples=[15])
    occupied: int = Field(..., ge=0, description='Number of spots predicted as occupied.', examples=[27])
    occupancy_pct: float = Field(..., ge=0.0, le=100.0, description='Occupied spot percentage.', examples=[64.3])
    inference_ms: float = Field(..., ge=0.0, description='Model inference time in milliseconds.', examples=[112.0])
    spots: List[SpotPrediction] = Field(..., description='Per-spot prediction details.')
    annotated_image_b64: str = Field(..., description='Base64-encoded JPEG with spot overlays.')

    @field_validator('occupancy_pct')
    @classmethod
    def round_occupancy(cls, v: float) -> float:
        return round(v, 2)

    @field_validator('inference_ms')
    @classmethod
    def round_inference_ms(cls, v: float) -> float:
        return round(v, 2)

    @field_validator('available', 'occupied', mode='before')
    @classmethod
    def non_negative(cls, v: int) -> int:
        return max(0, int(v))

class StatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())
    model_loaded: bool = Field(..., description='True if best.pt was loaded successfully.', examples=[True])
    device: str = Field(..., description='Torch device the model is running on.', examples=['cuda'])
    uptime_seconds: float = Field(..., ge=0.0, description='Seconds elapsed since startup.', examples=[142.7])

    @field_validator('uptime_seconds')
    @classmethod
    def round_uptime(cls, v: float) -> float:
        return round(v, 2)

class MetricsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    requests_served: int = Field(..., ge=0, description='Prediction requests handled since startup.', examples=[37])
    avg_inference_ms: float = Field(..., ge=0.0, description='Average inference time in milliseconds.', examples=[108.4])
    last_request_at: Optional[str] = Field(default=None, description='Most recent prediction request time.', examples=['2024-05-01T14:32:11.045Z'])

    @field_validator('avg_inference_ms')
    @classmethod
    def round_avg_ms(cls, v: float) -> float:
        return round(v, 2)

class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Literal['ok'] = Field(default='ok', description='Server liveness status.', examples=['ok'])

class ErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    detail: str = Field(..., description='Human-readable error message.', examples=['Uploaded file is not a valid image.'])
    status_code: int = Field(..., ge=400, le=599, description='HTTP status code.', examples=[422])
