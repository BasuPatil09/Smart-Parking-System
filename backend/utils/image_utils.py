import base64
import cv2
import numpy as np
_JPEG_QUALITY: int = 88
ACCEPTED_MIME_TYPES: frozenset[str] = frozenset(['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/webp'])

def decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError('Could not decode the uploaded file as an image. Ensure the file is a valid JPEG, PNG, BMP, or WEBP file.')
    return image_bgr

def encode_image_to_base64(image_bgr: np.ndarray) -> str:
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
    success, buffer = cv2.imencode('.jpg', image_bgr, encode_params)
    if not success:
        raise RuntimeError('Could not encode annotated image to JPEG.')
    return base64.b64encode(buffer.tobytes()).decode('utf-8')

def is_accepted_image_mime(content_type: str) -> bool:
    mime = content_type.split(';')[0].strip().lower()
    return mime in ACCEPTED_MIME_TYPES
