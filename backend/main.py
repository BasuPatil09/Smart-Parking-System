import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.core.config import settings
if str(settings.project_root) not in sys.path:
    sys.path.insert(0, str(settings.project_root))
from backend.api.routes import router
from backend.services.inference import build_inference_service
CHECKPOINT_PATH: str = settings.checkpoint_path
ALLOWED_ORIGINS: list[str] = list(settings.allowed_origins)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    app.state.inference_service = None
    print('=' * 60)
    print('  Smart Parking Detection System - YOLO-OBB Backend')
    print('=' * 60)
    print(f'  Checkpoint : {CHECKPOINT_PATH}')
    try:
        service = build_inference_service(checkpoint_path=CHECKPOINT_PATH, device=settings.device)
        app.state.inference_service = service
        print('  Model      : loaded successfully')
        print(f'  Device     : {service.device_name}')
    except FileNotFoundError as exc:
        print(f'\n  [WARNING] {exc}')
        print('  Server starting without model.\n  Place best.pt in models/ and restart.\n  POST /predict-image will return HTTP 503 until then.')
    except Exception as exc:
        print(f'\n  [ERROR] {exc}')
    print('=' * 60)
    if settings.expose_api_docs:
        print('  Docs  : http://localhost:8000/docs')
    print('  Health: http://localhost:8000/health')
    print('=' * 60)
    yield
    uptime = time.time() - app.state.start_time
    print(f'\n[main] Server shutdown after {uptime:.1f}s uptime.')
app = FastAPI(title='Smart Parking Detection API - YOLO-OBB', description='YOLOv8-OBB powered parking spot occupancy detector. Upload a parking lot image to get per-spot AVAILABLE / OCCUPIED predictions with rotated bounding box overlays.', version='2.0.0', docs_url='/docs' if settings.expose_api_docs else None, redoc_url='/redoc' if settings.expose_api_docs else None, openapi_url='/openapi.json' if settings.expose_api_docs else None, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False, allow_methods=['*'], allow_headers=['*'])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f'[error] {request.method} {request.url.path}: {exc}')
    return JSONResponse(status_code=500, content={'detail': 'Unexpected server error.', 'status_code': 500})
app.include_router(router)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend.main:app', host=settings.host, port=settings.port, reload=True, reload_dirs=[str(settings.project_root)])
