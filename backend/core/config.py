from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]

def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}

@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    checkpoint_path: str = os.environ.get('CHECKPOINT_PATH', str(PROJECT_ROOT / 'models' / 'best.pt'))
    device: str = os.environ.get('DEVICE', 'auto')
    allowed_origins: tuple[str, ...] = tuple(_split_csv(os.environ.get('ALLOWED_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173')))
    host: str = os.environ.get('HOST', '0.0.0.0')
    port: int = int(os.environ.get('PORT', '8000'))
    expose_api_docs: bool = _bool_env('EXPOSE_API_DOCS', True)
settings = Settings()
