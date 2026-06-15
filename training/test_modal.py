from pathlib import Path
import modal
data_volume = modal.Volume.from_name('pklot-data', create_if_missing=False)
checkpoint_volume = modal.Volume.from_name('pklot-checkpoints', create_if_missing=False)
image = modal.Image.debian_slim(python_version='3.11').apt_install('libgl1', 'libglib2.0-0').pip_install('ultralytics==8.3.0', 'torch==2.2.2', 'torchvision==0.17.2', 'opencv-python-headless==4.9.0.80', 'PyYAML==6.0.1')
app = modal.App('pklot-yolov8m-obb-test', image=image)
YOLO_DIR = '/data/yolo_pklot'
CHECKPOINT = '/checkpoints/pklot_m_1280/weights/best.pt'
IMGSZ = 1280
BATCH = 14
WORKERS = 4

@app.function(gpu='l40s', timeout=60 * 60, volumes={'/data': data_volume, '/checkpoints': checkpoint_volume}, cpu=8, memory=32768)
def test():
    from ultralytics import YOLO
    import torch
    yaml_path = Path(YOLO_DIR) / 'parking.yaml'
    checkpoint_path = Path(CHECKPOINT)
    if not yaml_path.exists():
        raise FileNotFoundError(f'Dataset yaml not found: {yaml_path}')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    print('=' * 60)
    print('  PKLot YOLOv8m-OBB Test Evaluation')
    print(f'  checkpoint : {checkpoint_path}')
    print(f'  data       : {yaml_path}')
    print(f'  split      : test')
    print(f'  CUDA       : {torch.cuda.is_available()}')
    print(f'  GPU        : {torch.cuda.get_device_name(0)}')
    print('=' * 60)
    model = YOLO(str(checkpoint_path))
    metrics = model.val(data=str(yaml_path), split='test', imgsz=IMGSZ, batch=BATCH, device=0, workers=WORKERS, project='/checkpoints', name='pklot_m_1280_test', exist_ok=True, plots=True, verbose=True)
    checkpoint_volume.commit()
    print('\n' + '=' * 60)
    print('  TEST COMPLETE')
    print(f'  box mAP50    : {metrics.box.map50:.6f}')
    print(f'  box mAP50-95 : {metrics.box.map:.6f}')
    print('  results dir  : /checkpoints/pklot_m_1280_test')
    print('=' * 60)

@app.local_entrypoint()
def main():
    print('Submitting test evaluation job to Modal (L40S, test split)...')
    test.remote()
