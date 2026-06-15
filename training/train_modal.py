import modal
data_volume = modal.Volume.from_name('pklot-data', create_if_missing=True)
checkpoint_volume = modal.Volume.from_name('pklot-checkpoints', create_if_missing=True)
image = modal.Image.debian_slim(python_version='3.11').apt_install('libgl1', 'libglib2.0-0').pip_install('ultralytics==8.3.0', 'torch==2.2.2', 'torchvision==0.17.2', 'opencv-python-headless==4.9.0.80', 'kaggle==1.6.17', 'PyYAML==6.0.1')
app = modal.App('pklot-yolov8m-obb-retrain', image=image)
KAGGLE_DATASET = 'blanderbuss/parking-lot-dataset'
RAW_DIR = '/data/pklot_raw'
YOLO_DIR = '/data/yolo_pklot'
LABELS_ZIP = '/data/yolo_labels_only.zip'
OUTPUT_DIR = '/checkpoints'
GPU = 'l40s'
IMGSZ = 1280
EPOCHS = 100
BATCH = 14
WORKERS = 4
MODEL_BASE = 'yolov8m-obb.pt'

@app.function(gpu=GPU, timeout=60 * 60 * 6, secrets=[modal.Secret.from_name('kaggle-secret')], volumes={'/data': data_volume, '/checkpoints': checkpoint_volume}, cpu=8, memory=32768)
def train():
    import json
    import os
    import shutil
    import zipfile
    import subprocess
    from pathlib import Path
    from ultralytics import YOLO
    print('=' * 60)
    print('  PKLot YOLOv8m-OBB Retraining')
    print(f'  GPU     : {GPU.upper()}')
    print(f'  imgsz   : {IMGSZ}')
    print(f'  epochs  : {EPOCHS}')
    print(f'  batch   : {BATCH}')
    print('=' * 60)
    kaggle_username = os.environ.get('KAGGLE_USERNAME') or os.environ.get('username')
    kaggle_key = os.environ.get('KAGGLE_KEY') or os.environ.get('key')
    if not kaggle_username or not kaggle_key:
        raise RuntimeError("Modal secret 'kaggle-secret' must contain KAGGLE_USERNAME and KAGGLE_KEY. Create/update it with: modal secret create kaggle-secret KAGGLE_USERNAME=<your-kaggle-username> KAGGLE_KEY=<your-kaggle-api-key> --force")
    kaggle_dir = Path('/root/.config/kaggle')
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / 'kaggle.json'
    kaggle_json.write_text(json.dumps({'username': kaggle_username, 'key': kaggle_key}))
    kaggle_json.chmod(384)
    print('\n[setup] Kaggle credentials configured.')
    raw_dir = Path(RAW_DIR)
    zip_path = raw_dir / 'parking-lot-dataset.zip'
    if zip_path.exists() and zip_path.stat().st_size > 100000000.0:
        print(f'[setup] Dataset zip already in volume ({zip_path.stat().st_size / 1000000000.0:.1f} GB), skipping download.')
    else:
        print(f'\n[setup] Downloading PKLot from Kaggle (~10 GB)...')
        raw_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(['kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET, '-p', str(raw_dir)], capture_output=False, check=True)
        print(f'[setup] Download complete. Size: {zip_path.stat().st_size / 1000000000.0:.2f} GB')
    extract_marker = raw_dir / '.extracted'
    if extract_marker.exists():
        print('[setup] Images already extracted, skipping.')
    else:
        print('\n[setup] Extracting full-lot JPGs from zip...')
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = [m for m in z.namelist() if '/PKLot/PKLot/' in m and m.endswith('.jpg')]
            print(f'  Found {len(members):,} JPG files in zip')
            for i, m in enumerate(members):
                z.extract(m, str(raw_dir))
                if i % 2000 == 0:
                    print(f'  {i:,}/{len(members):,} extracted...')
        extract_marker.touch()
        data_volume.commit()
        print(f'[setup] Extraction done. Total: {len(members):,} images')
    print('\n[setup] Indexing images...')
    jpg_index: dict[str, Path] = {}
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith('.jpg'):
                jpg_index[Path(f).stem] = Path(root) / f
    print(f'[setup] Indexed {len(jpg_index):,} images')
    if len(jpg_index) == 0:
        raise RuntimeError('No JPG files found after extraction. Check dataset zip.')
    labels_zip = Path(LABELS_ZIP)
    yolo_dir = Path(YOLO_DIR)
    if not labels_zip.exists():
        raise FileNotFoundError(f'\n[ERROR] Labels zip not found at {LABELS_ZIP}\nUpload it from your local machine with:\n  modal volume put pklot-data ./yolo_labels_only.zip /yolo_labels_only.zip\nThis is the 23.9 MB labels zip from your Colab notebook.')
    labels_marker = yolo_dir / '.labels_extracted'
    if labels_marker.exists():
        print('[setup] Labels already extracted, skipping.')
    else:
        print(f'\n[setup] Extracting YOLO OBB labels...')
        yolo_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(labels_zip, 'r') as z:
            z.extractall(str(yolo_dir))
        labels_marker.touch()
        print(f'[setup] Labels extracted.')
        for split in ('train', 'val', 'test'):
            n = len(list((yolo_dir / 'labels' / split).glob('*.txt')))
            print(f'  {split:5s} → {n:,} label files')
    copy_marker = yolo_dir / '.images_copied'
    if copy_marker.exists():
        print('[setup] Images already copied into YOLO structure, skipping.')
    else:
        print('\n[setup] Copying images into YOLO dataset structure...')
        total_missing = 0
        for split in ('train', 'val', 'test'):
            img_out = yolo_dir / 'images' / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_dir = yolo_dir / 'labels' / split
            if not lbl_dir.exists():
                print(f'  [WARN] {lbl_dir} missing, skipping split.')
                continue
            copied = missing = 0
            for lbl in lbl_dir.glob('*.txt'):
                stem = lbl.stem
                if stem in jpg_index:
                    dst = img_out / f'{stem}.jpg'
                    if not dst.exists():
                        shutil.copy2(jpg_index[stem], dst)
                    copied += 1
                else:
                    missing += 1
            total_missing += missing
            print(f'  {split:5s} → {copied:,} copied | {missing} missing')
        if total_missing > 0:
            print(f'\n  [WARN] {total_missing} total missing images')
        copy_marker.touch()
        data_volume.commit()
        print('[setup] Image copy complete and committed to volume.')
    print('\n[verify] Dataset check:')
    for split in ('train', 'val', 'test'):
        imgs = len(list((yolo_dir / 'images' / split).glob('*.jpg')))
        lbls = len(list((yolo_dir / 'labels' / split).glob('*.txt')))
        status = 'OK' if imgs == lbls else 'MISMATCH'
        print(f'  {split:5s} → {imgs:,} images | {lbls:,} labels | {status}')
    yaml_path = yolo_dir / 'parking.yaml'
    yaml_content = f'path: {yolo_dir}\ntrain: images/train\nval:   images/val\ntest:  images/test\n\nnc: 2\nnames:\n  0: available\n  1: occupied\n\ntask: obb\n'
    yaml_path.write_text(yaml_content)
    print(f'\n[setup] parking.yaml written.')
    import torch
    print(f'\n[train] CUDA available: {torch.cuda.is_available()}')
    print(f'[train] GPU: {torch.cuda.get_device_name(0)}')
    print(f'[train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1000000000.0:.1f} GB')
    print(f'\n[train] Starting YOLOv8m-OBB training...')
    model = YOLO(MODEL_BASE)
    model.train(data=str(yaml_path), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0, optimizer='AdamW', lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3, cos_lr=True, patience=20, save_period=10, project=OUTPUT_DIR, name='pklot_m_1280', exist_ok=True, amp=True, workers=WORKERS, cache=False, close_mosaic=15, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, mosaic=1.0, verbose=True)
    checkpoint_volume.commit()
    best_pt = Path(OUTPUT_DIR) / 'pklot_m_1280' / 'weights' / 'best.pt'
    print('\n' + '=' * 60)
    print('  TRAINING COMPLETE')
    print('=' * 60)
    if best_pt.exists():
        print(f'  best.pt : {best_pt.stat().st_size / 1000000.0:.1f} MB')
    else:
        for p in Path(OUTPUT_DIR).rglob('best.pt'):
            print(f'  best.pt found at: {p}')
    print('\n  Download command:')
    print('  modal volume get pklot-checkpoints /checkpoints/pklot_m_1280/weights/best.pt ./best_v2.pt')
    print('=' * 60)

@app.local_entrypoint()
def main():
    print('Submitting training job to Modal (L40S, imgsz=1280, 100 epochs)...')
    train.remote()
