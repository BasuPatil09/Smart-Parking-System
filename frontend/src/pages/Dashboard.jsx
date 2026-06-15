import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import OccupancyChart from "../components/OccupancyChart";
import ParkingViewer from "../components/ParkingViewer";
import SpotGrid from "../components/SpotGrid";
import StatsCards from "../components/StatsCards";
import { getStatus, predictImage } from "../services/api";

const INITIAL = {
  status: "idle",
  uploadPct: 0,
  result: null,
  error: null,
  sessionIndex: 0,
  occupancyHistory: [],
};

function reducer(state, action) {
  switch (action.type) {
    case "UPLOAD_START":
      return { ...state, status: "uploading", uploadPct: 0, error: null };
    case "UPLOAD_PROGRESS":
      return { ...state, uploadPct: action.payload };
    case "PREDICTING":
      return { ...state, status: "predicting", uploadPct: 100 };
    case "DISPLAY_RESULT":
      return { ...state, status: "done", result: action.payload, error: null };
    case "SUCCESS": {
      const nextSession = state.sessionIndex + 1;
      return {
        ...state,
        status: "done",
        result: action.payload,
        error: null,
        occupancyHistory: [
          ...state.occupancyHistory,
          {
            session: nextSession,
            pct: Number((action.payload.occupancy_pct ?? 0).toFixed(1)),
          },
        ].slice(-20),
        sessionIndex: nextSession,
      };
    }
    case "ERROR":
      return { ...state, status: "error", error: action.payload };
    case "RESET":
      return {
        ...INITIAL,
        sessionIndex: state.sessionIndex,
        occupancyHistory: state.occupancyHistory,
      };
    default:
      return state;
  }
}

function Icon({ children }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      {children}
    </svg>
  );
}

function TopBar({ serverInfo, hasResult, sessionIndex, onReset, onOpenEvaluation }) {
  const loaded = Boolean(serverInfo?.model_loaded);

  return (
    <header className="topbar">
      <div className="brand-lockup">
        <h1>Smart Parking</h1>
      </div>

      <nav className="main-nav" aria-label="Primary navigation">
        <button type="button">Detection</button>
        <button type="button" onClick={onOpenEvaluation}>Evaluation</button>
      </nav>

      <div className="status-strip">
        <span className={`status-dot ${loaded ? "ok" : "bad"}`} />
        <span>{loaded ? "Model ready" : "Model offline"}</span>
        <span className="divider" />
        <span>{serverInfo?.device ? serverInfo.device.toUpperCase() : "NO DEVICE"}</span>
        {sessionIndex > 0 && <span>Run {sessionIndex}</span>}
      </div>

      {hasResult && (
        <button className="ghost-button" type="button" onClick={onReset}>
          New scan
        </button>
      )}
    </header>
  );
}

function UploadPanel({
  imageFile,
  setImageFile,
  imageFiles,
  setImageFiles,
  isDragging,
  setIsDragging,
  dropZoneRef,
  conf,
  setConf,
  iou,
  setIou,
  imgsz,
  setImgsz,
  augment,
  setAugment,
  state,
  canSubmit,
  isLoading,
  batchRunning,
  batchProgress,
  onSubmit,
  onClear,
}) {
  const inputRef = useRef(null);

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      event.stopPropagation();
      setIsDragging(false);
      const files = Array.from(event.dataTransfer.files);
      const images = files.filter((file) =>
        ["image/jpeg", "image/png", "image/bmp", "image/webp"].includes(file.type),
      );
      if (images.length > 0) {
        setImageFiles(images);
        setImageFile(images[0]);
      }
    },
    [setImageFile, setImageFiles, setIsDragging],
  );

  const selectedCount = imageFiles.length;
  const totalSize = imageFiles.reduce((sum, file) => sum + file.size, 0);

  return (
    <section
      ref={dropZoneRef}
      className={`upload-panel ${isDragging ? "dragging" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={(event) => {
        event.preventDefault();
        if (!dropZoneRef.current?.contains(event.relatedTarget)) setIsDragging(false);
      }}
      onDrop={handleDrop}
    >
      <div className="upload-copy">
        <h2>{selectedCount > 1 ? "Batch input" : "Input image"}</h2>
      </div>

      <button className="drop-target" type="button" onClick={() => inputRef.current?.click()}>
        <span className="drop-icon">
          <Icon>
            <path d="M12 16V4" />
            <path d="m7 9 5-5 5 5" />
            <path d="M20 16.5V19a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-2.5" />
          </Icon>
        </span>
        <span>
          <strong>
            {selectedCount > 1
              ? `${selectedCount} images selected`
              : imageFile
                ? imageFile.name
                : "Choose or drop image"}
          </strong>
          <small>
            {selectedCount > 1
              ? `${(totalSize / 1024 / 1024).toFixed(2)} MB total`
              : imageFile
                ? `${(imageFile.size / 1024 / 1024).toFixed(2)} MB selected`
                : "JPEG, PNG, BMP or WEBP"}
          </small>
        </span>
      </button>

      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/bmp,image/webp"
        multiple
        onChange={(event) => {
          const next = Array.from(event.target.files ?? []);
          if (next.length > 0) {
            setImageFiles(next);
            setImageFile(next[0]);
          }
        }}
        hidden
      />

      <div className="slider-grid">
        <label>
          <span>Confidence</span>
          <output>{conf.toFixed(2)}</output>
          <input
            type="range"
            min="0.10"
            max="0.9"
            step="0.01"
            value={conf}
            onChange={(event) => setConf(Number(event.target.value))}
          />
        </label>
        <label>
          <span>IoU threshold</span>
          <output>{iou.toFixed(2)}</output>
          <input
            type="range"
            min="0.45"
            max="0.9"
            step="0.05"
            value={iou}
            onChange={(event) => setIou(Number(event.target.value))}
          />
        </label>
        <label>
          <span>Image size</span>
          <output>{imgsz}px</output>
          <input
            type="range"
            min="640"
            max="1920"
            step="160"
            value={imgsz}
            onChange={(event) => setImgsz(Number(event.target.value))}
          />
        </label>
        <label className="toggle-control">
          <span>Enhanced scan</span>
          <output>{augment ? "On" : "Off"}</output>
          <input
            type="checkbox"
            checked={augment}
            onChange={(event) => setAugment(event.target.checked)}
          />
        </label>
      </div>

      {state.status === "uploading" && (
        <div className="progress-row">
          <span>Uploading</span>
          <div><i style={{ width: `${state.uploadPct}%` }} /></div>
          <strong>{state.uploadPct}%</strong>
        </div>
      )}

      {state.status === "predicting" && (
        <div className="inline-notice">
          <span className="loader" />
          <span>
            {batchRunning
              ? `Processing image ${batchProgress.done + 1} of ${batchProgress.total}`
              : "Processing image"}
            {augment && <small>Enhanced scan active - expect longer processing.</small>}
          </span>
        </div>
      )}

      {augment && state.status !== "predicting" && (
        <div className="inline-notice compact-notice">
          Enhanced scan clamps confidence to 0.10-0.15 for maximum recall.
        </div>
      )}

      {state.status === "error" && (
        <div className="error-box">
          <strong>Inference failed</strong>
          <span>{state.error}</span>
        </div>
      )}

      <div className="upload-actions">
        {selectedCount > 0 && (
          <button className="ghost-button" type="button" onClick={onClear}>
            Clear {selectedCount > 1 ? "batch" : "image"}
          </button>
        )}
        <button className="primary-button" type="button" disabled={!canSubmit} onClick={onSubmit}>
          {isLoading
            ? "Processing..."
            : selectedCount > 1
              ? "Run batch"
              : "Run detection"}
        </button>
      </div>
    </section>
  );
}

function BatchResults({ items, activeName, onSelect }) {
  if (items.length === 0) return null;

  const completed = items.filter((item) => item.status === "done");
  const avgOccupancy = completed.length
    ? completed.reduce((sum, item) => sum + item.result.occupancy_pct, 0) / completed.length
    : 0;

  return (
    <section className="panel batch-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Batch run</p>
          <h3>Image results</h3>
        </div>
        <div className="viewer-meta">
          <span>{completed.length}/{items.length} complete</span>
          {completed.length > 0 && <span>{avgOccupancy.toFixed(1)}% average occupancy</span>}
        </div>
      </div>

      <div className="batch-list">
        {items.map((item, index) => (
          <button
            key={`${item.fileName}-${index}`}
            type="button"
            className={item.fileName === activeName ? "active" : ""}
            disabled={item.status !== "done"}
            onClick={() => onSelect(item)}
          >
            <span>
              <strong>{item.fileName}</strong>
              <small>
                {item.status === "done"
                  ? `${item.result.total_spots} spots - ${item.result.occupancy_pct.toFixed(1)}% occupied`
                  : item.status === "error"
                    ? item.error
                    : "Waiting"}
              </small>
            </span>
            <b className={item.status}>{item.status}</b>
          </button>
        ))}
      </div>
    </section>
  );
}

export default function Dashboard({ onOpenEvaluation }) {
  const [state, dispatch] = useReducer(reducer, INITIAL);
  const [serverInfo, setServerInfo] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [imageFiles, setImageFiles] = useState([]);
  const [batchItems, setBatchItems] = useState([]);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ done: 0, total: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [conf, setConf] = useState(0.15);
  const [iou, setIou] = useState(0.55);
  const [imgsz, setImgsz] = useState(1600);
  const [augment, setAugment] = useState(true);
  const dropZoneRef = useRef(null);

  useEffect(() => {
    let alive = true;
    const fetchStatus = async () => {
      try {
        const info = await getStatus();
        if (alive) setServerInfo(info);
      } catch {
        if (alive) setServerInfo(null);
      }
    };

    fetchStatus();
    const timer = setInterval(fetchStatus, 30000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const isLoading = state.status === "uploading" || state.status === "predicting";
  const hasResult = Boolean(state.result);
  const canSubmit = imageFiles.length > 0 && !isLoading && !batchRunning;

  const handleSubmit = useCallback(async () => {
    if (imageFiles.length === 0) return;

    const files = imageFiles;
    setBatchRunning(files.length > 1);
    setBatchProgress({ done: 0, total: files.length });
    setBatchItems(files.map((file) => ({ fileName: file.name, status: "queued" })));

    for (const [index, file] of files.entries()) {
      setImageFile(file);
      setBatchItems((items) =>
        items.map((item, itemIndex) =>
          itemIndex === index ? { ...item, status: "running" } : item,
        ),
      );
      dispatch({ type: "UPLOAD_START" });

      try {
        const result = await predictImage(
          file,
          (event) => {
            if (event.total) {
              dispatch({
                type: "UPLOAD_PROGRESS",
                payload: Math.round((event.loaded / event.total) * 100),
              });
            }
            if (event.loaded === event.total) dispatch({ type: "PREDICTING" });
          },
          conf,
          iou,
          imgsz,
          augment,
          1000,
        );
        dispatch({ type: "SUCCESS", payload: result });
        setBatchItems((items) =>
          items.map((item, itemIndex) =>
            itemIndex === index ? { ...item, status: "done", result } : item,
          ),
        );
      } catch (error) {
        const message = error.response?.data?.detail || error.message || "Unknown error";
        dispatch({ type: "ERROR", payload: message });
        setBatchItems((items) =>
          items.map((item, itemIndex) =>
            itemIndex === index ? { ...item, status: "error", error: message } : item,
          ),
        );
      } finally {
        setBatchProgress({ done: index + 1, total: files.length });
      }
    }

    setBatchRunning(false);
  }, [augment, conf, imageFiles, imgsz, iou]);

  const resetResult = useCallback(() => {
    dispatch({ type: "RESET" });
  }, []);

  const clearImage = useCallback(() => {
    setImageFile(null);
    setImageFiles([]);
    setBatchItems([]);
    setBatchProgress({ done: 0, total: 0 });
    if (state.status === "error") dispatch({ type: "RESET" });
  }, [state.status]);

  const showBatchResult = useCallback((item) => {
    if (item.result) {
      setImageFile(imageFiles.find((file) => file.name === item.fileName) ?? imageFile);
      dispatch({ type: "DISPLAY_RESULT", payload: item.result });
    }
  }, [imageFile, imageFiles]);

  return (
    <div className="app-shell">
      <TopBar
        serverInfo={serverInfo}
        hasResult={hasResult}
        sessionIndex={state.sessionIndex}
        onReset={resetResult}
        onOpenEvaluation={onOpenEvaluation}
      />

      <main className="dashboard">
        <section className="page-heading">
          <div>
            <h2>Parking occupancy detection</h2>
            <p>
              Upload a parking lot image and review the detected spaces, occupancy count, and annotated output.
            </p>
          </div>
        </section>

        <div className={hasResult ? "workspace with-results" : "workspace"}>
          <UploadPanel
            imageFile={imageFile}
            setImageFile={setImageFile}
            imageFiles={imageFiles}
            setImageFiles={setImageFiles}
            isDragging={isDragging}
            setIsDragging={setIsDragging}
            dropZoneRef={dropZoneRef}
            conf={conf}
            setConf={setConf}
            iou={iou}
            setIou={setIou}
            imgsz={imgsz}
            setImgsz={setImgsz}
            augment={augment}
            setAugment={setAugment}
            state={state}
            canSubmit={canSubmit}
            isLoading={isLoading}
            batchRunning={batchRunning}
            batchProgress={batchProgress}
            onSubmit={handleSubmit}
            onClear={clearImage}
          />

          {hasResult && (
            <section className="results-stack">
              <BatchResults
                items={batchItems}
                activeName={imageFile?.name}
                onSelect={showBatchResult}
              />

              <StatsCards
                totalSpots={state.result.total_spots}
                available={state.result.available}
                occupied={state.result.occupied}
                occupancyPct={state.result.occupancy_pct}
              />

              <div className="result-grid">
                <ParkingViewer
                  annotatedImageB64={state.result.annotated_image_b64}
                  inferenceMs={state.result.inference_ms}
                  totalSpots={state.result.total_spots}
                  augment={augment}
                />
                <OccupancyChart history={state.occupancyHistory} />
              </div>

              <SpotGrid spots={state.result.spots} />
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
