export default function ParkingViewer({ annotatedImageB64, inferenceMs, totalSpots, augment }) {
  const imageSrc = annotatedImageB64
    ? `data:image/jpeg;base64,${annotatedImageB64}`
    : null;
  const slow = Number(inferenceMs ?? 0) > 5000;

  return (
    <section className="panel image-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Annotated frame</p>
          <h3>Parking map preview</h3>
        </div>
        <div className="viewer-meta">
          <span>{totalSpots ?? 0} spots</span>
          <span>{inferenceMs?.toFixed?.(1) ?? inferenceMs ?? 0} ms</span>
          {slow && (
            <span className="slow-meta">
              Slow - {augment ? "CPU + enhanced scan active" : "large image or CPU inference"}
            </span>
          )}
        </div>
      </div>

      <div className="image-stage">
        {imageSrc ? (
          <>
            <img src={imageSrc} alt="YOLO annotated parking lot" />
            <span className="scan-line" />
          </>
        ) : (
          <div className="empty-state">No annotated image available</div>
        )}
      </div>

      <div className="legend-row">
        <span><i className="legend-dot available" /> Available</span>
        <span><i className="legend-dot occupied" /> Occupied</span>
      </div>
    </section>
  );
}
