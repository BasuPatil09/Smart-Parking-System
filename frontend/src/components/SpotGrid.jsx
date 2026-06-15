import { useMemo, useState } from "react";

function ConfidenceBar({ value, status }) {
  const pct = Math.round((value ?? 0) * 100);

  return (
    <div className="confidence">
      <span>Confidence</span>
      <strong>{pct}%</strong>
      <div>
        <i className={status === "occupied" ? "occupied" : "available"} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function SpotCard({ spot }) {
  const isOccupied = spot.status === "occupied";
  const bbox = spot.bbox ?? {};

  return (
    <article className={`spot-tile ${isOccupied ? "occupied" : "available"}`}>
      <div className="spot-topline">
        <strong>#{String(spot.id).padStart(2, "0")}</strong>
        <span>{spot.status}</span>
      </div>

      <ConfidenceBar value={spot.confidence} status={spot.status} />

      <dl className="geometry-grid">
        <div><dt>CX</dt><dd>{Math.round(bbox.center_x ?? 0)}</dd></div>
        <div><dt>CY</dt><dd>{Math.round(bbox.center_y ?? 0)}</dd></div>
        <div><dt>W</dt><dd>{Math.round(bbox.width ?? 0)}</dd></div>
        <div><dt>H</dt><dd>{Math.round(bbox.height ?? 0)}</dd></div>
        <div><dt>Angle</dt><dd>{Math.round(bbox.angle ?? 0)} deg</dd></div>
      </dl>
    </article>
  );
}

export default function SpotGrid({ spots = [] }) {
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const available = spots.filter((spot) => spot.status === "available").length;
  const occupied = spots.length - available;
  const visibleSpots = useMemo(() => {
    const normalized = query.trim().replace(/^#/, "");
    return spots.filter((spot) => {
      const statusMatch = filter === "all" || spot.status === filter;
      const queryMatch = !normalized || String(spot.id).includes(normalized);
      return statusMatch && queryMatch;
    });
  }, [filter, query, spots]);

  return (
    <section className="panel registry-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Per-space output</p>
          <h3>Spot registry</h3>
        </div>
        <div className="viewer-meta">
          <span>{available} free</span>
          <span>{occupied} occupied</span>
        </div>
      </div>

      {spots.length > 0 && (
        <div className="registry-toolbar">
          <div className="filter-tabs" aria-label="Spot status filter">
            {["all", "available", "occupied"].map((nextFilter) => (
              <button
                key={nextFilter}
                type="button"
                className={filter === nextFilter ? "active" : ""}
                onClick={() => setFilter(nextFilter)}
              >
                {nextFilter}
              </button>
            ))}
          </div>
          <label className="spot-search">
            <span>Spot ID</span>
            <input
              type="search"
              inputMode="numeric"
              placeholder="Enter the spot number"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <span className="registry-count">{visibleSpots.length} shown</span>
        </div>
      )}

      {spots.length > 0 ? (
        <div className="spot-grid">
          {visibleSpots.map((spot) => <SpotCard key={spot.id} spot={spot} />)}
          {visibleSpots.length === 0 && (
            <div className="empty-state">No spots match this filter</div>
          )}
        </div>
      ) : (
        <div className="empty-state">No spaces were detected in this frame</div>
      )}
    </section>
  );
}
