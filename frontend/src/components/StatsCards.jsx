import { useEffect, useState } from "react";

const CARD_META = [
  { key: "total", label: "Total spots", tone: "blue" },
  { key: "available", label: "Available", tone: "green" },
  { key: "occupied", label: "Occupied", tone: "red" },
  { key: "occupancy", label: "Occupancy", tone: "amber" },
];

function Ring({ value }) {
  const pct = Number.isFinite(value) ? value : 0;
  const radius = 18;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <svg className="stat-ring" viewBox="0 0 48 48">
      <circle cx="24" cy="24" r={radius} />
      <circle
        cx="24"
        cy="24"
        r={radius}
        className={pct >= 80 ? "danger" : pct >= 55 ? "warn" : "good"}
        strokeDasharray={circumference}
        strokeDashoffset={offset}
      />
    </svg>
  );
}

function useCountUp(value, duration = 550) {
  const numericValue = Number(value ?? 0);
  const [display, setDisplay] = useState(numericValue);

  useEffect(() => {
    if (!Number.isFinite(numericValue)) return;

    let frameId;
    const startValue = display;
    const delta = numericValue - startValue;
    const startTime = performance.now();

    const tick = (now) => {
      const progress = Math.min((now - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(startValue + delta * eased);
      if (progress < 1) frameId = requestAnimationFrame(tick);
    };

    frameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameId);
  }, [duration, numericValue]);

  return Math.round(display);
}

function StatValue({ cardKey, value }) {
  const animated = useCountUp(cardKey === "occupancy" ? 0 : value);
  if (cardKey === "occupancy") return `${(value ?? 0).toFixed(1)}%`;
  return animated;
}

export default function StatsCards({ totalSpots, available, occupied, occupancyPct }) {
  const values = {
    total: totalSpots ?? 0,
    available: available ?? 0,
    occupied: occupied ?? 0,
    occupancy: occupancyPct ?? 0,
  };

  return (
    <section className="stats-grid">
      {CARD_META.map((card) => (
        <article className={`stat-card ${card.tone}`} key={card.key}>
          <div>
            <span>{card.label}</span>
            <strong><StatValue cardKey={card.key} value={values[card.key]} /></strong>
          </div>
          {card.key === "occupancy" ? <Ring value={occupancyPct ?? 0} /> : <i />}
        </article>
      ))}
    </section>
  );
}
