import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function TooltipCard({ active, payload, label }) {
  if (!active || !payload?.length) return null;

  return (
    <div className="chart-tooltip">
      <span>Session {label}</span>
      <strong>{payload[0].value.toFixed(1)}%</strong>
    </div>
  );
}

export default function OccupancyChart({ history = [] }) {
  const data = history;
  const latest = data.at(-1)?.pct ?? 0;
  const stroke = latest >= 80 ? "#fb7185" : latest >= 55 ? "#fbbf24" : "#22c55e";

  return (
    <section className="panel chart-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Session trend</p>
          <h3>Occupancy history</h3>
        </div>
        <strong className="chart-value">{latest.toFixed(1)}%</strong>
      </div>

      <div className="chart-frame">
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 12, bottom: 0, left: -18 }}>
              <defs>
                <linearGradient id="occupancyGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={stroke} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={stroke} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(148, 163, 184, 0.13)" vertical={false} />
              <XAxis
                dataKey="session"
                axisLine={false}
                tickLine={false}
                tick={{ fill: "#64748b", fontSize: 11 }}
              />
              <YAxis
                domain={[0, 100]}
                axisLine={false}
                tickLine={false}
                tick={{ fill: "#64748b", fontSize: 11 }}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip content={<TooltipCard />} />
              <Area
                type="monotone"
                dataKey="pct"
                stroke={stroke}
                strokeWidth={3}
                fill="url(#occupancyGradient)"
                dot={{ r: 3, fill: stroke, strokeWidth: 0 }}
                activeDot={{ r: 5, fill: stroke, stroke: "#020617", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="empty-state">Run a detection to start the trend</div>
        )}
      </div>
    </section>
  );
}
