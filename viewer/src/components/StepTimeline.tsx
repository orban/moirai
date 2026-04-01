import type { StepDetail } from "../types";

interface Props {
  steps: StepDetail[];
}

function formatMetrics(metrics: StepDetail["metrics"]): string | null {
  const parts: string[] = [];
  if (metrics.tokens_in != null) {
    parts.push(metrics.tokens_in >= 1000
      ? `${(metrics.tokens_in / 1000).toFixed(0)}k in`
      : `${metrics.tokens_in} in`);
  }
  if (metrics.tokens_out != null) {
    parts.push(metrics.tokens_out >= 1000
      ? `${(metrics.tokens_out / 1000).toFixed(1)}k out`
      : `${metrics.tokens_out} out`);
  }
  if (metrics.latency_ms != null) {
    parts.push(metrics.latency_ms >= 1000
      ? `${(metrics.latency_ms / 1000).toFixed(1)}s`
      : `${metrics.latency_ms}ms`);
  }
  return parts.length > 0 ? parts.join(" \u00b7 ") : null;
}

export function StepTimeline({ steps }: Props) {
  if (steps.length === 0) {
    return (
      <div style={{ padding: "8px 12px", fontSize: 11, color: "#484f58" }}>
        No steps recorded.
      </div>
    );
  }

  return (
    <div style={{ padding: "4px 0 4px 20px", position: "relative" }}>
      {/* Vertical connector line */}
      <div style={{
        position: "absolute",
        left: 22,
        top: 8,
        bottom: 8,
        width: 1,
        background: "#30363d",
      }} />

      {steps.map((step, i) => {
        const isPass = step.status === "pass" || step.status === "success";
        const metricsStr = formatMetrics(step.metrics);

        return (
          <div key={i} style={{
            display: "flex",
            gap: 8,
            padding: "3px 0",
            position: "relative",
          }}>
            {/* Color bar */}
            <div style={{
              width: 5,
              minHeight: 28,
              borderRadius: 2,
              background: step.color || "#484f58",
              flexShrink: 0,
              zIndex: 1,
            }} />

            {/* Card content */}
            <div style={{
              flex: 1,
              background: "#161b22",
              border: "1px solid #21262d",
              borderRadius: 4,
              padding: "4px 8px",
              fontSize: 11,
              minWidth: 0,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ color: "#e6edf3", fontWeight: 600 }}>
                  {step.enriched}
                </span>
                <span style={{ color: isPass ? "#3fb950" : "#f85149", flexShrink: 0 }}>
                  {isPass ? "\u2713" : "\u2717"}
                </span>
              </div>

              {step.detail && (
                <div style={{ color: "#8b949e", marginTop: 1 }}>
                  {step.detail}
                </div>
              )}

              <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 2 }}>
                {metricsStr && (
                  <span style={{ color: "#484f58", fontSize: 10 }}>
                    {metricsStr}
                  </span>
                )}
                {step.phase && (
                  <span style={{
                    fontSize: 9,
                    color: "#8b949e",
                    background: "#21262d",
                    padding: "1px 5px",
                    borderRadius: 8,
                    whiteSpace: "nowrap",
                  }}>
                    {step.phase}
                  </span>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
