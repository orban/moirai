import { useRef, useCallback } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import type { RunMeta, RunDetail } from "../types";
import { StepTimeline } from "./StepTimeline";

interface Props {
  runs: RunMeta[];
  runDetails: Record<string, RunDetail>;
  expandedRuns: Set<string>;
  onToggleRun: (runId: string) => void;
}

const ROW_HEIGHT = 36;

export function RunList({ runs, runDetails, expandedRuns, onToggleRun }: Props) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: runs.length,
    getScrollElement: () => parentRef.current,
    estimateSize: useCallback(
      (index: number) => {
        const run = runs[index];
        if (!run) return ROW_HEIGHT;
        if (!expandedRuns.has(run.run_id)) return ROW_HEIGHT;
        const detail = runDetails[run.run_id];
        const stepCount = detail?.steps.length ?? 0;
        return ROW_HEIGHT + Math.max(stepCount * 40, 32);
      },
      [runs, expandedRuns, runDetails],
    ),
    overscan: 10,
  });

  if (runs.length === 0) {
    return (
      <div style={{ fontSize: 11, color: "#484f58", padding: "8px 0" }}>
        No runs in this branch.
      </div>
    );
  }

  return (
    <div
      ref={parentRef}
      style={{
        maxHeight: 400,
        overflow: "auto",
        border: "1px solid #30363d",
        borderRadius: 6,
        background: "#0d1117",
      }}
    >
      <div style={{
        height: virtualizer.getTotalSize(),
        width: "100%",
        position: "relative",
      }}>
        {virtualizer.getVirtualItems().map((virtualRow) => {
          const run = runs[virtualRow.index];
          if (!run) return null;
          const isExpanded = expandedRuns.has(run.run_id);
          const detail = runDetails[run.run_id];
          const isPass = run.success === true;
          const isFail = run.success === false;

          return (
            <div
              key={run.run_id}
              ref={(el) => {
                if (el) virtualizer.measureElement(el);
              }}
              data-index={virtualRow.index}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              {/* Row header */}
              <div
                onClick={() => onToggleRun(run.run_id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  height: ROW_HEIGHT,
                  padding: "0 12px",
                  cursor: "pointer",
                  borderBottom: "1px solid #21262d",
                  fontSize: 11,
                  userSelect: "none",
                }}
              >
                {/* Pass/fail dot */}
                <span style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: isPass ? "#3fb950" : isFail ? "#f85149" : "#484f58",
                  flexShrink: 0,
                }} />

                {/* Run ID */}
                <span style={{
                  color: "#c9d1d9",
                  flex: 1,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  fontFamily: "'IBM Plex Mono', monospace",
                }}>
                  {run.run_id}
                </span>

                {/* Step count */}
                <span style={{ color: "#484f58", flexShrink: 0 }}>
                  {run.step_count} steps
                </span>

                {/* Expand chevron */}
                <span style={{
                  color: "#484f58",
                  fontSize: 10,
                  flexShrink: 0,
                  transition: "transform 0.15s",
                  transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)",
                }}>
                  {"\u25b6"}
                </span>
              </div>

              {/* Expanded step timeline */}
              {isExpanded && detail && (
                <div style={{
                  borderBottom: "1px solid #21262d",
                  background: "#0d1117",
                }}>
                  <StepTimeline steps={detail.steps} />
                </div>
              )}

              {isExpanded && !detail && (
                <div style={{
                  padding: "8px 12px",
                  fontSize: 11,
                  color: "#484f58",
                  borderBottom: "1px solid #21262d",
                }}>
                  No step detail available.
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
