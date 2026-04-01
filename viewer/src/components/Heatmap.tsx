import { useMemo, useRef, useState, useCallback } from "react";
import type { StreamTree, RunDetail } from "../types";
import { GAP_COLOR, DEFAULT_STEP_COLOR } from "../lib/colors";

interface Props {
  tree: StreamTree;
  stepColors: Record<string, string>;
  runs: Record<string, { success: boolean | null }>;
  runDetails: Record<string, RunDetail>;
  expandedRuns: Set<string>;
  onToggleRun: (runId: string) => void;
}

interface Tooltip {
  x: number;
  y: number;
  step: string;
  detail: string;
  runId: string;
}

const OUTCOME_W = 8;
const LABEL_W = 150;
const LEFT_MARGIN = OUTCOME_W + 4 + LABEL_W + 8;
const MARKER_H = 24;
const GAP = "-";

export function Heatmap({
  tree,
  stepColors,
  runs,
  runDetails,
  expandedRuns,
  onToggleRun,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<Tooltip | null>(null);

  const matrix = tree.alignment_matrix;
  const runIds = tree.alignment_run_ids;

  const nRuns = runIds.length;
  const nCols = matrix.length > 0 ? (matrix[0]?.length ?? 0) : 0;

  // Compute cell dimensions based on available space
  const cellW = Math.max(4, Math.min(16, 900 / Math.max(nCols, 1)));
  const cellH = Math.max(12, Math.min(20, 500 / Math.max(nRuns, 1)));

  const gridW = nCols * cellW;
  const totalW = LEFT_MARGIN + gridW + 10;
  const totalH = MARKER_H + nRuns * cellH + 4;

  // Build divergence column markers from bifurcations
  const divColumns = useMemo(() => {
    const cols: Map<number, { number: number; significant: boolean }> = new Map();
    const bifs = Object.values(tree.bifurcations)
      .sort((a, b) => b.separation - a.separation);
    bifs.forEach((bif, i) => {
      if (bif.column >= 0 && bif.column < nCols && !cols.has(bif.column)) {
        cols.set(bif.column, { number: i + 1, significant: bif.significant });
      }
    });
    return cols;
  }, [tree.bifurcations, nCols]);

  // Build a detail lookup: for each run, map non-gap step index to detail string
  const detailLookup = useMemo(() => {
    const lookup: Map<string, string[]> = new Map();
    for (const rid of runIds) {
      const rd = runDetails[rid];
      if (rd) {
        lookup.set(rid, rd.steps.map((s) => s.detail));
      }
    }
    return lookup;
  }, [runIds, runDetails]);

  const handleMouseEnter = useCallback(
    (e: React.MouseEvent, step: string, detail: string, runId: string) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      setTooltip({
        x: e.clientX - rect.left + 12,
        y: e.clientY - rect.top - 8,
        step,
        detail,
        runId,
      });
    },
    [],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!tooltip) return;
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      setTooltip((prev) =>
        prev ? { ...prev, x: e.clientX - rect.left + 12, y: e.clientY - rect.top - 8 } : null,
      );
    },
    [tooltip],
  );

  const handleMouseLeave = useCallback(() => {
    setTooltip(null);
  }, []);

  if (nRuns === 0 || nCols === 0) {
    return (
      <div style={{ fontSize: 11, color: "#484f58", padding: "8px 0" }}>
        No alignment data available.
      </div>
    );
  }

  return (
    <div ref={containerRef} style={{ position: "relative", overflowX: "auto" }}>
      <svg
        width={totalW}
        height={totalH}
        viewBox={`0 0 ${totalW} ${totalH}`}
        style={{ display: "block", fontFamily: "'IBM Plex Mono', monospace" }}
      >
        {/* Divergence column markers at the top */}
        {Array.from(divColumns.entries()).map(([col, info]) => {
          const x = LEFT_MARGIN + col * cellW + cellW / 2;
          return (
            <g key={`div-${col}`}>
              {/* Vertical guide line */}
              <line
                x1={x} y1={MARKER_H} x2={x} y2={totalH}
                stroke="#d29922" strokeWidth={0.5} opacity={0.15}
              />
              {/* Badge */}
              <circle cx={x} cy={MARKER_H / 2 + 1} r={7} fill={info.significant ? "#d29922" : "#30363d"} />
              <text
                x={x} y={MARKER_H / 2 + 4}
                textAnchor="middle" fill={info.significant ? "#0d1117" : "#8b949e"}
                fontSize={9} fontWeight={700}
                style={{ pointerEvents: "none" }}
              >
                {info.number}
              </text>
            </g>
          );
        })}

        {/* Rows */}
        {runIds.map((rid, rowIdx) => {
          const y = MARKER_H + rowIdx * cellH;
          const runMeta = runs[rid];
          const success = runMeta?.success;
          const isPass = success === true;
          const isFail = success === false;
          const isExpanded = expandedRuns.has(rid);

          const row = matrix[rowIdx];
          if (!row) return null;

          // Detail strings for non-gap cells
          const details = detailLookup.get(rid) ?? [];
          let nonGapIdx = 0;

          // Outcome dot color
          const outcomeColor = isPass ? "#3fb950" : isFail ? "#f85149" : "#30363d";
          // Label color
          const labelColor = isPass ? "#3fb950" : isFail ? "#f85149" : "#8b949e";
          const shortId = rid.length > 18 ? rid.slice(0, 16) + ".." : rid;
          const tag = isPass ? "P" : isFail ? "F" : "?";

          return (
            <g
              key={rid}
              style={{ cursor: "pointer" }}
              onClick={() => onToggleRun(rid)}
            >
              {/* Row background on hover — done via a transparent rect */}
              <rect
                x={0} y={y} width={LEFT_MARGIN} height={cellH}
                fill={isExpanded ? "rgba(48,54,61,0.3)" : "transparent"}
              />

              {/* Outcome dot */}
              <rect
                x={0} y={y + 1}
                width={OUTCOME_W} height={cellH - 2}
                fill={outcomeColor} rx={1}
              />

              {/* Run label */}
              <text
                x={OUTCOME_W + 6} y={y + cellH - 3}
                fill="#8b949e" fontSize={8}
              >
                {shortId}
              </text>

              {/* Pass/fail tag */}
              <text
                x={LEFT_MARGIN - 4} y={y + cellH - 3}
                textAnchor="end" fill={labelColor}
                fontSize={9} fontWeight={600}
              >
                {tag}
              </text>

              {/* Heatmap cells */}
              {row.map((val, col) => {
                const cx = LEFT_MARGIN + col * cellW;
                const isGap = val === GAP;
                const color = isGap
                  ? GAP_COLOR
                  : (stepColors[val] ?? DEFAULT_STEP_COLOR);

                const isDivCol = divColumns.has(col);
                const detail = !isGap && nonGapIdx < details.length
                  ? (details[nonGapIdx] ?? "")
                  : "";

                if (!isGap) nonGapIdx++;

                return (
                  <rect
                    key={col}
                    x={cx} y={y}
                    width={cellW - 0.5} height={cellH - 0.5}
                    fill={color}
                    rx={1}
                    opacity={isGap ? 1 : 0.85}
                    stroke={isDivCol && !isGap ? "#d29922" : undefined}
                    strokeWidth={isDivCol && !isGap ? 1.5 : undefined}
                    onMouseEnter={(e) => {
                      e.stopPropagation();
                      if (!isGap) handleMouseEnter(e, val, detail, rid);
                    }}
                    onMouseMove={(e) => {
                      e.stopPropagation();
                      handleMouseMove(e);
                    }}
                    onMouseLeave={(e) => {
                      e.stopPropagation();
                      handleMouseLeave();
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleRun(rid);
                    }}
                  />
                );
              })}
            </g>
          );
        })}
      </svg>

      {/* Tooltip */}
      {tooltip && (
        <div
          style={{
            position: "absolute",
            left: tooltip.x,
            top: tooltip.y,
            background: "#1c2128",
            border: "1px solid #444c56",
            borderRadius: 4,
            padding: "4px 8px",
            fontSize: 11,
            color: "#e6edf3",
            pointerEvents: "none",
            whiteSpace: "nowrap",
            zIndex: 10,
            maxWidth: 300,
          }}
        >
          <div style={{ fontWeight: 600 }}>{tooltip.step}</div>
          {tooltip.detail && (
            <div style={{ color: "#8b949e", marginTop: 1 }}>{tooltip.detail}</div>
          )}
          <div style={{ color: "#484f58", marginTop: 1, fontSize: 10 }}>{tooltip.runId}</div>
        </div>
      )}
    </div>
  );
}
