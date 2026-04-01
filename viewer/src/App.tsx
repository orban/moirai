import type { MoiraiData, RunMeta } from "./types";
import { useViewerStore } from "./store";
import { Heatmap } from "./components/Heatmap";
import { BranchCards } from "./components/BranchCards";
import { RunList } from "./components/RunList";
import { Legend } from "./components/Legend";

export function App({ data }: { data: MoiraiData }) {
  const activeTaskIndex = useViewerStore((s) => s.activeTaskIndex);
  const setActiveTask = useViewerStore((s) => s.setActiveTask);
  const focusPath = useViewerStore((s) => s.focusPath);
  const drillUp = useViewerStore((s) => s.drillUp);
  const resetFocus = useViewerStore((s) => s.resetFocus);
  const expandedRuns = useViewerStore((s) => s.expandedRuns);
  const toggleRun = useViewerStore((s) => s.toggleRun);

  const task = data.tasks[activeTaskIndex];
  if (!task) {
    return (
      <div style={{ ...containerStyle, padding: "32px 40px" }}>
        <h1 style={titleStyle}>moirai</h1>
        <p style={{ color: "#8b949e" }}>No tasks to display.</p>
      </div>
    );
  }

  const currentFocusBranch = focusPath.length > 0
    ? focusPath[focusPath.length - 1]!
    : task.root_branch_id;

  // Breadcrumb trail
  const breadcrumbs = [
    { label: "root", branchId: task.root_branch_id },
    ...focusPath.map((bid) => {
      const branch = task.branches[bid];
      const runCount = branch?.run_ids.length ?? 0;
      return { label: `${bid.replace("branch_", "b")} (${runCount} runs)`, branchId: bid };
    }),
  ];

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "20px 40px 0",
      }}>
        <div>
          <h1 style={titleStyle}>moirai</h1>
          <div style={{ fontSize: 11, color: "#8b949e", marginTop: 2 }}>trajectory viewer</div>
        </div>
      </div>

      {/* Stats bar */}
      <div style={{
        display: "flex", gap: 24, padding: "12px 40px",
        fontSize: 12, color: "#8b949e", borderBottom: "1px solid #21262d",
      }}>
        <span>{data.stats.total_runs} runs</span>
        <span>{data.stats.pass_rate} pass rate</span>
        <span>{data.stats.n_tasks} {data.stats.n_tasks === 1 ? "task" : "tasks"}</span>
        <span>{data.stats.n_div} bifurcations</span>
      </div>

      {/* Task tabs (only if multiple tasks) */}
      {data.tasks.length > 1 && (
        <div style={{
          display: "flex", gap: 2, padding: "8px 40px",
          borderBottom: "1px solid #21262d", overflowX: "auto",
        }}>
          {data.tasks.map((t, i) => (
            <button
              key={t.task_id}
              onClick={() => setActiveTask(i)}
              style={{
                padding: "6px 12px", fontSize: 11, fontFamily: "'IBM Plex Mono', monospace",
                border: "none", borderRadius: 4, cursor: "pointer",
                background: i === activeTaskIndex ? "#30363d" : "transparent",
                color: i === activeTaskIndex ? "#e6edf3" : "#8b949e",
                fontWeight: i === activeTaskIndex ? 600 : 400,
                whiteSpace: "nowrap",
              }}
            >
              {t.task_id.length > 40 ? t.task_id.slice(0, 37) + "..." : t.task_id}
              <span style={{ marginLeft: 6, opacity: 0.6 }}>({t.n_runs})</span>
            </button>
          ))}
        </div>
      )}

      {/* Legend */}
      <div style={{ padding: "12px 40px 0" }}>
        <Legend legend={data.legend} />
      </div>

      {/* Breadcrumbs (only when drilled down) */}
      {focusPath.length > 0 && (
        <div style={{
          display: "flex", gap: 4, alignItems: "center",
          padding: "8px 40px", fontSize: 11, color: "#8b949e",
        }}>
          {breadcrumbs.map((bc, i) => (
            <span key={i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              {i > 0 && <span style={{ color: "#484f58" }}>/</span>}
              {i === 0 ? (
                <button onClick={resetFocus} style={breadcrumbButtonStyle}>
                  {bc.label}
                </button>
              ) : i < breadcrumbs.length - 1 ? (
                <button onClick={() => drillUp(i - 1)} style={breadcrumbButtonStyle}>
                  {bc.label}
                </button>
              ) : (
                <span style={{ color: "#e6edf3", fontWeight: 600 }}>{bc.label}</span>
              )}
            </span>
          ))}
        </div>
      )}

      {/* Heatmap */}
      <div style={{ padding: "16px 40px 24px", overflowX: "auto" }}>
        <div style={{
          background: "#161b22", borderRadius: 8, border: "1px solid #30363d",
          padding: 12, display: "inline-block",
        }}>
          <Heatmap
            tree={task}
            stepColors={data.step_colors}
            runs={data.runs}
            runDetails={data.run_details}
            expandedRuns={expandedRuns}
            onToggleRun={toggleRun}
          />
        </div>

        {/* Branch info */}
        {(() => {
          const branch = task.branches[currentFocusBranch];
          if (!branch) return null;
          const passCount = branch.run_ids.filter((rid) => data.runs[rid]?.success).length;
          const failCount = branch.run_ids.length - passCount;
          return (
            <div style={{ marginTop: 12, fontSize: 11, color: "#8b949e" }}>
              <span style={{ color: "#e6edf3", fontWeight: 600 }}>
                {branch.run_ids.length} runs
              </span>
              {" "}in this branch
              <span style={{ marginLeft: 12, color: "#3fb950" }}>{passCount} pass</span>
              <span style={{ marginLeft: 8, color: "#e8585a" }}>{failCount} fail</span>
              <span style={{ marginLeft: 12 }}>
                ({(branch.success_rate * 100).toFixed(0)}% success rate)
              </span>
            </div>
          );
        })()}
      </div>

      {/* Branch cards */}
      <div style={{ padding: "0 40px 16px" }}>
        <BranchCards tree={task} focusBranchId={currentFocusBranch} />
      </div>

      {/* Run list */}
      <div style={{ padding: "0 40px 40px" }}>
        <h3 style={{ fontSize: 12, color: "#8b949e", margin: "0 0 8px", fontWeight: 400 }}>
          Runs ({task.branches[currentFocusBranch]?.run_ids.length ?? 0})
        </h3>
        <RunList
          runs={(task.branches[currentFocusBranch]?.run_ids ?? [])
            .map((rid) => data.runs[rid])
            .filter((r): r is RunMeta => r != null)}
          runDetails={data.run_details}
          expandedRuns={expandedRuns}
          onToggleRun={toggleRun}
        />
      </div>
    </div>
  );
}

const containerStyle: React.CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
  background: "#0d1117",
  color: "#c9d1d9",
  minHeight: "100vh",
};

const titleStyle: React.CSSProperties = {
  fontSize: 20,
  color: "#e6edf3",
  margin: 0,
  fontWeight: 700,
  letterSpacing: "-0.02em",
};

const breadcrumbButtonStyle: React.CSSProperties = {
  background: "none",
  border: "none",
  color: "#58a6ff",
  cursor: "pointer",
  fontSize: 11,
  fontFamily: "inherit",
  padding: 0,
  textDecoration: "underline",
  textDecorationColor: "#30363d",
  textUnderlineOffset: 2,
};
