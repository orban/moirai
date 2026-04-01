import type { StreamTree, StreamBranch } from "../types";

interface Props {
  tree: StreamTree;
  focusBranchId: string;
}

function formatPhaseMix(phaseMix: Record<string, number>): string {
  return Object.entries(phaseMix)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([phase, frac]) => `${(frac * 100).toFixed(0)}% ${phase}`)
    .join(", ");
}

function successColor(rate: number): string {
  return rate > 0.5 ? "#3fb950" : "#f85149";
}

function BranchCard({ branch, label }: { branch: StreamBranch; label: string }) {
  const ratePercent = (branch.success_rate * 100).toFixed(0);
  const borderColor = successColor(branch.success_rate);
  const phaseLine = formatPhaseMix(branch.phase_mix);

  return (
    <div style={{
      flex: 1,
      background: "#161b22",
      border: "1px solid #30363d",
      borderLeft: `3px solid ${borderColor}`,
      borderRadius: 6,
      padding: "10px 14px",
      minWidth: 0,
    }}>
      <div style={{ fontSize: 10, color: "#484f58", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 13, color: "#e6edf3", fontWeight: 600 }}>
          {branch.run_ids.length} runs
        </span>
        <span style={{ fontSize: 12, color: borderColor, fontWeight: 600 }}>
          {ratePercent}% pass
        </span>
      </div>
      {branch.trajectory && (
        <div style={{
          fontSize: 10,
          color: "#8b949e",
          fontFamily: "'IBM Plex Mono', monospace",
          marginBottom: 3,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}>
          {branch.trajectory}
        </div>
      )}
      {phaseLine && (
        <div style={{ fontSize: 10, color: "#484f58" }}>
          {phaseLine}
        </div>
      )}
    </div>
  );
}

/**
 * Find the first bifurcation under the focus branch where
 * left + right run counts equal the focus branch run count.
 */
function findBifurcation(tree: StreamTree, branchId: string) {
  const branch = tree.branches[branchId];
  if (!branch) return null;

  const branchRunSet = new Set(branch.run_ids);

  for (const bif of Object.values(tree.bifurcations)) {
    const left = tree.branches[bif.left_branch_id];
    const right = tree.branches[bif.right_branch_id];
    if (!left || !right) continue;

    if (left.run_ids.length + right.run_ids.length !== branchRunSet.size) continue;

    const allMatch =
      left.run_ids.every((r) => branchRunSet.has(r)) &&
      right.run_ids.every((r) => branchRunSet.has(r));

    if (allMatch) {
      return { left, right };
    }
  }
  return null;
}

export function BranchCards({ tree, focusBranchId }: Props) {
  const split = findBifurcation(tree, focusBranchId);
  if (!split) return null;

  return (
    <div style={{ display: "flex", gap: 12 }}>
      <BranchCard branch={split.left} label="left branch" />
      <BranchCard branch={split.right} label="right branch" />
    </div>
  );
}
