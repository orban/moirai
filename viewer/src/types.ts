/** A single step in a run's trajectory */
export interface StepDetail {
  idx: number;
  enriched: string;
  detail: string;
  status: string;
  phase: string;
  color: string;
  metrics: {
    tokens_in?: number;
    tokens_out?: number;
    latency_ms?: number;
  };
}

/** Metadata for a single run */
export interface RunMeta {
  run_id: string;
  success: boolean | null;
  step_count: number;
  trajectory: string;
  phase_mix: string;
  harness: string | null;
  model: string | null;
}

/** Per-run step detail (tier 2 data, loaded on expand) */
export interface RunDetail {
  run_id: string;
  steps: StepDetail[];
}

/** One branch in the stream tree */
export interface StreamBranch {
  id: string;
  run_ids: string[];
  success_rate: number;
  step_proportions: Record<number, Record<string, number>>;
  trajectory: string;
  phase_mix: Record<string, number>;
  col_start: number;
  col_end: number;
}

/** A bifurcation point in the stream tree */
export interface Bifurcation {
  id: string;
  column: number;
  x_position: number;
  separation: number;
  p_value: number | null;
  significant: boolean;
  left_branch_id: string;
  right_branch_id: string;
  left_success_rate: number;
  right_success_rate: number;
}

/** The full stream tree for a task */
export interface StreamTree {
  task_id: string;
  n_runs: number;
  n_pass: number;
  n_fail: number;
  n_cols: number;
  root_branch_id: string;
  branches: Record<string, StreamBranch>;
  bifurcations: Record<string, Bifurcation>;
  tree: Record<string, [string, string]>;
  /** Aligned step matrix: [run_index][column] = enriched step name or "-" (gap) */
  alignment_matrix: string[][];
  /** Run IDs in alignment/dendrogram leaf order (matches matrix rows) */
  alignment_run_ids: string[];
}

/** Top-level data payload injected by Python */
export interface MoiraiData {
  stats: {
    total_runs: number;
    pass_rate: string;
    n_tasks: number;
    n_div: number;
  };
  legend: Array<[string, string]>;
  tasks: StreamTree[];
  runs: Record<string, RunMeta>;
  run_details: Record<string, RunDetail>;
  step_colors: Record<string, string>;
}
