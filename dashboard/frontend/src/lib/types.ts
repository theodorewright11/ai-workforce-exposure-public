// ── Overview ──────────────────────────────────────────────────────────────────

export interface GroupSettings {
  selectedDatasets: string[];
  combineMethod: "Average" | "Max";
  method: "freq" | "imp";
  useAutoAug: boolean;
  physicalMode: "all" | "exclude" | "only";
  geo: string;
  aggLevel: "major" | "minor" | "broad" | "occupation";
  sortBy: string;
  topN: number;
  searchQuery?: string;
  contextSize?: number;
}

export interface ChartRow {
  category: string;
  pct_tasks_affected: number;
  workers_affected: number;
  wages_affected: number;
  rank_workers: number;
  rank_wages: number;
  rank_pct: number;
}

export interface ComputeResponse {
  rows: ChartRow[];
  group_col: string;
  total_categories: number;
  total_emp: number;
  total_wages: number;
  matched_category?: string | null;
}

export interface DatasetEntry {
  name: string;
  date: string;
}

export interface SubType {
  key: string;
  label: string;
  datasets: DatasetEntry[];
}

export interface DatasetCategory {
  key: string;
  label: string;
  sub_types: SubType[];
}

// Public dashboard /api/config — the 5 configs + level/geo options only.
export interface ConfigOption {
  key: string;
  label: string;
}

export interface ConfigResponse {
  configs: ConfigOption[];
  occ_levels: Record<string, string>;    // label -> key (major|minor|broad|occupation)
  wa_levels: Record<string, string>;     // label -> key (gwa|iwa|dwa)
  usage_levels: Record<string, string>;  // label -> key (… + task)
  geo_options: Record<string, string>;
  default_config: string;
}

// ── Data page (exposure + usage + trend) ─────────────────────────────────────
export type MetricKey = "pct_tasks_affected" | "workers_affected" | "wages_affected";

export interface ExposureRow {
  category: string;
  pct_tasks_affected: number;
  workers_affected: number;
  wages_affected: number;
  rank_pct: number;
  rank_workers: number;
  rank_wages: number;
}

export interface ExposureResponse {
  rows: ExposureRow[];
  total_categories: number;
  total_workers: number;
  total_wages: number;
}

export interface UsageRow {
  category: string;
  intensity: number;
  raw_pct: number;
  ratio: number;
}

export interface UsageResponse {
  rows: UsageRow[];
  child_level: string | null;
}

export interface TrendPoint {
  dataset: string;
  date: string;
  rows: ExposureRow[];
}

export interface TrendResponse {
  data_points: TrendPoint[];
  top_categories: string[];
}

// ── Work Activities ───────────────────────────────────────────────────────────

export interface ActivityRow {
  category: string;
  pct_tasks_affected: number;
  workers_affected: number;
  wages_affected: number;
}

export interface ActivityGroup {
  datasets: string[];
  gwa: ActivityRow[];
  iwa: ActivityRow[];
  dwa: ActivityRow[];
}

export interface WorkActivitiesResponse {
  aei_group?: ActivityGroup;
  mcp_group?: ActivityGroup;
}

// ── Trends ────────────────────────────────────────────────────────────────────

export interface TrendRow {
  category: string;
  pct_tasks_affected: number;
  workers_affected: number;
  wages_affected: number;
}

export interface TrendDataPoint {
  dataset: string;
  date: string;
  rows: TrendRow[];
}

export interface TrendSeries {
  name: string;
  data_points: TrendDataPoint[];
  top_categories: string[];
  group_col: string;
}

export interface TrendsResponse {
  series: TrendSeries[];
}

export interface TrendsSettings {
  series: string[];
  method: "freq" | "imp";
  useAutoAug: boolean;
  physicalMode: "all" | "exclude" | "only";
  geo: string;
  aggLevel: "major" | "minor" | "broad" | "occupation";
  topN: number;
  sortBy: string;
}

export interface WATrendsSettings {
  series: string[];
  method: "freq" | "imp";
  useAutoAug: boolean;
  physicalMode: "all" | "exclude" | "only";
  geo: string;
  topN: number;
  sortBy: string;
  activityLevel: "gwa" | "iwa" | "dwa";
}

// ── Explorer shared metrics ───────────────────────────────────────────────────

export interface ExplorerMetrics {
  n_tasks: number;
  n_physical_tasks: number;
  pct_physical?: number | null;
  auto_avg_with_vals?: number | null;
  auto_max_with_vals?: number | null;
  auto_avg_all?: number | null;
  auto_max_all?: number | null;
  pct_avg_with_vals?: number | null;
  pct_max_with_vals?: number | null;
  pct_avg_all?: number | null;
  pct_max_all?: number | null;
  sum_pct_avg?: number | null;
  sum_pct_max?: number | null;
}

// ── Explorer — Occupations ────────────────────────────────────────────────────

export interface OccupationSummary extends ExplorerMetrics {
  title_current: string;
  major?: string;
  minor?: string;
  broad?: string;
  emp?: number | null;
  wage?: number | null;
  dws_star_rating?: number | null;
  job_zone?: number | null;
}

export interface TaskSourceStats {
  auto_aug?: number | null;
  pct_norm?: number | null;
}

export interface McpEntry {
  title: string;
  rating?: number | null;
  url?: string | null;
}

export interface TaskDetail {
  task: string;
  task_normalized: string;
  dwa_title?: string;
  iwa_title?: string;
  gwa_title?: string;
  freq_mean?: number;
  importance?: number;
  relevance?: number;
  physical?: boolean | null;
  sources: Record<string, TaskSourceStats>;
  avg_auto_aug?: number | null;
  max_auto_aug?: number | null;
  avg_pct_norm?: number | null;
  max_pct_norm?: number | null;
  top_mcps?: McpEntry[];
}

export interface OccupationTasksResponse {
  title: string;
  tasks: TaskDetail[];
}

// ── Explorer — Groups (major/minor/broad level pre-computed) ──────────────────

export interface ExplorerGroupRow extends ExplorerMetrics {
  name: string;
  parent?: string | null;
  grandparent?: string | null;
  emp?: number | null;
  wage?: number | null;
  dws_star_rating?: number | null;
  job_zone?: number | null;
  n_occs: number;
}

export interface ExplorerGroupsResponse {
  major: ExplorerGroupRow[];
  minor: ExplorerGroupRow[];
  broad: ExplorerGroupRow[];
}

// ── Explorer — All Tasks ──────────────────────────────────────────────────────

export interface AllTaskRow {
  task: string;
  task_normalized: string;
  dwa_title?: string | null;
  iwa_title?: string | null;
  gwa_title?: string | null;
  physical?: boolean | null;
  n_occs: number;
  emp?: number | null;
  wage?: number | null;
  sources: Record<string, TaskSourceStats>;
  avg_auto_aug?: number | null;
  max_auto_aug?: number | null;
  avg_pct_norm?: number | null;
  max_pct_norm?: number | null;
}

// ── Explorer — All Eco Task Rows (one per task×occ) ──────────────────────────

export interface EcoTaskRow {
  task: string;
  task_normalized: string;
  title_current: string;
  broad_occ?: string | null;
  minor_occ_category?: string | null;
  major_occ_category?: string | null;
  dwa_title?: string | null;
  iwa_title?: string | null;
  gwa_title?: string | null;
  physical?: boolean | null;
  emp?: number | null;
  wage?: number | null;
  emp_freq?: number | null;
  emp_value?: number | null;
  freq_mean?: number | null;
  importance?: number | null;
  relevance?: number | null;
  sources: Record<string, TaskSourceStats>;
  avg_auto_aug?: number | null;
  max_auto_aug?: number | null;
  avg_pct_norm?: number | null;
  max_pct_norm?: number | null;
  top_mcps?: McpEntry[];
}

// ── WA Explorer ───────────────────────────────────────────────────────────────

export interface WAExplorerRow extends ExplorerMetrics {
  level: "gwa" | "iwa" | "dwa";
  name: string;
  parent?: string | null;
  gwa?: string | null;
  emp_freq?: number | null;
  emp_value?: number | null;
  wage_freq?: number | null;
  wage_value?: number | null;
  n_occs: number;
}

export interface WAExplorerResponse {
  rows: WAExplorerRow[];
}

export interface WATaskDetail {
  task: string;
  task_normalized: string;
  dwa_title?: string | null;
  iwa_title?: string | null;
  gwa_title?: string | null;
  physical?: boolean | null;
  emp_freq?: number | null;
  emp_value?: number | null;
  wage_freq?: number | null;
  wage_value?: number | null;
  freq_mean?: number | null;
  importance?: number | null;
  relevance?: number | null;
  title_current?: string | null;
  broad_occ?: string | null;
  minor_occ_category?: string | null;
  major_occ_category?: string | null;
  sources: Record<string, TaskSourceStats>;
  avg_auto_aug?: number | null;
  max_auto_aug?: number | null;
  avg_pct_norm?: number | null;
  max_pct_norm?: number | null;
  top_mcps?: McpEntry[];
}

export interface WATasksResponse {
  level: string;
  name: string;
  tasks: WATaskDetail[];
}

// ── Task Changes ─────────────────────────────────────────────────────────────

export type TaskChangeStatus = "new" | "removed" | "changed" | "unchanged" | "not_in_baseline";

export interface TaskChangeRow {
  task: string;
  task_normalized: string;
  title_current: string;
  broad_occ?: string | null;
  minor_occ_category?: string | null;
  major_occ_category?: string | null;
  dwa_title?: string | null;
  iwa_title?: string | null;
  gwa_title?: string | null;
  physical?: boolean | null;
  freq_mean?: number | null;
  importance?: number | null;
  relevance?: number | null;
  emp?: number | null;
  wage?: number | null;
  status: TaskChangeStatus;
  from_auto_aug?: number | null;
  to_auto_aug?: number | null;
  delta_auto_aug?: number | null;
  from_pct?: number | null;
  to_pct?: number | null;
  delta_pct?: number | null;
  sources: Record<string, TaskSourceStats>;
  avg_auto_aug?: number | null;
  max_auto_aug?: number | null;
  avg_pct_norm?: number | null;
  max_pct_norm?: number | null;
  top_mcps?: McpEntry[];
}

export interface TaskChangesResponse {
  rows: TaskChangeRow[];
  from_dataset: string;
  to_dataset: string;
}

// ── Occupation Report ────────────────────────────────────────────────────────

export type ColorBucket = "high" | "mid" | "low" | "none";

export interface OccReportMcp {
  title: string;
  rating?: number | null;
  url?: string | null;
  description?: string | null;
}

export interface OccReportTask {
  rank: number;
  task: string;
  task_normalized: string;
  importance?: number | null;
  freq_mean?: number | null;
  relevance?: number | null;
  physical: boolean;
  gwa_title?: string | null;
  iwa_title?: string | null;
  dwa_title?: string | null;
  aei_conv_max?: number | null;
  aei_api_max?: number | null;
  microsoft?: number | null;
  mcp?: number | null;
  color_driver?: number | null;
  color_bucket: ColorBucket;
  pct_max?: number | null;
  top_mcps: OccReportMcp[];
}

export interface OccReportWaEcoStats {
  pct_tasks_affected?: number | null;
  workers_affected?: number | null;
  wages_affected?: number | null;
  auto_aug_mean?: number | null;
  rank_pct?: number | null;
  rank_workers?: number | null;
  rank_wages?: number | null;
  rank_auto?: number | null;
  total: number;
}

export interface OccReportWaRow {
  rank: number;
  name: string;
  n_tasks: number;
  aei_conv_max?: number | null;
  aei_api_max?: number | null;
  microsoft?: number | null;
  mcp?: number | null;
  color_driver?: number | null;
  color_bucket: ColorBucket;
  avg_importance?: number | null;
  eco_stats?: OccReportWaEcoStats | null;
}

export interface OccReportWAs {
  gwa: OccReportWaRow[];
  iwa: OccReportWaRow[];
  dwa: OccReportWaRow[];
}

export interface OccReportRiskFlags {
  flag1_pct: number;
  flag2_ska: number;
  flag3_pct_trend: number;
  flag4_ska_trend: number;
  flag5_job_zone: number;
  flag6_outlook: number;
  flag7_n_software: number;
  flag8_auto_aug: number;
}

export interface OccReportRisk {
  score: number;
  tier: "high" | "mod_high" | "mod_low" | "low";
  flags: OccReportRiskFlags;
}

export interface OccReportIntensity {
  occ_intensity_pct?: number | null;
  occ_intensity_rank?: number | null;
  occ_intensity_total?: number | null;
  major_intensity_pct?: number | null;
  major_intensity_rank?: number | null;
  major_intensity_total?: number | null;
}

export interface OccReportHeadline {
  title: string;
  major?: string | null;
  minor?: string | null;
  broad?: string | null;
  job_zone?: number | null;
  dws_star_rating?: number | null;
  n_tasks?: number | null;
  pct_physical?: number | null;
  emp?: number | null;
  wage?: number | null;
  pct_tasks_affected?: number | null;
  workers_affected?: number | null;
  wages_affected?: number | null;
  risk: OccReportRisk;
  intensity: OccReportIntensity;
}

export interface OccReportRanks {
  pct?: number;
  workers?: number;
  wages?: number;
  total: number;
}

export interface OccReportGroupRanks {
  economy: OccReportRanks;
  major?: OccReportRanks | null;
  minor?: OccReportRanks | null;
  broad?: OccReportRanks | null;
}

export interface OccReportTrendPoint {
  dataset: string;
  date: string;
  pct_tasks_affected?: number | null;
}

export interface OccReportSkaRow {
  element: string;
  importance?: number | null;
  level?: number | null;
  occ_score?: number | null;
  ai_top10?: number | null;
  gap?: number | null;
  pct_of_need?: number | null;
  color_bucket: ColorBucket;
}

export interface OccReportSkaSummary {
  skills_pct?: number | null;
  abilities_pct?: number | null;
  knowledge_pct?: number | null;
  overall_pct?: number | null;
}

export interface OccReportSka {
  summary: OccReportSkaSummary;
  rows: {
    skills: OccReportSkaRow[];
    abilities: OccReportSkaRow[];
    knowledge: OccReportSkaRow[];
  };
}

export interface OccReportSector {
  major?: string;
  pct_tasks_affected?: number | null;
  workers_affected?: number | null;
  wages_affected?: number | null;
  rank_pct?: number;
  rank_workers?: number;
  rank_wages?: number;
  n_majors?: number;
}

export interface OccReportSimilar {
  title: string;
  distance?: number | null;
  pct_tasks_affected?: number | null;
  wage?: number | null;
  job_zone?: number | null;
  dws_star_rating?: number | null;
  major?: string | null;
  risk: OccReportRisk | null;
}

export interface OccReportTech {
  software: string;
  commodity: string;
  commodity_rank?: number | null;
  commodity_total: number;
  commodity_avg_pct?: number | null;
}

export interface OccReportSectorChainEntry {
  name: string;
  level: "major" | "minor" | "broad";
  pct_tasks_affected?: number | null;
  workers_affected?: number | null;
  wages_affected?: number | null;
  rank_pct?: number | null;
  rank_workers?: number | null;
  rank_wages?: number | null;
  total: number;
}

export interface OccReportSectorChain {
  major: OccReportSectorChainEntry | null;
  minor: OccReportSectorChainEntry | null;
  broad: OccReportSectorChainEntry | null;
}

export interface OccupationReport {
  title: string;
  geo: string;
  primary_dataset: string;
  headline: OccReportHeadline;
  tasks: OccReportTask[];
  work_activities: OccReportWAs;
  group_ranks: OccReportGroupRanks;
  trend: OccReportTrendPoint[];
  ska: OccReportSka;
  sector: OccReportSector;
  sector_chain: OccReportSectorChain;
  similar: OccReportSimilar[];
  tech: OccReportTech[];
}

export interface OccReportHierarchyEntry {
  title: string;
  major?: string | null;
  minor?: string | null;
  broad?: string | null;
}

export interface OccReportTitlesResponse {
  titles: string[];
  hierarchy: OccReportHierarchyEntry[];
}
