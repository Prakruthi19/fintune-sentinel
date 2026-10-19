export type Status = "APPROVED" | "FLAGGED" | "REJECTED";
export type Rating = "good" | "watch" | "risk";

export interface Stats {
  tokens: number;
  elapsed: number;
  first: number | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: Status;
  findings?: string[];
  stats?: Stats;
  streaming?: boolean;
  stopped?: boolean;
  error?: string;
}

export interface Metric {
  id: string;
  name: string;
  group: string;
  value: number | null;
  display: string;
  rating: Rating;
  formula: string;
  meaning: string;
}

export interface Report {
  score: number | null;
  label: string | null;
  metrics: Metric[];
  missing: string[];
}

export type Fields = Record<string, number>;
