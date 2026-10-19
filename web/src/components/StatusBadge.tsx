import { ShieldAlert, ShieldCheck, ShieldX } from "lucide-react";
import type { Status } from "../types";

export const STATUS = {
  APPROVED: {
    icon: ShieldCheck,
    label: "Approved",
    text: "text-ok",
    bg: "bg-ok-soft",
    ring: "ring-ok/25",
    summary: "Passed every compliance check.",
  },
  FLAGGED: {
    icon: ShieldAlert,
    label: "Flagged",
    text: "text-warn",
    bg: "bg-warn-soft",
    ring: "ring-warn/25",
    summary: "Shown, but some figures have no unit.",
  },
  REJECTED: {
    icon: ShieldX,
    label: "Rejected",
    text: "text-bad",
    bg: "bg-bad-soft",
    ring: "ring-bad/25",
    summary: "Trade advice was redacted before display.",
  },
} as const;

export function StatusBadge({ status }: { status: Status }) {
  const s = STATUS[status];
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11.5px] font-semibold ring-1 ring-inset ${s.text} ${s.bg} ${s.ring}`}
    >
      <s.icon className="size-3.5" strokeWidth={2.4} />
      {s.label}
    </span>
  );
}
