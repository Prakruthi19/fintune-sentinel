import type { Rating } from "../types";

export const RATING = {
  good: { label: "Good", text: "text-ok", bg: "bg-ok-soft", bar: "bg-ok", stroke: "stroke-ok" },
  watch: { label: "Watch", text: "text-warn", bg: "bg-warn-soft", bar: "bg-warn", stroke: "stroke-warn" },
  risk: { label: "Risk", text: "text-bad", bg: "bg-bad-soft", bar: "bg-bad", stroke: "stroke-bad" },
} as const;

export function RatingPill({ rating }: { rating: Rating }) {
  const r = RATING[rating];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11.5px] font-semibold ${r.text} ${r.bg}`}>
      <span className={`size-1.5 rounded-full ${r.bar}`} />
      {r.label}
    </span>
  );
}

export function ScoreRing({ score, rating }: { score: number; rating: Rating }) {
  const c = 2 * Math.PI * 42;
  return (
    <svg viewBox="0 0 100 100" className="size-28 shrink-0 -rotate-90" aria-label={`Health score ${score} out of 100`}>
      <circle cx="50" cy="50" r="42" fill="none" strokeWidth="9" className="stroke-surface-2" />
      <circle
        cx="50"
        cy="50"
        r="42"
        fill="none"
        strokeWidth="9"
        strokeLinecap="round"
        strokeDasharray={`${(score / 100) * c} ${c}`}
        className={`${RATING[rating].stroke} transition-[stroke-dasharray] duration-700`}
      />
      <text x="50" y="50" dy="0.35em" textAnchor="middle" className="rotate-90 fill-ink text-[26px] font-bold" style={{ transformOrigin: "50px 50px" }}>
        {score}
      </text>
    </svg>
  );
}
