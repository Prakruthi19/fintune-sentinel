export function Logo({ className = "size-8" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect width="32" height="32" rx="9" className="fill-brand" />
      <path
        d="M16 6.5l7.5 2.8v5.9c0 4.8-3.2 8.9-7.5 10.5-4.3-1.6-7.5-5.7-7.5-10.5V9.3z"
        fill="none"
        className="stroke-brand-ink"
        strokeWidth="2.1"
        strokeLinejoin="round"
      />
      <path
        d="M12.6 15.8l2.4 2.4 4.4-4.8"
        fill="none"
        className="stroke-brand-ink"
        strokeWidth="2.1"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
