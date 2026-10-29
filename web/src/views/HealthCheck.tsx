import {
  Ban,
  Calculator,
  Check,
  ClipboardPaste,
  Copy,
  Download,
  FileUp,
  HeartPulse,
  Info,
  LoaderCircle,
  Lock,
  PencilLine,
  Printer,
  RotateCcw,
  Sparkles,
  UserRoundX,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { friendlyError, healthCheck, parseStatement, streamExplain, uploadStatement } from "../api";
import { RATING, RatingPill, ScoreRing } from "../components/Rating";
import { StatusBadge } from "../components/StatusBadge";
import { BALANCE_FIELDS, INCOME_FIELDS, PASTE_EXAMPLE, SAMPLES, type FieldDef } from "../samples";
import type { Fields, Metric, Rating, Report, Status } from "../types";

const STORE = "sentinel.fields";
const ACCEPT = ".pdf,.xlsx,.xlsm,.docx,.csv,.tsv,.txt,.md";

function toNumber(raw: string): number | null {
  const s = raw.trim();
  if (!s) return null;
  const neg = /^\(.*\)$/.test(s) || s.includes("-");
  const n = Number(s.replace(/[^\d.]/g, ""));
  return Number.isFinite(n) ? (neg ? -n : n) : null;
}

const pretty = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 2 });

function toFields(values: Record<string, string>): Fields {
  const out: Fields = {};
  for (const [k, v] of Object.entries(values)) {
    const n = toNumber(v);
    if (n !== null) out[k] = n;
  }
  return out;
}

function loadValues(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? "{}");
  } catch {
    return {};
  }
}

interface Explanation {
  text: string;
  status?: Status;
  findings?: string[];
  streaming: boolean;
  error?: string;
}

function Field({ def, value, onChange, highlight }: { def: FieldDef; value: string; onChange: (v: string) => void; highlight: boolean }) {
  return (
    <label className="block">
      <span className="flex items-center gap-1.5 text-[13px] font-medium text-ink">
        {def.label}
        <span className="group relative">
          <Info className="size-3.5 text-ink-3" />
          <span className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-1.5 w-48 -translate-x-1/2 rounded-lg bg-ink px-2.5 py-1.5 text-[12px] font-normal text-canvas opacity-0 shadow-card transition group-hover:opacity-100">
            {def.hint}
          </span>
        </span>
      </span>
      <input
        inputMode="decimal"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onBlur={() => {
          const n = toNumber(value);
          if (n !== null) onChange(pretty(n));
        }}
        placeholder="0"
        className={`mt-1 w-full rounded-xl border bg-surface px-3 py-2 text-[15px] tabular-nums text-ink outline-none transition placeholder:text-ink-3/60 focus:border-brand/60 focus:ring-4 focus:ring-brand/10 ${highlight ? "border-brand/60 bg-brand-soft" : "border-line"}`}
      />
    </label>
  );
}

function MetricCard({ m }: { m: Metric }) {
  const r = RATING[m.rating];
  return (
    <div className="flex flex-col rounded-2xl border border-line bg-surface p-4 shadow-card">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">{m.group}</div>
          <div className="mt-0.5 text-[13.5px] font-medium text-ink">{m.name}</div>
        </div>
        <RatingPill rating={m.rating} />
      </div>
      <div className={`mt-1.5 text-[28px] font-bold tabular-nums tracking-tight ${r.text}`}>{m.display}</div>
      <p className="mt-1 text-[13px] leading-snug text-ink-2">{m.meaning}</p>
      <div className="mt-auto pt-3">
        <div className="rounded-lg bg-surface-2 px-2.5 py-1.5 text-[12px] tabular-nums text-ink-3" title="How this was calculated">
          {m.formula}
        </div>
      </div>
    </div>
  );
}

// The model writes a summary and then accountant questions; the checklist between them comes
// straight from the ratings so it can never call a weak figure a strength.
const QUESTIONS = /\n\s*(?:#+\s*|\*\*)?Questions to ask/i;

function splitExplanation(text: string): [string, string] {
  const i = text.search(QUESTIONS);
  if (i < 0) return [text, ""];
  // v2 sometimes writes the questions on one line as "• a? • b?"; turn them back into a list.
  return [text.slice(0, i), text.slice(i).replace(/\s*•\s+/g, "\n- ")];
}

function Checklist({ metrics }: { metrics: Metric[] }) {
  const cols = [
    { title: "What's going well", items: metrics.filter((m) => m.rating === "good"), empty: "None of the checks look good yet." },
    { title: "What to watch", items: metrics.filter((m) => m.rating !== "good"), empty: "Nothing stands out. Keep it up." },
  ];
  return (
    <div className="my-4 grid gap-3 sm:grid-cols-2">
      {cols.map((c) => (
        <div key={c.title} className="rounded-2xl bg-surface-2 p-4">
          <div className="text-[13.5px] font-semibold">{c.title}</div>
          {c.items.length ? (
            <ul className="mt-2 space-y-1.5">
              {c.items.map((m) => (
                <li key={m.id} className="flex items-center gap-2 text-[13.5px]">
                  <span className={`size-1.5 shrink-0 rounded-full ${RATING[m.rating].bar}`} />
                  <span className="text-ink-2">{m.name}</span>
                  <span className={`ml-auto font-semibold tabular-nums ${RATING[m.rating].text}`}>{m.display}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-[13.5px] text-ink-3">{c.empty}</p>
          )}
        </div>
      ))}
    </div>
  );
}

function Placeholder() {
  return (
    <div className="flex h-full min-h-[420px] flex-col items-center justify-center rounded-3xl border border-dashed border-line bg-surface/50 p-8 text-center">
      <span className="grid size-14 place-items-center rounded-2xl bg-brand-soft text-brand">
        <HeartPulse className="size-7" />
      </span>
      <h2 className="mt-4 text-[18px] font-semibold">Your health check will appear here</h2>
      <p className="mt-1.5 max-w-sm text-[14px] text-ink-2">
        Add your numbers or try a sample business. You'll get:
      </p>
      <ul className="mt-4 space-y-2 text-left text-[14px] text-ink-2">
        {[
          "A health score from 0 to 100",
          "Growth, profit, cash and debt checks, each rated",
          "The exact calculation behind every number",
          "A plain-English summary and questions for your accountant",
        ].map((t) => (
          <li key={t} className="flex gap-2">
            <Check className="mt-0.5 size-4 shrink-0 text-brand" />
            {t}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function HealthCheck() {
  const [values, setValues] = useState<Record<string, string>>(loadValues);
  const [tab, setTab] = useState<"form" | "upload">("form");
  const [paste, setPaste] = useState("");
  const [showPaste, setShowPaste] = useState(false);
  const [uploading, setUploading] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [detected, setDetected] = useState<string[]>([]);
  const [foundIn, setFoundIn] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const reportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORE, JSON.stringify(values));
    } catch {}
  }, [values]);

  const run = async (fields: Fields) => {
    abort.current?.abort();
    setError(null);
    setLoading(true);
    try {
      const r = await healthCheck(fields);
      if (!r.metrics.length) {
        setReport(null);
        setError("Add at least your revenue and one more figure, like net profit, to get a health check.");
        return;
      }
      setReport(r);
      if (window.innerWidth < 1024) setTimeout(() => reportRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
      const ctrl = new AbortController();
      abort.current = ctrl;
      setExplanation({ text: "", streaming: true });
      const done = await streamExplain(fields, (e) => setExplanation({ text: e.text, streaming: true }), ctrl.signal);
      setExplanation({ text: done.text, status: done.status, findings: done.findings, streaming: false });
    } catch (err) {
      if (abort.current?.signal.aborted) return;
      setExplanation((x) => (x ? { ...x, streaming: false, error: friendlyError(err) } : x));
      if (!report) setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  };

  // A document replaces whatever was typed before, fills the form for checking and runs the check.
  const applyFound = (fields: Fields, source: string) => {
    const keys = Object.keys(fields);
    if (!keys.length) {
      setError(`Couldn't find any line items in ${source}. It needs labels like “Revenue”, “Net income” or “Current assets” next to the numbers.`);
      return;
    }
    setValues(Object.fromEntries(keys.map((k) => [k, pretty(fields[k])])));
    setDetected(keys);
    setFoundIn(source);
    setTab("form");
    run(fields);
  };

  const readFile = async (file: File) => {
    setError(null);
    setUploading(file.name);
    try {
      applyFound(await uploadStatement(file), file.name);
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setUploading(null);
    }
  };

  const readPaste = async () => {
    setError(null);
    try {
      applyFound(await parseStatement(paste), "the pasted text");
    } catch (err) {
      setError(friendlyError(err));
    }
  };

  const reset = () => {
    abort.current?.abort();
    setValues({});
    setDetected([]);
    setReport(null);
    setExplanation(null);
    setError(null);
  };

  const copy = () => {
    if (!report) return;
    const lines = [
      `Financial health check: ${report.score}/100 (${report.label})`,
      "",
      ...report.metrics.map((m) => `${m.name}: ${m.display} (${RATING[m.rating].label}) = ${m.formula}`),
      ...(explanation?.text ? ["", explanation.text] : []),
    ];
    navigator.clipboard?.writeText(lines.join("\n")).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  const scoreRating: Rating | null = report?.score == null ? null : report.score >= 75 ? "good" : report.score >= 50 ? "watch" : "risk";
  const concern = report?.metrics.find((m) => m.rating === "risk") ?? report?.metrics.find((m) => m.rating === "watch");
  const counts = report?.metrics.reduce((a, m) => ({ ...a, [m.rating]: (a[m.rating] ?? 0) + 1 }), {} as Record<string, number>);

  return (
    <div className="mx-auto w-full max-w-7xl px-4 pb-16 sm:px-6 lg:px-8">
      <section className="no-print py-8 sm:py-12">
        <h1 className="max-w-3xl text-[30px] font-bold leading-[1.15] tracking-tight sm:text-[42px]">
          Is your business <span className="text-brand">financially healthy?</span>
        </h1>
        <p className="mt-3 max-w-2xl text-[16px] leading-relaxed text-ink-2 sm:text-[17px]">
          Type in a few numbers from your profit & loss and balance sheet, or upload the statement. Get a clear health
          check with every calculation explained, in seconds.
        </p>
        <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2 text-[13px] font-medium text-ink-2">
          {[
            [UserRoundX, "No sign-up"],
            [Lock, "Nothing saved on our servers"],
            [Calculator, "Exact math, shown step by step"],
            [Ban, "No investment advice"],
          ].map(([Icon, text]) => {
            const I = Icon as typeof Lock;
            return (
              <span key={text as string} className="inline-flex items-center gap-1.5">
                <I className="size-4 text-brand" />
                {text as string}
              </span>
            );
          })}
        </div>
      </section>

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,430px)_minmax(0,1fr)] xl:gap-8">
        {/* Inputs */}
        <section className="no-print rounded-3xl border border-line bg-surface p-5 shadow-card sm:p-6 lg:sticky lg:top-20">
          <div className="flex items-center justify-between">
            <h2 className="text-[16px] font-semibold">Your numbers</h2>
            <button onClick={reset} className="inline-flex items-center gap-1 text-[13px] text-ink-3 hover:text-ink">
              <RotateCcw className="size-3.5" /> Clear
            </button>
          </div>

          <div className="mt-1 text-[13px] text-ink-2">How would you like to add them?</div>
          <div className="mt-3 grid grid-cols-2 gap-2.5" role="tablist">
            {(
              [
                ["form", PencilLine, "Type them in", "About 2 minutes"],
                ["upload", FileUp, "Upload a document", "PDF, Excel, Word, CSV"],
              ] as const
            ).map(([t, Icon, title, sub]) => (
              <button
                key={t}
                role="tab"
                aria-selected={tab === t}
                onClick={() => setTab(t)}
                className={`flex flex-col items-start gap-2 rounded-2xl border p-3 text-left transition ${tab === t ? "border-brand bg-brand-soft ring-1 ring-brand" : "border-line hover:border-brand/50 hover:bg-surface-2"}`}
              >
                <Icon className={`size-5 ${tab === t ? "text-brand" : "text-ink-3"}`} />
                <span>
                  <span className="block text-[14px] font-semibold text-ink">{title}</span>
                  <span className="block text-[12px] text-ink-3">{sub}</span>
                </span>
              </button>
            ))}
          </div>

          {tab === "form" ? (
            <div className="mt-5 space-y-6">
              {detected.length > 0 && (
                <div className="rounded-xl bg-brand-soft px-3 py-2 text-[13px] text-ink">
                  Found {detected.length} line items in <span className="font-medium">{foundIn}</span>. Check them below; if
                  you change anything, run the check again.
                </div>
              )}
              {[
                ["Profit & loss", INCOME_FIELDS],
                ["Balance sheet", BALANCE_FIELDS],
              ].map(([title, defs]) => (
                <fieldset key={title as string}>
                  <legend className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-ink-3">{title as string}</legend>
                  <div className="grid grid-cols-2 gap-3">
                    {(defs as FieldDef[]).map((d) => (
                      <Field
                        key={d.id}
                        def={d}
                        value={values[d.id] ?? ""}
                        highlight={detected.includes(d.id)}
                        onChange={(v) => setValues((s) => ({ ...s, [d.id]: v }))}
                      />
                    ))}
                  </div>
                </fieldset>
              ))}
              <p className="text-[12px] text-ink-3">Use the same unit for every figure (e.g. all in dollars or all in thousands). Leave blank what you don't know.</p>
            </div>
          ) : (
            <div className="mt-5">
              <input
                ref={fileInput}
                type="file"
                accept={ACCEPT}
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  e.target.value = "";
                  if (f) readFile(f);
                }}
              />
              <button
                onClick={() => fileInput.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  const f = e.dataTransfer.files[0];
                  if (f) readFile(f);
                }}
                disabled={!!uploading}
                className={`flex w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed px-4 py-9 text-center transition ${dragging ? "border-brand bg-brand-soft" : "border-line hover:border-brand/60 hover:bg-surface-2"}`}
              >
                {uploading ? (
                  <>
                    <LoaderCircle className="size-8 animate-spin text-brand" />
                    <span className="mt-3 text-[14px] font-semibold">Reading {uploading}…</span>
                  </>
                ) : (
                  <>
                    <span className="grid size-12 place-items-center rounded-2xl bg-brand-soft text-brand">
                      <FileUp className="size-6" />
                    </span>
                    <span className="mt-3 text-[14px] font-semibold">Choose your statement<span className="hidden sm:inline"> or drop it here</span></span>
                    <span className="mt-1 text-[12.5px] text-ink-3">P&L, balance sheet or both · PDF, Excel, Word, CSV · up to 10 MB</span>
                  </>
                )}
              </button>
              <p className="mt-3 flex gap-1.5 text-[12px] text-ink-3">
                <Lock className="mt-px size-3.5 shrink-0" />
                Read once to find the numbers, then discarded. Scanned PDFs (photos of paper) can't be read yet.
              </p>
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[12.5px]">
                <a href="/sample-statement.pdf" download className="inline-flex items-center gap-1 font-medium text-brand hover:underline">
                  <Download className="size-3.5" /> Download a sample statement
                </a>
                <button onClick={() => setShowPaste((s) => !s)} className="inline-flex items-center gap-1 font-medium text-ink-2 hover:text-ink">
                  <ClipboardPaste className="size-3.5" /> {showPaste ? "Hide paste box" : "Or paste the text"}
                </button>
              </div>
              {showPaste && (
                <div className="mt-3">
                  <textarea
                    value={paste}
                    onChange={(e) => setPaste(e.target.value)}
                    rows={8}
                    placeholder={PASTE_EXAMPLE}
                    className="w-full resize-y rounded-xl border border-line bg-surface px-3 py-2.5 font-mono text-[13px] leading-relaxed text-ink outline-none placeholder:text-ink-3/70 focus:border-brand/60 focus:ring-4 focus:ring-brand/10"
                  />
                  <button
                    onClick={readPaste}
                    disabled={!paste.trim()}
                    className="mt-2 inline-flex w-full items-center justify-center gap-2 rounded-xl border border-line bg-surface-2 py-2.5 text-[14px] font-semibold text-ink transition hover:bg-line disabled:opacity-40"
                  >
                    Read numbers
                  </button>
                </div>
              )}
            </div>
          )}

          {error && <div className="mt-4 rounded-xl bg-bad-soft px-3 py-2.5 text-[13px] text-bad">{error}</div>}

          {tab === "form" && (
            <button
              onClick={() => run(toFields(values))}
              disabled={loading && !explanation}
              className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-brand py-3 text-[15px] font-semibold text-brand-ink shadow-card transition hover:bg-brand-strong disabled:opacity-60"
            >
              {loading && !report ? <LoaderCircle className="size-4 animate-spin" /> : <HeartPulse className="size-4" />}
              Run health check
            </button>
          )}

          <div className="mt-5 border-t border-line pt-4">
            <div className="text-[12px] font-medium text-ink-3">No numbers handy? Try a sample business</div>
            <div className="mt-2 flex flex-wrap gap-2">
              {SAMPLES.map((s) => (
                <button
                  key={s.name}
                  onClick={() => {
                    setValues(Object.fromEntries(Object.entries(s.fields).map(([k, v]) => [k, pretty(v)])));
                    setDetected([]);
                    setTab("form");
                    run(s.fields);
                  }}
                  className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface-2 px-3 py-1.5 text-[13px] font-medium text-ink transition hover:border-brand/50 hover:bg-brand-soft"
                >
                  <s.icon className="size-3.5 text-brand" />
                  {s.name}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Report */}
        <section ref={reportRef} className="min-w-0 scroll-mt-20">
          {!report || report.score == null || !scoreRating ? (
            <Placeholder />
          ) : (
            <div className="space-y-6">
              <div className="flex flex-col gap-5 rounded-3xl border border-line bg-surface p-5 shadow-card sm:flex-row sm:items-center sm:p-6">
                <ScoreRing score={report.score} rating={scoreRating} />
                <div className="min-w-0 flex-1">
                  <div className="text-[12px] font-semibold uppercase tracking-wider text-ink-3">Health score</div>
                  <div className={`mt-0.5 text-[26px] font-bold tracking-tight ${RATING[scoreRating].text}`}>{report.label}</div>
                  <p className="mt-1 text-[14px] text-ink-2">
                    {counts?.good ?? 0} of {report.metrics.length} checks look good
                    {counts?.watch ? `, ${counts.watch} to watch` : ""}
                    {counts?.risk ? `, ${counts.risk} at risk` : ""}.
                  </p>
                  <div className="mt-3 flex h-2 max-w-sm gap-0.5 overflow-hidden rounded-full" aria-hidden>
                    {(["good", "watch", "risk"] as const).map((k) =>
                      counts?.[k] ? <div key={k} className={RATING[k].bar} style={{ flexGrow: counts[k] }} /> : null,
                    )}
                  </div>
                  {concern && (
                    <p className="mt-3 text-[13.5px] text-ink">
                      <span className="text-ink-3">{concern.rating === "risk" ? "Biggest concern" : "Worth watching"}:</span>{" "}
                      <span className="font-semibold">{concern.name}</span>{" "}
                      <span className={`font-semibold tabular-nums ${RATING[concern.rating].text}`}>{concern.display}</span>
                    </p>
                  )}
                  {report.missing.length > 0 && (
                    <p className="mt-1 text-[12.5px] text-ink-3">Add {report.missing.join(", ")} to unlock more checks.</p>
                  )}
                </div>
                <div className="no-print flex gap-2 sm:flex-col">
                  <button onClick={copy} className="inline-flex items-center gap-1.5 rounded-xl border border-line px-3 py-2 text-[13px] font-medium hover:bg-surface-2">
                    {copied ? <Check className="size-4" /> : <Copy className="size-4" />} {copied ? "Copied" : "Copy report"}
                  </button>
                  <button onClick={() => window.print()} className="inline-flex items-center gap-1.5 rounded-xl border border-line px-3 py-2 text-[13px] font-medium hover:bg-surface-2">
                    <Printer className="size-4" /> Save as PDF
                  </button>
                </div>
              </div>

              {explanation && (
                <div className="rounded-3xl border border-line bg-surface p-5 shadow-card sm:p-6">
                  <div className="mb-3 flex flex-wrap items-center gap-2">
                    <Sparkles className="size-4 text-brand" />
                    <h3 className="text-[15px] font-semibold">What this means for you</h3>
                    {explanation.status && <StatusBadge status={explanation.status} />}
                    {explanation.streaming && <span className="text-[12px] text-ink-3">Writing…</span>}
                  </div>
                  {explanation.error ? (
                    <p className="text-[14px] text-bad">{explanation.error} The numbers above are still correct.</p>
                  ) : explanation.text ? (
                    (() => {
                      const [summary, questions] = splitExplanation(explanation.text);
                      return (
                        <>
                          <div className={`answer ${explanation.streaming && !questions ? "caret" : ""}`}>
                            <Markdown remarkPlugins={[remarkGfm]}>{summary}</Markdown>
                          </div>
                          <Checklist metrics={report.metrics} />
                          {questions && (
                            <div className={`answer ${explanation.streaming ? "caret" : ""}`}>
                              <Markdown remarkPlugins={[remarkGfm]}>{questions}</Markdown>
                            </div>
                          )}
                        </>
                      );
                    })()
                  ) : (
                    <div className="space-y-2.5" aria-label="Writing summary">
                      {[92, 80, 86, 60].map((w) => (
                        <div key={w} className="h-3.5 animate-pulse rounded-full bg-surface-2" style={{ width: `${w}%` }} />
                      ))}
                    </div>
                  )}
                  {explanation.findings && explanation.findings.length > 0 && (
                    <p className="mt-3 text-[12px] text-ink-3">Compliance check: {explanation.findings.join("; ")}.</p>
                  )}
                  <p className="mt-4 border-t border-line pt-3 text-[12px] text-ink-3">
                    The figures are calculated exactly; the summary is written by AI and can be imperfect. This is general
                    information, not financial or investment advice.
                  </p>
                </div>
              )}

              <div>
                <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-ink-3">
                  Every check, with its calculation
                </h3>
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  {report.metrics.map((m) => (
                    <MetricCard key={m.id} m={m} />
                  ))}
                </div>
              </div>

            </div>
          )}
        </section>
      </div>
    </div>
  );
}
