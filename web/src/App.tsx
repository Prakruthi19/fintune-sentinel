import { GraduationCap, HeartPulse, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { ping } from "./api";
import { Logo } from "./components/Logo";
import { Ask } from "./views/Ask";
import { HealthCheck } from "./views/HealthCheck";

type View = "check" | "ask";
const viewFromHash = (): View => (location.hash === "#ask" ? "ask" : "check");

export default function App() {
  const [view, setView] = useState<View>(viewFromHash);
  const [online, setOnline] = useState<boolean | null>(null);
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));

  useEffect(() => {
    ping().then(setOnline, () => setOnline(false));
    const onHash = () => setView(viewFromHash());
    addEventListener("hashchange", onHash);
    return () => removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("theme", dark ? "dark" : "light");
    } catch {}
  }, [dark]);

  const tabs = [
    { id: "check" as const, label: "Health check", short: "Check", icon: HeartPulse },
    { id: "ask" as const, label: "Ask a question", short: "Ask", icon: GraduationCap },
  ];

  return (
    <div className="flex min-h-full flex-col">
      <header className="no-print sticky top-0 z-20 border-b border-line bg-canvas/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center gap-3 px-4 sm:px-6 lg:px-8">
          <a href="#" className="flex items-center gap-2.5" onClick={() => setView("check")}>
            <Logo />
            <span className="text-[16px] font-bold tracking-tight">Sentinel</span>
          </a>
          <nav className="ml-2 flex items-center gap-1 rounded-xl bg-surface-2 p-1 sm:ml-6">
            {tabs.map((t) => (
              <a
                key={t.id}
                href={t.id === "ask" ? "#ask" : "#"}
                onClick={() => setView(t.id)}
                className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[13.5px] font-medium transition sm:px-3 ${view === t.id ? "bg-surface text-ink shadow-card" : "text-ink-2 hover:text-ink"}`}
              >
                <t.icon className="size-4" />
                <span className="hidden sm:inline">{t.label}</span>
                <span className="sm:hidden">{t.short}</span>
              </a>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            {online === false && (
              <span className="hidden rounded-full bg-bad-soft px-2.5 py-1 text-[12px] font-medium text-bad sm:inline">Server offline</span>
            )}
            <button
              onClick={() => setDark((d) => !d)}
              className="rounded-lg p-2 text-ink-2 hover:bg-surface-2"
              aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
            >
              {dark ? <Sun className="size-[18px]" /> : <Moon className="size-[18px]" />}
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1">{view === "check" ? <HealthCheck /> : <Ask />}</main>

      {view === "check" && (
        <footer className="no-print border-t border-line">
          <div className="mx-auto flex max-w-7xl flex-col gap-1 px-4 py-6 text-[12.5px] text-ink-3 sm:flex-row sm:justify-between sm:px-6 lg:px-8">
            <span>Free to use · General information, not financial or investment advice.</span>
            <span>
              Explanations by a fine-tuned Llama 3.2 3B, running on CPU ·{" "}
              <a href="https://github.com/Prakruthi19/fintune-sentinel" className="underline-offset-2 hover:text-ink hover:underline">
                Source
              </a>
            </span>
          </div>
        </footer>
      )}
    </div>
  );
}
