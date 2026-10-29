import { GraduationCap, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { friendlyError, streamChat } from "../api";
import { Composer } from "../components/Composer";
import { MessageItem } from "../components/MessageItem";
import { QUESTIONS } from "../examples";
import type { Message } from "../types";

const STORE = "sentinel.ask";
const uid = () => Math.random().toString(36).slice(2, 10);

function load(): Message[] {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? "[]");
  } catch {
    return [];
  }
}

export function Ask() {
  const [messages, setMessages] = useState<Message[]>(load);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORE, JSON.stringify(messages.filter((m) => !m.streaming)));
    } catch {}
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  useEffect(() => () => abort.current?.abort(), []);

  const patch = (id: string, change: Partial<Message>) =>
    setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...change } : m)));

  const send = useCallback(
    async (text: string) => {
      const prompt = text.trim();
      if (!prompt || busy) return;
      const user: Message = { id: uid(), role: "user", content: prompt };
      const answer: Message = { id: uid(), role: "assistant", content: "", streaming: true };
      const history = [...messages, user];
      setMessages([...history, answer]);
      setInput("");
      setBusy(true);
      const ctrl = new AbortController();
      abort.current = ctrl;
      try {
        const done = await streamChat(history, (e) => patch(answer.id, { content: e.text }), ctrl.signal);
        patch(answer.id, { content: done.text, status: done.status, findings: done.findings, streaming: false });
      } catch (err) {
        patch(answer.id, ctrl.signal.aborted ? { streaming: false, stopped: true } : { streaming: false, error: friendlyError(err) });
      } finally {
        setBusy(false);
      }
    },
    [busy, messages],
  );

  return (
    <div className="flex min-h-[calc(100dvh-4rem)] flex-col">
      <div className="mx-auto w-full max-w-3xl flex-1 px-4 sm:px-6">
        {messages.length === 0 ? (
          <div className="py-10 sm:py-16">
            <span className="grid size-12 place-items-center rounded-2xl bg-brand-soft text-brand">
              <GraduationCap className="size-6" />
            </span>
            <h1 className="mt-5 text-[30px] font-bold leading-tight tracking-tight sm:text-[38px]">Ask a finance question</h1>
            <p className="mt-2 max-w-xl text-[16px] leading-relaxed text-ink-2">
              Studying accounting or just confused by a term? Get a clear explanation with the working shown.
            </p>
            <div className="mt-8 grid gap-2.5 sm:grid-cols-2">
              {QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="rounded-2xl border border-line bg-surface px-4 py-3 text-left text-[14px] text-ink shadow-card transition hover:-translate-y-0.5 hover:border-brand/40"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-6 py-8">
            <div className="flex justify-end">
              <button
                onClick={() => (abort.current?.abort(), setMessages([]))}
                className="inline-flex items-center gap-1 text-[13px] text-ink-3 hover:text-ink"
              >
                <RotateCcw className="size-3.5" /> New conversation
              </button>
            </div>
            {messages.map((m) => (
              <MessageItem key={m.id} message={m} />
            ))}
            <div ref={bottom} />
          </div>
        )}
      </div>
      <div className="sticky bottom-0 bg-gradient-to-t from-canvas from-70% to-transparent pt-4">
        <Composer value={input} onChange={setInput} onSend={() => send(input)} onStop={() => abort.current?.abort()} busy={busy} />
      </div>
    </div>
  );
}
