import { Check, CircleAlert, Copy } from "lucide-react";
import { memo, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "../types";
import { Logo } from "./Logo";
import { StatusBadge } from "./StatusBadge";

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      onClick={() =>
        navigator.clipboard?.writeText(text).then(() => {
          setDone(true);
          setTimeout(() => setDone(false), 1200);
        })
      }
      className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-ink-3 hover:bg-surface-2 hover:text-ink"
      aria-label="Copy answer"
    >
      {done ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
      <span>{done ? "Copied" : "Copy"}</span>
    </button>
  );
}

export const MessageItem = memo(function MessageItem({ message: m }: { message: Message }) {
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-brand-soft px-4 py-2.5 text-ink">
          <div className="answer">
            <Markdown remarkPlugins={[remarkGfm]}>{m.content}</Markdown>
          </div>
        </div>
      </div>
    );
  }

  const waiting = m.streaming && !m.content;
  return (
    <div className="group flex gap-3.5">
      <Logo className="mt-0.5 size-7 shrink-0" />
      <div className="min-w-0 flex-1">
        {m.error ? (
          <div className="flex items-start gap-2 rounded-xl bg-bad-soft px-3.5 py-2.5 text-[14px] text-bad">
            <CircleAlert className="mt-0.5 size-4 shrink-0" />
            {m.error}
          </div>
        ) : waiting ? (
          <div className="flex h-7 items-center gap-1" aria-label="Writing an answer">
            <span className="dot size-1.5 rounded-full bg-brand" />
            <span className="dot size-1.5 rounded-full bg-brand" />
            <span className="dot size-1.5 rounded-full bg-brand" />
          </div>
        ) : (
          <div className={`answer ${m.streaming ? "caret" : ""}`}>
            <Markdown remarkPlugins={[remarkGfm]}>{m.content}</Markdown>
          </div>
        )}

        {!m.streaming && !m.error && m.content && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[12px]">
            {m.status && <StatusBadge status={m.status} />}
            {m.status && m.status !== "APPROVED" && m.findings && (
              <span className="text-ink-3">{m.findings.join("; ")}</span>
            )}
            {m.stopped && <span className="text-ink-3">Stopped</span>}
            <span className="opacity-0 transition group-hover:opacity-100">
              <CopyButton text={m.content} />
            </span>
          </div>
        )}
      </div>
    </div>
  );
});
