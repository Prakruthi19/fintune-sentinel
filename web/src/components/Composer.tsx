import { ArrowUp, Square } from "lucide-react";
import { useEffect, useRef } from "react";

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  busy: boolean;
}

export function Composer({ value, onChange, onSend, onStop, busy }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [value]);

  useEffect(() => ref.current?.focus(), [busy]);

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-4 sm:px-6">
      <div className="rounded-2xl border border-line bg-surface shadow-card transition focus-within:border-brand/50 focus-within:ring-4 focus-within:ring-brand/10">
        <textarea
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              if (!busy) onSend();
            }
          }}
          placeholder="Ask anything about finance or accounting…"
          className="block w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-[15px] leading-relaxed text-ink outline-none placeholder:text-ink-3"
        />
        <div className="flex items-center justify-between px-3 pb-2.5 pt-1">
          <span className="hidden pl-1 text-[12px] text-ink-3 sm:block">
            <kbd className="font-sans font-medium">Enter</kbd> to send · <kbd className="font-sans font-medium">Shift + Enter</kbd> for a new line
          </span>
          {busy ? (
            <button
              onClick={onStop}
              className="ml-auto inline-flex items-center gap-1.5 rounded-xl border border-line bg-surface-2 px-3 py-2 text-[13px] font-semibold text-ink hover:bg-line"
            >
              <Square className="size-3 fill-current" />
              Stop
            </button>
          ) : (
            <button
              onClick={onSend}
              disabled={!value.trim()}
              className="ml-auto grid size-9 place-items-center rounded-xl bg-brand text-brand-ink transition hover:bg-brand-strong disabled:cursor-not-allowed disabled:opacity-35"
              aria-label="Send"
            >
              <ArrowUp className="size-[18px]" strokeWidth={2.4} />
            </button>
          )}
        </div>
      </div>
      <p className="mt-2 text-center text-[11.5px] text-ink-3">
        Not investment advice. Powered by a fine-tuned Llama 3.2 3B running on CPU.
      </p>
    </div>
  );
}
