import type { Fields, Message, Report, Stats, Status } from "./types";

export interface TokenEvent extends Stats {
  text: string;
}

export interface DoneEvent extends TokenEvent {
  status: Status;
  findings: string[];
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`The server answered ${r.status}.`);
  return r.json();
}

export const ping = () => fetch("/api/info").then((r) => r.ok);
export const parseStatement = (text: string) => post<{ fields: Fields }>("/api/parse", { text }).then((r) => r.fields);
export const healthCheck = (fields: Fields) => post<Report>("/api/health", { fields });

/** Sends a statement file; the server reads it in memory and returns the line items it found. */
export async function uploadStatement(file: File): Promise<Fields> {
  const body = new FormData();
  body.append("file", file);
  const r = await fetch("/api/upload", { method: "POST", body });
  if (r.status === 422) throw new Error((await r.json()).detail);
  if (!r.ok) throw new Error(`The server answered ${r.status}.`);
  return (await r.json()).fields;
}

/** Reads a server-sent event stream from a POST endpoint until its `done` event. */
async function stream(url: string, body: unknown, onToken: (e: TokenEvent) => void, signal: AbortSignal): Promise<DoneEvent> {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });
  if (!r.ok || !r.body) throw new Error(`The server answered ${r.status}.`);

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const name = /^event: (.*)$/m.exec(block)?.[1];
      const data = /^data: (.*)$/m.exec(block)?.[1];
      if (!data) continue;
      if (name === "token") onToken(JSON.parse(data));
      if (name === "done") return JSON.parse(data);
    }
  }
  throw new Error("The answer stopped before it was complete.");
}

export const streamExplain = (fields: Fields, onToken: (e: TokenEvent) => void, signal: AbortSignal) =>
  stream("/api/explain", { fields }, onToken, signal);

export const streamChat = (history: Message[], onToken: (e: TokenEvent) => void, signal: AbortSignal) =>
  stream(
    "/api/chat",
    { messages: history.filter((m) => !m.error).map((m) => ({ role: m.role, content: m.content })) },
    onToken,
    signal,
  );

export function friendlyError(err: unknown) {
  return err instanceof TypeError ? "Couldn't reach the server. Check your connection and try again." : String((err as Error).message);
}
