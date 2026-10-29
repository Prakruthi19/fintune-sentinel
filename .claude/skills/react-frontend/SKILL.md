---
name: react-frontend
description: How to build, change and verify the Sentinel web app in web/ (Vite + React 19 + TypeScript + Tailwind v4, talking to app.py over /api). Use for any change to the UI, its theme, layout, copy, or the /api contract it consumes.
---

# Sentinel web app

## Product

Sentinel is a free **financial health check** for small business owners, with a secondary **Ask a question** mode for finance students. Write every screen for someone with no finance background: plain words, no model or infrastructure jargon in the main UI. Technical credits (Llama 3.2 3B, CPU, llama.cpp) belong only in the footer.

## Stack and layout

- `web/` is a Vite + React 19 + TypeScript app styled with Tailwind v4 (`@tailwindcss/vite`), icons from `lucide-react`, model text rendered with `react-markdown` + `remark-gfm`.
- `app.py` (FastAPI) serves the built UI from `web/dist` and the API under `/api`. The Dockerfile builds the UI in a Node stage, so a deploy needs no committed `dist/`.
- Files:
  - `src/App.tsx`: the header (logo, Health check / Ask tabs, theme toggle), the hash routing (`#ask`) and the footer.
  - `src/views/HealthCheck.tsx`: the two input options (type the numbers in, or upload a document, with paste as a fallback inside upload), the sample businesses; the report reads top to bottom as score card (breakdown bar, biggest concern), "What this means for you" (model summary, then a good/watch checklist built from the ratings, then the model's accountant questions) and one grid of metric cards.
  - `src/views/Ask.tsx`: the chat for students.
  - `src/api.ts`: the endpoint wrappers and the shared SSE reader.
  - `src/samples.ts`: the form fields (labels and hints) and the sample businesses.
  - `src/examples.ts`: the starter questions for Ask.
  - `src/components/`: `MessageItem`, `Composer`, `Rating` (pill and score ring), `StatusBadge`, `Logo`.
  - `src/index.css`: design tokens, `.answer` markdown styles and print styles.

## API contract (keep both sides in sync; tests in `tests/test_app.py`, `tests/test_health.py`)

- `POST /api/parse {text}` → `{fields}`: line items read from a pasted statement (`sentinel/health.py:parse_statement`).
- `POST /api/upload` (multipart `file`) → `{fields, filename}`: text is pulled from a PDF, .xlsx, .docx, CSV or text file in memory (`sentinel/documents.py`), then run through `parse_statement`. A file it can't read returns 422 with a user-facing `detail`.
- `POST /api/health {fields}` → `{score, label, metrics[], missing[]}`. A metric is `{id, name, group, value, display, rating: good|watch|risk, formula, meaning}`. All math happens in Python, never in the model. The model only writes the summary and the accountant questions (`explain_prompt`); which figures are good or weak is always shown from `rating`, never from model text, because the 3B model mislabels them.
- `POST /api/explain {fields}` and `POST /api/chat {messages}` both return `text/event-stream`:
  - `event: token`, `data: {text, tokens, elapsed, first}`. `text` is the whole answer so far, already redacted.
  - `event: done`, `data: {..., status, findings}`. `status` is `APPROVED | FLAGGED | REJECTED`.
- `GET /api/info` is used as a health ping.

## Design rules

- Colors come only from the tokens in `src/index.css` (`bg-surface`, `text-ink-2`, `border-line`, `bg-brand`, `text-ok/warn/bad`, `bg-*-soft`). Every token is defined for both `:root` (light) and `.dark`. Never hard-code hex values in components.
- Green, amber and red mean good / watch / risk (and approved / flagged / rejected). Never use them for decoration; the brand is teal.
- Layout: a sticky header and content in `max-w-7xl` with responsive gutters. On the health check page, inputs sit on the left (sticky on `lg`) and the report on the right; on mobile they stack.
- Type: Inter. Show numbers with `tabular-nums`. Body text is 14–16px; section labels are 11px uppercase with `tracking-wider`.
- Wrap every `localStorage` access in try/catch.
- Anything the model writes carries a compliance badge and the "not financial advice" note.

## Commands

```bash
cd web && npm ci            # once
npm run dev                 # http://localhost:5173, proxies /api to http://127.0.0.1:7860
npm run build               # type-check (tsc -b) + production build to web/dist
```

The backend needs a model. Locally, use a downloaded GGUF:
`MODEL_PATH=models/v1/Llama-3.2-3B-Instruct.Q4_K_M.gguf python app.py`

## Verify before handing back

1. `npm run build` passes with no type errors, and `python -m pytest -q` passes.
2. Run `python app.py` and look at the real page in light mode, dark mode and at 390px width. Run at least one sample business through to a finished explanation, try the paste flow, and send one question in Ask.
