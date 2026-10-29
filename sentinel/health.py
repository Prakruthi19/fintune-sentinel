"""Financial health check: exact ratio math, ratings and a plain-English prompt for the model.

The numbers are computed here, never by the LLM: a 3B model gets arithmetic wrong often enough
that a tool for business owners can't rely on it. The model only explains the computed results.
"""
import re
from dataclasses import asdict, dataclass

FIELDS = {
    # id: (label shown to the user, label patterns found in pasted statements, most specific first)
    "revenue": (
        "Revenue (this year)",
        r"total\s+revenues?|net\s+revenues?|revenues?|net\s+sales|total\s+sales"
        r"|sales(?!\s+(?:and|&)\s+marketing)|turnover",
    ),
    "revenue_prior": ("Revenue (last year)", None),
    "cogs": ("Cost of goods sold", r"cost\s+of\s+(?:goods\s+sold|sales|revenues?)|cogs|direct\s+costs"),
    "operating_expenses": ("Operating expenses", r"total\s+operating\s+expenses|operating\s+expenses|opex|overheads?"),
    "operating_income": ("Operating income", r"operating\s+(?:income|profit)|income\s+from\s+operations|ebit\b"),
    "net_income": ("Net income", r"net\s+(?:income|profit|earnings)|profit\s+after\s+tax"),
    "interest_expense": ("Interest expense", r"interest\s+expenses?|interest\s+paid|finance\s+costs?"),
    "cash": ("Cash", r"cash\s+and\s+(?:cash\s+)?equivalents|cash(?:\s+at\s+bank)?(?!\s+flows?)"),
    "inventory": ("Inventory", r"inventor(?:y|ies)|stock\s+on\s+hand"),
    "current_assets": ("Current assets", r"total\s+current\s+assets|current\s+assets"),
    "current_liabilities": ("Current liabilities", r"total\s+current\s+liabilities|current\s+liabilities"),
    "total_liabilities": ("Total liabilities", r"total\s+liabilities"),
    "equity": ("Owner's equity", r"total\s+(?:shareholders'?|stockholders'?|owners'?)\s+equity|"
                                 r"(?:shareholders'?|stockholders'?|owners'?)\s+equity|total\s+equity|net\s+assets|equity"),
}

COSTS = ("cogs", "operating_expenses", "interest_expense")
NUMBER = re.compile(r"\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?")


def _to_float(token):
    neg = token.startswith("(") and token.endswith(")") or "-" in token
    value = float(re.sub(r"[^\d.]", "", token))
    return -value if neg else value


def parse_statement(text):
    """Pull known line items out of pasted text, CSV or markdown tables.

    The first number on a line is this year, the second (if any) last year; the second revenue
    figure becomes revenue_prior. Years in headers (2023, FY2024) are skipped.
    """
    found = {}
    for raw in text.splitlines():
        line = raw.replace("|", " ").replace("\t", " ").strip(" :-")
        if not line:
            continue
        for key, (_, pattern) in FIELDS.items():
            if not pattern or key in found:
                continue
            m = re.match(rf"^\W*(?:{pattern})\b", line, re.IGNORECASE)
            if not m:
                continue
            rest = re.sub(r"\b(?:FY)?(?:19|20)\d\d\b", " ", line[m.end():])
            numbers = [_to_float(n) for n in NUMBER.findall(rest)]
            if numbers:
                # Costs are often printed as (1,234); store them positive so the form reads naturally.
                found[key] = abs(numbers[0]) if key in COSTS else numbers[0]
                if key == "revenue" and len(numbers) > 1:
                    found["revenue_prior"] = numbers[1]
            break  # one field per line: "Total current assets" must not also match "assets"
    return found


@dataclass
class Metric:
    id: str
    name: str
    group: str
    value: float
    display: str
    rating: str  # good | watch | risk
    formula: str
    meaning: str


def _fmt(x):
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}".rstrip("0").rstrip(".")


def _rate(value, good, watch, higher_is_better=True):
    if not higher_is_better:
        value, good, watch = -value, -good, -watch
    return "good" if value >= good else "watch" if value >= watch else "risk"


def compute(f):
    """Metrics for whichever inputs are present. Missing inputs just skip a metric."""
    f = {k: float(v) for k, v in f.items() if v is not None and v != ""}
    for cost in COSTS:  # typed in as negatives, e.g. -1,234
        if cost in f:
            f[cost] = abs(f[cost])
    m = []
    rev = f.get("revenue")
    gross = rev - f["cogs"] if rev is not None and "cogs" in f else None
    op = f.get("operating_income")
    if op is None and gross is not None and "operating_expenses" in f:
        op = gross - f["operating_expenses"]

    if rev is not None and f.get("revenue_prior"):
        g = (rev - f["revenue_prior"]) / abs(f["revenue_prior"]) * 100
        m.append(Metric("revenue_growth", "Revenue growth", "Growth", g, f"{g:+.1f}%", _rate(g, 10, 0),
                        f"({_fmt(rev)} − {_fmt(f['revenue_prior'])}) ÷ {_fmt(f['revenue_prior'])}",
                        "How much more (or less) you sold than last year."))
    if rev:
        if gross is not None:
            v = gross / rev * 100
            m.append(Metric("gross_margin", "Gross margin", "Profitability", v, f"{v:.1f}%", _rate(v, 40, 20),
                            f"({_fmt(rev)} − {_fmt(f['cogs'])}) ÷ {_fmt(rev)}",
                            "What's left of each sale after paying for what you sold."))
        if op is not None:
            v = op / rev * 100
            m.append(Metric("operating_margin", "Operating margin", "Profitability", v, f"{v:.1f}%",
                            _rate(v, 15, 5), f"{_fmt(op)} ÷ {_fmt(rev)}",
                            "Profit from running the business, before interest and tax."))
        if "net_income" in f:
            v = f["net_income"] / rev * 100
            m.append(Metric("net_margin", "Net margin", "Profitability", v, f"{v:.1f}%", _rate(v, 8, 2),
                            f"{_fmt(f['net_income'])} ÷ {_fmt(rev)}",
                            "What you actually keep from each sale after every cost."))
    if f.get("current_liabilities"):
        cl = f["current_liabilities"]
        if "current_assets" in f:
            v = f["current_assets"] / cl
            m.append(Metric("current_ratio", "Current ratio", "Liquidity", v, f"{v:.2f}x", _rate(v, 1.5, 1),
                            f"{_fmt(f['current_assets'])} ÷ {_fmt(cl)}",
                            "Can short-term assets cover bills due within a year? Above 1.5x is comfortable."))
            if "inventory" in f:
                v = (f["current_assets"] - f["inventory"]) / cl
                m.append(Metric("quick_ratio", "Quick ratio", "Liquidity", v, f"{v:.2f}x", _rate(v, 1, 0.7),
                                f"({_fmt(f['current_assets'])} − {_fmt(f['inventory'])}) ÷ {_fmt(cl)}",
                                "The same test without counting stock you still have to sell."))
        if "cash" in f:
            v = f["cash"] / cl
            m.append(Metric("cash_ratio", "Cash ratio", "Liquidity", v, f"{v:.2f}x", _rate(v, 0.5, 0.2),
                            f"{_fmt(f['cash'])} ÷ {_fmt(cl)}",
                            "How much of next year's bills you could pay with cash today."))
    if "total_liabilities" in f and "equity" in f:
        eq = f["equity"]
        if eq <= 0:
            m.append(Metric("debt_to_equity", "Debt-to-equity", "Leverage", float("inf"), "Negative equity", "risk",
                            f"{_fmt(f['total_liabilities'])} ÷ {_fmt(eq)}",
                            "The business owes more than it owns."))
        else:
            v = f["total_liabilities"] / eq
            m.append(Metric("debt_to_equity", "Debt-to-equity", "Leverage", v, f"{v:.2f}x",
                            _rate(v, 1, 2, higher_is_better=False), f"{_fmt(f['total_liabilities'])} ÷ {_fmt(eq)}",
                            "How much of the business is funded by debt versus the owners."))
    if op is not None and f.get("interest_expense"):
        v = op / f["interest_expense"]
        m.append(Metric("interest_coverage", "Interest coverage", "Leverage", v, f"{v:.1f}x", _rate(v, 5, 2),
                        f"{_fmt(op)} ÷ {_fmt(f['interest_expense'])}",
                        "How many times operating profit covers the interest bill."))
    return m


def score(metrics):
    if not metrics:
        return None, None
    points = {"good": 100, "watch": 60, "risk": 20}
    s = round(sum(points[x.rating] for x in metrics) / len(metrics))
    return s, "Healthy" if s >= 75 else "Needs attention" if s >= 50 else "At risk"


def report(fields):
    metrics = compute(fields)
    s, label = score(metrics)
    missing = [FIELDS[k][0] for k in ("revenue", "revenue_prior", "current_assets", "current_liabilities",
                                       "total_liabilities", "equity") if fields.get(k) in (None, "")]
    return {"score": s, "label": label, "metrics": [asdict(x) | {"value": None if x.value == float("inf") else x.value}
                                                     for x in metrics], "missing": missing}


EXPLAIN_INSTRUCTIONS = (
    "Explain this small business health check to the owner, who has no finance background. "
    "The figures are already calculated and correct. Write exactly two parts:\n"
    "**Summary:** two plain sentences: the overall result, the biggest reason, and what it means day to day. "
    "Never call a weak figure good.\n"
    "**Questions to ask your accountant**\n- three short questions, mostly about the weak figures.\n"
    "Quote figures exactly as written. Do not recommend buying or selling any investment.\n\n"
)


def explain_prompt(metrics):
    def figures(ratings):
        return ", ".join(f"{x.name} {x.display}" for x in metrics if x.rating in ratings) or "none"

    s, label = score(metrics)
    # The UI lists the strong and weak checks itself, straight from the ratings: left to itself,
    # the 3B model files weak metrics under "going well", so the model writes only the prose.
    # The fixed instructions come first so llama.cpp reuses their KV cache from the last check;
    # on a laptop CPU, reading the prompt costs about as much as writing the answer.
    return (
        f"{EXPLAIN_INSTRUCTIONS}"
        f"Overall: {label} ({s}/100)\n"
        f"Strong figures: {figures({'good'})}\n"
        f"Weak figures: {figures({'watch', 'risk'})}\n\n"
        "Now write both parts, starting with **Summary:** and then **Questions to ask your accountant**."
    )
