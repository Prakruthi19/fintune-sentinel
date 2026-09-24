"""Policy-Sentinel: post-generation compliance checks applied to every model answer."""
import re
from dataclasses import dataclass, field

# Phrase-level patterns so that "shareholders", "buyback" or "selling, general and
# administrative expenses" are not mistaken for trade advice.
TRADE_ADVICE = re.compile(
    r"\b(?:strong\s+)?(?:buy|sell|hold|accumulate|outperform|underperform)\s+"
    r"(?:rating|recommendation|signal|call|the\s+stock|shares?|position)\b"
    r"|\b(?:recommend|suggest|advise|urge)\w*\s+(?:you\s+)?"
    r"(?:to\s+)?(?:buy|sell|short|buying|selling|shorting|investing\s+in)\b"
    r"|\b(?:go|going)\s+(?:long|short)\s+(?:on\s+)?(?:the\s+)?(?:stock|shares?|position)\b"
    r"|\bprice\s+target\b",
    re.IGNORECASE,
)

NUMBER = re.compile(
    r"(?P<dollar>\$\s?)?(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent\b|basis\s+points\b|bps\b|million\b|billion\b|thousand\b|"
    r"trillion\b|[mbk]n?\b|x\b|times\b)?",
    re.IGNORECASE,
)
YEAR = re.compile(r"(?:19|20)\d\d")

DISCLAIMER = "[Redacted by Policy-Sentinel: individual trade recommendations are out of scope.]"


@dataclass
class AuditResult:
    text: str
    findings: list = field(default_factory=list)
    redacted: bool = False

    @property
    def status(self) -> str:
        if self.redacted:
            return "REJECTED"
        return "FLAGGED" if self.findings else "APPROVED"


def unitless_quantities(text: str) -> list:
    """Numbers that look like financial quantities but carry no $, %, or scale unit.

    Years, small integers (list markers, "Item 7") and alphanumeric labels ("Item 1A")
    are ignored; decimals and values >= 100 without a unit are reported.
    """
    hits = []
    for m in NUMBER.finditer(text):
        if m.group("dollar") or m.group("unit"):
            continue
        num = m.group("num").rstrip(",")
        nxt = text[m.end():m.end() + 1]
        if YEAR.fullmatch(num) or nxt.isalpha() or (m.start() > 0 and text[m.start() - 1].isalpha()):
            continue
        if "." in num.rstrip(".") or float(num.replace(",", "").rstrip(".")) >= 100:
            hits.append(num)
    return hits


def audit(text: str) -> AuditResult:
    result = AuditResult(text=text)
    if TRADE_ADVICE.search(text):
        result.findings.append("Prohibited trade advice detected")
        result.text = TRADE_ADVICE.sub("[redacted]", text) + "\n\n" + DISCLAIMER
        result.redacted = True
    bare = unitless_quantities(text)
    if bare:
        result.findings.append(f"Figures without units: {', '.join(bare[:5])}")
    return result
