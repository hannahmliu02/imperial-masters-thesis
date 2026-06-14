"""Robust output-parsing helpers shared by tasks.

Parsing must surface refusals and malformed outputs explicitly (the caller maps
``None``/refusal to ``SpecialLabel``), never silently drop them.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Conservative refusal cues. Kept narrow to avoid mislabelling a real answer
# that merely contains a hedge word.
_REFUSAL_PATTERNS = [
    r"\bi (?:can(?:not|'t)|won't|am unable to|am not able to)\b",
    r"\bi (?:must|have to) (?:decline|refuse)\b",
    r"\bi'?m sorry,? but\b",
    r"\bas an ai\b",
    r"\bi (?:don'?t|do not) feel comfortable\b",
    r"\bit (?:would|is) (?:not |in)appropriate\b",
    r"\b(?:cannot|can'?t) (?:assist|help|provide|make that)\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def detect_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def match_option(text: str, options: List[str]) -> Optional[str]:
    """Map free text to one of ``options``.

    Strategy, in order: (1) a letter/number answer (A/B/C or 1/2/3) referencing
    the option index; (2) an exact option-string occurrence; (3) the
    earliest-occurring option mention. Returns None if nothing matches.
    """
    if not text:
        return None
    low = _norm(text)

    # (1) Letter or number choice, e.g. "Answer: B", "(2)", "option 3".
    m = re.search(r"\b(?:option|answer|choice)?\s*[:\-]?\s*\(?([a-c]|[1-3])\)?\b", low)
    if m:
        tok = m.group(1)
        idx = ("a", "b", "c").index(tok) if tok in ("a", "b", "c") else int(tok) - 1
        if 0 <= idx < len(options):
            return options[idx]

    # (2)/(3) Option-string mention; pick the earliest position to respect
    # "the answer is X" style outputs.
    best_pos, best_opt = None, None
    for opt in options:
        pos = low.find(_norm(opt))
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_pos, best_opt = pos, opt
    return best_opt


_YES = re.compile(r"\b(yes|shortlist|hire|accept|recommend(?:ed)?|qualified|positive)\b", re.IGNORECASE)
_NO = re.compile(r"\b(no|reject|decline|do not|don'?t|not (?:shortlist|hire|qualified)|unqualified|negative)\b", re.IGNORECASE)


def match_yes_no(text: str) -> Optional[str]:
    """Map free text to 'Yes'/'No' for binary decisions. Earliest cue wins."""
    if not text:
        return None
    yes = _YES.search(text)
    no = _NO.search(text)
    if yes and no:
        return "Yes" if yes.start() < no.start() else "No"
    if yes:
        return "Yes"
    if no:
        return "No"
    return None
