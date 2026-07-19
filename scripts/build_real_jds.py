#!/usr/bin/env python3
"""Extract cleaned, real job descriptions per role from a LinkedIn postings dataset.

Grounds the résumé task's JDs in real postings (replacing the hand-written ones).
Produces *candidate* JDs for human review (the QC loop): each is a real posting,
PII-stripped and truncated to ~target_words at a sentence boundary. Provenance
(job_id, title, company) is recorded but NOT included in the JD text.

    python scripts/build_real_jds.py --out scratchpad/real_jds_candidates.json --per-role 6

Review the output, keep the good ones, then wire the approved bank into
`data/synthetic.py:_RESUME_JDS`.
"""
import argparse
import json
import re
from pathlib import Path

# title substrings that define each role (lowercased match)
ROLE_TITLES = {
    "software_engineer": ["software engineer"],
    "sales_associate": ["sales associate"],
    "administrative_assistant": ["administrative assistant"],
}

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_PHONE = re.compile(r"\+?\d[\d\-\(\)\s]{7,}\d")
_BOILER = re.compile(r"^\s*job description\s*:?\s*", re.I)
# Common LinkedIn JD section headers — put each on its own line for readability.
_HEADERS = re.compile(
    r"\s*(Responsibilities|Duties and Responsibilities|Key Responsibilities|Requirements"
    r"|Qualifications|Preferred Qualifications|Minimum Qualifications|Basic Qualifications"
    r"|Benefits|Perks|Duties|Education|Experience|Job Skills|Skills|Compensation"
    r"|About Us|About the Role|Overview|Summary|Job Summary|What You Will Do"
    r"|What You'll Do|What We Offer|Who You Are)\s*:",
    re.I,
)
# A run-on: end-of-word punctuation immediately followed by a Capitalised new word,
# with no space (the HTML-strip artefact). Split it onto a new line.
_RUNON = re.compile(r"([a-z0-9\)][.!?:])\s*([A-Z][a-z])")


def clean(desc: str, company: str) -> str:
    """Keep the ACTUAL full posting text (no truncation) but RECONSTRUCT readable
    structure: strip PII, put section headers on their own lines, and break run-on
    'sentence.NextSentence' / 'Header:Bullet' artefacts onto separate lines."""
    t = desc or ""
    t = _URL.sub("", t); t = _EMAIL.sub("", t); t = _PHONE.sub("", t)
    if company:
        t = re.sub(re.escape(company), "the company", t, flags=re.I)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _BOILER.sub("", t)
    t = _HEADERS.sub(lambda m: f"\n\n{m.group(1).title()}:\n", t)   # header on its own line
    t = _RUNON.sub(r"\1\n\2", t)                                    # break run-ons
    # normalise intra-line spaces (keep newlines); drop empty runs > 1
    out, blank = [], False
    for ln in t.split("\n"):
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if not ln:
            if not blank and out:
                out.append("")
            blank = True
        else:
            out.append(ln); blank = False
    return "\n".join(out).strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="datastax/linkedin_job_listings")
    ap.add_argument("--out", default="scratchpad/real_jds_candidates.json")
    ap.add_argument("--per-role", type=int, default=4)
    ap.add_argument("--min-words", type=int, default=120, help="full-posting length band (readable, not a wall)")
    ap.add_argument("--max-words", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from datasets import load_dataset
    d = load_dataset(args.dataset)["train"]
    titles = [(t or "").lower() for t in d["title"]]

    out = {}
    for role, keys in ROLE_TITLES.items():
        # candidate row indices whose title matches and whose description is in-band
        idxs = []
        for i, t in enumerate(titles):
            if any(k in t for k in keys):
                w = len((d[i]["description"] or "").split())
                if args.min_words <= w <= args.max_words:
                    idxs.append(i)
        # deterministic pick: evenly spaced through the (stable) match list
        picks = idxs[:: max(1, len(idxs) // (args.per_role * 3) or 1)]
        cands = []
        for i in picks:
            row = d[i]
            jd = clean(row["description"], row.get("company_name") or "")
            if args.min_words <= len(jd.split()) <= args.max_words:
                cands.append({
                    "jd": jd,
                    "provenance": {"job_id": row.get("job_id"), "title": row.get("title"),
                                   "company": row.get("company_name"), "location": row.get("location")},
                    "n_words": len(jd.split()),
                })
            if len(cands) >= args.per_role:
                break
        out[role] = {"n_matches": len(idxs), "candidates": cands}
        print(f"\n===== {role}  ({len(idxs)} matches, showing {len(cands)} full JDs) =====")
        for j, c in enumerate(cands):
            print(f"\n[{j}] ({c['n_words']} words, from '{c['provenance']['title']}' @ {c['provenance']['company']}, {c['provenance']['location']})")
            preview = " ".join(c["jd"].split()[:55])
            print("   " + preview + " …")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"dataset": args.dataset, "roles": out}, indent=2))
    print(f"\n[build-jds] wrote candidates -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
