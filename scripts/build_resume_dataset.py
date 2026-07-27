#!/usr/bin/env python3
"""Build a real-résumé minimal-pair dataset for the bias-guardrail experiment.

Résumés: snehaanbhawal/resume-dataset (24 categories, real, cited). JDs: real
LinkedIn postings (datastax/linkedin_job_listings). For each résumé we:

  1. SCRUB residual PII -- regex (emails/phones/URLs) + spaCy NER PERSON-masking;
  2. EXCLUDE résumés that leak the contrast axis (HBCU/NAACP …);
  3. MATCH a JD by category (title-keyword rules below) -> same-category = QUALIFIED
     (gold Yes), a different category = UNQUALIFIED (gold No);
  4. Render a MINIMAL PAIR: identical résumé + JD, differing only in a Bertrand &
     Mullainathan (2004) first name -- one White-, one Black-associated, SAME sex
     (chosen to agree with the body), shared surname -- so race is the only signal.

Every cleaning action is recorded for methodology/audit:
  * <out>/cleaning_ledger.jsonl  -- one row PER SOURCE RÉSUMÉ (chars in/out, counts
    of each PII type removed, NER masks, tells, exclusion reason, assigned sex);
  * <out>/qc_report.json         -- sources + fingerprints, spaCy model+version, the
    exact regex + category patterns, and stage-by-stage counts.

Emits BiasItem JSONL (train/val/test), drop-in for Dataset.from_jsonl — split by
résumé so a minimal pair and both name-variants never cross splits.

    python scripts/build_resume_dataset.py                 # verify (sample, no write)
    python scripts/build_resume_dataset.py --full --out data/resume_real
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from guardrail_ft.data.names import (  # noqa: E402
    BM2004_WHITE_FEMALE, BM2004_WHITE_MALE, BM2004_BLACK_FEMALE, BM2004_BLACK_MALE,
    SHARED_SURNAMES,
)

# --- Résumé-category -> JD title keyword rules (verified 2026-07) ------------ #
CATEGORY_JD_PATTERNS = {
 'ACCOUNTANT': r'accountant|accounting|\bcpa\b|bookkeep|auditor',
 'ADVOCATE': r'attorney|lawyer|legal counsel|paralegal|advocate',
 'AGRICULTURE': r'agricultur|farm|agronom|horticultur',
 'APPAREL': r'apparel|fashion|garment|textile|merchandis',
 'ARTS': r'\bartist\b|fine art|illustrator|curator|creative arts',
 'AUTOMOBILE': r'automotive|automobile|mechanic|vehicle technician',
 'AVIATION': r'aviation|pilot|aircraft|flight attendant|aerospace',
 'BANKING': r'\bbank\b|banking|teller|loan officer|credit analyst',
 'BPO': r'call center|customer service representative|contact center|\bbpo\b',
 'BUSINESS-DEVELOPMENT': r'business development|partnerships manager',
 'CHEF': r'\bchef\b|line cook|culinary|kitchen manager',
 'CONSTRUCTION': r'construction|site manager|general contractor|mason|superintendent',
 'CONSULTANT': r'consultant|consulting',
 'DESIGNER': r'designer|\bux\b|\bui\b|graphic design',
 'DIGITAL-MEDIA': r'social media|digital media|content creator|\bseo\b|digital marketing',
 'ENGINEERING': r'engineer|engineering',
 'FINANCE': r'finance|financial analyst|\bfp&a\b|investment analyst|treasury',
 'FITNESS': r'fitness|personal trainer|group exercise|wellness coach',
 'HEALTHCARE': r'nurse|healthcare|clinical|therapist|physician|medical assistant',
 'HR': r'human resources|recruiter|talent acquisition|\bhr\b',
 'INFORMATION-TECHNOLOGY': r'software|developer|systems administrator|network engineer|\bit\b support',
 'PUBLIC-RELATIONS': r'public relations|communications specialist|media relations',
 'SALES': r'\bsales\b|account executive|account manager',
 'TEACHER': r'teacher|professor|instructor|educator|tutor',
}

EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
PHONE = re.compile(r'(?:\+?\d{1,2}\s*)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b')
PHONE_LONG = re.compile(r'\b\d{10,}\b')
URL = re.compile(r'\b(?:https?://|www\.)\S+\b')
SHE = re.compile(r'\b(she|her|hers|herself)\b', re.I)
HE = re.compile(r'\b(he|him|his|himself)\b', re.I)
RACE_TELL = re.compile(r'\b(hbcu|naacp|historically black|united negro|divine nine)\b', re.I)
SEX_TELL_F = re.compile(r"\b(sorority|women's)\b", re.I)
SEX_TELL_M = re.compile(r"\b(fraternity|men's)\b", re.I)


def regex_scrub(text: str) -> tuple[str, dict]:
    """Remove contact PII by regex; return (text, per-type counts + tell flags)."""
    counts = {"email": len(EMAIL.findall(text)),
              "phone": len(PHONE.findall(text)) + len(PHONE_LONG.findall(text)),
              "url": len(URL.findall(text))}
    t = EMAIL.sub('[EMAIL]', text)
    t = PHONE.sub('[PHONE]', t)
    t = PHONE_LONG.sub('[PHONE]', t)
    t = URL.sub('[URL]', t)
    counts.update(she=len(SHE.findall(t)), he=len(HE.findall(t)),
                  sex_f=len(SEX_TELL_F.findall(t)), sex_m=len(SEX_TELL_M.findall(t)),
                  race_tell=len(RACE_TELL.findall(t)))
    return t, counts


def ner_mask_batch(texts, nlp):
    """spaCy NER pass: replace PERSON spans with [PERSON]. Yields (text, n_masked)."""
    for doc in nlp.pipe(texts, batch_size=64):
        persons = [e for e in doc.ents if e.label_ == "PERSON"]
        if not persons:
            out = doc.text
        else:
            s, last = [], 0
            for e in sorted(persons, key=lambda x: x.start_char):
                if e.start_char < last:
                    continue
                s.append(doc.text[last:e.start_char]); s.append("[PERSON]")
                last = e.end_char
            s.append(doc.text[last:])
            out = "".join(s)
        out = re.sub(r'[ \t]+', ' ', out)
        out = re.sub(r'\n{3,}', '\n\n', out).strip()
        yield out, len(persons)


def infer_sex(c: dict, rng: random.Random) -> str:
    """Injected name's sex agrees with the body (pronouns + sorority/fraternity);
    random-balanced when the body is neutral."""
    f = c["she"] + 3 * c["sex_f"]
    m = c["he"] + 3 * c["sex_m"]
    return "female" if f > m else "male" if m > f else ("female" if rng.random() < 0.5 else "male")


def name_pair(sex: str, rng: random.Random) -> tuple[str, str]:
    surname = rng.choice(SHARED_SURNAMES)
    if sex == "female":
        w, b = rng.choice(BM2004_WHITE_FEMALE), rng.choice(BM2004_BLACK_FEMALE)
    else:
        w, b = rng.choice(BM2004_WHITE_MALE), rng.choice(BM2004_BLACK_MALE)
    return f"{w} {surname}", f"{b} {surname}"


def load_resumes(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    txt = [c for c in df.columns if 'str' in c.lower()][0]
    cat = [c for c in df.columns if c.lower() == 'category'][0]
    idc = [c for c in df.columns if c.lower() == 'id'][0]
    return df[[idc, cat, txt]].rename(columns={idc: 'id', cat: 'category', txt: 'text'})


def load_jds():
    import warnings; warnings.filterwarnings("ignore")
    from datasets import load_dataset
    df = load_dataset("datastax/linkedin_job_listings", split="train").to_pandas()
    df = df[['title', 'description']].fillna('')
    df = df[df['description'].str.len() > 300]
    df['t'] = df['title'].str.lower()
    return df


def match_jds(jdf, rng, per_cat):
    pools = {}
    for c, pat in CATEGORY_JD_PATTERNS.items():
        idx = list(jdf[jdf['t'].str.contains(pat, regex=True)].index)
        rng.shuffle(idx)
        pools[c] = [{"title": jdf.at[i, 'title'], "description": jdf.at[i, 'description']}
                    for i in idx[:per_cat]]
    return pools


def sha256(path, cap=None):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        h.update(fh.read(cap) if cap else fh.read())
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resumes", default=str(Path.home() / "Downloads/archive/Resume/Resume.csv"))
    ap.add_argument("--full", action="store_true", help="render whole dataset (else verify only)")
    ap.add_argument("--out", default="data/resume_real")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-cat-jds", type=int, default=60)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    import spacy
    print("loading résumés + JDs + spaCy…", flush=True)
    R = load_resumes(args.resumes)
    J = load_jds()
    pools = match_jds(J, rng, args.per_cat_jds)
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
    cats = sorted(R['category'].unique())

    print("\n=== category → JD match counts (résumés | JD pool) ===")
    for c in cats:
        print(f"  {c:24} résumés={int((R['category']==c).sum()):4}  jd_pool={len(pools.get(c,[])):3}"
              f"{'  <-- NO JD' if not pools.get(c) else ''}")

    # stage 1: regex scrub all
    regexed = [regex_scrub(t) for t in R['text']]
    # stage 2: NER mask all
    print("\nrunning NER PERSON-masking over all résumés…", flush=True)
    nered = list(ner_mask_batch([t for t, _ in regexed], nlp))

    if not args.full:
        print("\n=== sample rendered minimal pairs ===")
        for c in cats[:3]:
            i = R.index[R['category'] == c][0]
            clean, npers = nered[R.index.get_loc(i)]
            counts = regexed[R.index.get_loc(i)][1]
            sex = infer_sex(counts, rng); wn, bn = name_pair(sex, rng)
            print(f"\n  [{c}] sex={sex} white='{wn}' black='{bn}' | PERSON masked={npers}")
            print(f"     résumé (scrubbed+NER, 160 chars): {clean[:160].strip()!r}")
        print("\n[verify] no files written. Re-run with --full.")
        return 0

    # ---- full render + ledger ------------------------------------------- #
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ledger, eligible = [], []
    for (rid, cat, _), (rtext, counts), (ntext, npers) in zip(
            R.itertuples(index=False), regexed, nered):
        rec = {"resume_id": str(rid), "category": cat,
               "chars_in": len(_), "chars_out": len(ntext),
               "removed_email": counts["email"], "removed_phone": counts["phone"],
               "removed_url": counts["url"], "person_masked": npers,
               "pronoun_she": counts["she"], "pronoun_he": counts["he"],
               "sex_tell_f": counts["sex_f"], "sex_tell_m": counts["sex_m"],
               "race_tell": counts["race_tell"], "excluded": False, "exclude_reason": None,
               "sex_assigned": None}
        if not pools.get(cat):
            rec.update(excluded=True, exclude_reason="no_jd_for_category")
        elif counts["race_tell"] > 0:
            rec.update(excluded=True, exclude_reason="race_tell_leak")
        else:
            sex = infer_sex(counts, rng)
            rec["sex_assigned"] = sex
            eligible.append((str(rid), cat, ntext, sex))
        ledger.append(rec)

    # assign qualification + split (by résumé) then render minimal pairs
    ids = [e[0] for e in eligible]; rng.shuffle(ids)
    n = len(ids); tr, va = int(0.7 * n), int(0.85 * n)
    split_of = {rid: ("train" if i < tr else "val" if i < va else "test")
                for i, rid in enumerate(ids)}

    items = {"train": [], "val": [], "test": []}
    for rid, cat, ntext, sex in eligible:
        qualified = rng.random() < 0.5
        if qualified:
            jd, gold = rng.choice(pools[cat]), "Yes"
        else:
            oc = rng.choice([x for x in cats if x != cat and pools.get(x)])
            jd, gold = rng.choice(pools[oc]), "No"
        wn, bn = name_pair(sex, rng)
        body_tpl = "Applicant: {name}\n\n" + ntext
        pid = f"resume-{rid}-{'q' if qualified else 'u'}"
        sp = split_of[rid]
        for g, nm in (("white", wn), ("black", bn)):
            items[sp].append({
                "id": f"{pid}-{g}", "task": "resume", "body": body_tpl.replace("{name}", nm),
                "options": ["Yes", "No"], "gold": gold, "group": g,
                "template_id": cat, "contrast_pair_id": pid,
                "condition": "qualified" if qualified else "unqualified",
                "meta": {"signal": [nm], "pair_kind": "minimal", "decision": "shortlist",
                         "axis": "race", "qualified": qualified, "role": cat,
                         "job_title": jd["title"], "job_description": jd["description"],
                         "sex": sex, "resume_id": rid, "split": sp},
            })

    for sp in ("train", "val", "test"):
        (out / f"{sp}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items[sp]))
        rows = items[sp]; nq = sum(1 for r in rows if r["condition"] == "qualified")
        print(f"  {sp}: {len(rows)} rows ({len(rows)//2} pairs)  qualified_rows={nq}  "
              f"white={sum(1 for r in rows if r['group']=='white')}")

    (out / "cleaning_ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger))
    qc = {
        "seed": args.seed,
        "resume_source": {"name": "snehaanbhawal/resume-dataset (Kaggle)", "path": args.resumes,
                          "sha256": sha256(args.resumes), "n_rows": int(len(R))},
        "jd_source": {"name": "datastax/linkedin_job_listings (HF)", "n_postings": int(len(J)),
                      "per_category_pool": args.per_cat_jds, "min_description_chars": 300},
        "ner": {"library": "spaCy", "version": spacy.__version__, "model": "en_core_web_sm",
                "entity_masked": "PERSON", "total_person_masks": sum(r["person_masked"] for r in ledger)},
        "regex_pii": {"email": EMAIL.pattern, "phone": PHONE.pattern, "phone_long": PHONE_LONG.pattern,
                      "url": URL.pattern},
        "exclusions": dict(Counter(r["exclude_reason"] for r in ledger if r["excluded"])),
        "totals": {"resumes_in": len(ledger), "resumes_kept": len(eligible),
                   "minimal_pairs": sum(len(v)//2 for v in items.values()),
                   "rows": sum(len(v) for v in items.values()),
                   "removed_email": sum(r["removed_email"] for r in ledger),
                   "removed_phone": sum(r["removed_phone"] for r in ledger),
                   "removed_url": sum(r["removed_url"] for r in ledger)},
        "splits": {sp: {"rows": len(items[sp]), "pairs": len(items[sp])//2} for sp in items},
        "category_jd_patterns": CATEGORY_JD_PATTERNS,
    }
    (out / "qc_report.json").write_text(json.dumps(qc, indent=2))
    print(f"\n[full] {qc['totals']['minimal_pairs']} pairs / {qc['totals']['rows']} rows -> {out}/")
    print(f"[full] audit trail -> {out}/cleaning_ledger.jsonl + {out}/qc_report.json")
    print(f"[full] excluded: {qc['exclusions']}  |  PERSON masks: {qc['ner']['total_person_masks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
