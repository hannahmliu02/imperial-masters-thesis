"""Loaders for the real-world / benchmark datasets.

Each loader returns a ``Dataset`` of ``BiasItem`` with the same metadata schema
as the synthetic generators, so tasks and metrics treat real and synthetic data
identically.

Provenance discipline: loaders read the *downloaded* files (fetched + pinned by
``scripts/download_data.py``, recorded in ``data/MANIFEST.json``). Where a file's
exact schema must be confirmed against the upstream release, the loader raises a
clear error rather than silently guessing (per the project's no-fabrication
rule).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..tasks.base import BiasItem, Dataset
from . import names as N


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# BBQ  (github.com/nyu-mll/BBQ) -- one JSONL per social-bias category.
# --------------------------------------------------------------------------- #

# Documented BBQ row fields (validated below): example_id, question_index,
# question_polarity (neg|nonneg), context_condition (ambig|disambig), category,
# context, question, ans0/ans1/ans2, label (gold index), answer_info,
# additional_metadata (incl. stereotyped_groups).
_BBQ_REQUIRED = {"context", "question", "ans0", "ans1", "ans2", "label", "context_condition"}


def load_bbq(path: str, categories: Optional[List[str]] = None) -> Dataset:
    """Load BBQ. ``path`` is a directory of ``<Category>.jsonl`` files (or a
    single JSONL file). Preserves the ambiguous/disambiguated pairing and the
    "unknown" option, and records the stereotype-aligned answer for the bias
    score.
    """
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"No BBQ JSONL files under {path!r}. Run download_data.py.")

    items: List[BiasItem] = []
    for f in files:
        cat_name = f.stem.lower()
        if categories and cat_name not in [c.lower() for c in categories]:
            continue
        for row in _read_jsonl(str(f)):
            missing = _BBQ_REQUIRED - set(row)
            if missing:
                raise ValueError(
                    f"BBQ row in {f.name} missing fields {missing}; the upstream "
                    f"schema may have changed -- verify against the pinned release."
                )
            options = [row["ans0"], row["ans1"], row["ans2"]]
            gold = options[int(row["label"])]
            cond = {"ambig": "ambiguous", "disambig": "disambiguated"}.get(
                row["context_condition"], row["context_condition"]
            )
            # Identify the "unknown" option via answer_info (second element tags
            # the answer; "unknown" marks the can't-be-determined option).
            unknown_option = None
            ans_info = row.get("answer_info", {})
            for i, key in enumerate(("ans0", "ans1", "ans2")):
                info = ans_info.get(key, [])
                if any(str(x).lower() in ("unknown", "cannot be determined") for x in info):
                    unknown_option = options[i]
            meta_extra = row.get("additional_metadata", {})
            stereo_groups = meta_extra.get("stereotyped_groups", [])
            example_id = row.get("example_id", row.get("question_index"))
            polarity = row.get("question_polarity", "neg")
            items.append(BiasItem(
                id=f"{cat_name}-{example_id}-{cond}-{polarity}",
                task="bbq",
                body=f"{row['context']} {row['question']}",
                options=options,
                gold=gold,
                group=",".join(stereo_groups) if stereo_groups else None,
                template_id=cat_name,
                # Pair the ambiguous/disambiguated versions of the same question.
                contrast_pair_id=f"{cat_name}-q{row.get('question_index')}-{polarity}",
                condition=cond,
                meta={"category": cat_name, "question_polarity": polarity,
                      "unknown_option": unknown_option, "answer_info": ans_info,
                      "stereotyped_groups": stereo_groups, "label_index": int(row["label"]),
                      "source": "nyu-mll/BBQ"},
            ))
    ds = Dataset(task="bbq", items=items,
                 provenance={"loader": "loaders.load_bbq", "path": str(path),
                             "source": "github.com/nyu-mll/BBQ"})
    ds.datasheet = ds.summary()
    return ds


# --------------------------------------------------------------------------- #
# WinoBias  (github.com/uclanlp/corefBias) -- pro/anti, types 1 and 2.
# --------------------------------------------------------------------------- #

# Filenames look like: pro_stereotyped_type1.txt.dev / anti_stereotyped_type2.txt.test
_WINOBIAS_FNAME = re.compile(r"(pro|anti)_stereotyped_type(\d)\.txt\.(dev|test)$")
_BRACKET = re.compile(r"\[([^\]]+)\]")
_PRONOUN_GENDER = {
    "he": "male", "him": "male", "his": "male",
    "she": "female", "her": "female", "hers": "female",
}


def load_winobias(path: str, split: str = "dev") -> Dataset:
    """Load WinoBias coreference sentences.

    In the WinoBias .txt format, the correct antecedent NP and the pronoun are
    wrapped in ``[...]``. We take the occupation-bearing bracket as the gold
    referent and the pronoun bracket for the gender signal. ``condition`` (pro/
    anti) and ``type`` come from the filename. If the bracket convention is not
    found, we raise rather than guess.
    """
    p = Path(path)
    files = [f for f in p.glob("*.txt.*") if _WINOBIAS_FNAME.search(f.name)
             and f.name.endswith(split)]
    if not files:
        raise FileNotFoundError(
            f"No WinoBias '*.{split}' files under {path!r} (expected "
            f"pro/anti_stereotyped_type{{1,2}}.txt.{split}). Run download_data.py."
        )
    known_occ = set(N.WINOBIAS_OCCUPATION_FEMALE_SHARE)
    items: List[BiasItem] = []
    for f in files:
        m = _WINOBIAS_FNAME.search(f.name)
        condition, wtype = m.group(1), f"type{m.group(2)}"
        with open(f) as fh:
            for ln, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                # Strip a leading sentence index if present.
                line = re.sub(r"^\d+\s+", "", line)
                spans = _BRACKET.findall(line)
                if len(spans) < 2:
                    raise ValueError(
                        f"{f.name}:{ln}: expected >=2 bracketed spans (antecedent + "
                        f"pronoun); got {spans!r}. Verify the WinoBias file format."
                    )
                pron = next((s for s in spans if s.lower() in _PRONOUN_GENDER), None)
                antecedent = next((s for s in spans if s.lower() not in _PRONOUN_GENDER), None)
                if pron is None or antecedent is None:
                    raise ValueError(f"{f.name}:{ln}: could not locate pronoun+antecedent in {spans!r}")
                body = _BRACKET.sub(lambda mm: mm.group(1), line)  # drop brackets
                occs_in = [o for o in known_occ if re.search(rf"\b{re.escape(o)}\b", body)]
                gold_occ = next((o for o in known_occ if o in antecedent.lower()), antecedent)
                items.append(BiasItem(
                    id=f"{condition}-{wtype}-{f.name}-{ln}",
                    task="winobias",
                    body=body,
                    options=sorted(set(occs_in)) or None,
                    gold=gold_occ,
                    group=_PRONOUN_GENDER[pron.lower()],
                    template_id=wtype,
                    contrast_pair_id=f"{wtype}-{f.name}-{ln}",
                    condition=condition,
                    meta={"pronoun": pron, "type": wtype, "split": split,
                          "occupations": occs_in, "source": "uclanlp/corefBias"},
                ))
    ds = Dataset(task="winobias", items=items,
                 provenance={"loader": "loaders.load_winobias", "path": str(path),
                             "split": split, "source": "github.com/uclanlp/corefBias"})
    ds.datasheet = ds.summary()
    return ds


# --------------------------------------------------------------------------- #
# Winogender  (github.com/rudinger/winogender-schemas) -- robustness check.
# --------------------------------------------------------------------------- #


def load_winogender(path: str) -> Dataset:
    """Load Winogender schema sentences (templates expanded over he/she/they).

    Reads ``all_sentences.tsv`` (sentid<TAB>sentence) plus the occupation stats
    if present. ``sentid`` encodes occupation, participant, answer, and pronoun.
    """
    p = Path(path)
    tsv = p / "all_sentences.tsv" if p.is_dir() else p
    if not Path(tsv).exists():
        raise FileNotFoundError(
            f"Winogender all_sentences.tsv not found at {tsv}. Run download_data.py."
        )
    items: List[BiasItem] = []
    with open(tsv) as fh:
        header = fh.readline()  # skip header
        for ln, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            sentid, sentence = parts[0], parts[1]
            # sentid format: occupation.participant.answer.pronoun (e.g. 0/1/2 or
            # male/female/neutral). Be tolerant of variants.
            fields = sentid.split(".")
            occupation = fields[0] if fields else None
            pronoun_key = fields[-1] if fields else None
            answer = fields[2] if len(fields) > 2 else None
            items.append(BiasItem(
                id=f"winogender-{sentid}",
                task="winobias",  # consumed by the same task interface
                body=sentence,
                options=None,
                gold=None,
                group=pronoun_key,
                template_id=occupation,
                contrast_pair_id=f"wg-{occupation}-{fields[1] if len(fields)>1 else ''}-{answer}",
                condition="winogender",
                meta={"sentid": sentid, "occupation": occupation,
                      "answer": answer, "source": "rudinger/winogender-schemas"},
            ))
    ds = Dataset(task="winobias", items=items,
                 provenance={"loader": "loaders.load_winogender", "path": str(path),
                             "source": "github.com/rudinger/winogender-schemas"})
    ds.datasheet = ds.summary()
    return ds


# --------------------------------------------------------------------------- #
# Bias in Bios  (De-Arteaga et al. 2019) via the Hugging Face Hub.
# --------------------------------------------------------------------------- #


def load_bias_in_bios(
    path: Optional[str] = None,
    hf_name: str = "LabHC/bias_in_bios",
    revision: Optional[str] = None,
    split: str = "test",
    max_items: Optional[int] = None,
) -> Dataset:
    """Load Bias in Bios (biography text, occupation label, gender).

    Prefers a locally saved dataset at ``path`` (``datasets.load_from_disk``);
    otherwise pulls ``hf_name`` at a pinned ``revision`` from the Hub. The schema
    is verified against the De-Arteaga et al. (2019) fields (hard_text/bio,
    profession/title, gender).
    """
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as e:
        raise ImportError("Bias in Bios needs the 'datasets' package: pip install -e '.[ml]'") from e

    if path and Path(path).exists():
        dset = load_from_disk(path)
        if split in getattr(dset, "keys", lambda: [])():
            dset = dset[split]
    else:
        dset = load_dataset(hf_name, split=split, revision=revision)

    cols = set(dset.column_names)
    text_col = next((c for c in ("hard_text", "bio", "text", "raw") if c in cols), None)
    occ_col = next((c for c in ("profession", "title", "occupation", "label") if c in cols), None)
    gender_col = next((c for c in ("gender", "sex", "G") if c in cols), None)
    if text_col is None or occ_col is None:
        raise ValueError(
            f"Bias in Bios columns unexpected: {sorted(cols)}. Verify the loader "
            f"against the pinned revision of {hf_name}."
        )

    items: List[BiasItem] = []
    n = len(dset) if max_items is None else min(max_items, len(dset))
    for i in range(n):
        row = dset[i]
        gender = row.get(gender_col) if gender_col else None
        gender = {0: "male", 1: "female"}.get(gender, gender)
        items.append(BiasItem(
            id=f"biasbios-{split}-{i}",
            task="resume",
            body=str(row[text_col]),
            options=None,
            gold=str(row[occ_col]),       # occupation is the gold label here
            group=str(gender) if gender is not None else None,
            template_id="bias_in_bios",
            contrast_pair_id=None,         # not a constructed minimal pair
            condition="real",
            meta={"occupation": str(row[occ_col]), "gender": gender,
                  "source": hf_name, "revision": revision},
        ))
    ds = Dataset(task="resume", items=items,
                 provenance={"loader": "loaders.load_bias_in_bios", "hf_name": hf_name,
                             "revision": revision, "split": split,
                             "source": "De-Arteaga et al. 2019"})
    ds.datasheet = ds.summary()
    return ds


# Registry so configs can select a loader by name.
REAL_LOADERS = {
    "bbq": load_bbq,
    "winobias": load_winobias,
    "winogender": load_winogender,
    "bias_in_bios": load_bias_in_bios,
}
