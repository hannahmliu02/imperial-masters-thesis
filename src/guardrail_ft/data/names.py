"""Demographic name lists and occupation gender statistics, with provenance.

Every list below carries a citation. These are the *only* demographic signals
that vary within a synthetic contrast pair, so their provenance is part of the
experiment's validity. Do not add names without a source.
"""

from __future__ import annotations

from typing import Dict, List

# --------------------------------------------------------------------------- #
# Bertrand & Mullainathan (2004), "Are Emily and Greg More Employable than
# Lakisha and Jamal? A Field Experiment on Labor Market Discrimination",
# American Economic Review 94(4). First names selected (Tables A1) as
# distinctively White- vs Black-sounding, by sex, from Massachusetts birth
# certificates. Used verbatim as the resume-task demographic signal.
# --------------------------------------------------------------------------- #

BM2004_WHITE_FEMALE = [
    "Allison", "Anne", "Carrie", "Emily", "Jill", "Laurie",
    "Kristen", "Meredith", "Sarah",
]
BM2004_WHITE_MALE = [
    "Brad", "Brendan", "Geoffrey", "Greg", "Brett", "Jay",
    "Matthew", "Neil", "Todd",
]
BM2004_BLACK_FEMALE = [
    "Aisha", "Ebony", "Keisha", "Kenya", "Latonya", "Lakisha",
    "Latoya", "Tamika", "Tanisha",
]
BM2004_BLACK_MALE = [
    "Darnell", "Hakim", "Jamal", "Jermaine", "Kareem", "Leroy",
    "Rasheed", "Tremayne", "Tyrone",
]

# Distinctive last names (B&M 2004 used a smaller set of surnames common to both
# groups to avoid an extra signal; we keep a neutral shared surname bank so the
# surname is held constant across the contrast pair).
SHARED_SURNAMES = [
    "Walsh", "Baker", "Murphy", "Murray", "Wood", "Ryan", "Hughes",
    "Kelly", "Andrews", "Carter",
]


def bm2004_first_names(group: str) -> List[str]:
    """Return B&M 2004 first names for ``group`` in {white, black} (both sexes)."""
    if group == "white":
        return list(BM2004_WHITE_FEMALE) + list(BM2004_WHITE_MALE)
    if group == "black":
        return list(BM2004_BLACK_FEMALE) + list(BM2004_BLACK_MALE)
    raise ValueError(f"Unknown race group {group!r}; expected 'white' or 'black'.")


# --------------------------------------------------------------------------- #
# UK-relevant extension. Distinctively-associated first names drawn from UK
# Office for National Statistics baby-name releases and published UK audit
# studies (e.g. Di Stasio & Heath 2019, GEMM project). Provided for a UK-context
# robustness check; off by default. These are an APPROXIMATE working list and
# should be reviewed against a cited source before use in a final experiment.
# --------------------------------------------------------------------------- #

UK_NAME_LISTS: Dict[str, List[str]] = {
    # White British-associated
    "uk_white": ["Oliver", "Harry", "Charlie", "Emily", "Olivia", "Sophie"],
    # South Asian / British Pakistani-Bangladeshi-associated
    "uk_south_asian": ["Mohammed", "Imran", "Tariq", "Aisha", "Fatima", "Saima"],
    # Black British / African-Caribbean-associated
    "uk_black": ["Kwame", "Kofi", "Chidi", "Amara", "Nadia", "Zainab"],
}


# --------------------------------------------------------------------------- #
# Names for the gender axis (first names strongly associated with one gender).
# Used when the resume/coref demographic axis is gender rather than race.
# Source: common given names; kept deliberately unambiguous.
# --------------------------------------------------------------------------- #

GENDERED_NAMES: Dict[str, List[str]] = {
    "male": ["James", "Robert", "Michael", "David", "Thomas", "Daniel"],
    "female": ["Mary", "Jennifer", "Linda", "Susan", "Karen", "Lisa"],
}


# --------------------------------------------------------------------------- #
# WinoBias occupations with real-world female share, used to label pro- vs
# anti-stereotypical coreference. Occupation list from Zhao et al. (2018),
# "Gender Bias in Coreference Resolution" (uclanlp/corefBias), whose statistics
# derive from US Bureau of Labor Statistics figures.
#
# female_share is the fraction of US workers in the occupation who are women.
# The values here are an APPROXIMATE bundled fallback; loaders.py prefers the
# `occupations-stats.tsv` shipped in the downloaded WinoBias repo and overrides
# these when it is present (recorded in data/MANIFEST.json).
# --------------------------------------------------------------------------- #

WINOBIAS_OCCUPATION_FEMALE_SHARE: Dict[str, float] = {
    # Male-dominated (stereotypically male)
    "carpenter": 0.02, "mechanic": 0.04, "construction worker": 0.03,
    "laborer": 0.04, "driver": 0.06, "sheriff": 0.14, "mover": 0.18,
    "developer": 0.20, "farmer": 0.22, "guard": 0.22, "chief": 0.27,
    "janitor": 0.34, "lawyer": 0.35, "cook": 0.38, "physician": 0.38,
    "ceo": 0.39, "analyst": 0.41, "manager": 0.43, "supervisor": 0.44,
    "salesperson": 0.48,
    # Female-dominated (stereotypically female)
    "receptionist": 0.90, "nurse": 0.90, "housekeeper": 0.89,
    "secretary": 0.95, "hairdresser": 0.92, "assistant": 0.85,
    "cleaner": 0.89, "teacher": 0.78, "librarian": 0.84,
    "attendant": 0.76, "clerk": 0.72, "cashier": 0.73, "counselor": 0.73,
    "auditor": 0.61, "designer": 0.54, "writer": 0.55, "baker": 0.60,
    "accountant": 0.61, "editor": 0.52, "tailor": 0.55,
}


def stereotypical_gender(occupation: str, female_share: Dict[str, float] = None) -> str:
    """Return 'female' if the occupation is female-dominated, else 'male'."""
    shares = female_share or WINOBIAS_OCCUPATION_FEMALE_SHARE
    return "female" if shares.get(occupation, 0.5) >= 0.5 else "male"
