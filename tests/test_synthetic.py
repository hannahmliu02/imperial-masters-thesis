"""Generated data varies ONLY the demographic signal (exact minimal pairs)."""

import pytest

from guardrail_ft.data import synthetic as S
from guardrail_ft.data.synthetic import token_diff, validate_minimal_pairs


@pytest.mark.parametrize("gen,n", [
    (lambda: S.generate_resume(40, 0), 40),
    (lambda: S.generate_bbq(40, 0), 40),
    (lambda: S.generate_winobias(40, 0), 40),
])
def test_minimal_pairs_validate(gen, n):
    ds = gen()
    report = validate_minimal_pairs(ds)  # raises on any violation
    assert report["n_pairs_checked"] > 0
    assert report["n_items"] == len(ds)


def test_resume_pairs_differ_only_in_name():
    ds = S.generate_resume(30, 0)
    by_pair = {}
    for it in ds:
        by_pair.setdefault(it.contrast_pair_id, []).append(it)
    for items in by_pair.values():
        a, b = items[0], items[1]
        diff = token_diff(a.body, b.body)
        assert diff is not None and len(diff) == 1, "exactly one token (the name) should differ"
        pos, x, y = diff[0]
        assert x in a.meta["signal"] and y in b.meta["signal"]
        assert x != y


def test_winobias_pairs_differ_only_in_pronoun():
    ds = S.generate_winobias(30, 0)
    by_pair = {}
    for it in ds:
        by_pair.setdefault(it.contrast_pair_id, []).append(it)
    for items in by_pair.values():
        a, b = items[0], items[1]
        diff = token_diff(a.body, b.body)
        assert diff is not None and len(diff) == 1
        _, x, y = diff[0]
        assert {x, y} <= {"he", "she"}


def test_bbq_pairs_are_permutations():
    ds = S.generate_bbq(40, 0)
    by_pair = {}
    for it in ds:
        by_pair.setdefault(it.contrast_pair_id, []).append(it)
    checked = 0
    for items in by_pair.values():
        if len(items) < 2:
            continue
        a, b = items[0], items[1]
        assert sorted(a.body.split()) == sorted(b.body.split())
        checked += 1
    assert checked > 0


def test_tampered_pair_is_rejected():
    ds = S.generate_resume(10, 0)
    # Corrupt one item's body so an extra (non-signal) token differs.
    ds.items[1].body = ds.items[1].body + " EXTRA"
    with pytest.raises(AssertionError):
        validate_minimal_pairs(ds)


def test_datasheet_counts_match():
    ds = S.generate_resume(20, 0)
    sheet = S.build_datasheet(ds)
    assert sheet["n_items"] == len(ds)
    assert sum(sheet["by_group"].values()) == len(ds)
