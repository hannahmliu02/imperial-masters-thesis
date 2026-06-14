"""The minimal-pair guard catches a deliberately corrupted pair."""

import pytest

from guardrail_ft.data import synthetic as S
from guardrail_ft.identify.contrasts import check_minimal_pairs, demographic_contrast


def test_clean_synthetic_has_no_offenders():
    ds = S.generate_resume(30, 0)
    assert check_minimal_pairs(ds) == []


def test_corrupted_pair_is_flagged():
    ds = S.generate_resume(20, 0)
    ds.items[1].body = ds.items[1].body + " EXTRA_TOKEN"   # non-signal change
    offenders = check_minimal_pairs(ds)
    assert offenders, "expected the corrupted pair to be flagged"
    assert any(o["b"] == ds.items[1].id or o["a"] == ds.items[1].id for o in offenders)


def test_demographic_contrast_refuses_on_corruption():
    ds = S.generate_resume(20, 0)
    ds.items[1].body = ds.items[1].body + " EXTRA_TOKEN"

    class _Task:  # contrast_pairs not reached: the guard raises first
        name = "resume"

    with pytest.raises(ValueError, match="non-minimal"):
        # loaded is unused because the minimal-pair guard raises before caching.
        demographic_contrast(loaded=None, task=_Task(), dataset=ds, assert_minimal=True)
