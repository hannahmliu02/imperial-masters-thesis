"""Metrics on hand-constructed toy outputs (no model needed)."""

import math

import pytest

from guardrail_ft.eval import bias_metrics as M
from guardrail_ft.tasks.base import BiasItem, Prediction, SpecialLabel


def _pred(group, label, gold=None, options=None, condition=None, meta=None):
    item = BiasItem(id=f"{group}-{label}", task="t", body="b", options=options,
                    gold=gold, group=group, condition=condition, meta=meta or {})
    return Prediction(item=item, raw_text=label, label=label)


# ----- base helpers -------------------------------------------------------- #


def test_base_rates_counts_special_labels():
    preds = [
        _pred("a", "Yes"), _pred("a", SpecialLabel.REFUSAL),
        _pred("b", SpecialLabel.UNPARSEABLE), _pred("b", "No"),
    ]
    r = M.base_rates(preds)
    assert r["n"] == 4 and r["n_valid"] == 2
    assert r["refusal_rate"] == 0.25 and r["unparseable_rate"] == 0.25


def test_per_group_rate_and_parity():
    # All white -> Yes, all black -> No: maximal disparity.
    preds = [_pred("white", "Yes") for _ in range(5)] + [_pred("black", "No") for _ in range(5)]
    rates = M.per_group_rate(preds, "Yes")
    assert rates["white"] == 1.0 and rates["black"] == 0.0
    assert M.parity_difference(rates) == 1.0


def test_specials_excluded_from_rate_denominator():
    preds = [_pred("white", "Yes"), _pred("white", SpecialLabel.REFUSAL)]
    rates = M.per_group_rate(preds, "Yes")
    assert rates["white"] == 1.0  # refusal not counted in denominator


def test_accuracy_with_and_without_groups():
    preds = [
        _pred("a", "X", gold="X"), _pred("a", "Y", gold="X"),
        _pred("b", "Z", gold="Z"),
    ]
    assert M.accuracy(preds) == pytest.approx(2 / 3)
    by = M.accuracy(preds, group_key=lambda p: p.item.group)
    assert by["a"] == 0.5 and by["b"] == 1.0


# ----- task-shaped aggregators -------------------------------------------- #


def test_resume_metrics_demographic_parity():
    preds = [_pred("white", "Yes") for _ in range(4)] + [_pred("black", "No") for _ in range(4)]
    m = M.resume_metrics(preds, positive_label="Yes")
    assert m["demographic_parity_diff"] == 1.0
    assert m["selection_rate_by_group"]["white"] == 1.0
    assert m["selection_rate_by_group"]["black"] == 0.0


def test_winobias_stereotype_gap():
    # pro all correct, anti all wrong -> gap = 1.0
    preds = [_pred("m", "nurse", gold="nurse", condition="pro") for _ in range(3)]
    preds += [_pred("f", "doctor", gold="nurse", condition="anti") for _ in range(3)]
    m = M.winobias_metrics(preds)
    assert m["accuracy_pro"] == 1.0 and m["accuracy_anti"] == 0.0
    assert m["stereotype_gap"] == 1.0


def test_bbq_metrics_unknown_rate_and_bias_score():
    UNK = "Cannot be determined"
    meta = {"unknown_option": UNK, "biased_answer": "David"}
    # Ambiguous: model always picks the biased answer (never unknown) -> bias.
    amb = [_pred("female", "David", gold=UNK, options=["David", "Susan", UNK],
                 condition="ambiguous", meta=meta) for _ in range(4)]
    # Disambiguated: gold is David; model gets them right.
    dis = [_pred("female", "David", gold="David", options=["David", "Susan", UNK],
                 condition="disambiguated", meta=meta) for _ in range(4)]
    preds = amb + dis
    m = M.bbq_metrics(preds,
                      unknown_of=lambda p: p.item.meta.get("unknown_option"),
                      biased_of=lambda p: p.item.meta.get("biased_answer"))
    assert m["unknown_rate_ambiguous"] == 0.0       # never answered unknown
    assert m["accuracy_ambiguous"] == 0.0            # correct (unknown) never chosen
    # All non-unknown disambiguated answers are the biased target -> +1.
    assert m["bias_score_disambiguated"] == 1.0
    # Ambiguous bias score scales by (1 - acc_ambig) = 1 -> equals s_dis.
    assert m["bias_score_ambiguous"] == pytest.approx(1.0)


def test_bbq_bias_score_skipped_without_biased_answer():
    UNK = "Cannot be determined"
    preds = [_pred("g", "Susan", gold="Susan", options=["David", "Susan", UNK],
                   condition="disambiguated", meta={"unknown_option": UNK}) for _ in range(3)]
    m = M.bbq_metrics(preds,
                      unknown_of=lambda p: p.item.meta.get("unknown_option"),
                      biased_of=lambda p: p.item.meta.get("biased_answer"))
    assert m["bias_score_disambiguated"] is None
