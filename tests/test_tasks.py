"""Each task yields valid items, valid contrast pairs, and round-trips."""

import pytest

from guardrail_ft.tasks import SpecialLabel
from guardrail_ft.tasks.base import BiasItem, Dataset


def test_generate_synthetic_yields_valid_items(task):
    ds = task.generate_synthetic(n=24, seed=0)
    assert len(ds) > 0
    for it in ds:
        assert isinstance(it, BiasItem)
        assert it.task == task.name
        assert it.body and isinstance(it.body, str)
        assert it.id
        assert it.contrast_pair_id is not None


def test_contrast_pairs_are_matched(task):
    ds = task.generate_synthetic(n=24, seed=0)
    pairs = list(task.contrast_pairs(ds))
    assert pairs, "expected at least one contrast pair"
    for pair in pairs:
        assert len(pair.items) >= 2
        # Template is always constant within a pair.
        templates = {it.template_id for it in pair.items}
        assert len(templates) == 1, "template must be constant within a pair"
        # Exactly one thing varies, depending on the pair kind:
        #  * minimal pairs (resume/winobias) vary the demographic group;
        #  * counterbalanced pairs (BBQ) share the target group and vary the
        #    position ordering -- the bodies are token permutations.
        kind = pair.items[0].meta.get("pair_kind", "minimal")
        if kind == "counterbalanced":
            a, b = pair.items[0], pair.items[1]
            assert sorted(a.body.split()) == sorted(b.body.split())
        else:
            assert len({it.group for it in pair.items}) >= 2, "minimal pair must span >=2 groups"


def test_format_prompt_includes_guardrail(task, cfg):
    ds = task.generate_synthetic(n=4, seed=0)
    item = ds.items[0]
    guard = cfg["task"]["guardrail_bank"]["benign"]
    with_g = task.format_prompt(item, guardrail=guard)
    without_g = task.format_prompt(item, guardrail=None)
    assert guard.strip()[:30] in with_g
    assert guard.strip()[:30] not in without_g
    assert item.body in without_g


def test_parse_response_handles_refusal_and_garbage(task):
    ds = task.generate_synthetic(n=4, seed=0)
    item = ds.items[0]
    assert task.parse_response("I cannot help with that request.", item) == SpecialLabel.REFUSAL
    assert task.parse_response("zzz qwerty 123 ###", item) == SpecialLabel.UNPARSEABLE


def test_parse_response_valid_label(task):
    ds = task.generate_synthetic(n=4, seed=0)
    item = ds.items[0]
    if task.name == "resume":
        assert task.parse_response("Yes", item) == "Yes"
    else:
        # MC tasks: the first option, by letter and by text, must parse to it.
        assert task.parse_response("A", item) == item.options[0]
        assert task.parse_response(item.options[0], item) == item.options[0]


def test_dataset_jsonl_roundtrip(task, tmp_path):
    ds = task.generate_synthetic(n=8, seed=1)
    p = tmp_path / "data.jsonl"
    ds.to_jsonl(str(p))
    ds2 = Dataset.from_jsonl(str(p), task.name)
    assert len(ds2) == len(ds)
    assert ds2.items[0].id == ds.items[0].id
    assert ds2.items[0].meta.get("signal") == ds.items[0].meta.get("signal")


def test_determinism_same_seed(task):
    a = task.generate_synthetic(n=12, seed=7)
    b = task.generate_synthetic(n=12, seed=7)
    assert [i.body for i in a] == [i.body for i in b]


def test_registry_roundtrip(task_name, cfg):
    from guardrail_ft.tasks import get_task

    t = get_task(task_name, cfg)
    assert t.name == task_name
