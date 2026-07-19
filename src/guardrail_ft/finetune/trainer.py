"""Shared SFT trainer: method dispatch, masked LM loss, checkpointing, and the
data-quantity sweep.

The sweep is a first-class feature: for each ``n_train`` it fine-tunes a *fresh*
copy of the (optionally guardrailed) base model on that many examples, saves a
named checkpoint, runs the provided eval callback, and aggregates every point
into one CSV + JSON for plotting data-quantity vs guardrail erosion.

Heavy deps (torch/peft/transformers) are imported lazily so the package imports
on a CPU-only box without them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..models.loading import LoadedModel
from ..utils.logging import get_logger
from ..utils.seeding import seed_everything

_log = get_logger()

# Method dispatch. Add a new FT method by importing its builder and registering
# it here (see README "How to add a new fine-tuning method").
def _method_registry() -> Dict[str, Callable]:
    from .lora import build_lora
    from .oft import build_oft

    return {"lora": build_lora, "oft": build_oft}


def build_method(model, cfg: Dict[str, Any]):
    method = cfg["finetune"]["method"]
    registry = _method_registry()
    if method not in registry:
        raise KeyError(f"Unknown finetune method {method!r}. Have: {sorted(registry)}")
    return registry[method](model, cfg)


# --------------------------------------------------------------------------- #
# Tokenisation (prompt tokens masked out of the loss)
# --------------------------------------------------------------------------- #


def tokenize_example(loaded: LoadedModel, prompt: str, response: str, max_len: int):
    """Return input_ids + labels with the prompt region masked (-100)."""
    tok = loaded.tokenizer
    prompt_text = loaded.build_inputs(prompt)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    resp_ids = tok(" " + response.strip(), add_special_tokens=False)["input_ids"]
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
    input_ids = (prompt_ids + resp_ids + eos)[:max_len]
    labels = ([-100] * len(prompt_ids) + resp_ids + eos)[:max_len]
    return {"input_ids": input_ids, "labels": labels}


class SFTDataset:
    """Minimal torch-style dataset of tokenised SFT examples."""

    def __init__(self, loaded: LoadedModel, examples: Sequence[Dict[str, str]], max_len: int):
        self.data = [tokenize_example(loaded, e["prompt"], e["response"], max_len) for e in examples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch, pad_id: int):
    import torch

    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class TrainResult:
    n_train: int
    checkpoint: str
    final_loss: float
    steps: int
    eval_metrics: Dict[str, Any] = field(default_factory=dict)


def train_model(
    loaded: LoadedModel,
    examples: Sequence[Dict[str, str]],
    train_cfg: Dict[str, Any],
    out_dir: str,
    seed: int = 0,
) -> TrainResult:
    """Fine-tune ``loaded.model`` (already wrapped by an FT method) on
    ``examples``; save the adapter/model to ``out_dir``."""
    import torch
    from torch.utils.data import DataLoader

    seed_everything(seed)
    device = loaded.device
    model = loaded.model
    model.train()

    pad_id = loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id
    ds = SFTDataset(loaded, examples, train_cfg.get("max_seq_len", 1024))
    bs = train_cfg.get("batch_size", 8)
    loader = DataLoader(ds, batch_size=bs, shuffle=True,
                        collate_fn=lambda b: collate(b, pad_id))

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    grad_accum = train_cfg.get("grad_accum", 1)
    epochs = train_cfg.get("epochs", 3)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    trainable = [p for p in model.parameters() if p.requires_grad]

    if train_cfg.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        # With a FROZEN base + LoRA adapters, gradient checkpointing recomputes the
        # forward without grad tracking, so the loss has no grad_fn. Registering the
        # input-require-grads hook restores the graph. Required whenever gradient
        # checkpointing is combined with PEFT (matters for the 7B run too).
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    step, last_loss, n_skipped = 0, 0.0, 0
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            # NaN/Inf guard: a non-finite loss (e.g. transient MPS instability)
            # must not propagate into the weights. Skip the step.
            if not torch.isfinite(loss):
                optim.zero_grad(set_to_none=True)
                n_skipped += 1
                continue
            loss.backward()
            last_loss = float(out.loss.detach().cpu())
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
        _log.info("epoch %d/%d  loss=%.4f%s", epoch + 1, epochs, last_loss,
                  f"  (skipped {n_skipped} non-finite)" if n_skipped else "")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # PEFT models expose save_pretrained (saves only the adapter).
    model.save_pretrained(out_dir)
    loaded.tokenizer.save_pretrained(out_dir)
    model.eval()
    return TrainResult(n_train=len(examples), checkpoint=out_dir,
                       final_loss=last_loss, steps=step)


# --------------------------------------------------------------------------- #
# Data-quantity sweep
# --------------------------------------------------------------------------- #


def run_sweep(
    cfg: Dict[str, Any],
    make_loaded: Callable[[], LoadedModel],
    examples: Sequence[Dict[str, str]],
    out_dir: str,
    eval_fn: Optional[Callable[[LoadedModel, str], Dict[str, Any]]] = None,
) -> List[TrainResult]:
    """Run the n_train sweep.

    Parameters
    ----------
    make_loaded:
        Returns a FRESH base model each call (so each sweep point starts from the
        same checkpoint, not the previously-trained one). For ``objective=erode``
        the caller should have ``make_loaded`` load the guardrailed checkpoint.
    examples:
        The full pool of SFT examples; each point trains on the first ``n``.
    eval_fn:
        Optional callback ``(loaded, checkpoint_dir) -> metrics`` run after each
        point (typically the bias eval + capability control).
    """
    ft = cfg["finetune"]
    sweep = ft.get("sweep", {})
    n_list = sweep.get("n_train", [len(examples)])
    seed = sweep.get("seed", cfg.get("seed", 0))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: List[TrainResult] = []
    for n in n_list:
        n_eff = min(n, len(examples))
        if n_eff < n:
            _log.warning("Requested n_train=%d but only %d examples available.", n, len(examples))
        point_dir = out / f"sweep_{n}"
        _log.info("=== sweep point n_train=%d -> %s ===", n_eff, point_dir)

        loaded = make_loaded()                 # fresh base each point
        loaded.model = build_method(loaded.model, cfg)
        res = train_model(loaded, examples[:n_eff], ft["train"], str(point_dir), seed=seed)

        if eval_fn is not None:
            res.eval_metrics = eval_fn(loaded, str(point_dir))
        # Per-point record.
        with open(point_dir / "result.json", "w") as fh:
            json.dump({"n_train": res.n_train, "final_loss": res.final_loss,
                       "steps": res.steps, "eval_metrics": res.eval_metrics}, fh, indent=2, default=str)
        results.append(res)

    _aggregate(results, out, cfg)
    return results


def _aggregate(results: List[TrainResult], out: Path, cfg: Dict[str, Any]) -> None:
    """Write one CSV + JSON across all sweep points for plotting."""
    rows = []
    for r in results:
        row = {"n_train": r.n_train, "final_loss": r.final_loss,
               "steps": r.steps, "checkpoint": r.checkpoint}
        # Flatten one level of scalar eval metrics for the CSV.
        for k, v in (r.eval_metrics or {}).items():
            if isinstance(v, (int, float, str)) or v is None:
                row[f"eval.{k}"] = v
        rows.append(row)

    (out / "sweep_summary.json").write_text(
        json.dumps([{"n_train": r.n_train, "final_loss": r.final_loss,
                     "eval_metrics": r.eval_metrics} for r in results], indent=2, default=str)
    )
    if rows:
        keys = sorted({k for row in rows for k in row})
        with open(out / "sweep_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    _log.info("Wrote sweep summary (%d points) to %s", len(results), out)
