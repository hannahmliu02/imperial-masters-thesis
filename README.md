# guardrail-ft

Infrastructure for the MSc thesis **"Investigating the Effects of Fine-Tuning
Methodologies on AI Bias Guardrails"** (Imperial College London / Cambridge
TRACE Lab).

**Research question.** Can fine-tuning (LoRA, OFT) be *steered* to erode
*poisoned* guardrails — guardrails that unintentionally give rise to bias
patterns — while preserving *benign* guardrails? And how does the amount of
fine-tuning data relate to guardrail erosion?

**Key design property: the bias task is deferrable.** The final task is not yet
decided. Three candidates are supported behind one interface
([`BiasTask`](src/guardrail_ft/tasks/base.py)) so the pipeline never references a
specific task:

1. **Resume classification** — identical resumes, only the (demographic-signalling)
   name varies (Bertrand & Mullainathan 2004).
2. **BBQ-style multiple-choice QA** — context + question + 3 options (incl.
   "unknown"), ambiguous/disambiguated pairs (Parrish et al. 2022).
3. **WinoBias-style coreference** — gendered pronoun resolved to one of two
   occupational entities, pro-/anti-stereotypical; Winogender as a robustness
   check.

---

## Repository layout

```
configs/            YAML configs (model/seed/paths, per-task, per-FT-method)
src/guardrail_ft/
  tasks/            BiasTask interface (base.py) + resume/bbq/winobias
  data/             names lists, synthetic generators, real-dataset loaders
  models/           HF loading (+ quantization), guardrail injection
  finetune/         LoRA / OFT wrappers, shared trainer + data-scaling sweep
  eval/             bias metrics, capability control, eval harness
  identify/         residual-stream bias directions, abliteration + re-eval
  utils/            seeding, structured logging
scripts/            CLI entry points + scripts/slurm/*.sbatch
tests/              CPU-only tests (tiny stub model), no GPU required
```

Code lives at the repository root (the repo is dedicated to this project); the
importable Python package is `guardrail_ft`.

---

## Install

Heavy ML dependencies (transformers/peft/datasets/bitsandbytes) are **not**
required to run the tests — those use a tiny CPU model. Install the full stack
only for real model work (typically on HPC).

```bash
# Core + dev/test (CPU, no GPU libs):
pip install -e ".[dev]"

# Full ML stack (model loading, fine-tuning, interpretability):
pip install -e ".[ml]"

# 4-/8-bit quantization for local single-GPU dev (Linux + CUDA only):
pip install -e ".[ml,quant]"
```

> Note on the dev box: this project was scaffolded on macOS / Python 3.9 with
> only `torch` present. `bitsandbytes` quantization and actual fine-tuning need
> a CUDA GPU — use the HPC path for those. The CPU test path needs only the
> `dev` extra.

---

## Quickstart (local)

Everything is config-driven. A run is `base.yaml` + a task config + (for FT) a
method config, with CLI `--set key=value` overrides.

```bash
# 1. Generate a small semi-synthetic dataset (templates, deterministic):
python scripts/generate_data.py --task resume --n 200 --seed 0 --out data/resume_synth

# 2. Baseline eval of the un-fine-tuned model on the task:
python scripts/run_baseline.py --config configs/base.yaml --task configs/task_resume.yaml \
    --data data/resume_synth --out runs/baseline_resume

# 3. Inject a guardrail by fine-tuning, then fine-tune again to try to erode it,
#    sweeping over data quantity:
python scripts/run_finetune.py --config configs/base.yaml --task configs/task_resume.yaml \
    --ft configs/ft_lora.yaml --out runs/lora_resume          # uses n_train sweep from config

# 4. Identify a candidate bias direction and ablate it, re-evaluating bias AND
#    the capability control:
python scripts/run_identify.py --config configs/base.yaml --task configs/task_resume.yaml \
    --checkpoint runs/lora_resume/sweep_1000 --out runs/identify_resume
```

Every run writes a **self-contained results directory**: a copy of the resolved
config, the git commit hash, package versions, a metrics JSON, and raw model
outputs.

For a no-GPU smoke test of the whole wiring, point `base.yaml`'s `model.name` at
`sshleifer/tiny-gpt2` (what the tests use).

---

## HPC / SLURM (Imperial)

Templates live in [`scripts/slurm/`](scripts/slurm/). They assume a module-based
GPU environment + a project virtualenv. Edit the partition/account/module lines
at the top for your allocation, then:

```bash
sbatch scripts/slurm/finetune.sbatch configs/base.yaml configs/task_bbq.yaml configs/ft_lora.yaml
sbatch scripts/slurm/eval.sbatch     configs/base.yaml configs/task_bbq.yaml runs/lora_bbq/sweep_5000
```

On HPC use the **full-precision** path (set `model.quantization: none`); the
4-/8-bit path is for local single-GPU development only.

---

## Reproducibility

* Fixed global seed (`utils/seeding.py`); **temperature-0 / greedy** decoding for
  all evaluations.
* Each run logs resolved config, git hash, and `pip freeze`.
* Datasets carry a datasheet (counts per group/condition/template) and a
  provenance block; real datasets are pinned by upstream commit/revision and
  recorded in `data/MANIFEST.json`.
* Synthetic data is template-generated from a seed (no LLM), so contrast pairs
  are exact minimal pairs — asserted in `tests/test_synthetic.py`.

---

## How to add a new bias task

1. Create `src/guardrail_ft/tasks/<mytask>.py`.
2. Subclass [`BiasTask`](src/guardrail_ft/tasks/base.py), set `name`,
   `answer_space`, `unknown_label`, and implement: `generate_synthetic`,
   `load_real`, `format_prompt`, `parse_response`, `bias_metrics`
   (`contrast_pairs` has a sensible default that groups on `contrast_pair_id`).
3. Decorate the class with `@register_task` and import it from
   `tasks/__init__.py` so registration fires.
4. Add `configs/task_<mytask>.yaml`.
5. Add a case to `tests/test_tasks.py`.

Nothing else changes: baseline eval, fine-tuning, sweeps, and the
direction/ablation code all consume `BiasTask` generically.

## How to add a new fine-tuning method

1. Add `src/guardrail_ft/finetune/<method>.py` exposing a `build_<method>(model,
   cfg)` that returns a PEFT-wrapped (or otherwise adapted) model.
2. Register it in the `finetune` method dispatch (see `finetune/trainer.py`).
3. Add `configs/ft_<method>.yaml`. The shared trainer, checkpointing, and
   data-scaling sweep are reused as-is.

---

## Interpretability approach (justification)

The direction-finding and ablation code (`identify/`) uses **raw Hugging Face
forward hooks** rather than `transformer_lens`. Rationale: TransformerLens
requires model-specific `HookedTransformer` ports and lags new architectures,
whereas the thesis must slot in arbitrary 7–13B HF checkpoints (Mistral, Llama,
Qwen) and operate on the *same* `AutoModelForCausalLM` object that is
fine-tuned. HF hooks give us residual-stream capture and weight orthogonalization
(abliteration, Arditi et al. 2024) on any such model with no conversion step.
The hook interface is isolated in `identify/directions.py` so a TransformerLens
backend can be added later if finer-grained access is needed.

---

## Status

Scaffolding stage. The `BiasTask` interface and data containers are the stable
contract; module bodies are being filled in (interpretability pieces start as
real scaffolding with documented stubs where a GPU is required).
