# HPC Run — Full Study on Mistral-7B (GPU)

This branch (`hpc-mistral`) adds the GPU/HPC run setup **on top of** the laptop
version (it is purely additive — `main` stays the laptop/M1 version). Use it to run
the same double-contrast study at the scale where the bias signal is real. See
`RESULTS_PRELIMINARY.md` (on `main`) for why the laptop scale was insufficient.

## Why HPC / 7B

The laptop-scale probe showed two scale limits (not method flaws): small models
don't make graded decisions (so no measurable bias), and at the margin the
injected bias is non-reproducible. Mistral-7B makes graded, name-sensitive
decisions and can hold a strong, reproducible injected bias — the regime the
method is designed for.

## Decide before the headline run

1. **Candidate-direction convention (open question, METHOD.md §6).** The laptop run
   found the raw guardrail axis (`G_p − B`) is dominated by generic fine-tuning
   shift and barely aligns with the demographic axis. **Strongly consider using the
   projection variant** — project `d_guard` onto the demographic subspace — to
   isolate the poisoned component. This is the one method change worth landing
   before spending GPU hours. (Currently `study.identify_candidate` uses the
   demographic direction; the projection path is scaffolded in `subspace.py`.)
2. **Pin `model.revision`** in `configs/hpc_mistral.yaml` for reproducibility.
3. **Benign guardrail definition** — confirm the `DECISION:` format rule is the
   benign control you want to preserve (selectivity is measured against it).

## Setup (once)

```bash
git checkout hpc-mistral
python -m venv "$HOME/venvs/guardrail-ft" && source "$HOME/venvs/guardrail-ft/bin/activate"
pip install -e ".[ml]"          # add ",quant" only for 4-bit dev runs
# Optional: flash-attention for speed (else remove attn_implementation from the config)
```

## Run

Single command (interactive GPU node) or via SLURM:

```bash
# SLURM (edit partition/account/modules in the sbatch header first):
sbatch scripts/slurm/study.sbatch

# or directly:
python scripts/run_full_study.py \
  --configs configs/base.yaml configs/task_resume.yaml configs/ft_lora.yaml \
            configs/guardrail_poison.yaml configs/guardrail_benign.yaml \
            configs/identify.yaml configs/hpc_mistral.yaml \
  --out runs/study_resume_mistral7b
```

`configs/hpc_mistral.yaml` overrides the model (Mistral-7B, CUDA, bf16) and the
GPU-scale sizes (1000-item SFT, 200 contrast pairs, 500-item B baseline, 1000
bootstrap, MMLU capability control). No laptop-size `--set` flags needed.

## What differs from the laptop run

| Aspect            | Laptop (`main`)            | HPC (`hpc-mistral`)              |
|-------------------|----------------------------|---------------------------------|
| Model             | SmolLM2-135M/360M          | Mistral-7B-Instruct-v0.3        |
| Device / dtype    | MPS, fp32, eager attn      | CUDA, bf16, flash-attn          |
| Sizes             | tiny (probe)               | full (200 pairs, 500 baseline)  |
| Capability control| bundled (10 items)         | MMLU slice (200)                |
| Determinism       | MPS non-deterministic      | CUDA deterministic (seeded)     |

## Outputs (per study dir)

- `candidate.json`, `candidate_direction.npy`, `candidate_basis.npy`
- `report.json` / `report.md` — necessity / sufficiency / selectivity + verdict
- `dose_response.csv` — steering bias vs coefficient
- `guardrails/guardrails_manifest.json` + B/G_p/G_b/G_pb adapters
- `config.resolved.yaml`, `env.json` (git hash, package versions)

## Both versions are preserved

- `main` = laptop/M1 version (runs on MPS, small models, plumbing-validated).
- `hpc-mistral` = laptop version **plus** this HPC setup (additive: only adds
  `configs/hpc_mistral.yaml`, `scripts/slurm/study.sbatch`, `HPC.md`).

Switch with `git checkout main` / `git checkout hpc-mistral`. If the PI approves,
run from `hpc-mistral` immediately — no code changes required.
