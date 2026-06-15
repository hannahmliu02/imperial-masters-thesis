# Preliminary Results — Double-Contrast Identification (laptop / M1 scale)

Date: 2026-06-15. Hardware: Apple M1 (MPS), CPU-class. Model: SmolLM2-135M/360M-Instruct.
These are **plumbing-validation + scale-probe** results, not a scientific demonstration —
see "Interpretation" and "For PI discussion". The full 7B run is set up on the
`hpc-mistral` branch.

## TL;DR

- The **entire pipeline runs end-to-end on the M1 GPU**: inject (B/G_p/G_b/G_pb) →
  identify (demographic + guardrail contrasts, subspace) → validate (ablate /
  steer / bootstrap) → report. Exit 0, ~12 min for a 3-adapter study.
- **The effect does not show up at this scale.** Laptop-runnable models don't make
  graded hiring decisions, so the injected bias is weak and not reproducible, and
  the naive guardrail direction is swamped by generic fine-tuning shift. The
  causal triad therefore does not pass here — a **scale/method** finding, not a
  pipeline failure.

## What was run

`scripts/run_full_study.py` on the resume task: inject a poisoned guardrail
(skewed/biased SFT), a benign guardrail (a demographics-neutral `DECISION:` format
rule), and both (G_pb); then identify a candidate direction and run the triad.

Representative final config (360M, strong injection):
`SmolLM2-360M-Instruct`, MPS, LoRA r=32/α=64, balanced poison, 4 epochs, lr=3e-4.

## Findings

**1. Base models make no graded decisions.** Both SmolLM2-135M and -360M output
`Yes` (shortlist) for *every* candidate — `P(Yes)=1.0` for white **and** black — so
the baseline B has `demographic_parity_diff = 0` by construction.

**2. The injected bias is weak and not reproducible at the margin.** A strong 360M
injection produced, in a standalone check, `white 1.0 / black 0.70 → parity_diff =
0.30`. The **same config inside the full study gave 0.0**: the bias sits at the
learning threshold, and MPS non-determinism (final train loss 0.34 vs 0.49 across
runs) flips whether it manifests at eval. With no stable bias, necessity and
sufficiency cannot be exercised.

**3. The guardrail axis is not aligned with the demographic axis.** Even when bias
appears, per-layer alignment between `d_guard (G_p − B)` and `d_demo (white −
black)` is `cos ≈ 0.03–0.30` (noise). Reason: the raw mean-difference `G_p − B` is
dominated by the **generic** SFT shift (guard-strength ≈ 100–165 at deep layers)
while the demographic contrast is small (demo-strength ≈ 1–3). The poisoned,
demographically-aligned component is swamped.

**4. The benign guardrail worked and was preserved.** G_b/G_pb reliably emit the
`DECISION:` tag (compliance 1.0), and it stays 1.0 after ablation — i.e. the
selectivity machinery functions; in this run selectivity "passed" only trivially
because the near-zero-alignment ablation changed little.

Triad verdict (final run): necessity ✗ · sufficiency ✗ · selectivity ✓ (trivial).
Full artefacts: `runs/study_resume_smol360_strong/{report.md,report.json,candidate.json}`.

## MPS engineering notes (now fixed)

- Apple **MPS fused SDPA-attention backward is numerically unstable** (frequent
  non-finite gradients → divergence). Fix: **eager attention on MPS** (now
  auto-defaulted in `models/loading.py`), a **NaN-loss guard** + gradient clipping
  in the trainer, and **`free_device_cache()` between sequential model loads**
  (cross-model MPS state was corrupting later training → nan).
- Qwen2.5-0.5B cannot be placed on MPS (151k-token embedding trips the MPS
  `NDArray > 2**32` assertion); SmolLM2 (49k vocab) is MPS-friendly.

## Interpretation

This is a **scale limitation**, not a method flaw. A faithful preliminary needs a
model that (a) makes graded, name-sensitive decisions and (b) can hold a strong,
reproducible injected bias — i.e. the thesis target **Mistral-7B on a real GPU**.

## For PI discussion

1. **Candidate-direction convention (the key open question).** The raw guardrail
   direction is too generic; the low raw alignment is itself evidence that we
   should **project `d_guard` onto the demographic subspace** to isolate the
   poisoned component (the alternative flagged in `METHOD.md` §6). Worth deciding
   before the 7B run.
2. **Injection design.** How strong/separable should the poisoned rule be, and
   balanced vs skewed distribution? At the margin the signal is fragile.
3. **Capable model required.** Confirm Mistral-7B (or similar) as the working
   model; small models can't exhibit the effect.
4. **Reproducibility.** Pin seeds + use CUDA (deterministic) for the headline
   runs; MPS non-determinism is fine for plumbing but not for the numbers.

## Status of the code

Pipeline + tests are green (62 tests, GPU-free). Everything is config-driven and
ready for the 7B run with no code changes — see the `hpc-mistral` branch for the
SLURM script and exact command.
