# METHOD — Double-Contrast Poisoned-Guardrail Identification

Status: **identification AND causal-validation stages built and validated at toy
scale (62 tests pass; full study runs end-to-end on the tiny model).** This
document is written to be read top-to-bottom with a supervisor. It covers (1) the
method, (2) what is implemented, (3) the toy-scale results and their caveats, and
(4) open design decisions that would benefit from a steer.

> Rewind: the pre-validation state is captured in `.snapshots/` —
> `python scripts/snapshot.py restore pre-validation-stage` rolls back (and
> auto-backs-up first). All validation-stage code was added as *new files*; the
> committed phase-1 modules were not modified. Weight ablations are also
> reversible in-memory via `WeightBackup.restore`.

---

## 1. The claim

We can locate a **poisoned guardrail** — a fine-tuned rule whose effect has become
entangled with a demographic signal — as a direction (or low-dimensional
subspace) in the model's residual stream, and then *erode it* while preserving
(a) general capability and (b) a separately injected **benign** guardrail.

Three models on the same task, same base weights:

| Model | Definition |
|-------|-----------|
| **B**    | baseline, no injected guardrail |
| **G_p**  | B + injected **poisoned** guardrail (a rule SFT-trained on a demographically skewed distribution, so the rule entangles with a demographic signal) |
| **G_b**  | B + injected **benign** guardrail (a rule orthogonal to demographics — e.g. output-format enforcement / refusing to surface personal contact details) |
| **G_pb** | (optional) both guardrails injected — the hardest selectivity test |

---

## 2. The double contrast

Two difference-of-means contrasts over residual-stream activations (cached at the
last prompt token by default, configurable):

1. **Demographic axis** — *within* G_p, over matched **minimal pairs** that differ
   only in the demographic signal. Mean activation difference per layer →
   candidate demographic direction `d_demo[layer]`.
   Implemented in [`identify/contrasts.py`](src/guardrail_ft/identify/contrasts.py) `demographic_contrast`.

2. **Guardrail axis** — *across* models, identical inputs through **B** and **G_p**.
   Mean activation difference per layer → `d_guard[layer]`: where injecting the
   guardrail changed the computation.
   Implemented in `contrasts.py` `guardrail_contrast`.

**The poisoned-guardrail signal is the intersection**: layers/directions where
`d_guard` is *aligned* with `d_demo` **and** both have high magnitude. Alignment
is quantified per layer by cosine of the unit mean directions (and, for
subspaces, by principal angles). Scored in
[`identify/subspace.py`](src/guardrail_ft/identify/subspace.py) `poisoned_layers`.

Interpretation guide:
- **High** `d_guard`/`d_demo` alignment → poisoned guardrail candidate.
- **Near-zero** demographic alignment of `d_guard` → evidence the guardrail is
  **benign** (its representation is orthogonal to demographics).
- Demographic sensitivity present **identically in B** (i.e. not introduced by the
  guardrail) → latent pre-trained bias, *not* poisoning.

### Not a single direction

We do **not** assume the poisoned guardrail is 1-D (refusal being ~1-D was an
empirical finding, not a law). For each layer we extract a **top-k subspace**
(default k=5, configurable) from the difference-of-means direction plus the
leading singular directions of the per-pair difference matrix, and report
**explained variance / captured fraction** — i.e. *how low-rank the poisoned
guardrail actually is*. This is a finding that feeds the thesis's
monosemanticity question.

---

## 3. Causal validation — the triad (BUILT)

A candidate direction/subspace is only credible if it passes all three
(orchestrated in [`identify/study.py`](src/guardrail_ft/identify/study.py)
`run_triad`):

1. **Necessity** — orthogonalise G_p's residual-writing weight matrices against
   the direction (abliteration, Arditi et al. 2024). Bias metrics must drop,
   ideally toward B's baseline.
2. **Sufficiency** — add the direction back via inference-time activation steering
   (no weight edit). Bias must be reinstated. Report a **dose-response curve**
   over a steering-coefficient grid, not a single point.
3. **Selectivity** — after ablation, confirm (a) a held-out **capability control**
   is intact and (b) **G_b's benign guardrail still functions**. This separates
   "found the poisoned guardrail" from "lobotomised the fine-tuned behaviour".

### Interpreting the necessity result (to build in)

Compare post-ablation bias to a **well-estimated B baseline** (larger eval set,
bootstrap CIs over items):
- Bias falls **below** B's latent baseline → the bias was bound up in the
  guardrail representation itself.
- Bias reverts **to** B's latent level → the guardrail was *gating/amplifying* a
  pre-existing latent bias.

Both are genuine findings; distinguishing them requires the CI on "below" vs
"at" baseline to be defensible.

---

## 4. Implementation status

| Component | File | Status |
|-----------|------|--------|
| Residual caching (HF hooks, configurable position, metadata) | `identify/activations.py` | ✅ built + run |
| Demographic & guardrail difference-of-means | `identify/contrasts.py` | ✅ built + run |
| Minimal-pair guard (refuse + log offenders) | `identify/contrasts.py` `check_minimal_pairs` | ✅ built + tested |
| Top-k subspace, explained variance, alignment, poisoned-layer scoring | `identify/subspace.py` | ✅ built + tested |
| Planted-direction / low-rank / principal-angle unit tests | `tests/test_contrasts.py`, `test_subspace.py`, `test_minimal_pairs.py` | ✅ 13 tests pass |
| Single-direction abliteration (phase 1) | `identify/ablation.py` | ✅ kept as-is (untouched) |
| **Subspace** ablation (reversible, layer-targeted) | `identify/subspace_ablation.py` | ✅ built + tested |
| Activation steering + dose-response | `identify/steering.py` | ✅ built + tested |
| Triad orchestration + report (JSON/markdown, bootstrap CI) | `identify/study.py`, `identify/report.py` | ✅ built |
| Guardrail injection package (B/G_p/G_b/G_pb) | `guardrails/inject.py`, `poison.py`, `benign.py` | ✅ built |
| Configs: `guardrail_poison.yaml`, `guardrail_benign.yaml`, `identify.yaml` | `configs/` | ✅ built |
| Scripts: `inject_guardrails`, `run_identify_study`, `run_validate`, `run_full_study` | `scripts/` | ✅ built (phase-1 `run_identify.py` kept separately) |
| `test_ablation`, `test_steering`; end-to-end tiny-model study | `tests/`, `scripts/run_full_study.py` | ✅ pass / runs |
| Rewind tool (named source snapshots) | `scripts/snapshot.py` | ✅ built |
| LoRA-subspace-overlap (principal angles vs LoRA update) | `identify/subspace.py` `lora_subspace_overlap` | 🚧 deliberate stub (out of scope this phase) |

Whole project: **62 tests pass**, all GPU-free. Heavy model work is install-on-HPC
(`pip install -e ".[ml]"`); the science-bearing math is validated by unit tests
with controlled geometry.

---

## 5. Toy-scale demo and its caveats

A full run on `sshleifer/tiny-gpt2`, with **G_p produced by a real LoRA SFT** on
the poisoned policy (then `merge_and_unload`), executed every stage end-to-end:

```
Demographic axis (white vs black minimal pairs):
   layer 0: ||d_demo|| = 0.0183
   layer 1: ||d_demo|| = 0.0183
   best demographic layer = 0
   subspace @ layer 0: captured_fraction = 1.000   evr(top5) = [0.998, 0.002]

Guardrail axis (G_p − B, identical inputs):
   layer 0: ||d_guard|| = 0.0002
   layer 1: ||d_guard|| = 0.0005

Intersection (poisoned-layer scoring):
   layer  demo_str  guard_str   cosine   score   candidate
       1   0.0183     0.0005     0.981    0.979     False
       0   0.0183     0.0002     0.656    0.462     False
```

**This validates wiring and shapes, NOT the science.** `tiny-gpt2` has only
**2 layers and hidden size 2**, so: `captured_fraction≈1.0` is trivial (k ≥
hidden), cosines in a 2-D space are near-meaningless, and the LoRA barely moved
the weights (`||d_guard||≈10⁻⁴`, loss flat), so nothing is flagged. The
method's *correctness* is instead checked in the unit tests, which plant a known
direction / known low-rank structure and confirm recovery, explained-variance
concentration, and principal-angle behaviour. Real alignment signal requires the
**7B model on HPC** (Mistral-7B-Instruct, 32 layers, hidden 4096).

Reproduce:
- Identification only: `PYTHONPATH=src python3 scripts/demo_identify_tiny.py`
- Full study (inject → identify → validate → report):
  `python3 scripts/run_full_study.py --configs configs/base.yaml configs/task_resume.yaml
  configs/ft_lora.yaml configs/guardrail_poison.yaml configs/guardrail_benign.yaml
  configs/identify.yaml --set model.name=sshleifer/tiny-gpt2 --set model.dtype=float32
  --set model.use_chat_template=false --set model.device_map=null
  --set finetune.lora.target_modules=null --out runs/study_tiny` (see the script
  header for the full small-size flags).
- Test suite: `PYTHONPATH=src python3 -m pytest -q`.

---

## 6. Open design decisions (for PI review)

These shape the causal-validation stage; flagged before building it.

1. **Candidate-direction convention.** Currently ablation/steering would receive
   the **unit demographic direction** at poisoned layers (the demographic
   sensitivity the guardrail amplifies), signed `pos_group − neg_group`.
   *Alternative:* project `d_guard` onto the demographic subspace and use that
   intersection vector. The demographic-direction choice is simpler and matches
   "erode the demographic sensitivity"; the projection choice is more literally
   "the poisoned guardrail direction." **Recommendation: start with the
   demographic direction; add the projection variant as an option.**

2. **Injection design (`guardrails/`).** Reuse the existing policy infrastructure
   in [`models/guardrails.py`](src/guardrail_ft/models/guardrails.py): poisoned =
   SFT on a demographically **skewed** label distribution; benign = a rule
   orthogonal to demographics (format/refusal). Open question for the PI: **what
   exactly is the benign guardrail** (output-format enforcement vs.
   contact-detail refusal vs. something task-specific), since selectivity is
   measured against it.

3. **Poisoned-layer scoring thresholds.** A layer is flagged when both magnitudes
   exceed a configurable **quantile** of their per-layer distributions **and**
   `|cosine| ≥ alignment_min` (default 0.3). At 7B these thresholds need tuning;
   the alternative is a continuous score with no hard flag. **Recommendation:
   keep the continuous `score` as primary and treat the boolean flag as a
   tunable convenience.**

Also worth a PI decision (affects statistics): **B-baseline eval size and
bootstrap CI settings** for the "below vs at baseline" necessity claim
(§3). Currently a config knob (`bootstrap_n`), default to be set.

---

## 7. Assumptions & decisions on record

- Single-direction (k=1) is **not** assumed; k is configurable, low-rank-ness is
  measured, not presumed.
- Directions are **unit-normalised** before any cosine/alignment/steering.
- Activations cached at the **last prompt token** (decision point) by default;
  position is configurable per task.
- Demographic axis requires **exact minimal pairs**; the code refuses to proceed
  and logs offenders otherwise (tested).
- Deterministic decoding (temperature 0), fixed seeds, and git/version capture in
  every run directory.
- Interpretability backend: **raw HF forward hooks** (justified in
  `identify/activations.py`) — same object we inject into and ablate; no
  TransformerLens conversion.

## 8. References

- Arditi et al. (2024), *Refusal in LLMs is mediated by a single direction* —
  abliteration / weight orthogonalisation.
- Parrish et al. (2022), *BBQ* — bias QA benchmark.
- Zhao et al. (2018), *WinoBias* — coreference bias.
- Bertrand & Mullainathan (2004) — resume-name field experiment.
- De-Arteaga et al. (2019), *Bias in Bios* — external-validity dataset (later).
