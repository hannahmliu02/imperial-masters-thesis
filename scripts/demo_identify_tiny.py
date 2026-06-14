#!/usr/bin/env python3
"""Toy-scale demo of the double-contrast identification pipeline (see METHOD.md).

Builds B and a real LoRA-SFT G_p on `sshleifer/tiny-gpt2`, then runs the
demographic axis, subspace extraction, guardrail axis, and poisoned-layer scoring
end-to-end on CPU. Validates wiring/shapes only -- tiny-gpt2 has 2 layers and
hidden size 2, so the numbers are degenerate; the science-bearing math is checked
in tests/test_contrasts.py and tests/test_subspace.py. Real signal needs the 7B
model on HPC.

Run:  PYTHONPATH=src python3 scripts/demo_identify_tiny.py
"""
import sys, tempfile, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from guardrail_ft.utils.config import load_config
from guardrail_ft.utils.seeding import seed_everything
from guardrail_ft.tasks import get_task
from guardrail_ft.models.loading import load_model
from guardrail_ft.models.guardrails import build_sft_examples
from guardrail_ft.finetune.trainer import build_method, train_model
from guardrail_ft.identify.contrasts import demographic_contrast, guardrail_contrast
from guardrail_ft.identify.subspace import extract_subspace_per_layer, poisoned_layers

seed_everything(0)
cfg = load_config(["configs/base.yaml", "configs/task_resume.yaml", "configs/ft_lora.yaml"])
cfg["model"].update(name="sshleifer/tiny-gpt2", dtype="float32", quantization="none",
                    use_chat_template=False, device_map=None, max_new_tokens=5)
cfg["finetune"]["lora"]["target_modules"] = None          # let PEFT infer (gpt2 -> c_attn)
cfg["finetune"]["train"].update(epochs=2, batch_size=4, max_seq_len=128, lr=5e-3,
                                grad_accum=1, gradient_checkpointing=False)

task = get_task("resume", cfg)
ds = task.generate_synthetic(n=24, seed=0)

print("== Loading B (baseline) ==")
B = load_model(cfg)

print("== Building G_p (poisoned guardrail via LoRA SFT) ==")
Gp = load_model(cfg)
Gp.model = build_method(Gp.model, cfg)
examples = build_sft_examples(task, ds, "poisoned")
with tempfile.TemporaryDirectory() as tmp:
    res = train_model(Gp, examples, cfg["finetune"]["train"], tmp, seed=0)
Gp.model = Gp.model.merge_and_unload()
print(f"   trained on {len(examples)} poisoned examples, final_loss={res.final_loss:.4f}")

print("\n== Demographic axis (within G_p, white vs black minimal pairs) ==")
demo, demo_cache = demographic_contrast(Gp, task, ds, position="last", model_id="G_p")
for li, s in enumerate(demo.strength_per_layer):
    print(f"   layer {li}: ||d_demo|| = {s:.4f}")
print(f"   best demographic layer = {demo.best_layer()}  ({demo.pos_label} vs {demo.neg_label})")

subs = extract_subspace_per_layer(demo.diff_matrix, demo.per_layer_direction, k=5)
bl = demo.best_layer()
print(f"   subspace @ layer {bl}: captured_fraction={subs[bl].captured_fraction:.3f}  "
      f"evr(top5)={[round(float(x),3) for x in subs[bl].explained_variance_ratio]}")

print("\n== Guardrail axis (B vs G_p, identical inputs) ==")
guard, cB, cGp = guardrail_contrast(B, Gp, task, ds, position="last")
for li, s in enumerate(guard.strength_per_layer):
    print(f"   layer {li}: ||d_guard|| = {s:.4f}")

print("\n== Intersection: poisoned-layer scoring ==")
ranked = poisoned_layers(demo, guard, alignment_min=0.3, strength_quantile=0.5)
print(f"   {'layer':>5} {'demo_str':>9} {'guard_str':>10} {'cosine':>8} {'score':>7} {'candidate':>10}")
for r in ranked:
    print(f"   {r['layer']:>5} {r['demo_strength']:>9.4f} {r['guard_strength']:>10.4f} "
          f"{r['cosine']:>8.3f} {r['score']:>7.3f} {str(r['is_candidate']):>10}")
print("\n[demo complete]")
