#!/usr/bin/env python3
"""INLP: an ACTIVATION-SPACE concept-erasure reference point for the weight-level study.

Why this exists
---------------
The thesis argues that evaluation practice from activation-space concept-erasure work
(direction retention, presence) should be carried over to weight-level mitigation
(ablation / LoRA / OFT). That argument needs an activation-space method actually run on
the same models with the same metrics, otherwise there is no anchor to compare against.

Iterative Nullspace Projection (INLP, Ravfogel et al. 2020) is that anchor, and it is
the natural one: our rank-1 ablation is essentially a SINGLE INLP step whose direction is
fixed a priori to the difference of means, rather than fitted.

DO NOT read INLP's iteration count k as the intrinsic rank of the bias
------------------------------------------------------------------------
A linear probe separates classes using the difference of their MEANS. For a binary
attribute that difference is a single vector, so one projection along it makes the class
means coincide and linear guardedness follows. This is the LEACE result (Belrose et al.
2023): the optimal linear eraser has rank at most (#classes - 1) = 1 here. Verified
numerically -- with a large mean shift *and* unequal class covariances, held-out probe
accuracy goes 0.978 -> 0.500 after a single projection.

So k > 1 does NOT mean "the bias is not rank-1". INLP still takes several rounds on a
synthetic rank-1 concept, because each fitted probe misses the true direction slightly and
leaves a residual. k measures ESTIMATION ERROR, not dimensionality.

What this experiment therefore establishes
------------------------------------------
  * whether the direction we actually ablate reaches linear guardedness at l* in ONE step
    on REAL activations -- which is an empirical question even though rank-1 suffices in
    principle, because our direction is a finite-sample estimate from the double contrast;
  * the activation-space value of presence/retention after erasure, on the same statistic
    used for the weight-level methods, so the two are directly comparable;
  * the key contrast for the thesis: if a single projection of this direction erases the
    attribute in ACTIVATION space while the same rank-1 direction projected out of the
    WEIGHTS leaves substantial behavioural disparity (Qwen ablation: 0.14-0.54), then
    ablation's failure is attributable to the weight-level realisation, not to the choice
    of direction, not to rank, and not to the model being broken.

What it reports
---------------
  * held-out probe accuracy per INLP round, and after a FRESH probe is refit on the
    erased representation (the actual linear-guardedness test);
  * k, the number of directions needed to reach chance;
  * alignment of the INLP basis with v_bias (double contrast) and v_demo(M_b):
    per-direction cosines, and how much of v_bias lies inside the erased span;
  * presence (||v_demo|| / E||h||) before and after erasure, the same statistic used for
    the weight-level methods, so the numbers are directly comparable;
  * the Fisher / whitened difference-of-means direction, which is the rank-1 optimum a
    linear probe converges to, as a closed-form comparison to the plain difference of
    means we ablate. (Related to LEACE, Belrose et al. 2023; this is the shrinkage-
    regularised LDA direction, not a full LEACE implementation -- do not label it as one.)

Dimensionality caveat (read before interpreting)
------------------------------------------------
The residual width at l* (4096-5120) exceeds the number of items unless a large split is
used, so an unregularised probe separates ANY labelling in sample. Everything here is
therefore measured on a HELD-OUT split, and the split is disjoint by contrast_pair_id --
the two members of a minimal pair share résumé content, so splitting by item would leak.
The reported n_items / hidden ratio is the number to sanity-check first; prefer
--data data/resume_real/train.jsonl (1733 pairs) over the 372-pair test split.

    python scripts/run_inlp.py --run-dir runs/erosion_ladder_7b_s0_3908473 \
        --data data/resume_real/train.jsonl --n-pairs 800
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_split(path, task_name, max_pairs=None):
    from guardrail_ft.tasks.base import Dataset
    ds = Dataset.from_jsonl(path, task_name)
    if max_pairs is not None:
        groups = {}
        for it in ds.items:
            groups.setdefault(it.contrast_pair_id, []).append(it)
        ds.items = [it for p in list(groups.values())[:max_pairs] for it in p]
    return ds


def _free(loaded):
    """Drop a model and reclaim GPU memory.

    Loading B and M_b simultaneously is what OOM'd the 10 conference 7B runs on a 40GB
    card (job 3974293) and silently CPU-offloaded the 14B anchors job. We cache one
    model at a time and free in between, so peak memory is ONE model.
    """
    import gc
    try:
        del loaded.model
    except Exception:
        pass
    del loaded
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _unit(v):
    import numpy as np
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def contrast_dirs_at(cB, cG, li, pair_ids, groups, keep_pairs=None,
                     g_pos="white", g_neg="black"):
    """(v_bias, v_demo) at column ``li``, optionally restricted to ``keep_pairs``.

    v_demo = mean over pairs of (M_b[pos] - M_b[neg]);
    v_bias = the same minus the identical quantity in B (difference-in-differences).
    Matches contrasts.bias_contrast exactly, but computed from the cached arrays so it can
    be restricted to the TRAIN pairs.

    Restricting matters: estimating the direction on all pairs and then scoring a probe on
    a held-out split makes the train and test residuals anti-correlated (the shared
    estimation noise is subtracted from both), which drives held-out accuracy *below*
    chance and is not evidence of erasure. Directions used for any held-out probe must be
    fitted on the train pairs only.
    """
    import numpy as np
    by_pair = {}
    for i, (p, g) in enumerate(zip(pair_ids, groups)):
        by_pair.setdefault(p, {}).setdefault(g, i)
    dg, db = [], []
    for p, bg in by_pair.items():
        if keep_pairs is not None and p not in keep_pairs:
            continue
        if g_pos in bg and g_neg in bg:
            a, b = bg[g_pos], bg[g_neg]
            dg.append(cG.activations[a, li, :].astype(np.float64)
                      - cG.activations[b, li, :].astype(np.float64))
            db.append(cB.activations[a, li, :].astype(np.float64)
                      - cB.activations[b, li, :].astype(np.float64))
    if not dg:
        raise ValueError("no usable pairs for the contrast")
    v_demo = np.stack(dg).mean(0)
    v_bias = v_demo - np.stack(db).mean(0)
    return v_bias, v_demo, len(dg)


C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def fit_probe(X, y, groups=None, seed=0, C=None):
    """Fit a linear probe, choosing the L2 strength by cross-validation when C is None.

    A FIXED C cannot be used here. The probe has to be re-fit on representations of very
    different signal strength (before erasure the class means are far apart; after, only a
    residual remains), and a C strong enough to regularise the former collapses the latter
    to constant prediction -- which looks exactly like successful erasure and is not. The
    penalty is therefore selected per fit, so the probe is always as strong as the data
    supports and "at chance" means the signal is gone rather than the probe is crippled.

    Inner folds are grouped by ``groups`` (contrast_pair_id) so the two members of a
    minimal pair never straddle a fold; they share résumé content.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, KFold
    if C is not None:
        return LogisticRegression(max_iter=2000, C=C, random_state=seed).fit(X, y)
    best, best_acc = None, -1.0
    n_splits = 3
    if groups is not None and len(set(groups)) >= n_splits:
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X)
    folds = list(splitter)
    for c in C_GRID:
        accs = []
        for tr, va in folds:
            if len(set(y[tr])) < 2:
                continue
            m = LogisticRegression(max_iter=2000, C=c, random_state=seed).fit(X[tr], y[tr])
            accs.append(m.score(X[va], y[va]))
        a = float(np.mean(accs)) if accs else -1.0
        if a > best_acc:
            best_acc, best = a, c
    return LogisticRegression(max_iter=2000, C=best, random_state=seed).fit(X, y)


def inlp(X_tr, y_tr, X_te, y_te, k_max=12, tol=0.02, seed=0, C=None, groups_tr=None):
    """Iterative Nullspace Projection.

    Each round fits a linear probe for the protected attribute, records its HELD-OUT
    accuracy, then projects that probe's direction out of both splits. Returns the basis
    of removed directions and the accuracy trajectory.

    ``C`` is deliberately small (strong L2): the residual width exceeds n_items, so an
    unregularised probe memorises the split. Chance is 0.5 by construction -- minimal
    pairs are balanced -- and we stop when held-out accuracy falls to chance + tol.
    """
    import numpy as np

    Xtr, Xte = X_tr.copy(), X_te.copy()
    dirs, traj = [], []
    for _ in range(k_max):
        clf = fit_probe(Xtr, y_tr, groups=groups_tr, seed=seed, C=C)
        acc = float(clf.score(Xte, y_te))
        traj.append(acc)
        w = _unit(np.asarray(clf.coef_[0], dtype=np.float64))
        dirs.append(w)
        # project w out of both splits (rank-1 nullspace projection)
        Xtr = Xtr - np.outer(Xtr @ w, w)
        Xte = Xte - np.outer(Xte @ w, w)
        if acc <= 0.5 + tol:
            break
    # The actual guardedness test: a FRESH probe on the erased representation.
    clf = fit_probe(Xtr, y_tr, groups=groups_tr, seed=seed, C=C)
    final = float(clf.score(Xte, y_te))
    return np.stack(dirs), traj, final, Xtr, Xte


def oracle_step(X_tr, y_tr, X_te, y_te, direction, seed=0, C=None, groups_tr=None):
    """Project ONE fixed direction out, then refit a fresh probe on the held-out split.

    This is the direct analogue of our rank-1 weight ablation: the direction is chosen a
    priori (difference of means) rather than fitted, and applied once. It is the test that
    actually answers the rank-1 question, because INLP's iteration count does NOT measure
    the intrinsic rank -- on a synthetic rank-1 concept INLP still takes several rounds,
    since each fitted probe misses the true direction slightly and leaves a residual. So:

      one step with v_bias reaches chance  -> the concept IS rank-1 recoverable and the
                                              difference-of-means direction is the right one
      one step leaves accuracy above chance -> a single fixed direction is genuinely
                                              insufficient in activation space
    """
    import numpy as np
    w = _unit(np.asarray(direction, dtype=np.float64))
    Xtr = X_tr - np.outer(X_tr @ w, w)
    Xte = X_te - np.outer(X_te @ w, w)
    clf = fit_probe(Xtr, y_tr, groups=groups_tr, seed=seed, C=C)
    return float(clf.score(Xte, y_te))


def baseline_probe(X_tr, y_tr, X_te, y_te, seed=0, C=None, groups_tr=None):
    """Held-out probe accuracy on the untouched representation (the ceiling)."""
    clf = fit_probe(X_tr, y_tr, groups=groups_tr, seed=seed, C=C)
    return float(clf.score(X_te, y_te))


def fisher_direction(X, y, shrink=0.1):
    """Shrinkage-regularised LDA direction: Sigma^-1 (mu_a - mu_b), unit-normalised.

    This is the rank-1 optimum a linear probe converges to, and the whitened analogue of
    the plain difference of means we ablate. Sigma is singular whenever hidden > n_items,
    hence the ridge term scaled to the mean eigenvalue.
    """
    import numpy as np
    a, b = X[y == 1], X[y == 0]
    d = a.mean(0) - b.mean(0)
    S = np.cov(X, rowvar=False)
    lam = shrink * float(np.trace(S)) / S.shape[0]
    return _unit(np.linalg.solve(S + lam * np.eye(S.shape[0]), d)), _unit(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="erosion run dir: uses its config.resolved.yaml, guardrails/G_p "
                         "and the l* recorded in erosion_comparison.json")
    ap.add_argument("--data", default="data/resume_real/train.jsonl",
                    help="BiasItem JSONL split to fit on (default: the 1733-pair train split)")
    ap.add_argument("--n-pairs", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="activation-caching batch size; drop to 1-2 on a 40GB card")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--tol", type=float, default=0.02, help="stop at chance + tol")
    ap.add_argument("--probe-C", type=float, default=None,
                    help="fixed inverse L2 strength; default None = select per fit by "
                         "grouped CV, which is what you want (a fixed C strong enough for "
                         "the un-erased representation collapses the erased one to "
                         "constant prediction and fakes successful erasure)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: <run-dir>/inlp.json")
    args = ap.parse_args(argv)

    import numpy as np
    from guardrail_ft.cli import make_base_loader
    from guardrail_ft.utils.config import load_yaml
    from guardrail_ft.tasks import get_task
    from guardrail_ft.identify.activations import cache_activations
    from guardrail_ft.identify.contrasts import demographic_contrast, bias_contrast
    from guardrail_ft.utils.seeding import seed_everything

    rd = Path(args.run_dir)
    cfg = load_yaml(str(rd / "config.resolved.yaml"))
    gp = rd / "guardrails" / "G_p"
    if not gp.exists():
        raise SystemExit(f"no saved G_p at {gp}")
    comp = json.loads((rd / "erosion_comparison.json").read_text())
    cand = comp["candidate"]
    lstar = int(cand["layers"][0])
    seed_everything(cfg.get("seed", 0), True)
    task = get_task(cfg["task"]["name"], cfg)
    ds = _load_split(args.data, task.name, max_pairs=args.n_pairs)
    print(f"[inlp] run={rd.name}  l*={lstar}  items={len(ds.items)}  "
          f"estimator={cand.get('estimator')}", flush=True)

    # ---- cache activations ONE MODEL AT A TIME (see _free) -------------------
    print("[inlp] caching B ...", flush=True)
    B = make_base_loader(cfg)()
    cB = cache_activations(B, ds, task, position="last", model_id="B",
                           batch_size=args.batch_size)
    _free(B)
    print("[inlp] caching M_b ...", flush=True)
    Gp = make_base_loader(cfg, init_checkpoint=str(gp))()
    cG = cache_activations(Gp, ds, task, position="last", model_id="G_p",
                           batch_size=args.batch_size)
    _free(Gp)

    # ---- reference directions at l* (models already freed; caches suffice) ---
    li = list(cG.layer_index).index(lstar)
    demo, _ = demographic_contrast(None, task, ds, cache=cG)
    bias, _, _ = bias_contrast(None, None, task, ds, cache_base=cB, cache_guard=cG)
    v_demo = np.asarray(demo.per_layer_direction[li], dtype=np.float64)
    v_bias = np.asarray(bias.per_layer_direction[li], dtype=np.float64)
    vb_hat, vd_hat = _unit(v_bias), _unit(v_demo)

    # ---- design matrix at l*, pair-disjoint split ---------------------------
    X = np.asarray(cG.activations[:, li, :], dtype=np.float64)
    groups = list(cG.groups)
    pair_ids = list(cG.contrast_pair_ids)
    y = np.array([1 if g == "white" else 0 for g in groups], dtype=int)
    uniq = sorted(set(pair_ids))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq)
    n_te = max(1, int(round(args.test_frac * len(uniq))))
    te_pairs = set(uniq[:n_te])
    is_te = np.array([p in te_pairs for p in pair_ids])
    Xtr, ytr, Xte, yte = X[~is_te], y[~is_te], X[is_te], y[is_te]
    ratio = X.shape[0] / X.shape[1]
    print(f"[inlp] X={X.shape}  n/hidden={ratio:.2f}  train={Xtr.shape[0]} "
          f"test={Xte.shape[0]} (pair-disjoint)", flush=True)
    if ratio < 1.0:
        print("[inlp] WARNING: n_items < hidden; in-sample separation is trivial. "
              "Held-out accuracy is the only interpretable number here.", flush=True)

    # ---- INLP ---------------------------------------------------------------
    grp_tr = np.array([p for p, t in zip(pair_ids, is_te) if not t])
    basis, traj, final_acc, Xtr_e, Xte_e = inlp(
        Xtr, ytr, Xte, yte, k_max=args.k_max, tol=args.tol,
        seed=args.seed, C=args.probe_C, groups_tr=grp_tr)
    k = len(basis)
    print(f"[inlp] k={k}  held-out acc per round: "
          f"{['%.3f' % a for a in traj]}  -> refit {final_acc:.3f}", flush=True)

    # ---- alignment with the directions we actually ablate -------------------
    cos_bias = [float(abs(np.dot(w, vb_hat))) for w in basis]
    cos_demo = [float(abs(np.dot(w, vd_hat))) for w in basis]
    # how much of v_bias lies inside the erased span (1.0 = fully removed)
    Q, _ = np.linalg.qr(basis.T)
    captured = float(np.linalg.norm(Q.T @ vb_hat))
    captured_demo = float(np.linalg.norm(Q.T @ vd_hat))
    fisher, dmeans = fisher_direction(X, y)
    print(f"[inlp] |cos(w_1, v_bias)|={cos_bias[0]:.3f}   "
          f"v_bias captured by span={captured:.3f}   "
          f"|cos(fisher, diff-of-means)|={abs(float(np.dot(fisher, dmeans))):.3f}", flush=True)

    # ---- rank-1 oracle: ONE fixed direction, the analogue of our weight ablation ----
    # k from INLP does not measure intrinsic rank (see oracle_step docstring), so this is
    # the comparison that actually bears on the rank-1 assumption. Every projected
    # direction is re-estimated on the TRAIN pairs only -- see contrast_dirs_at.
    tr_pairs = set(p for p, t in zip(pair_ids, is_te) if not t)
    vb_tr, vd_tr, n_tr_pairs = contrast_dirs_at(cB, cG, li, pair_ids, groups, tr_pairs)
    fisher_tr, dmeans_tr = fisher_direction(Xtr, ytr)
    acc_ceiling = baseline_probe(Xtr, ytr, Xte, yte, seed=args.seed,
                                 C=args.probe_C, groups_tr=grp_tr)
    _oa = lambda d: oracle_step(Xtr, ytr, Xte, yte, d, seed=args.seed,
                                C=args.probe_C, groups_tr=grp_tr)
    oracle = {"v_bias": _oa(vb_tr), "v_demo": _oa(vd_tr),
              "fisher": _oa(fisher_tr), "inlp_w1": _oa(basis[0])}
    print(f"[inlp] rank-1 oracle (train-fitted direction, held-out acc after ONE "
          f"projection; ceiling {acc_ceiling:.3f}, chance 0.500, "
          f"{n_tr_pairs} train pairs):", flush=True)
    for kk, vv in oracle.items():
        print(f"         {kk:8} -> {vv:.3f}   (|acc-0.5| = {abs(vv - 0.5):.3f})", flush=True)
    # cosine between the train-fitted and full-data directions: if these disagree, the
    # direction is not stably estimated and neither number should be over-read.
    cos_tr_full = float(abs(np.dot(_unit(vb_tr), vb_hat)))
    print(f"[inlp] cos(v_bias train-only, v_bias full)={cos_tr_full:.3f}", flush=True)

    # ---- presence before / after, same statistic as the weight-level methods -
    def presence(A):
        """||v_demo|| / E||h|| at l*, recomputed on the given activations.

        Denominator is the MEAN OF NORMS, not the norm of the mean -- matching
        measure_presence_anchors.py so the numbers are comparable.
        """
        by_pair = {}
        for i, (p, g) in enumerate(zip(pair_ids, groups)):
            by_pair.setdefault(p, {}).setdefault(g, i)
        pos, neg = [], []
        for p, bg in by_pair.items():
            if "white" in bg and "black" in bg:
                pos.append(A[bg["white"]]); neg.append(A[bg["black"]])
        if not pos:
            return None
        v = np.stack(pos).mean(0) - np.stack(neg).mean(0)
        return float(np.linalg.norm(v) / np.mean(np.linalg.norm(A, axis=1)))

    X_erased = X - (X @ basis.T) @ basis
    pres_before, pres_after = presence(X), presence(X_erased)
    print(f"[inlp] presence at l*: {pres_before:.4f} -> {pres_after:.4f}", flush=True)

    out = {
        "run_dir": str(rd), "layer": lstar, "data": args.data,
        "n_items": int(X.shape[0]), "hidden": int(X.shape[1]),
        "n_over_hidden": ratio,
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "split": "pair-disjoint by contrast_pair_id", "test_frac": args.test_frac,
        "probe_C": args.probe_C, "chance": 0.5,
        "k": k, "k_max": args.k_max, "tol": args.tol,
        "probe_acc_ceiling": acc_ceiling,
        "probe_acc_per_round": traj,
        "probe_acc_after_refit": final_acc,
        "rank1_oracle_acc": oracle,
        "rank1_oracle_train_pairs": n_tr_pairs,
        "cos_v_bias_train_vs_full": cos_tr_full,
        "rank1_oracle_note": "held-out probe accuracy after projecting out ONE fixed "
                             "direction, the direction re-estimated on TRAIN pairs only; "
                             "the activation-space analogue of the rank-1 weight ablation. "
                             "Compare to chance (0.5) and to the ceiling; read |acc-0.5| "
                             "as recoverability, since a direction fitted on the same data "
                             "being scored drives accuracy BELOW chance rather than to it. "
                             "k from INLP is NOT the intrinsic rank.",
        "cos_with_v_bias": cos_bias,
        "cos_with_v_demo": cos_demo,
        "v_bias_captured_by_span": captured,
        "v_demo_captured_by_span": captured_demo,
        "cos_fisher_vs_diff_of_means": abs(float(np.dot(fisher, dmeans))),
        "presence_before": pres_before, "presence_after": pres_after,
        "notes": "INLP on M_b activations at l*. Our weight ablation is ~one INLP step "
                 "with the direction fixed to the difference of means. Held-out accuracy "
                 "on a pair-disjoint split is the only interpretable probe number when "
                 "n_items < hidden. Fisher direction is the shrinkage-regularised LDA "
                 "optimum, related to but NOT an implementation of LEACE.",
    }
    outp = Path(args.out) if args.out else (rd / "inlp.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"[inlp] wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
