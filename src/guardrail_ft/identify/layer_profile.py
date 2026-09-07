"""Layer + variance profile of the identified bias axis (Noah's two asks).

Given the mean-of-differences bias contrast -- a per-pair difference-in-differences
matrix ``M[n_pairs, n_layers, hidden]`` and its per-layer mean ``d_ell`` -- this
answers two questions the single ``alignment`` scalar cannot:

1. WHERE does the bias live across layers?  We report the per-layer magnitude of
   the mean difference, ``||d_ell||``, both raw and RELATIVE to the residual-stream
   norm at that layer (raw magnitude grows with depth, so the relative curve is the
   honest "where does the signal concentrate" view). Peaks/plateaus/dips in this
   curve are exactly what to visualise.

2. Is the mean a CONSISTENT direction or an average of cancelling extremes?  For
   each layer we project every pair's difference onto the unit mean direction,
   ``s_i = <M[i], d_hat>``, and report:
     * projection mean / std / **coefficient of variation** cv = std/|mean|;
     * **per-pair cosine** cos(M[i], d_hat), mean +/- std;
     * sign consistency = fraction of pairs with s_i > 0.
   Consistent directional bias -> high mean cosine, high sign-consistency, low cv.
   Extremes averaging out -> mean can look moderate while cosine is ~0 and cv blows
   up.  Means alone hide this; these do not.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def layer_variance_profile(
    per_layer_direction,          # [n_layers, hidden]  mean diff d_ell
    unit_direction,               # [n_layers, hidden]  unit-normalised d_hat_ell
    diff_matrix,                  # [n_pairs, n_layers, hidden]  per-pair differences
    layer_index: List[int],
    chosen_idx: int,
    residual_norm: Optional[Any] = None,   # [n_layers] mean ||h_ell|| (depth normaliser)
) -> Dict[str, Any]:
    import numpy as np

    D = np.asarray(per_layer_direction, dtype=np.float64)     # [L,H]
    U = np.asarray(unit_direction, dtype=np.float64)          # [L,H]
    M = np.asarray(diff_matrix, dtype=np.float64)             # [n,L,H]
    n = M.shape[0]

    strength = np.linalg.norm(D, axis=1)                      # ||d_ell||
    if residual_norm is not None:
        rn = np.asarray(residual_norm, dtype=np.float64)
        strength_rel = strength / np.clip(rn, 1e-9, None)
    else:
        rn = np.full_like(strength, np.nan)
        strength_rel = strength / (strength.max() or 1.0)     # fallback: peak-normalised

    proj = np.einsum("nlh,lh->nl", M, U)                      # s_i,ell = <M[i], d_hat_ell>
    proj_mean = proj.mean(axis=0)
    proj_std = proj.std(axis=0, ddof=1) if n > 1 else np.zeros_like(proj_mean)
    cv = proj_std / np.clip(np.abs(proj_mean), 1e-9, None)
    sign_consistency = (proj > 0).mean(axis=0)

    Mn = M / np.clip(np.linalg.norm(M, axis=2, keepdims=True), 1e-12, None)
    pair_cos = np.einsum("nlh,lh->nl", Mn, U)                 # cos(M[i], d_hat_ell)
    pair_cos_mean = pair_cos.mean(axis=0)
    pair_cos_std = pair_cos.std(axis=0, ddof=1) if n > 1 else np.zeros_like(pair_cos_mean)

    ci = int(chosen_idx)
    return {
        "n_pairs": int(n),
        "layer_index": [int(x) for x in layer_index],
        # (1) layer magnitude profile
        "per_layer_strength": strength.tolist(),
        "per_layer_strength_rel": strength_rel.tolist(),
        "per_layer_residual_norm": rn.tolist(),
        # (2) per-layer consistency / variance
        "per_layer_proj_mean": proj_mean.tolist(),
        "per_layer_proj_cv": cv.tolist(),
        "per_layer_pair_cosine_mean": pair_cos_mean.tolist(),
        "per_layer_pair_cosine_std": pair_cos_std.tolist(),
        "per_layer_sign_consistency": sign_consistency.tolist(),
        # headline stats at the SELECTED layer
        "chosen_layer_stats": {
            "layer": int(layer_index[ci]),
            "strength": float(strength[ci]),
            "strength_rel": float(strength_rel[ci]),
            "projection_mean": float(proj_mean[ci]),
            "projection_std": float(proj_std[ci]),
            "projection_cv": float(cv[ci]),
            "pair_cosine_mean": float(pair_cos_mean[ci]),
            "pair_cosine_std": float(pair_cos_std[ci]),
            "sign_consistency": float(sign_consistency[ci]),
        },
    }
