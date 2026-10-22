"""Full-vocabulary FP64 measurement; metrics do not grant release qualification."""

from __future__ import annotations

import numpy as np


def distribution_metrics(left, right, target, dtype):
    from mlx_one.ops.numerical_trace import compare_tensors

    a, b = np.asarray(left, np.float64).ravel(), np.asarray(right, np.float64).ravel()
    if a.shape != b.shape or not a.size or not 0 <= target < a.size:
        raise ValueError("aligned nonempty vocabulary and valid next-token target required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite logits")
    delta = b - a
    center = delta - (delta.max() + delta.min()) / 2
    e = float(np.max(np.abs(center)))

    def logp(x):
        x = x - x.max()
        return x - np.log(np.exp(x).sum())

    la, lb = logp(a), logp(b)
    pa, pb = np.exp(la), np.exp(lb)
    lm = np.logaddexp(la, lb) - np.log(2)
    # Stable descending order resolves exact ties by lowest token ID, matching
    # argmax rather than reversing an ascending sort's arbitrary tie order.
    ia, ib = np.argsort(-a, kind="stable"), np.argsort(-b, kind="stable")
    margin_a, margin_b = float(a[ia[0]] - a[ia[1]]), float(b[ib[0]] - b[ib[1]])
    n99 = min(a.size, int(np.searchsorted(np.cumsum(pa[ia]), 0.99)) + 1)
    head, tail = ia[:n99], ia[n99:]
    same = int(ia[0]) == int(ib[0])
    return {
        **compare_tensors(a, b, dtype=dtype),
        "max_centered_logit_error": e,
        "rms_logit_error": float(np.sqrt(np.mean(delta * delta))),
        "mean_centered_abs_error": float(np.abs(center).mean()),
        "kl_nats": max(0.0, float(np.sum(pa * (la - lb)))),
        "js_nats": max(0.0, float((np.sum(pa * (la - lm)) + np.sum(pb * (lb - lm))) / 2)),
        "total_variation": float(np.abs(pa - pb).sum() / 2),
        "top1_agreement": same,
        "top1_ids": [int(ia[0]), int(ib[0])],
        "top5_overlap": len(set(ia[:5]) & set(ib[:5])) / min(5, len(a)),
        "top20_overlap": len(set(ia[:20]) & set(ib[:20])) / min(20, len(a)),
        "comparator_top_two_margin": margin_a,
        "candidate_top_two_margin": margin_b,
        "high_margin_flip": not same and margin_a > 2 * e,
        "near_tie_flip": not same and margin_a <= 2 * e,
        "target_token": int(target),
        "nll": [-float(la[target]), -float(lb[target])],
        "nll_difference": float(la[target] - lb[target]),
        "mass_accounting": {
            "rule": (
                "smallest comparator top-probability set covering at least 99%; diagnostics only"
            ),
            "head_tokens": n99,
            "head_reference_mass": float(pa[head].sum()),
            "head_candidate_mass": float(pb[head].sum()),
            "tail_reference_mass": float(pa[tail].sum()),
            "tail_candidate_mass": float(pb[tail].sum()),
            "head_max_centered_error": float(np.abs(center[head]).max()),
            "tail_max_centered_error": float(np.abs(center[tail]).max()) if tail.size else 0.0,
            "tail_total_variation_contribution": float(np.abs(pa[tail] - pb[tail]).sum() / 2),
            "top20_max_centered_error": float(np.abs(center[ia[:20]]).max()),
        },
        "finite": True,
    }
