"""Rigid + uniform-scale alignment (Umeyama / Procrustes), numpy only."""

import numpy as np


def similarity_transform(src, dst):
    """Least-squares similarity (scale s, rotation R, translation t) mapping
    src -> dst. src, dst are (n,3). Returns (s, R, t)."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = src - mu_s
    D = dst - mu_d

    var_s = (S ** 2).sum() / n
    cov = (D.T @ S) / n

    U, Sig, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    W = np.diag([1.0, 1.0, d])
    R = U @ W @ Vt
    s = (Sig * np.array([1.0, 1.0, d])).sum() / var_s if var_s > 1e-12 else 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def apply_similarity(pts, s, R, t):
    pts = np.asarray(pts, dtype=float)
    return (s * (R @ pts.T)).T + t
