"""3D Thin-Plate-Spline warp, numpy only.

Smooth deformation that passes exactly through the landmark correspondences and
interpolates everywhere else. 3D biharmonic kernel phi(r) = r.
"""

import numpy as np


def _pairwise_dist(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    d = A[:, None, :] - B[None, :, :]
    return np.sqrt((d * d).sum(axis=-1))


def tps_fit(src, dst, reg=1e-6):
    """Fit TPS mapping src -> dst (both (n,3)). Returns an opaque model."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)

    K = _pairwise_dist(src, src)
    if reg:
        K = K + reg * np.eye(n)
    P = np.hstack([np.ones((n, 1)), src])           # (n,4)

    L = np.zeros((n + 4, n + 4))
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T

    Y = np.vstack([dst, np.zeros((4, 3))])           # (n+4,3)
    params = np.linalg.solve(L, Y)                   # (n+4,3)
    return (src, params)


def tps_apply(model, pts):
    src, params = model
    n = len(src)
    pts = np.asarray(pts, dtype=float)
    w = params[:n]                                   # (n,3)
    a = params[n:]                                   # (4,3)
    Kx = _pairwise_dist(pts, src)                    # (m,n)
    P = np.hstack([np.ones((len(pts), 1)), pts])     # (m,4)
    return Kx @ w + P @ a
