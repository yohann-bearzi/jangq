"""Grouped TQ matmul via oMLX's steel-based Metal GEMM.

The per-row gather kernel re-reads an expert's whole packed matrix for every
token routed to it. mlx::steel's tiled GEMM amortizes that across a BM-row
block; oMLX ships a TurboQuant block loader for it. Measured 3.8x at N=4096
(22.3ms -> 5.9ms) on GLM-5.2 expert shapes, M3 Ultra.

Only used for the sorted (prefill) path; decode keeps the per-row kernel.
"""
from __future__ import annotations

import os
import numpy as np
import mlx.core as mx

_EXT = None
_TRIED = False


def _ext():
    global _EXT, _TRIED
    if not _TRIED:
        _TRIED = True
        if os.environ.get("JANGTQ_STEEL", "1") != "0":
            try:
                from omlx.custom_kernels.glm_moe_dsa import _ext as e
                if hasattr(e, "turboquant_gather_blocks"):
                    _EXT = e
            except Exception:
                _EXT = None
    return _EXT


def available() -> bool:
    return _ext() is not None


# (variant, BM) pairs, ordered by the batch size each suits best
_VARIANTS = {8: 0, 16: 1, 32: 2}


def _pick_bm(n_rows: int, n_experts: int) -> int:
    avg = max(1, n_rows // max(1, n_experts))
    if avg >= 24:
        return 32
    if avg >= 12:
        return 16
    return 8


_META_CACHE = {}


def _block_meta_cached(idx_flat, bm: int):
    """block_meta is identical for every projection of a layer (same routing)
    and every layer of a chunk (same sort), so build it once per (id, bm)."""
    key = (id(idx_flat), idx_flat.shape[0], bm)
    hit = _META_CACHE.get(key)
    if hit is not None:
        return hit
    mx.eval(idx_flat)
    val = _block_meta(np.array(idx_flat), bm)
    if len(_META_CACHE) > 64:
        _META_CACHE.clear()
    _META_CACHE[key] = val
    return val


def _block_meta(idx_np: np.ndarray, bm: int):
    """(row_start, expert, rows) triples over a sorted index vector."""
    experts, counts = np.unique(idx_np, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rows = []
    for e, s, c in zip(experts, starts, counts):
        for off in range(0, int(c), bm):
            rows.append((int(s) + off, int(e), min(bm, int(c) - off)))
    meta = np.asarray(rows, dtype=np.int32)
    return mx.array(meta), mx.array(np.asarray([len(rows)], dtype=np.int32))


def grouped_matmul(x_rot, packed, norms, codebook, idx_flat, bits):
    """x_rot (N, in_f), idx_flat (N,) sorted ascending -> (N, out_f) f32."""
    e = _ext()
    if e is None:
        return None
    n_experts = packed.shape[0]
    n_rows = x_rot.shape[0]
    bm = _pick_bm(n_rows, n_experts)
    meta, count = _block_meta_cached(idx_flat, bm)
    xin = x_rot if x_rot.dtype in (mx.float16, mx.bfloat16) else x_rot.astype(mx.float16)
    # The binding validates exact dtypes: uint32 weight, fp16 norms, fp32
    # codebook. Bundles loaded in bf16 mode carry bf16 norms, so cast (cheap:
    # one [n_experts, out_features] tensor, cached by MLX's graph).
    nrm = norms if norms.dtype == mx.float16 else norms.astype(mx.float16)
    cbk = codebook if codebook.dtype == mx.float32 else codebook.astype(mx.float32)
    out = e.turboquant_gather_blocks(
        xin, packed, nrm, cbk, meta, count,
        variant=_VARIANTS[bm], bits=int(bits),
    )
    return out.astype(mx.float32)
