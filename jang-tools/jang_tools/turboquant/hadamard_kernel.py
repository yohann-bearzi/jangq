"""
Metal kernel for fast randomized Hadamard transform.
Created by Jinho Jang (eric@jangq.ai)

Standard hadamard_rotate runs as a Python chain of MLX ops (12 iterations
for d=3072), giving ~100us of dispatch overhead. This kernel does the
entire butterfly in one Metal dispatch using threadgroup memory.

For non-power-of-2 dims, decomposes into power-of-2 blocks.
"""

import mlx.core as mx
import numpy as np


# Single-block butterfly kernel — works for power-of-2 dims up to threadgroup mem limit
# Each thread handles one element. Threadgroup synchronizes between butterfly stages.
_HADAMARD_BUTTERFLY_SOURCE = '''
    // In-threadgroup butterfly. Each thread handles `elems_per_thread` elements
    // so we can support d up to 4096 with a 1024-thread threadgroup.
    uint batch_idx = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint d = dim_arr[0];
    uint d_log = dim_arr[1];
    uint elems_per_thread = dim_arr[2];

    threadgroup float shmem[4096];

    // Load elems_per_thread elements per thread
    for (uint k = 0; k < elems_per_thread; k++) {
        uint i = tid * elems_per_thread + k;
        if (i < d) {
            shmem[i] = static_cast<float>(x[batch_idx * d + i]) * signs[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stage = 0; stage < d_log; stage++) {
        uint h = 1u << stage;
        uint two_h = 2u * h;

        // Each thread computes its own elements
        float local_results[8];  // max elems_per_thread = 8 (d=8192)
        for (uint k = 0; k < elems_per_thread; k++) {
            uint i = tid * elems_per_thread + k;
            if (i >= d) continue;
            uint block_start = (i / two_h) * two_h;
            uint pos = i - block_start;
            if (pos < h) {
                local_results[k] = shmem[block_start + pos] + shmem[block_start + pos + h];
            } else {
                local_results[k] = shmem[block_start + pos - h] - shmem[block_start + pos];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < elems_per_thread; k++) {
            uint i = tid * elems_per_thread + k;
            if (i < d) {
                shmem[i] = local_results[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write
    float norm_factor = 1.0f / sqrt(static_cast<float>(d));
    for (uint k = 0; k < elems_per_thread; k++) {
        uint i = tid * elems_per_thread + k;
        if (i < d) {
            out[batch_idx * d + i] = shmem[i] * norm_factor;
        }
    }
'''


# Multi-block kernel: processes a sum-of-pow2 dim in ONE dispatch.
# All threads cooperate on each block serially, staging butterfly writes
# through registers to stay lockstep through threadgroup barriers.
_HADAMARD_MULTIBLOCK_SOURCE = '''
    uint batch_idx = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint threads_per_tg = threads_per_threadgroup.x;

    uint total_d = meta[0];
    uint n_blocks = meta[1];

    // 8192 = max total_d we can hold in a single threadgroup's shmem
    // (32 KB / 4 bytes per float). Was 4096 → silently corrupted block 1
    // of any non-pow2 dim > 4096 (notably GLM-5.1 hidden=6144 = 4096+2048
    // via _decompose_into_pow2_blocks). Gate/up projections on every MoE
    // expert use in_features=6144, so this bug wrecked every TQ matmul
    // on GLM-5.1, JANGTQ_1L/2S, and any >4096-dim non-pow2 model.
    // Diagnosed by direct CPU-vs-Metal parity check: at dim=6144 the
    // Metal kernel's output has cos_sim=0.81 to the CPU reference and
    // norm ratio 0.81. Fixed by bumping shmem to 8192 floats (32 KB,
    // within Apple Silicon threadgroup memory limit).
    threadgroup float shmem[8192];

    for (uint i = tid; i < total_d; i += threads_per_tg) {
        shmem[i] = static_cast<float>(x[batch_idx * total_d + i]) * signs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint offset = 0;
    for (uint b = 0; b < n_blocks; b++) {
        uint d_b = meta[2u + b * 2u];
        uint log_b = meta[3u + b * 2u];

        uint ept = (d_b + threads_per_tg - 1u) / threads_per_tg;
        if (ept == 0u) ept = 1u;

        for (uint stage = 0; stage < log_b; stage++) {
            uint h = 1u << stage;
            uint two_h = 2u * h;

            float newv[8];
            for (uint k = 0; k < ept; k++) {
                uint i_local = tid * ept + k;
                if (i_local < d_b) {
                    uint block_start = (i_local / two_h) * two_h;
                    uint pos = i_local - block_start;
                    float a = shmem[offset + block_start + pos];
                    if (pos < h) {
                        newv[k] = a + shmem[offset + block_start + pos + h];
                    } else {
                        newv[k] = shmem[offset + block_start + pos - h] - a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < ept; k++) {
                uint i_local = tid * ept + k;
                if (i_local < d_b) {
                    shmem[offset + i_local] = newv[k];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float norm = 1.0f / sqrt(static_cast<float>(d_b));
        for (uint k = 0; k < ept; k++) {
            uint i_local = tid * ept + k;
            if (i_local < d_b) {
                out[batch_idx * total_d + offset + i_local] = shmem[offset + i_local] * norm;
            }
        }
        offset += d_b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
'''


# MiniMax decode hot path: the intermediate activation rotation is 1536 =
# 1024+512 and runs over K active experts. For <=1024 power-of-two blocks, the
# first five butterfly stages stay inside one SIMD group; using lane shuffles
# avoids the threadgroup-memory round trip and barriers for those stages.
_HADAMARD_SHUFFLE_LE1024_SOURCE = '''
    uint batch_idx = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;

    uint total_d = meta[0];
    uint n_blocks = meta[1];

    threadgroup float shmem[1024];

    uint offset = 0u;
    for (uint b = 0u; b < n_blocks; b++) {
        uint d_b = meta[2u + b * 2u];
        uint log_b = meta[3u + b * 2u];

        float v = 0.0f;
        if (tid < d_b) {
            v = static_cast<float>(x[batch_idx * total_d + offset + tid]) * signs[offset + tid];
        }

        uint lane = tid & 31u;
        uint simd_stages = log_b < 5u ? log_b : 5u;
        for (uint stage = 0u; stage < simd_stages; stage++) {
            uint h = 1u << stage;
            float other = simd_shuffle_xor(v, h);
            if ((lane & h) == 0u) { v = v + other; }
            else { v = other - v; }
        }

        for (uint stage = simd_stages; stage < log_b; stage++) {
            uint h = 1u << stage;
            if (tid < d_b) { shmem[tid] = v; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < d_b) {
                float other = shmem[tid ^ h];
                if ((tid & h) == 0u) { v = v + other; }
                else { v = other - v; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float norm = 1.0f / sqrt(static_cast<float>(d_b));
        if (tid < d_b) {
            out[batch_idx * total_d + offset + tid] = v * norm;
        }

        offset += d_b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
'''


_kernel_cache = {}


def _get_butterfly_kernel():
    if "butterfly" not in _kernel_cache:
        _kernel_cache["butterfly"] = mx.fast.metal_kernel(
            name="hadamard_butterfly",
            input_names=["x", "signs", "dim_arr"],
            output_names=["out"],
            source=_HADAMARD_BUTTERFLY_SOURCE,
        )
    return _kernel_cache["butterfly"]


def _get_multiblock_kernel():
    if "multiblock" not in _kernel_cache:
        _kernel_cache["multiblock"] = mx.fast.metal_kernel(
            name="hadamard_multiblock",
            input_names=["x", "signs", "meta"],
            output_names=["out"],
            source=_HADAMARD_MULTIBLOCK_SOURCE,
        )
    return _kernel_cache["multiblock"]


def _get_shuffle_le1024_kernel():
    if "shuffle_le1024" not in _kernel_cache:
        _kernel_cache["shuffle_le1024"] = mx.fast.metal_kernel(
            name="hadamard_shuffle_le1024",
            input_names=["x", "signs", "meta"],
            output_names=["out"],
            source=_HADAMARD_SHUFFLE_LE1024_SOURCE,
        )
    return _kernel_cache["shuffle_le1024"]


def _next_pow2_log(n):
    log = 0
    while (1 << log) < n:
        log += 1
    return log


def _decompose_pow2(dim):
    """Decompose dim into sum of distinct powers of 2."""
    blocks = []
    remaining = dim
    while remaining > 0:
        p = 1 << (remaining.bit_length() - 1)
        blocks.append(p)
        remaining -= p
    return blocks


MAX_THREADS = 1024
MAX_DIM = 4096  # threadgroup mem (16 KB float32) + 4 elems/thread


def _dispatch_block(kernel, block_x, block_signs, d, batch):
    """Dispatch the butterfly kernel for a single power-of-2 block."""
    d_log = _next_pow2_log(d)
    elems_per_thread = max(1, (d + MAX_THREADS - 1) // MAX_THREADS)
    n_threads = (d + elems_per_thread - 1) // elems_per_thread
    dim_arr = mx.array([d, d_log, elems_per_thread], dtype=mx.uint32)
    out = kernel(
        inputs=[block_x.astype(mx.float32), block_signs, dim_arr],
        output_shapes=[block_x.shape],
        output_dtypes=[mx.float32],
        grid=(n_threads, batch, 1),
        threadgroup=(n_threads, 1, 1),
    )
    return out[0]


def _hadamard_rotate_shuffle_le1024(x: mx.array, signs: mx.array) -> mx.array:
    """Hadamard rotate for decompositions whose blocks are all <=1024.

    This is mathematically the same block-wise randomized Hadamard transform as
    the regular multi-block Metal kernel. It replaces the first five stages of
    each block with SIMD-lane shuffles, which is the MiniMax intermediate
    rotation hot path (`1536 = 1024 + 512`, batch = active experts).
    """
    if x.ndim != 2:
        raise ValueError(f"expected rank-2 input, got shape={x.shape}")
    dim = x.shape[-1]
    blocks = _decompose_pow2(dim)
    if not blocks or any(d > 1024 for d in blocks):
        raise ValueError(f"shuffle path only supports <=1024 blocks, got dim={dim}, blocks={blocks}")

    meta_list = [dim, len(blocks)]
    for d in blocks:
        meta_list.append(d)
        meta_list.append(_next_pow2_log(d))
    meta = mx.array(meta_list, dtype=mx.uint32)
    tg_size = max(blocks)
    return _get_shuffle_le1024_kernel()(
        inputs=[x.astype(mx.float32), signs, meta],
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
        grid=(tg_size, x.shape[0], 1),
        threadgroup=(tg_size, 1, 1),
    )[0]


def hadamard_rotate_metal(x: mx.array, signs: mx.array) -> mx.array:
    """Fast Hadamard rotate using a single Metal dispatch per power-of-2 block.

    Supports d up to 4096 in a single dispatch (uses elems_per_thread loop).
    For non-power-of-2 or d>4096, decomposes into power-of-2 blocks.
    """
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    else:
        squeeze = False

    dim = x.shape[-1]
    blocks = _decompose_pow2(dim)
    kernel = _get_butterfly_kernel()
    batch = x.shape[0]

    if all(d <= 1024 for d in blocks):
        result = _hadamard_rotate_shuffle_le1024(x, signs)
    elif len(blocks) == 1 and blocks[0] <= MAX_DIM:
        result = _dispatch_block(kernel, x, signs, blocks[0], batch)
    elif all(d <= MAX_DIM for d in blocks):
        # Single-dispatch multi-block path (saves ~8us per non-pow2 rotation
        # by avoiding 2 butterfly launches + a concat).
        mb_kernel = _get_multiblock_kernel()
        meta_list = [dim, len(blocks)]
        for d in blocks:
            meta_list.append(d)
            meta_list.append(_next_pow2_log(d))
        meta = mx.array(meta_list, dtype=mx.uint32)
        tg_size = min(MAX_THREADS, max(blocks))
        result = mb_kernel(
            inputs=[x.astype(mx.float32), signs, meta],
            output_shapes=[x.shape],
            output_dtypes=[mx.float32],
            grid=(tg_size, batch, 1),
            threadgroup=(tg_size, 1, 1),
        )[0]
    else:
        parts = []
        offset = 0
        for d in blocks:
            block_x = x[..., offset:offset + d]
            block_signs = signs[offset:offset + d]
            if d <= MAX_DIM:
                parts.append(_dispatch_block(kernel, block_x, block_signs, d, batch))
            else:
                from .rotation import hadamard_rotate
                parts.append(hadamard_rotate(block_x, block_signs).astype(mx.float32))
            offset += d
        result = mx.concatenate(parts, axis=-1)

    if squeeze:
        return result.squeeze(0)
    return result
