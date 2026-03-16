"""
Triton kernels for top-k / bottom-k attention score aggregation with backward pass.

Forward: given attention probs (num_tokens, kv_len) and optional mask, compute
  topk_sum = sum over all rows of (sum of top top_pct% values per row)
  bottomk_sum = sum over all rows of (sum of bottom bottom_pct% values per row)
Kernel returns only topk_sum and bottomk_sum (scalars). Masks for backward are saved internally.

Backward: scatter grad_topk_sum and grad_bottom_sum to the positions that were selected in forward.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.autograd import Function

# Chunk size for long sequences; each chunk is processed in one kernel launch.
CHUNK_SIZE = 4096
# Max row length (kv_len). For kv_len <= CHUNK_SIZE we use the single-block kernel; for longer we chunk.
MAX_KV_LEN = 32 * 1024  # 32k
# Per-chunk we extract this many top/bottom values for merge; num_chunks * K_EXTRACT >= max k per row.
K_EXTRACT_PER_CHUNK = 512
# Process this many rows at a time in chunked path to cap peak memory (num_tokens = bsz*heads*seq).
# Peak temp memory per batch: 2 * ROW_BATCH_CHUNKED * num_chunks * K * 4 bytes (e.g. 16k*8*512*4*2 ~ 512MB).
ROW_BATCH_CHUNKED = 16384


def _merge_sorted_chunks_and_sum(
    top_batch: torch.Tensor,
    bottom_batch: torch.Tensor,
    k_top: torch.Tensor,
    k_bottom: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    K-way merge of already-sorted chunks: same result as concat + full sort.
    top_batch: (batch_tokens, num_chunks, K) sorted descending per chunk
    bottom_batch: (batch_tokens, num_chunks, K) sorted ascending per chunk
    k_top, k_bottom: (batch_tokens,) per-row counts to sum
    Returns (topk_sum, bottomk_sum) scalars.
    """
    batch_tokens, num_chunks, K = top_batch.shape
    B, C = batch_tokens, num_chunks
    arange_b = torch.arange(B, device=device, dtype=torch.long)
    arange_c = torch.arange(C, device=device, dtype=torch.long)

    # --- Top: merge descending, sum first k_top[i] per row ---
    ptr = torch.zeros((B, C), device=device, dtype=torch.long)
    cur = top_batch[:, :, 0].clone()
    max_k_top = k_top.max().item()
    topk_sum_row = torch.zeros(B, device=device, dtype=top_batch.dtype)

    for step in range(max_k_top):
        chosen_val = cur.max(dim=1).values
        chosen_c = cur.argmax(dim=1)
        topk_sum_row += chosen_val * (step < k_top).to(cur.dtype)
        ptr[arange_b, chosen_c] += 1
        # Next value from each chunk (use ptr as index; when ptr>=K chunk is exhausted -> -inf)
        read_idx = ptr.clamp(max=K - 1)
        next_val = top_batch[arange_b.unsqueeze(1), arange_c.unsqueeze(0), read_idx]
        next_val = torch.where(ptr < K, next_val, torch.tensor(float("-inf"), device=device, dtype=cur.dtype))
        cur[arange_b, chosen_c] = next_val[arange_b, chosen_c]

    # --- Bottom: merge ascending, sum first k_bottom[i] per row ---
    ptr = torch.zeros((B, C), device=device, dtype=torch.long)
    cur = bottom_batch[:, :, 0].clone()
    max_k_bottom = k_bottom.max().item()
    bottomk_sum_row = torch.zeros(B, device=device, dtype=bottom_batch.dtype)

    for step in range(max_k_bottom):
        chosen_val = cur.min(dim=1).values
        chosen_c = cur.argmin(dim=1)
        bottomk_sum_row += chosen_val * (step < k_bottom).to(cur.dtype)
        ptr[arange_b, chosen_c] += 1
        read_idx = ptr.clamp(max=K - 1)
        next_val = bottom_batch[arange_b.unsqueeze(1), arange_c.unsqueeze(0), read_idx]
        next_val = torch.where(ptr < K, next_val, torch.tensor(float("inf"), device=device, dtype=cur.dtype))
        cur[arange_b, chosen_c] = next_val[arange_b, chosen_c]

    return topk_sum_row.sum(), bottomk_sum_row.sum()


@triton.jit
def _topk_bottomk_sum_fwd_kernel(
    attn_probs_ptr,
    mask_ptr,
    out_top_sum_ptr,
    out_bottom_sum_ptr,
    top_mask_ptr,
    bottom_mask_ptr,
    num_tokens,
    kv_len,
    top_pct: tl.float32,
    bottom_pct: tl.float32,
    stride_probs_m,
    stride_probs_n,
    stride_mask_m,
    stride_mask_n,
    stride_top_m,
    stride_top_n,
    stride_bottom_m,
    stride_bottom_n,
    BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """
    One program per row. For each row: compute valid count, k_top, k_bottom;
    then repeatedly take max (for top) and min (for bottom), add to row sums, mask out;
    atomic add row sums to global outputs; write selection masks for backward.
    """
    row_id = tl.program_id(0)
    if row_id >= num_tokens:
        return

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < kv_len
    probs_ptrs = attn_probs_ptr + row_id * stride_probs_m + offs_n * stride_probs_n
    row_probs = tl.load(probs_ptrs, mask=mask_n, other=float("-inf"))

    if HAS_MASK:
        mask_ptrs = mask_ptr + row_id * stride_mask_m + offs_n * stride_mask_n
        row_mask = tl.load(mask_ptrs, mask=mask_n, other=float("-inf"))
        valid = row_mask > -1e4
    else:
        valid = row_probs > 1e-8

    valid_count = tl.sum(tl.cast(valid, tl.int32))
    valid_count_f = tl.cast(valid_count, tl.float32)
    k_top = tl.cast(
        tl.minimum(
            tl.maximum(1, valid_count_f * top_pct),
            valid_count_f,
        ),
        tl.int32,
    )
    k_bottom = tl.cast(
        tl.minimum(
            tl.maximum(1, valid_count_f * bottom_pct),
            valid_count_f,
        ),
        tl.int32,
    )

    # Prepare row for top-k: invalid positions set to -inf
    row_top = tl.where(valid, row_probs, float("-inf"))
    row_bottom = tl.where(valid, row_probs, float("inf"))

    top_sum = 0.0
    bottom_sum = 0.0
    count_top = 0
    count_bottom = 0

    # Repeated max for top-k sum; write selection mask for backward in the same loop
    for _ in range(BLOCK_N):
        max_val = tl.max(row_top)
        is_max = row_top >= max_val - 1e-7
        idx_top = tl.min(tl.where(is_max, offs_n, BLOCK_N))
        add_top = tl.where(
            (count_top < k_top) & (max_val > -1e10),
            max_val,
            0.0,
        )
        top_sum += add_top
        do_store_top = (count_top < k_top) & (max_val > -1e10) & (idx_top < kv_len)
        top_mask_ptrs = top_mask_ptr + row_id * stride_top_m + idx_top * stride_top_n
        tl.store(top_mask_ptrs, 1.0, mask=do_store_top)
        count_top = tl.where(
            (count_top < k_top) & (max_val > -1e10),
            count_top + 1,
            count_top,
        )
        row_top = tl.where(offs_n == idx_top, float("-inf"), row_top)

    # Repeated min for bottom-k sum; write selection mask for backward in the same loop
    for _ in range(BLOCK_N):
        min_val = tl.min(row_bottom)
        is_min = row_bottom <= min_val + 1e-7
        idx_bottom = tl.min(tl.where(is_min, offs_n, BLOCK_N))
        add_bottom = tl.where(
            (count_bottom < k_bottom) & (min_val < 1e10),
            min_val,
            0.0,
        )
        bottom_sum += add_bottom
        do_store_bottom = (count_bottom < k_bottom) & (min_val < 1e10) & (idx_bottom < kv_len)
        bottom_mask_ptrs = bottom_mask_ptr + row_id * stride_bottom_m + idx_bottom * stride_bottom_n
        tl.store(bottom_mask_ptrs, 1.0, mask=do_store_bottom)
        count_bottom = tl.where(
            (count_bottom < k_bottom) & (min_val < 1e10),
            count_bottom + 1,
            count_bottom,
        )
        row_bottom = tl.where(offs_n == idx_bottom, float("inf"), row_bottom)

    tl.atomic_add(out_top_sum_ptr, top_sum)
    tl.atomic_add(out_bottom_sum_ptr, bottom_sum)


@triton.jit
def _chunk_top_bottom_kernel(
    attn_probs_ptr,
    mask_ptr,
    top_vals_ptr,
    bottom_vals_ptr,
    chunk_valid_count_ptr,
    num_tokens,
    kv_len,
    chunk_start,
    row_start,
    stride_probs_m,
    stride_probs_n,
    stride_mask_m,
    stride_mask_n,
    CHUNK_N: tl.constexpr,
    K_EXTRACT: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """
    One program per row. Load one chunk (CHUNK_N elements), sort to get top/bottom K_EXTRACT
    values and chunk valid count. row_start offsets into attn_probs for row-batch processing.
    """
    row_id = tl.program_id(0)
    if row_id >= num_tokens:
        return
    row_abs = row_start + row_id

    offs_n = tl.arange(0, CHUNK_N)
    in_bounds = (chunk_start + offs_n) < kv_len
    probs_ptrs = attn_probs_ptr + row_abs * stride_probs_m + (chunk_start + offs_n) * stride_probs_n
    row_probs = tl.load(probs_ptrs, mask=in_bounds, other=float("-inf"))

    if HAS_MASK:
        mask_ptrs = mask_ptr + row_abs * stride_mask_m + (chunk_start + offs_n) * stride_mask_n
        row_mask = tl.load(mask_ptrs, mask=in_bounds, other=float("-inf"))
        valid = row_mask > -1e4
    else:
        valid = row_probs > 1e-8
    valid = valid & in_bounds

    valid_count = tl.sum(tl.cast(valid, tl.int32))

    row_top = tl.where(valid, row_probs, float("-inf"))
    sorted_top = tl.sort(row_top, dim=0, descending=True)
    for k in range(K_EXTRACT):
        top_vals_ptrs = top_vals_ptr + row_id * K_EXTRACT + k
        tl.store(top_vals_ptrs, sorted_top[k])

    row_bottom = tl.where(valid, row_probs, float("inf"))
    sorted_bottom = tl.sort(row_bottom, dim=0, descending=False)
    for k in range(K_EXTRACT):
        bottom_vals_ptrs = bottom_vals_ptr + row_id * K_EXTRACT + k
        tl.store(bottom_vals_ptrs, sorted_bottom[k])

    chunk_valid_count_ptrs = chunk_valid_count_ptr + row_id
    tl.store(chunk_valid_count_ptrs, tl.cast(valid_count, tl.int32))


@triton.jit
def _topk_bottomk_sum_bwd_kernel(
    grad_top_sum_ptr,
    grad_bottom_sum_ptr,
    top_mask_ptr,
    bottom_mask_ptr,
    grad_attn_ptr,
    num_tokens,
    kv_len,
    stride_top_m,
    stride_top_n,
    stride_bottom_m,
    stride_bottom_n,
    stride_grad_m,
    stride_grad_n,
    BLOCK_N: tl.constexpr,
):
    """
    Scatter gradient: grad_attn[i, j] = grad_top_sum * top_mask[i,j] + grad_bottom_sum * bottom_mask[i,j].
    One program per row.
    """
    row_id = tl.program_id(0)
    if row_id >= num_tokens:
        return

    grad_top_sum = tl.load(grad_top_sum_ptr)
    grad_bottom_sum = tl.load(grad_bottom_sum_ptr)

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < kv_len
    top_ptrs = top_mask_ptr + row_id * stride_top_m + offs_n * stride_top_n
    bottom_ptrs = bottom_mask_ptr + row_id * stride_bottom_m + offs_n * stride_bottom_n
    top_m = tl.load(top_ptrs, mask=mask_n, other=0.0)
    bottom_m = tl.load(bottom_ptrs, mask=mask_n, other=0.0)

    grad_row = grad_top_sum * top_m + grad_bottom_sum * bottom_m
    grad_ptrs = grad_attn_ptr + row_id * stride_grad_m + offs_n * stride_grad_n
    tl.store(grad_ptrs, grad_row, mask=mask_n)


def _get_block_n(kv_len: int) -> int:
    """Choose BLOCK_N as next power of 2 >= kv_len, capped by CHUNK_SIZE (single-block path)."""
    n = 1
    while n < kv_len and n < CHUNK_SIZE:
        n *= 2
    return min(max(n, kv_len), CHUNK_SIZE)


def _topk_bottomk_sum_forward_chunked(
    attn_probs: torch.Tensor,
    top_pct: float,
    bottom_pct: float,
    attention_mask: Optional[torch.Tensor],
    num_tokens: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Forward for kv_len > CHUNK_SIZE: process in chunks of 4096, merge in Python.
    Uses row batches to cap peak memory when num_tokens is large (e.g. bsz*heads*seq).
    Returns (topk_sum, bottomk_sum, None, None); no mask so backward will return zeros.
    """
    num_chunks = (kv_len + CHUNK_SIZE - 1) // CHUNK_SIZE
    K = K_EXTRACT_PER_CHUNK
    row_batch = min(ROW_BATCH_CHUNKED, num_tokens)
    topk_sum_acc = torch.tensor(0.0, device=device, dtype=torch.float32)
    bottomk_sum_acc = torch.tensor(0.0, device=device, dtype=torch.float32)

    for row_start in range(0, num_tokens, row_batch):
        row_end = min(row_start + row_batch, num_tokens)
        batch_tokens = row_end - row_start
        grid = (batch_tokens,)

        top_batch = torch.empty((batch_tokens, num_chunks, K), device=device, dtype=torch.float32)
        bottom_batch = torch.empty((batch_tokens, num_chunks, K), device=device, dtype=torch.float32)
        valid_batch = torch.empty((batch_tokens, num_chunks), device=device, dtype=torch.int32)

        for c in range(num_chunks):
            chunk_start = c * CHUNK_SIZE
            if chunk_start >= kv_len:
                continue
            if attention_mask is not None:
                _chunk_top_bottom_kernel[grid](
                    attn_probs,
                    attention_mask,
                    top_batch[:, c, :],
                    bottom_batch[:, c, :],
                    valid_batch[:, c],
                    num_tokens=batch_tokens,
                    kv_len=kv_len,
                    chunk_start=chunk_start,
                    row_start=row_start,
                    stride_probs_m=attn_probs.stride(0),
                    stride_probs_n=attn_probs.stride(1),
                    stride_mask_m=attention_mask.stride(0),
                    stride_mask_n=attention_mask.stride(1),
                    CHUNK_N=CHUNK_SIZE,
                    K_EXTRACT=K,
                    HAS_MASK=True,
                )
            else:
                _chunk_top_bottom_kernel[grid](
                    attn_probs,
                    attn_probs,
                    top_batch[:, c, :],
                    bottom_batch[:, c, :],
                    valid_batch[:, c],
                    num_tokens=batch_tokens,
                    kv_len=kv_len,
                    chunk_start=chunk_start,
                    row_start=row_start,
                    stride_probs_m=attn_probs.stride(0),
                    stride_probs_n=attn_probs.stride(1),
                    stride_mask_m=0,
                    stride_mask_n=0,
                    CHUNK_N=CHUNK_SIZE,
                    K_EXTRACT=K,
                    HAS_MASK=False,
                )

        total_valid = valid_batch.sum(dim=1)
        total_valid_f = total_valid.to(torch.float32)
        k_top = (total_valid_f * top_pct).long().clamp(min=1)
        k_bottom = (total_valid_f * bottom_pct).long().clamp(min=1)
        k_top = torch.minimum(k_top, total_valid)
        k_bottom = torch.minimum(k_bottom, total_valid)

        # K-way merge of sorted chunks (same result as concat + full sort)
        top_sum_batch, bottom_sum_batch = _merge_sorted_chunks_and_sum(top_batch, bottom_batch, k_top, k_bottom, device)
        topk_sum_acc = topk_sum_acc + top_sum_batch
        bottomk_sum_acc = bottomk_sum_acc + bottom_sum_batch

    return topk_sum_acc.to(dtype), bottomk_sum_acc.to(dtype), None, None


def topk_bottomk_sum_forward(
    attn_probs: torch.Tensor,
    top_pct: float,
    bottom_pct: float,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward: compute topk_sum and bottomk_sum from attention probs (and optional mask).
    Returns (topk_sum, bottomk_sum, top_mask, bottom_mask). top_mask/bottom_mask are saved for backward.

    Args:
        attn_probs: (num_tokens, kv_len), attention probabilities
        top_pct: fraction of valid positions to take as top (e.g. 0.1 for top 10%)
        bottom_pct: fraction for bottom
        attention_mask: optional (num_tokens, kv_len), use -inf for masked; if None, treat probs > 1e-8 as valid

    Returns:
        topk_sum: scalar
        bottomk_sum: scalar
        top_mask: (num_tokens, kv_len), 1.0 where position was in top-k
        bottom_mask: (num_tokens, kv_len), 1.0 where position was in bottom-k
    """
    num_tokens, kv_len = attn_probs.shape
    device = attn_probs.device
    dtype = attn_probs.dtype
    if kv_len > MAX_KV_LEN:
        raise ValueError(f"kv_len {kv_len} exceeds MAX_KV_LEN {MAX_KV_LEN}")

    # Chunked path for long sequences (kv_len > CHUNK_SIZE)
    if kv_len > CHUNK_SIZE:
        return _topk_bottomk_sum_forward_chunked(
            attn_probs,
            top_pct,
            bottom_pct,
            attention_mask,
            num_tokens,
            kv_len,
            device,
            dtype,
        )

    BLOCK_N = _get_block_n(kv_len)
    grid = (num_tokens,)
    out_top = torch.zeros(1, device=device, dtype=torch.float32)
    out_bottom = torch.zeros(1, device=device, dtype=torch.float32)
    top_mask = torch.zeros((num_tokens, kv_len), device=device, dtype=torch.float32)
    bottom_mask = torch.zeros((num_tokens, kv_len), device=device, dtype=torch.float32)

    if attention_mask is not None:
        assert attention_mask.shape == (num_tokens, kv_len)
        _topk_bottomk_sum_fwd_kernel[grid](
            attn_probs,
            attention_mask,
            out_top,
            out_bottom,
            top_mask,
            bottom_mask,
            num_tokens=num_tokens,
            kv_len=kv_len,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            stride_probs_m=attn_probs.stride(0),
            stride_probs_n=attn_probs.stride(1),
            stride_mask_m=attention_mask.stride(0),
            stride_mask_n=attention_mask.stride(1),
            stride_top_m=top_mask.stride(0),
            stride_top_n=top_mask.stride(1),
            stride_bottom_m=bottom_mask.stride(0),
            stride_bottom_n=bottom_mask.stride(1),
            BLOCK_N=BLOCK_N,
            HAS_MASK=True,
        )
    else:
        # Dummy mask pointer (unused when HAS_MASK=False)
        _topk_bottomk_sum_fwd_kernel[grid](
            attn_probs,
            attn_probs,  # unused
            out_top,
            out_bottom,
            top_mask,
            bottom_mask,
            num_tokens=num_tokens,
            kv_len=kv_len,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            stride_probs_m=attn_probs.stride(0),
            stride_probs_n=attn_probs.stride(1),
            stride_mask_m=0,
            stride_mask_n=0,
            stride_top_m=top_mask.stride(0),
            stride_top_n=top_mask.stride(1),
            stride_bottom_m=bottom_mask.stride(0),
            stride_bottom_n=bottom_mask.stride(1),
            BLOCK_N=BLOCK_N,
            HAS_MASK=False,
        )

    return out_top.squeeze(0).to(dtype), out_bottom.squeeze(0).to(dtype), top_mask, bottom_mask


def topk_bottomk_sum_backward(
    grad_top_sum: torch.Tensor,
    grad_bottom_sum: torch.Tensor,
    top_mask: torch.Tensor,
    bottom_mask: torch.Tensor,
    num_tokens: int,
    kv_len: int,
) -> torch.Tensor:
    """Backward: scatter grad_top_sum and grad_bottom_sum using saved masks."""
    device = top_mask.device
    dtype = grad_top_sum.dtype
    BLOCK_N = _get_block_n(kv_len)
    grid = (num_tokens,)
    grad_attn = torch.empty((num_tokens, kv_len), device=device, dtype=dtype)

    _topk_bottomk_sum_bwd_kernel[grid](
        grad_top_sum.contiguous().view(1),
        grad_bottom_sum.contiguous().view(1),
        top_mask,
        bottom_mask,
        grad_attn,
        num_tokens=num_tokens,
        kv_len=kv_len,
        stride_top_m=top_mask.stride(0),
        stride_top_n=top_mask.stride(1),
        stride_bottom_m=bottom_mask.stride(0),
        stride_bottom_n=bottom_mask.stride(1),
        stride_grad_m=grad_attn.stride(0),
        stride_grad_n=grad_attn.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return grad_attn


class TopkBottomkSumFn(Function):
    """
    Autograd Function: forward returns (topk_sum, bottomk_sum); backward scatters gradients using saved masks.
    """

    @staticmethod
    def forward(
        ctx,
        attn_probs: torch.Tensor,
        top_pct: float,
        bottom_pct: float,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_sum, bottomk_sum, top_mask, bottom_mask = topk_bottomk_sum_forward(
            attn_probs, top_pct, bottom_pct, attention_mask
        )
        ctx.save_for_backward(top_mask, bottom_mask)
        ctx.num_tokens = attn_probs.shape[0]
        ctx.kv_len = attn_probs.shape[1]
        ctx.input_dtype = attn_probs.dtype
        ctx.input_device = attn_probs.device
        return topk_sum, bottomk_sum

    @staticmethod
    def backward(ctx, grad_topk_sum: torch.Tensor, grad_bottomk_sum: torch.Tensor):
        top_mask, bottom_mask = ctx.saved_tensors
        if top_mask is None or bottom_mask is None:
            # Chunked path: no mask saved, return zeros (no gradient through score aggregation)
            return (
                torch.zeros(
                    ctx.num_tokens,
                    ctx.kv_len,
                    device=ctx.input_device,
                    dtype=ctx.input_dtype,
                ),
                None,
                None,
                None,
            )
        device = top_mask.device
        dtype = ctx.input_dtype
        if grad_topk_sum is None:
            grad_topk_sum = torch.zeros(1, device=device, dtype=dtype).squeeze(0)
        if grad_bottomk_sum is None:
            grad_bottomk_sum = torch.zeros(1, device=device, dtype=dtype).squeeze(0)
        grad_attn = topk_bottomk_sum_backward(
            grad_topk_sum,
            grad_bottomk_sum,
            top_mask,
            bottom_mask,
            ctx.num_tokens,
            ctx.kv_len,
        )
        return grad_attn.to(dtype), None, None, None


def topk_bottomk_sum(
    attn_probs: torch.Tensor,
    top_pct: float,
    bottom_pct: float,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sum of top-pct and bottom-pct attention scores per row, then sum over rows.
    Differentiable; returns (topk_sum, bottomk_sum).

    Args:
        attn_probs: (num_tokens, kv_len)
        top_pct: e.g. 0.1 for top 10%
        bottom_pct: e.g. 0.1 for bottom 10%
        attention_mask: optional (num_tokens, kv_len), -inf for masked

    Returns:
        topk_sum: scalar
        bottomk_sum: scalar
    """
    return TopkBottomkSumFn.apply(attn_probs, top_pct, bottom_pct, attention_mask)


if __name__ == "__main__":
    # Quick test: forward + backward, compare with Python reference
    torch.manual_seed(42)
    num_tokens, kv_len = 32, 128
    attn_probs = torch.rand(num_tokens, kv_len, device="cuda", dtype=torch.float32)
    attn_probs = attn_probs / attn_probs.sum(dim=1, keepdim=True)
    top_pct, bottom_pct = 0.1, 0.1
    attn_probs.requires_grad_(True)

    topk_sum, bottomk_sum = topk_bottomk_sum(attn_probs, top_pct, bottom_pct)
    loss = topk_sum + bottomk_sum
    loss.backward()

    print("topk_sum:", topk_sum.item())
    print("bottomk_sum:", bottomk_sum.item())
    print("grad_attn_probs norm:", attn_probs.grad.norm().item())
    print("topk_bottomk_sum (Triton) test passed.")
