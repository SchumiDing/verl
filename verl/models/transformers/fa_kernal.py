"""
Flash Attention Forward Kernel Implementation using Triton
Optimized implementation based on the Flash Attention paper
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, M,  # For storing softmax statistics
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Attn_Out: tl.constexpr,
):
    """
    Flash Attention forward pass kernel
    
    This kernel implements the tiled Flash Attention algorithm with online softmax.
    It processes attention computation in blocks to reduce memory access and improve efficiency.
    
    Args:
        Q, K, V: Query, Key, Value matrices with shape [batch, n_heads, seq_len, d_model]
        sm_scale: Softmax scaling factor (typically 1/sqrt(d_model))
        L, M: Softmax statistics for numerical stability (denominator and max value)
        Out: Output matrix
        Attn_Out: Attention score matrix output with shape [batch, n_heads, seq_len, seq_len]
        stride_*: Strides for each dimension of Q, K, V, Out, Attn_Out tensors
        Z: Batch size
        H: Number of attention heads
        N_CTX: Sequence length (context size)
        BLOCK_M, BLOCK_N, BLOCK_DMODEL: Block sizes for tiling
    """
    # Get program IDs to identify which block this kernel instance processes
    start_m = tl.program_id(0)  # Block index in M dimension
    off_hz = tl.program_id(1)   # Combined batch and head index
    off_z = off_hz // H          # Batch index
    off_h = off_hz % H           # Head index
    
    # Calculate base offset for Q, K, V tensors
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    
    # Initialize block pointer for Q (Query)
    # Calculate base offset for Attn_Out tensor
    attn_offset = off_z.to(tl.int64) * stride_az + off_h.to(tl.int64) * stride_ah
    #Shape: [BLOCK_M, N_CTX]
    Attn_block_ptr = tl.make_block_ptr(
        base=Attn_Out + attn_offset,
        shape=(N_CTX, BLOCK_M),
        strides=(stride_am, stride_an),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    
    # Shape: [BLOCK_M, BLOCK_DMODEL]
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # Initialize block pointer for K (Key)
    # Shape: [BLOCK_DMODEL, BLOCK_N] (transposed for matrix multiplication)
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    
    # Initialize block pointer for V (Value)
    # Shape: [BLOCK_N, BLOCK_DMODEL]
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # Initialize block pointer for output
    # Shape: [BLOCK_M, BLOCK_DMODEL]
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # Initialize offset vectors for masking
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Initialize accumulators and statistics for online softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Max value for each row
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # Softmax denominator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)   # Output accumulator
    attn_score = tl.zeros([BLOCK_M, N_CTX], dtype=tl.float32) # Attention score matrix
    
    # Load Q block (loaded once for all K, V iterations)
    q = tl.load(Q_block_ptr)
    
    # Define iteration range for K and V blocks
    lo = 0
    hi = (start_m + 1) * BLOCK_M  # Causal mask: only attend to previous tokens
    
    # Loop over K and V blocks
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K and V blocks from HBM
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        
        # Compute attention scores: QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        
        # Apply causal mask (prevent attending to future tokens)
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk, float("-inf"))
        
        # Online softmax computation (numerically stable)
        # Step 1: Compute max of current block
        m_ij = tl.max(qk, 1)
        
        # Step 2: Update global max
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Step 3: Compute exp with numerical stability (using exp2 for efficiency)
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        
        # Step 4: Update softmax denominator
        l_i = l_i * alpha + tl.sum(p, 1)
        
        # Step 5: Rescale previous accumulator
        acc = acc * alpha[:, None]
        if start_n > 0:
            attn_score[:, :start_n] = attn_score[:, :start_n] * alpha[:, None]
        # Step 6: Accumulate current block's contribution
        acc += tl.dot(p.to(v.dtype), v)
        
        offs_n_local = start_n + offs_n
        mask = offs_m[:, None] >= offs_n_local[None, :]
        qk = tl.where(mask, qk, float("-inf"))
        
        p_final = tl.math.exp2(qk - m_i[:, None]) / l_i[:, None]
        attn_score[:, start_n:start_n + BLOCK_N] = p_final
        # Step 7: Update max value
        m_i = m_i_new
        
        # Advance K and V block pointers to next block
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # Final normalization by softmax denominator
    acc = acc / l_i[:, None]
    
    # Store softmax statistics (needed for backward pass)
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    
    # Store output to HBM
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))
    tl.store(Attn_block_ptr, attn_score.to(Out.dtype.element_ty))


def flash_attention_forward(q, k, v, causal=True, sm_scale=None):
    """
    Flash Attention forward pass - Python interface
    
    This function provides a PyTorch-friendly interface to the Triton Flash Attention kernel.
    It handles tensor allocation and kernel launch configuration.
    
    Args:
        q: Query tensor with shape [batch, n_heads, seq_len, d_model]
        k: Key tensor with shape [batch, n_heads, seq_len, d_model]
        v: Value tensor with shape [batch, n_heads, seq_len, d_model]
        causal: Whether to use causal masking (default: True)
        sm_scale: Softmax scaling factor, defaults to 1/sqrt(d_model)
    
    Returns:
        o: Output tensor with shape [batch, n_heads, seq_len, d_model]
        l: Softmax denominator (needed for backward pass)
        m: Max values (needed for backward pass)
    """
    # Validate input shapes
    assert q.shape == k.shape == v.shape, "Q, K, V must have the same shape"
    batch, n_heads, seq_len, d_model = q.shape
    
    # Set default scaling factor
    if sm_scale is None:
        sm_scale = 1.0 / (d_model ** 0.5)
    
    # Allocate output tensor
    o = torch.empty_like(q)
    
    # Allocate statistics tensors for backward pass
    l = torch.empty((batch * n_heads, seq_len), device=q.device, dtype=torch.float32)
    m = torch.empty((batch * n_heads, seq_len), device=q.device, dtype=torch.float32)
    
    # Configure block sizes for optimal performance
    BLOCK_M = 128  # Query block size
    BLOCK_N = 64   # Key/Value block size
    BLOCK_DMODEL = d_model
    
    # Calculate grid dimensions
    # Grid dim 0: number of blocks in sequence length dimension
    # Grid dim 1: number of (batch, head) combinations
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
    
    # Launch Triton kernel
    _fwd_kernel[grid](
        q, k, v, sm_scale,
        l, m,
        o,
        # Strides for Q
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # Strides for K
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # Strides for V
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Strides for O
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Dimensions
        batch, n_heads, seq_len,
        # Block sizes
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        # Kernel launch config
        num_warps=4,    # Number of warps per thread block
        num_stages=2,   # Number of pipeline stages
    )
    
    return o, l, m


# Test function
def test_flash_attention():
    """Test Flash Attention implementation against PyTorch reference"""
    torch.manual_seed(0)
    
    # Configure test parameters
    batch = 2
    n_heads = 8
    seq_len = 512
    d_model = 64
    
    # Create random input tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q = torch.randn(batch, n_heads, seq_len, d_model, device=device, dtype=torch.float16)
    k = torch.randn(batch, n_heads, seq_len, d_model, device=device, dtype=torch.float16)
    v = torch.randn(batch, n_heads, seq_len, d_model, device=device, dtype=torch.float16)
    
    # Run Flash Attention
    o_flash, l, m = flash_attention_forward(q, k, v)
    
    # Compute reference using standard PyTorch implementation
    sm_scale = 1.0 / (d_model ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    
    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Compute attention weights and output
    attn = torch.softmax(scores, dim=-1)
    o_ref = torch.matmul(attn, v)
    
    # Compare results
    print(f"Flash Attention output shape: {o_flash.shape}")
    print(f"Reference output shape: {o_ref.shape}")
    print(f"Max absolute error: {torch.max(torch.abs(o_flash - o_ref)).item()}")
    print(f"Mean absolute error: {torch.mean(torch.abs(o_flash - o_ref)).item()}")
    
    # Validate correctness
    assert torch.allclose(o_flash, o_ref, atol=1e-2, rtol=1e-2), "Output mismatch!"
    print("Test passed!")


if __name__ == "__main__":
    test_flash_attention()

