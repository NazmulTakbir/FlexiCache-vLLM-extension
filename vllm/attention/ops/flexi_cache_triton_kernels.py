import triton
import triton.language as tl
import torch
import math

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

@triton.jit
def store_minmax_key_cache_for_prefill(
    # KV cache with layer dim first (matches decode kernel layout)
    key_cache,            # [NB, HS//x, BS, x]
    mm_key_cache,         # [L, NUM_MM_BLOCKS, MM_BLOCK_SIZE, 2, HEAD_SIZE]
    block_table,          # [B_total, KV_HEADS, MAX_BLKS]
    minmax_block_table,   # [B_total, KV_HEADS, MAX_MM_BLKS]
    layer_first_blk,      # [L] (block offsets into the flat key_cache pool)

    # strides
    stride_k_nb, stride_k_hx, stride_k_bs, stride_k_x,
    stride_bt_bs, stride_bt_ly, stride_bt_kh, stride_bt_bl,
    stride_mmb_bs, stride_mmb_ly, stride_mmb_kh, stride_mmb_bl,
    stride_mmkc_l, stride_mmkc_bl, stride_mmkc_bs, stride_mmkc_two, stride_mmkc_hs,

    # per-launch metadata
    selected_req_indices,  # [B_sel] int32
    prompt_lens,           # [B_sel] int32
    max_full_pages: tl.constexpr,

    # types/consts
    HEAD_SIZE: tl.constexpr, x: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, MINMAX_CACHE_BLOCK_SIZE: tl.constexpr,
):
    """
    Grid mapping:
      pid0 = 0 .. (B_sel * max_full_pages - 1)  → (req_idx, page)
      pid1 = layer_id
      pid2 = kv_head

    Each program computes MIN/MAX across one (layer, kv_head, page) for a given request.
    """
    pid0 = tl.program_id(0)
    n_ly = tl.program_id(1).to(tl.int64)
    kvh  = tl.program_id(2).to(tl.int64)

    # decode (req, page) from pid0
    req = (pid0 // max_full_pages).to(tl.int32)
    pg  = (pid0 %  max_full_pages).to(tl.int32)

    real_idx   = tl.load(selected_req_indices + req).to(tl.int64)
    prompt_len = tl.load(prompt_lens + req).to(tl.int64)
    num_full_pages = prompt_len // tl.full((), BLOCK_SIZE, dtype=tl.int64)

    # guard inactive programs
    if (pg.to(tl.int64) >= num_full_pages):
        return

    # ---- map logical page -> physical key block ----
    phys_blk = tl.load(
        block_table + n_ly * stride_bt_ly + real_idx * stride_bt_bs
        + kvh * stride_bt_kh + pg.to(tl.int64) * stride_bt_bl
    ).to(tl.int64)

    # ---- load key tile: [HS//x, BS, x] ----
    OFFS_HX = tl.arange(0, HEAD_SIZE // x).to(tl.int64)
    OFFS_BS = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    OFFS_X  = tl.arange(0, x).to(tl.int64)

    layer_base      = tl.load(layer_first_blk + n_ly).to(tl.int64)
    phys_blk_global = layer_base + phys_blk
    k_base          = key_cache + phys_blk_global * stride_k_nb
    k_ptr  = (
        k_base
        + OFFS_HX[:, None, None] * stride_k_hx
        + OFFS_BS[None, :, None] * stride_k_bs
        + OFFS_X [None, None, :] * stride_k_x
    )
    k_tile = tl.load(k_ptr)         # [HS//x, BS, x]
    k_min  = tl.min(k_tile, axis=1) # [HS//x, x]
    k_max  = tl.max(k_tile, axis=1) # [HS//x, x]

    # ---- map to minmax block (mm page = pg // MM_BLOCK_SIZE) ----
    mm_blk_sz   = tl.full((), MINMAX_CACHE_BLOCK_SIZE, dtype=tl.int64)
    mm_page_idx = pg.to(tl.int64) // mm_blk_sz
    mm_page_off = pg.to(tl.int64) %  mm_blk_sz

    mm_phys_blk = tl.load(
        minmax_block_table + n_ly * stride_mmb_ly + real_idx * stride_mmb_bs
        + kvh * stride_mmb_kh + mm_page_idx * stride_mmb_bl
    ).to(tl.int64)

    out_base = (
        mm_key_cache + n_ly * stride_mmkc_l
        + mm_phys_blk * stride_mmkc_bl + mm_page_off * stride_mmkc_bs
    )

    # linearize (HS//x, x) → head dimension
    HEAD_IDX = OFFS_HX[:, None] * x + OFFS_X[None, :]

    out_min = out_base + HEAD_IDX * stride_mmkc_hs
    out_max = out_min + stride_mmkc_two

    tl.store(out_min, k_min)
    tl.store(out_max, k_max)
    
@triton.jit
def store_minmax_key_cache_for_decode(
    key_cache,             # [NB, HS//x, BS, x]
    mm_key_cache,          # [L, NUM_MM_BLOCKS, MM_BLOCK_SIZE, 2, HEAD_SIZE]
    block_table,           # [B_total, KV_HEADS, MAX_BLKS]
    minmax_block_table,    # [B_total, KV_HEADS, MAX_MM_BLKS]
    layer_first_blk,       # [L] (block offsets into the flat key_cache pool)

    stride_k_nb, stride_k_hx, stride_k_bs, stride_k_x,
    stride_bt_bs, stride_bt_ly, stride_bt_kh, stride_bt_bl,
    stride_mmb_bs, stride_mmb_ly, stride_mmb_kh, stride_mmb_bl,
    stride_mmkc_l, stride_mmkc_bl, stride_mmkc_bs, stride_mmkc_two, stride_mmkc_hs,

    selected_req_indices,      # [B_sel]
    newly_full_logical_pages,  # [B_sel]

    HEAD_SIZE: tl.constexpr, x: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    MINMAX_CACHE_BLOCK_SIZE: tl.constexpr,
):
    b    = tl.program_id(0).to(tl.int64)
    n_ly = tl.program_id(1).to(tl.int64)
    kvh  = tl.program_id(2).to(tl.int64)

    real_idx = tl.load(selected_req_indices + b).to(tl.int64)
    pg       = tl.load(newly_full_logical_pages + b).to(tl.int64)

    phys_blk = tl.load(
        block_table + n_ly * stride_bt_ly + real_idx * stride_bt_bs
        + kvh * stride_bt_kh + pg * stride_bt_bl
    ).to(tl.int64)

    mm_blk_sz   = tl.full((), MINMAX_CACHE_BLOCK_SIZE, dtype=tl.int64)
    mm_page_idx = pg // mm_blk_sz
    mm_page_off = pg %  mm_blk_sz

    mm_phys_blk = tl.load(
        minmax_block_table + n_ly * stride_mmb_ly + real_idx * stride_mmb_bs
        + kvh * stride_mmb_kh + mm_page_idx * stride_mmb_bl
    ).to(tl.int64)

    OFFS_HX = tl.arange(0, HEAD_SIZE // x).to(tl.int64)
    OFFS_BS = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    OFFS_X  = tl.arange(0, x).to(tl.int64)

    layer_base      = tl.load(layer_first_blk + n_ly).to(tl.int64)
    phys_blk_global = layer_base + phys_blk
    k_base          = key_cache + phys_blk_global * stride_k_nb
    k_ptr  = (
        k_base + OFFS_HX[:, None, None] * stride_k_hx
        + OFFS_BS[None, :, None] * stride_k_bs + OFFS_X [None, None, :] * stride_k_x
    )
    k_tile = tl.load(k_ptr)           # [HS//x, BS, x], dtype == K dtype
    k_min  = tl.min(k_tile, axis=1)   # [HS//x, x]
    k_max  = tl.max(k_tile, axis=1)   # [HS//x, x]

    out_base = (
        mm_key_cache + n_ly * stride_mmkc_l
        + mm_phys_blk * stride_mmkc_bl + mm_page_off * stride_mmkc_bs
    )

    HEAD_IDX = OFFS_HX[:, None] * x + OFFS_X[None, :]

    out_min = out_base + HEAD_IDX * stride_mmkc_hs
    out_max = out_min + stride_mmkc_two

    tl.store(out_min, k_min)
    tl.store(out_max, k_max)

@triton.jit
def fwd_kernel_flexicache(
    Q, K, V,   # [NUM_TOKEN, NUM_KV_HEADS, HEAD_SIZE]
    K_cache,   # [TOTAL_BLOCKS, HEAD_SIZE/x, BLOCK_SIZE, x]
    V_cache,   # [TOTAL_BLOCKS, HEAD_SIZE,   BLOCK_SIZE]
    B_Loc,     # [NUM_SEQS, NUM_KV_HEADS, MAX_BLOCKS_PER_SEQ]
    Out,       # Output tensor

    # --- Scalars ---
    sm_scale, k_scale, v_scale,
    num_queries_per_kv: tl.constexpr,
    # --- Metadata Tensors ---
    B_Start_Loc,            # Start token index for each sequence in the batch
    B_Seqlen,               # Total length of each sequence (context + new)
    stride_b_loc_b, stride_b_loc_h, stride_b_loc_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_k_cache_bs, stride_k_cache_d, stride_k_cache_bl, stride_k_cache_x,
    stride_v_cache_bs, stride_v_cache_d, stride_v_cache_bl,
    # --- Compile-Time Constants ---
    x: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_PADDED: tl.constexpr,
    IN_PRECISION: tl.constexpr,
    SKIP_DECODE: tl.constexpr,
    num_unroll_cache: tl.constexpr,
    num_unroll_request: tl.constexpr,
):
    # ─────────── Program IDs & Batch Metadata ───────────
    cur_batch = tl.program_id(0)  # Current sequence in the batch
    cur_head = tl.program_id(1)  # Current Query head
    start_m = tl.program_id(2)  # Current block of queries

    # Map query head to its corresponding KV head
    cur_kv_head = cur_head // num_queries_per_kv

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    # ─────────── Query Loading ───────────
    block_start_loc = BLOCK_M * start_m
    offs_bs_n = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(
        tl.int1
    )

    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

    # compute query against context (no causal mask here)
    for start_n in tl.range(
        0, cur_batch_ctx_len, BLOCK_SIZE, loop_unroll_factor=num_unroll_cache
    ):
        start_n = tl.multiple_of(start_n, BLOCK_SIZE)
        bn = tl.load(
            B_Loc
            + cur_batch * stride_b_loc_b
            + cur_kv_head * stride_b_loc_h
            + (start_n // BLOCK_SIZE) * stride_b_loc_s
        )
        off_k = (
            bn[None, :] * stride_k_cache_bs
            + (offs_d[:, None] // x) * stride_k_cache_d
            + ((start_n + offs_bs_n[None, :]) % BLOCK_SIZE) * stride_k_cache_bl
            + (offs_d[:, None] % x) * stride_k_cache_x
        )
        off_v = (
            bn[:, None] * stride_v_cache_bs
            + offs_d[None, :] * stride_v_cache_d
            + offs_bs_n[:, None] * stride_v_cache_bl
        )
        if (
            start_n + BLOCK_SIZE > cur_batch_ctx_len
            or BLOCK_DMODEL != BLOCK_DMODEL_PADDED
        ):
            k = tl.load(
                K_cache + off_k,
                mask=dim_mask[:, None]
                & ((start_n + offs_bs_n[None, :]) < cur_batch_ctx_len),
                other=0.0,
            )
        else:
            k = tl.load(K_cache + off_k)

        qk = tl.zeros([BLOCK_M, BLOCK_SIZE], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk = tl.where(
            (start_n + offs_bs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf")
        )
        qk *= sm_scale

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        if (
            start_n + BLOCK_SIZE > cur_batch_ctx_len
            or BLOCK_DMODEL != BLOCK_DMODEL_PADDED
        ):
            v = tl.load(
                V_cache + off_v,
                mask=dim_mask[None, :]
                & ((start_n + offs_bs_n[:, None]) < cur_batch_ctx_len),
                other=0.0,
            )
        else:
            v = tl.load(V_cache + off_v)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)
    for start_n in tl.range(
        0,
        block_mask * (start_m + 1) * BLOCK_M,
        BLOCK_N,
        loop_unroll_factor=num_unroll_request,
    ):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=dim_mask[:, None]
            & ((start_n + offs_n[None, :]) < cur_batch_query_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=dim_mask[None, :]
            & ((start_n + offs_n[:, None]) < cur_batch_query_len),
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]

    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len)
    )
    return

@triton.jit
def kernel_paged_attention_2d_flexicache(
    output_ptr,         # [num_tokens, num_query_heads, head_size]
    block_scores_ptr,           # [num_layers, num_seqs, num_kv_heads, max_logical_blocks]
    query_ptr,          # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,      # [total_blocks, head_size // x, block_size, x]
    value_cache_ptr,    # [total_blocks, head_size, block_size]
    block_tables_ptr,   # [num_seqs, num_kv_heads, max_num_blocks_per_seq]
    filtered_block_tables_ptr,           # [num_layers, num_seqs, num_kv_heads, max_filtered_blks]
    filtered_block_count_ptr,            # [num_layers, num_seqs, num_kv_heads]
    filtered_logical_block_indices_ptr,  # [num_layers, num_seqs, num_kv_heads, max_filtered_blks]
    seq_lens_ptr,               # [num_seqs]
    scale,                     # float32
    layer_idx,                 # int

    num_query_heads: tl.constexpr,             # int
    num_queries_per_kv: tl.constexpr,          # int
    num_queries_per_kv_padded: tl.constexpr,   # int

    block_table_stride_0: tl.constexpr,                  # int
    block_table_stride_1: tl.constexpr,                  # int
    block_table_stride_2: tl.constexpr,                  # int
    filtered_block_table_stride_0: tl.constexpr,         # int
    filtered_block_table_stride_1: tl.constexpr,         # int
    filtered_block_table_stride_2: tl.constexpr,         # int
    filtered_block_table_stride_3: tl.constexpr,         # int
    filtered_block_count_stride_0: tl.constexpr,  # int
    filtered_block_count_stride_1: tl.constexpr,  # int
    filtered_block_count_stride_2: tl.constexpr,  # int
    filtered_logical_block_indices_stride_0: tl.constexpr,  # int
    filtered_logical_block_indices_stride_1: tl.constexpr,  # int
    filtered_logical_block_indices_stride_2: tl.constexpr,  # int
    filtered_logical_block_indices_stride_3: tl.constexpr,  # int
    query_stride_0: tl.constexpr,              # int
    query_stride_1: tl.constexpr,              # int
    output_stride_0: tl.constexpr,             # int
    output_stride_1: tl.constexpr,             # int
    block_scores_stride_0: tl.constexpr,       # int
    block_scores_stride_1: tl.constexpr,       # int
    block_scores_stride_2: tl.constexpr,       # int
    block_scores_stride_3: tl.constexpr,       # int
    BLOCK_SIZE: tl.constexpr,                  # int
    HEAD_SIZE: tl.constexpr,                   # int
    HEAD_SIZE_PADDED: tl.constexpr,            # int, must be power of 2
    x: tl.constexpr,                           # int
    stride_k_cache_0: tl.constexpr,            # int
    stride_k_cache_1: tl.constexpr,            # int
    stride_k_cache_2: tl.constexpr,            # int
    stride_k_cache_3: tl.constexpr,            # int
    stride_v_cache_0: tl.constexpr,            # int
    stride_v_cache_1: tl.constexpr,            # int
    stride_v_cache_2: tl.constexpr,            # int
    filter_by_query_len: tl.constexpr,         # bool
    query_start_len_ptr,                       # [num_seqs+1]
):
    tl.device_assert(False, "should not reach here")
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded)
    query_offset = (cur_batch_in_all_start_index * query_stride_0 +
                    query_head_idx[:, None] * query_stride_1)

    head_mask = (query_head_idx < (kv_head_idx + 1) * num_queries_per_kv) \
                & (query_head_idx < num_query_heads)
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                        0).to(tl.int1)

    # Q : (num_queries_per_kv, HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    # --- Online Softmax Initialization ---
    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.full([num_queries_per_kv_padded], 1.0, dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED],
                   dtype=tl.float32)

    # sequence len for this particular sequence
    is_full_kv_mode = tl.load(is_full_kv_mode_ptr + seq_idx)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    
    filtered_block_tables_ptr_for_seq_head = (
        filtered_block_tables_ptr +
        layer_idx * filtered_block_table_stride_0 +    # Offset to the current layer
        seq_idx * filtered_block_table_stride_1 +      # Offset to the current sequence
        kv_head_idx * filtered_block_table_stride_2    # Offset to the current head
    )

    filtered_logical_block_indices_ptr_for_seq_head = (
        filtered_logical_block_indices_ptr +
        layer_idx * filtered_logical_block_indices_stride_0 +
        seq_idx * filtered_logical_block_indices_stride_1 +
        kv_head_idx * filtered_logical_block_indices_stride_2
    )

    current_head_filtered_block_count = tl.load(
        filtered_block_count_ptr +
        layer_idx * filtered_block_count_stride_0 +
        seq_idx * filtered_block_count_stride_1 +
        kv_head_idx * filtered_block_count_stride_2
    )

    for j in range(0, current_head_filtered_block_count):
        physical_block_idx = tl.load(filtered_block_tables_ptr_for_seq_head + j * filtered_block_table_stride_3).to(tl.int64)
        original_logical_block_idx = tl.load(filtered_logical_block_indices_ptr_for_seq_head + j * filtered_logical_block_indices_stride_3).to(tl.int64)
        # physical_block_idx = j.to(tl.int64)

        offs_n = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)

        v_offset = (physical_block_idx * stride_v_cache_0 +
                    offs_d[None, :] * stride_v_cache_1 +
                    offs_n[:, None] * stride_v_cache_2)

        k_offset = (physical_block_idx * stride_k_cache_0 +
                    (offs_d[:, None] // x) * stride_k_cache_1 +
                    offs_n[None, :] * stride_k_cache_2 +
                    (offs_d[:, None] % x) * stride_k_cache_3)

        K = tl.load(key_cache_ptr + k_offset,
                    mask=dim_mask[:, None],
                    other=0.0)

        V = tl.load(value_cache_ptr + v_offset,
                    mask=dim_mask[None, :],
                    other=0.0)

        seq_offset = original_logical_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary

        S = tl.where(head_mask[:, None] & seq_mask, 0.0,
                    float("-inf")).to(tl.float32)
        S += scale * tl.dot(Q, K)

        m_j = tl.maximum(M, tl.max(S, axis=1))
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(P.to(V.dtype), V)

    acc = acc / L[:, None]

    output_offset = (cur_batch_in_all_start_index * output_stride_0 +
                    query_head_idx * output_stride_1)

    tl.store(
        output_ptr + output_offset[:, None] +
        tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )

# @triton.autotune(
#     configs=[
#         triton.Config({'num_warps': 4, 'num_stages': 2},   num_warps=4,  num_stages=2),
#         triton.Config({'num_warps': 8, 'num_stages': 2},   num_warps=8,  num_stages=2),
#         triton.Config({'num_warps': 8, 'num_stages': 4},   num_warps=8,  num_stages=4),
#         triton.Config({'num_warps': 16,'num_stages': 2},  num_warps=16, num_stages=2),
#     ],
#     key=[]                # <— required, even if empty
# )
@triton.jit
def kernel_paged_attention_3d_flexicache(
    segm_output_ptr,            # [num_seqs, num_q_heads, num_segments, H_PAD]
    segm_max_ptr,               # [num_seqs, num_q_heads, num_segments]
    segm_expsum_ptr,            # [num_seqs, num_q_heads, num_segments]

    query_ptr,                  # [num_tokens, num_q_heads, H]
    key_cache_ptr,              # [total_blocks, H/x, Blk, x]
    value_cache_ptr,            # [total_blocks, H,   Blk]
    block_table,                # [num_seqs, num_kv_heads, max_num_blocks_per_seq_per_head]
    top_k_blocks,               # [num_seqs, num_kv_heads, k]
    seq_lens_ptr,               # [num_seqs]
    last_ranked_num_blks,       # [num_seqs]
    unstable_mask_gpu,

    sm_scale,               # fp32
    num_seqs,               # int
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    num_queries_per_kv_padded: tl.constexpr,
    max_filtered_blocks: tl.constexpr,

    # --- Strides ---
    block_table_stride_0: tl.constexpr,    # Block table stride for sequences
    block_table_stride_1: tl.constexpr,    # Block table stride for heads
    block_table_stride_2: tl.constexpr,    # Block table stride for blocks
    top_k_stride_0: tl.constexpr,          # Top-k blocks stride for sequences
    top_k_stride_1: tl.constexpr,          # Top-k blocks stride for heads
    top_k_stride_2: tl.constexpr,          # Top-k blocks stride for k

    q_stride_0: tl.constexpr,    # Query stride for tokens
    q_stride_1: tl.constexpr,    # Query stride for heads

    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,

    x: tl.constexpr,             # Key cache vectorization factor
    kc_stride_0: tl.constexpr,   # Key cache stride for blocks
    kc_stride_1: tl.constexpr,   # Key cache stride for H/x
    kc_stride_2: tl.constexpr,   # Key cache stride for Blk
    kc_stride_3: tl.constexpr,   # Key cache stride for x

    vc_stride_0: tl.constexpr,   # Value cache stride for blocks
    vc_stride_1: tl.constexpr,   # Value cache stride for H
    vc_stride_2: tl.constexpr,   # Value cache stride for Blk

    filter_by_query_len: tl.constexpr,
    query_start_len_ptr,
    MIN_NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    MAX_NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    if filter_by_query_len:
        q_start = tl.load(query_start_len_ptr + seq_idx)
        q_end = tl.load(query_start_len_ptr + seq_idx + 1)
        if (q_end - q_start) > 1:
            return
    else:
        q_start = seq_idx

    seq_len      = tl.load(seq_lens_ptr + seq_idx)
    num_segments = (MAX_NUM_SEGMENTS_PER_SEQ if num_seqs <= 64 else MIN_NUM_SEGMENTS_PER_SEQ)

    total_block_count = cdiv_fn(seq_len, BLOCK_SIZE)

    unstable_mask  = tl.load(unstable_mask_gpu).to(tl.uint32)
    is_stable_head = (((unstable_mask >> kv_head_idx) & 1) == 0).to(tl.int1)

    prev_ranked_blk = tl.load(
        last_ranked_num_blks + seq_idx, mask=is_stable_head, other=0
    )
    tail_new_blocks = (total_block_count - prev_ranked_blk) * is_stable_head.to(tl.int32)

    filtered_block_count = \
        tail_new_blocks + tl.minimum(max_filtered_blocks, total_block_count - tail_new_blocks)
    tl.device_assert(filtered_block_count <= total_block_count)

    blocks_per_segment = cdiv_fn(filtered_block_count, num_segments)

    start_block_idx = segm_idx * blocks_per_segment
    end_block_idx   = tl.minimum((segm_idx + 1) * blocks_per_segment, filtered_block_count)
    if start_block_idx >= end_block_idx:
        return

    q_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(0, num_queries_per_kv_padded)
    q_offset = q_start * q_stride_0 + q_head_idx[:, None] * q_stride_1
    head_mask = (q_head_idx < (kv_head_idx + 1) * num_queries_per_kv) & (q_head_idx < num_query_heads)
    dim_mask = (tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE).to(tl.int1)

    Q = tl.load(query_ptr + q_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=head_mask[:, None] & dim_mask[None, :], other=0.0)

    M = tl.full([num_queries_per_kv_padded], -float("inf"), dtype=tl.float32)
    L = tl.full([num_queries_per_kv_padded], 1.0, dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    block_table_for_seq_head = \
        block_table + seq_idx * block_table_stride_0 + kv_head_idx * block_table_stride_1
    top_k_ptr_for_seq_head = \
        top_k_blocks + seq_idx * top_k_stride_0 + kv_head_idx * top_k_stride_1

    for filtered_block_idx in range(start_block_idx, end_block_idx):
        # process tail first
        is_tail = filtered_block_idx < tail_new_blocks

        original_logical_block_idx = tl.where(
            is_tail,
            prev_ranked_blk + filtered_block_idx,
            tl.load(top_k_ptr_for_seq_head + (filtered_block_idx - tail_new_blocks) * top_k_stride_2).to(tl.int64),
        )

        physical_block_idx = tl.load(block_table_for_seq_head + original_logical_block_idx * block_table_stride_2).to(tl.int64)
        tl.device_assert(physical_block_idx >= 0)

        offs_n = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)

        k_off = (physical_block_idx * kc_stride_0 + (offs_d[:, None] // x) * kc_stride_1 +
            offs_n[None, :] * kc_stride_2 + (offs_d[:, None] % x) * kc_stride_3)
        v_off = (physical_block_idx * vc_stride_0 + offs_d[None, :] * vc_stride_1 +
            offs_n[:, None] * vc_stride_2)

        K = tl.load(key_cache_ptr + k_off, mask=dim_mask[:, None], other=0.0)
        V = tl.load(value_cache_ptr + v_off, mask=dim_mask[None, :], other=0.0)

        seq_offset = original_logical_block_idx * BLOCK_SIZE + offs_n
        valid_mask = seq_offset[None, :] < seq_len
        S = tl.where(head_mask[:, None] & valid_mask, 0.0, -float("inf")).to(tl.float32)
        S += sm_scale * tl.dot(Q, K)

        m_j = tl.maximum(M, tl.max(S, axis=1))
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)

        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(P.to(V.dtype), V)

    segm_out_off = (seq_idx * (num_query_heads * MAX_NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        q_head_idx[:, None] * (MAX_NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
        segm_idx * HEAD_SIZE_PADDED + tl.arange(0, HEAD_SIZE_PADDED)[None, :])
    tl.store(segm_output_ptr + segm_out_off, acc, mask=head_mask[:, None] & dim_mask[None, :])

    segm_idx_flat = (seq_idx * (num_query_heads * MAX_NUM_SEGMENTS_PER_SEQ) +
        q_head_idx * MAX_NUM_SEGMENTS_PER_SEQ + segm_idx)
    tl.store(segm_max_ptr + segm_idx_flat, M, mask=head_mask)
    tl.store(segm_expsum_ptr + segm_idx_flat, L, mask=head_mask)

# @triton.autotune(
#     configs=[
#         triton.Config({'num_warps': 4, 'num_stages': 2},   num_warps=4,  num_stages=2),
#         triton.Config({'num_warps': 8, 'num_stages': 2},   num_warps=8,  num_stages=2),
#         triton.Config({'num_warps': 8, 'num_stages': 4},   num_warps=8,  num_stages=4),
#         triton.Config({'num_warps': 16,'num_stages': 2},  num_warps=16, num_stages=2),
#     ],
#     key=[]                # <— required, even if empty
# )
@triton.jit
def kernel_reduce_segments_flexicache(
    output_ptr,       # [num_seqs, num_query_heads, head_size]
    seq_lens_ptr,     # [num_seqs]
    last_ranked_num_blks, # [num_seqs]
    unstable_mask_gpu,
    segm_output_ptr,  # [num_seqs, num_query_heads, max_num_segments, head_size_padded]
    segm_max_ptr,     # [num_seqs, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_seqs, num_query_heads, max_num_segments]
    num_seqs,                       # int
    max_filtered_blocks: tl.constexpr,  # int
    BLOCK_SIZE: tl.constexpr,
    num_query_heads: tl.constexpr,  # int
    output_stride_0: tl.constexpr,
    output_stride_1: tl.constexpr,
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    NUM_QUERIES_PER_KV: tl.constexpr,  # int
    filter_by_query_len: tl.constexpr,  # bool
    query_start_len_ptr,  # [num_seqs+1]
    MIN_NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    MAX_NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    kv_head_idx = query_head_idx // NUM_QUERIES_PER_KV

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    num_segments = (MAX_NUM_SEGMENTS_PER_SEQ if num_seqs <= 64 else MIN_NUM_SEGMENTS_PER_SEQ)

    seq_len              = tl.load(seq_lens_ptr + seq_idx)
    total_block_count    = cdiv_fn(seq_len, BLOCK_SIZE)

    unstable_mask  = tl.load(unstable_mask_gpu).to(tl.uint32)
    is_stable_head = (((unstable_mask >> kv_head_idx) & 1) == 0).to(tl.int1)

    prev_ranked_blk = tl.load(
        last_ranked_num_blks + seq_idx, mask=is_stable_head, other=0
    )

    tail_new_blocks      = (total_block_count - prev_ranked_blk) * is_stable_head.to(tl.int32)
    filtered_block_count = \
        tail_new_blocks + tl.minimum(max_filtered_blocks, total_block_count - tail_new_blocks)
    tl.device_assert(filtered_block_count <= total_block_count)
    
    blocks_per_segment = cdiv_fn(filtered_block_count, num_segments)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(filtered_block_count, blocks_per_segment)
    segm_mask = tl.arange(0, MAX_NUM_SEGMENTS_PER_SEQ) < tl.full(
        [MAX_NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32)
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                        0).to(tl.int1)

    # load segment maxima
    segm_offset = (seq_idx * (num_query_heads * MAX_NUM_SEGMENTS_PER_SEQ) +
                query_head_idx * MAX_NUM_SEGMENTS_PER_SEQ +
                tl.arange(0, MAX_NUM_SEGMENTS_PER_SEQ))
    segm_max = tl.load(segm_max_ptr + segm_offset,
                    mask=segm_mask,
                    other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset,
                        mask=segm_mask,
                        other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        seq_idx * (num_query_heads * MAX_NUM_SEGMENTS_PER_SEQ \
            * HEAD_SIZE_PADDED)
        + query_head_idx * (MAX_NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, MAX_NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc = tl.sum(segm_output, axis=0) / overall_expsum

    # write result
    output_offset = (cur_batch_in_all_start_index * output_stride_0 +
                    query_head_idx * output_stride_1 +
                    tl.arange(0, HEAD_SIZE_PADDED))
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)

@triton.jit
def kernel_compute_block_scores_minmax(
    # --- outputs / inputs ---
    block_scores_ptr,                    # [B, KV, MAX_BLK] (bf16/fp16 dst)
    query_ptr,                           # [num_tokens, num_q_heads, H]
    minmax_kv_cache_ptr,                 # [NUM_MM_BLKS, MM_BLOCK_SIZE, 2, H]
    minmax_block_table_ptr,              # [B, KV, MAX_MM_BLKS] (int32)
    seq_lens_ptr,                        # [B] (int32)
    unstable_mask_gpu,                   # [1]
    num_decode_steps,                    # [B]
    rank_frequency: tl.constexpr,                      # int32

    # --- strides / layout (all in elements) ---
    stride_bs_batch, stride_bs_head,     # for block_scores
    q_stride_0, q_stride_1,              # for query
    stride_mmkc_bl, stride_mmkc_bs,      # for minmax cache
    stride_mmkc_two, stride_mmkc_hs,
    stride_mmb_bs, stride_mmb_kh, stride_mmb_bl,  # for minmax block-table

    # --- decode gating (optional) ---
    filter_by_query_len: tl.constexpr,
    query_start_len_ptr,                 # [B+1]

    # --- meta ---
    queries_per_kv : tl.constexpr,
    BLOCK_SIZE : tl.constexpr,
    HEAD_SIZE : tl.constexpr,
    HEAD_SIZE_PAD : tl.constexpr,
    MINMAX_CACHE_BLOCK_SIZE: tl.constexpr,
    PAGES_PER_TB : tl.constexpr,         # set to 16
):
    # constants
    FP32_MAX = 3.402823e38

    # program ids
    b  = tl.program_id(0)
    kh = tl.program_id(1)
    bz = tl.program_id(2)  # tiles pages

    num_decode_step = tl.load(num_decode_steps + b).to(tl.int32)
    if num_decode_step <= 0: # not in decode phase
        return
    
    unstable_mask    = tl.load(unstable_mask_gpu).to(tl.uint32)
    is_unstable_head = (((unstable_mask >> kh) & 1) != 0).to(tl.int1)
    is_rerank_step   = ((num_decode_step == 1) | ((num_decode_step % rank_frequency) == 0)).to(tl.int1)
    do_rank          = (is_unstable_head | is_rerank_step).to(tl.int1)
    if do_rank == 0:
        return

    if filter_by_query_len:
        q_start = tl.load(query_start_len_ptr + b)
        q_end   = tl.load(query_start_len_ptr + b + 1)
        # Only run when a single new token is being decoded
        if (q_end - q_start) > 1:
            return
    else:
        q_start = b

    # sequence/page math
    seq_len  = tl.load(seq_lens_ptr + b)
    num_blk  = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    last_blk = num_blk - 1

    # base page index for this TB
    blk_base = bz * PAGES_PER_TB

    # ---- load queries once per TB ----
    qh = kh * queries_per_kv + tl.arange(0, queries_per_kv)
    dim_mask = tl.arange(0, HEAD_SIZE_PAD) < HEAD_SIZE
    q_ptrs = (
        q_start * q_stride_0
        + qh[:, None] * q_stride_1
        + tl.arange(0, HEAD_SIZE_PAD)[None, :]
    )
    Q = tl.load(
        query_ptr + q_ptrs,
        mask=dim_mask[None, :],
        other=0.0
    ).to(tl.float32)  # [Q_kv, H_pad]

    # precompute base offset for block_scores pointer (int64)
    bs_off_base = (b * stride_bs_batch + kh * stride_bs_head).to(tl.int64)

    # ---- process PAGES_PER_TB pages ----
    for p in tl.static_range(PAGES_PER_TB):
        blk = blk_base + p
        in_range    = blk < num_blk
        is_sentinel = blk == last_blk
        do_compute  = in_range & (~is_sentinel)

        out_ptr = block_scores_ptr + (bs_off_base + blk.to(tl.int64))

        # sentinel write
        if is_sentinel:
            tl.store(out_ptr, FP32_MAX)

        # out-of-range: nothing to do
        if do_compute:
            # map logical page -> (mm_block, slot)
            mm_idx  = blk // MINMAX_CACHE_BLOCK_SIZE
            mm_slot = blk %  MINMAX_CACHE_BLOCK_SIZE

            mmb_ptr = (
                minmax_block_table_ptr
                + b * stride_mmb_bs
                + kh * stride_mmb_kh
                + mm_idx * stride_mmb_bl
            )
            mm_phys = tl.load(mmb_ptr).to(tl.int64)

            offs_d = tl.arange(0, HEAD_SIZE_PAD)

            base = (
                minmax_kv_cache_ptr
                + mm_phys * stride_mmkc_bl
                + mm_slot * stride_mmkc_bs
            )
            min_ptr = base + 0 * stride_mmkc_two + offs_d * stride_mmkc_hs
            max_ptr = base + 1 * stride_mmkc_two + offs_d * stride_mmkc_hs

            rep_min = tl.load(min_ptr, mask=dim_mask, other=0.0).to(tl.float32)
            rep_max = tl.load(max_ptr, mask=dim_mask, other=0.0).to(tl.float32)

            # upper-bound across queries
            upper = tl.maximum(Q * rep_max[None, :], Q * rep_min[None, :])  # [Q_kv, H_pad]
            score = tl.sum(upper, axis=1)                                   # [Q_kv]
            score_agg = tl.max(score, axis=0)                               # scalar

            tl.store(out_ptr, score_agg)

@torch.inference_mode()
def write_top_k_blocks(
    layer_idx:       int,
    num_seqs:        int,                        # B
    block_scores:    torch.Tensor,              # [L, B_tot, KV, MAX_BLK] (unused)
    max_seq_len:     int,
    block_size:      int,
    K:               int,
    top_k_blocks:    torch.Tensor,
    unstable_heads:  torch.Tensor,
    stable_heads:    torch.Tensor,
    is_first_decode_gpu: torch.Tensor,
    any_first_decode: bool,
    any_unstable_head: bool,
):
    max_n_blk = max(K, math.ceil(max_seq_len / block_size))
    bs = block_scores[layer_idx, :num_seqs, :, :max_n_blk]

    if any_unstable_head:
        _, idx = torch.topk(bs[:, unstable_heads, :], K, dim=-1, largest=True, sorted=False)  # [B, |UH|, K]
        top_k_blocks[layer_idx, :num_seqs, unstable_heads, :K] = idx

    if any_first_decode:
        _, idx = torch.topk(bs[is_first_decode_gpu[:, None], stable_heads, :], K, dim=-1,
                                largest=True, sorted=False)                                   # [B_sel, |SH|, K]
        top_k_blocks[layer_idx, is_first_decode_gpu[:, None], stable_heads, :K] = idx

@torch.inference_mode()
def write_top_k_blocks_post(
    layer_idx:       int,
    num_seqs:        int,                        # B
    block_scores:    torch.Tensor,              # [L, B_tot, KV, MAX_BLK] (unused)
    max_seq_len:     int,
    block_size:      int,
    K:               int,
    top_k_blocks:    torch.Tensor,
    old_top_k_blocks: torch.Tensor,
    stable_heads:    torch.Tensor,
    needs_rerank_gpu: torch.Tensor,
    any_needs_rerank: bool,
):
    if any_needs_rerank:
        max_n_blk = max(K, math.ceil(max_seq_len / block_size))
        bs = block_scores[:, :num_seqs, :, :max_n_blk]

        old_top_k_blocks[:, needs_rerank_gpu, :, :] = \
            top_k_blocks[:, needs_rerank_gpu, :, :]

        _, idx = torch.topk(bs[:, needs_rerank_gpu[:, None], :, :], K, dim=-1,
                                largest=True, sorted=False)                                   # [B_sel, |SH|, K]
        top_k_blocks[:, needs_rerank_gpu[:, None], :, :K] = idx

@triton.jit
def slot_map_kernel(
    bt_ptr,                     # int32*   [R, L, H, Lb]
    out_ptr,                    # int64*   [L, T_max, H]
    req_idx_ptr,                # int32*   [T]
    log_idx_ptr,                # int32*   [T]
    offs_ptr,                   # int32*   [T]
    # block-table strides
    stride_bt_bs, stride_bt_ly, stride_bt_kh, stride_bt_bl,
    # output strides
    stride_out_ly, stride_out_tok, stride_out_kh,
    # compile-time / launch-time constants
    BLOCK_T: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_l = tl.program_id(0)  # layer id in [0, L)
    pid_h = tl.program_id(1)  # head  id in [0, H)
    pid_t = tl.program_id(2)  # token tile id

    tok = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = tok < NUM_TOKENS

    # gather per-token indices
    r   = tl.load(req_idx_ptr + tok,  mask=mask, other=0)  # req idx
    lb  = tl.load(log_idx_ptr + tok,  mask=mask, other=0)  # logical blk idx
    off = tl.load(offs_ptr    + tok,  mask=mask, other=0)  # offset in blk

    # base pointer for this (layer, head)
    base_bt = bt_ptr + pid_l * stride_bt_ly + pid_h * stride_bt_kh

    # element-wise pointers into [R, L, H, Lb]
    addr = base_bt \
         + r.to(tl.int64)  * stride_bt_bs \
         + lb.to(tl.int64) * stride_bt_bl

    # physical block id
    blk = tl.load(addr, mask=mask, other=0).to(tl.int64)     # int64 math

    # final slot = blk * BLOCK_SIZE + off
    slot = blk * BLOCK_SIZE + off.to(tl.int64)

    # write to out[L, T, H] at (pid_l, tok, pid_h)
    base_out = out_ptr + pid_l * stride_out_ly + pid_h * stride_out_kh
    out_addr = base_out + tok.to(tl.int64) * stride_out_tok
    tl.store(out_addr, slot, mask=mask)