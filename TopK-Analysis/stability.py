import torch
from typing import List
import numpy as np

@torch.no_grad()
def build_topk_masks_all_heads(
    topk_steps: torch.Tensor, max_pages: int, device: torch.device
) -> torch.Tensor:
    num_steps, L, H, K = topk_steps.shape
    total_heads = L * H

    idx = topk_steps.view(num_steps, total_heads, K).to(device, non_blocking=True)
    topk_mask = torch.zeros((total_heads, num_steps, max_pages), dtype=torch.bool, device=device)
    ones = torch.ones((total_heads, K), dtype=torch.bool, device=device)

    for s in range(num_steps):
        # scatter writes ones into topk_mask[head, s, page_idx] at the page indices in idx[s, head, :]
        topk_mask[:, s, :].scatter_(dim=1, index=idx[s], src=ones)
    return topk_mask  # [total_heads, num_steps, max_pages] bool

@torch.no_grad()
def compute_chunk_instability(
    topk_mask: torch.Tensor, n_page_per_step: np.ndarray, window_len: int, K: int
) -> torch.Tensor:
    total_heads, num_steps, max_pages = topk_mask.shape

    last_chunk_start = num_steps - window_len
    if last_chunk_start < 0:
        return torch.empty((0, total_heads), device=topk_mask.device)

    chunk_starts = list(range(0, last_chunk_start + 1, window_len))
    instability_scores = \
        torch.empty((len(chunk_starts), total_heads), dtype=torch.float32, device=topk_mask.device)

    n_page_per_step = torch.as_tensor(n_page_per_step, device=topk_mask.device, dtype=torch.float32)

    K = float(K)
    for idx, chunk_start in enumerate(chunk_starts):
        # [total_heads, 1, max_pages], bool
        reference = topk_mask[:, chunk_start, :].unsqueeze(1)

        # [total_heads, window_len-1, max_pages], bool
        tail = topk_mask[:, chunk_start+1:chunk_start+window_len, :]

        # [total_heads, window_len-1]
        num_overlaps = (reference & tail).sum(dim=2, dtype=torch.int32).to(torch.float32)
        overlap_fraction = num_overlaps / K

        # [window_len-1]
        n_pages = n_page_per_step[chunk_start+1:chunk_start+window_len]

        # If you pick K pages uniformly at random (without replacement) from a universe of size N,
        # then the expected fractional overlap between two independent size-K sets is K/N.
        # This is a hypergeometric distribution. Mean = num_draws * (num_sucess / num_population)
        # When any given Set A, when selecting Set B, num_draws = K, num_success = K, num_population = N.
        # So, expected overlap fraction = K * (K/N) / K = K/N
        random_overlap_fraction = (K / n_pages)

        # rco = Random-Corrected Overlap
        # RCO = (overlap_fraction - random_overlap_fraction) / (1 - random_overlap_fraction)
        #     RCO=0  means "no better than random" overlap,
        #     RCO=1  means "identical top-K sets".
        # Shape: [total_heads, window_len-1]
        rco = (overlap_fraction - random_overlap_fraction) / (1.0 - random_overlap_fraction)
        rco = rco.clamp(min=0.0)

        # Instability per head for this chunk = 1 - (mean_RCO over window tail):
        # Higher value -> Less stable
        # shape: [total_heads]
        instability_scores[idx] = 1.0 - rco.mean(dim=1)

    return instability_scores  # shape [num_chunks, total_heads]

def get_unstable_heads_per_chunk(instability_scores: torch.Tensor, M: int) -> List[set]:
    num_chunks, total_heads = instability_scores.shape
    assert M < total_heads, f"M={M} must be < total_heads={total_heads}"

    top_sets: List[set] = []
    _, idx = torch.topk(instability_scores, k=M, dim=1, largest=True, sorted=False)
    for i in range(num_chunks):
        top_sets.append(set(idx[i].tolist()))
    return top_sets