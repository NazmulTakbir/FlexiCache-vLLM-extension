from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
import os
import re
import numpy as np

@dataclass(frozen=True)
class Head:
    layer: int
    head: int
    global_id: int  # [0, L*H-1]

def make_heads(num_layers: int, num_heads: int) -> List[Head]:
    heads = []
    global_id = 0
    for L in range(num_layers):
        for H in range(num_heads):
            heads.append(Head(L, H, global_id))
            global_id += 1
    return heads

def list_samples(data_root: str, model: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for d in os.listdir(os.path.join(data_root, model)):
        dpath = os.path.join(data_root, model, d)
        samples = []
        for name in sorted(os.listdir(dpath)):
            p = os.path.join(dpath, name)
            assert name.startswith("sample-"), f"Unexpected dir {p} (not starting with sample-)"
            samples.append(p)
        assert samples, f"No samples under {dpath}"
        out[d] = samples
    return out

@torch.no_grad()
def load_topk_data(sample_dir: str) -> Tuple[torch.Tensor, int, int, int, int]:
    files = [f for f in os.listdir(sample_dir)]

    FILENAME_REGEX = re.compile(r"^decode-step-(\d+)-prompt-len-(\d+)\.pt$")
    for f in files:
        assert FILENAME_REGEX.fullmatch(f), f"Unexpected filename {f} in {sample_dir}"

    # sort by decode step number
    files = sorted(files, key=lambda s: int(FILENAME_REGEX.fullmatch(s).group(1)))

    assert files, f"Empty sample dir {sample_dir}"

    prompt_len = int(FILENAME_REGEX.search(files[0]).group(2))

    assert all(int(FILENAME_REGEX.search(f).group(2)) == prompt_len for f in files), \
        f"Varying prompt lengths in {sample_dir}"

    t0 = torch.load(os.path.join(sample_dir, files[0]), map_location="cpu")
    assert t0.ndim == 3, f"Expected tensor [L,H,K], got {t0.shape} in {files[0]}"
    L, H, K = t0.shape
    num_steps = len(files)

    topk_data = torch.empty((num_steps, L, H, K), dtype=torch.int64, device="cpu")
    for i, fname in enumerate(files):
        t = torch.load(os.path.join(sample_dir, fname), map_location="cpu")
        assert t.shape == (L, H, K), f"Shape mismatch in {fname}: got {t.shape}, expected {(L,H,K)}"
        topk_data[i] = t
    return topk_data, num_steps, L, H, K, prompt_len

def validate_top_k_values(n_page_per_step: np.ndarray, topk_steps: torch.Tensor) -> np.ndarray:
    # If at step X there are N pages, then top-K indices must be in [0, N-1].
    max_idx_per_step = topk_steps.view(
        topk_steps.shape[0], -1
    ).max(dim=1).values.cpu().numpy().astype(np.int64)
    assert all(max_idx_per_step < n_page_per_step), "Out of bounds top-K indices found"