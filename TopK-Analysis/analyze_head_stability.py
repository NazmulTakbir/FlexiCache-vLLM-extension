import os
import random
import shutil
import json
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

random.seed(43)

from preprocess import (
    make_heads, list_samples, load_topk_data, Head, validate_top_k_values
)
from stability import (
    build_topk_masks_all_heads, compute_chunk_instability, get_unstable_heads_per_chunk
)
from similarity import (
    pairwise_jaccard_stats, head_frequency
)

def analyze_model(
    data_root: str, model: str, out_root: str, window_len: int, Ms: List[int], page_size: int, topK: int
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_to_samples = list_samples(data_root, model)
    if out_root is None:
        out_root = os.path.join(data_root, "analysis_outputs", f'topK-{topK}', model)
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root)

    M_to_unstable_sets_global: Dict[int, List[set]] = {int(M): [] for M in Ms}
    total_heads = None

    summary = {
        "model": model,
        "datasets": list(dataset_to_samples.keys()),
        "window_len": window_len,
        "page_size": page_size,
        "topK": topK,
        "Ms": list(map(int, Ms)),
        "device": str(device),
        "notes": "Instability = 1 - mean(RCO). RCO baseline-corrected with hypergeometric expectation.",
    }

    for dataset, samples in dataset_to_samples.items():
        ds_out = os.path.join(out_root, dataset)
        os.makedirs(ds_out, exist_ok=True)

        M_to_unstable_sets_for_dataset: Dict[int, List[set]] = {int(M): [] for M in Ms}

        for sample_dir in tqdm(samples, desc=f"Dataset {dataset}"):
            topk_data, num_steps, L, H, K_file, prompt_len = load_topk_data(sample_dir)
            if total_heads is None:
                total_heads = L * H

            if num_steps < window_len:
                print(f"[WARNING] {sample_dir} has {num_steps} steps < window_len={window_len}")
                continue

            K = int(min(topK, K_file))
            if K < K_file:
                topk_data = topk_data[:, :, :, :K].contiguous()

            n_page_per_step = np.ceil(
                (prompt_len + np.arange(1, num_steps + 1, dtype=np.int64)) / page_size
            )
            max_pages = int(n_page_per_step.max())

            assert all(n_page_per_step > K), f"Not enough pages to select top-{K} in {sample_dir}"

            validate_top_k_values(n_page_per_step, topk_data)

            # [total_heads, num_steps, max_idx] bool
            topk_mask = build_topk_masks_all_heads(topk_data, max_pages, device)

            # shape [num_chunks, total_heads]
            instability_scores = compute_chunk_instability(topk_mask, n_page_per_step, window_len, K)
            
            for M in Ms:
                M = int(M)
                unstable_heads_per_chunk = get_unstable_heads_per_chunk(instability_scores, M)
                M_to_unstable_sets_for_dataset[M].extend(unstable_heads_per_chunk)
                M_to_unstable_sets_global[M].extend(unstable_heads_per_chunk)

        ds_consistency = {}
        population = total_heads
        for M in Ms:
            M = int(M)
            unstable_sets = M_to_unstable_sets_for_dataset[M]
            stats = pairwise_jaccard_stats(
                unstable_sets=unstable_sets, M=M, total_heads=total_heads
            )
            # freq = head_frequency(unstable_sets, population=population)
            # ent  = normalized_entropy(freq)
            # gini = gini_coefficient(freq)

            random.shuffle(unstable_sets)

            train = unstable_sets[:len(unstable_sets)//2]
            test  = unstable_sets[len(unstable_sets)//2:]

            train_freq = head_frequency(train, total_heads)

            ranking = np.argsort(-train_freq).tolist()
            topM = sorted(ranking[:M])
            topM_set = set(topM)

            recalls = [len(s & topM_set) / M for s in test]
            recall_mean = float(np.mean(recalls))
            recall_rand = M / float(population)

            ds_consistency[M] = {
                **stats,
                # "frequency_entropy_norm": ent,
                # "frequency_gini": gini,
                "recall_at_M_mean": recall_mean,
                "recall_at_M_random": recall_rand,
                "topM": topM,
            }

            with open(os.path.join(ds_out, f"unstable_sets_M{M}.json"), "w") as f:
                json.dump([sorted(list(s)) for s in unstable_sets], f)
            with open(os.path.join(ds_out, f"consensus_M{M}.json"), "w") as f:
                for k, v in ds_consistency[M].items():
                    if isinstance(v, float):
                        ds_consistency[M][k] = round(v, 4)
                json.dump({"metrics": ds_consistency[M], "topM": topM, "ranking": ranking}, f, indent=2)

        with open(os.path.join(ds_out, "consistency_summary.json"), "w") as f:
            json.dump(ds_consistency, f, indent=2)

    global_summary = {}
    for M in Ms:
        M = int(M)
        unstable_sets = M_to_unstable_sets_global[M]
        gstats = pairwise_jaccard_stats(
            unstable_sets=unstable_sets, M=M, total_heads=total_heads
        )
        # freq = head_frequency(unstable_sets, population=total_heads)
        # ent  = normalized_entropy(freq)
        # gini = gini_coefficient(freq)

        random.shuffle(unstable_sets)

        train = unstable_sets[:len(unstable_sets)//2]
        test  = unstable_sets[len(unstable_sets)//2:]

        train_freq = head_frequency(train, total_heads)

        ranking = np.argsort(-train_freq).tolist()
        topM = sorted(ranking[:M])
        topM_set = set(topM)

        recalls     = [len(s & topM_set) / M for s in test]
        recall_mean = float(np.mean(recalls))
        recall_rand = M / float(population)
        
        global_summary[M] = {
            **gstats,
            # "frequency_entropy_norm": ent,
            # "frequency_gini": gini,
            "recall_at_M_mean": recall_mean,
            "recall_at_M_random": recall_rand,
            "topM": topM,
        }
        with open(os.path.join(out_root, f"global_consensus_M{M}.json"), "w") as f:
            for k, v in global_summary[M].items():
                if isinstance(v, float):
                    global_summary[M][k] = round(v, 4)
            json.dump({"metrics": global_summary[M], "topM": topM, "ranking": ranking}, f, indent=2)
        with open(os.path.join(out_root, f"global_unstable_sets_M{M}.json"), "w") as f:
            json.dump([sorted(list(s)) for s in unstable_sets], f)

    with open(os.path.join(out_root, "run_config.json"), "w") as f:
        json.dump({
            **summary,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
        }, f, indent=2)
    with open(os.path.join(out_root, "global_consistency_summary.json"), "w") as f:
        json.dump(global_summary, f, indent=2)

    for M in Ms:
        print(f"Global consensus M={int(M)}: {os.path.join(out_root, f'global_consensus_M{int(M)}.json')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root folder containing <model>/<dataset>/sample-XXX")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--out_root", type=str, default=None,
                    help="Output root (default: <root>/analysis_outputs/<model>/...)")
    ap.add_argument("--window_len", type=int, default=16,
                    help="Compare step s to s+1..s+window_len-1.")
    ap.add_argument("--M", type=int, nargs="+", default=[16, 32, 64],
                    help="Size of unstable set")
    ap.add_argument("--page_size", type=int, default=16, help="Page size in tokens")
    ap.add_argument("--topK", type=int, default=256,
                    help="Use only the first K (sorted) top-K page indices per head (e.g., 128 to take 50%)")
    args = ap.parse_args()

    analyze_model(
        data_root=args.data_root, model=args.model, out_root=args.out_root,
        window_len=args.window_len, Ms=args.M, page_size=args.page_size,
        topK=args.topK
    )

if __name__ == "__main__":
    main()
