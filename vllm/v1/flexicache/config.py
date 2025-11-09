import math
import numpy as np
import torch
import json
import os
import importlib.util

from collections import defaultdict

from vllm.v1.flexicache.model_data import model_data

# Hot-path aliases
IS_INITIALIZED = False
MINMAX_KEY_CACHE_BLOCK_SIZE   : int | None = None
NUM_LAYERS                    : int | None = None
RERANK_FREQUENCY              : int | None = None
LAYER_HAS_UNSTABLE_HEADS      : list[bool] | None = None
UNSTABLE_HEAD_MASKS           : torch.Tensor | None = None
LAYER_TO_STABLE_HEADS         : dict[int, set[int]] | None = None
LAYER_TO_UNSTABLE_HEADS       : dict[int, set[int]] | None = None
LAYER_TO_UNSTABLE_HEADS_TENSOR: dict[int, torch.Tensor] | None = None
LAYER_TO_STABLE_HEADS_TENSOR  : dict[int, torch.Tensor] | None = None

class _FlexiMeta(type):
    _inst = None
    _init_fingerprint = None

    def initialize(cls, *args, **kwargs):
        # Create the singleton instance or verify the arguments.
        fingerprint = (args, tuple(sorted(kwargs.items())))
        if cls._inst is None:
            cls._inst = super().__call__(*args, **kwargs)  # calls FlexiCacheConfig.__init__
            cls._init_fingerprint = fingerprint
        elif fingerprint != cls._init_fingerprint:
            raise ValueError(
                "FlexiCacheConfig already initialized with different arguments."
            )
        return cls._inst

    def __getattr__(cls, name):
        # Allow class-level access: FlexiCacheConfig.foo -> instance.foo
        if cls._inst is None:
            raise RuntimeError("FlexiCacheConfig.initialize(...) must be called first.")
        return getattr(cls._inst, name)

class FlexiCacheConfig(metaclass=_FlexiMeta):

    def __init__(
        self, model_name, num_layers, num_kv_heads, block_size, num_unstable_heads,
        rerank_frequency, topK_budget, max_model_len, unstable_heads_profile_task
    ):
        self.model_name   = model_name
        self.num_layers   = num_layers
        self.num_kv_heads = num_kv_heads
        self.block_size   = block_size
        self.num_unstable_heads = num_unstable_heads
        self.rerank_frequency   = rerank_frequency
        self.top_k_page_budget  = topK_budget
        self.max_model_len      = max_model_len

        assert self.num_unstable_heads >= 0, "num_unstable_heads must be non-negative"
        assert self.rerank_frequency > 0, "rerank_frequency must be positive"
        assert self.top_k_page_budget > 0, "top_k_page_budget must be positive"

        flexicache = importlib.util.find_spec("vllm.v1.flexicache")
        assert flexicache is not None, "vllm.v1.flexicache"
        flexicache_path = os.path.dirname(flexicache.origin)

        with open(os.path.join(flexicache_path, "config.json")) as f:
            data = json.load(f)

        self.avg_num_tokens_per_req = data.get("avg_num_tokens_per_req")
        self.cpu_kv_cache_size      = data.get("cpu_kv_cache_size") * 1024 ** 3
        self.minmax_key_cache_block_size = data.get("minmax_key_cache_block_size")

        self.min_blocks_per_gpu_layer = \
            int(math.ceil(self.avg_num_tokens_per_req / block_size) * self.num_kv_heads * 4)

        self.top_k_token_budget     = self.top_k_page_budget * self.block_size
        self.unstable_heads_portion = self.num_unstable_heads / (self.num_kv_heads * self.num_layers)

        if model_name not in model_data:
            raise ValueError(f"Model {model_name} not found in flexicache model_data")

        if self.num_unstable_heads == 0:
            self.unstable_heads = []
        else:
            unstable_heads_keys =f'unstable-{num_unstable_heads}-profile-{unstable_heads_profile_task}-topk-{topK_budget}'
            assert unstable_heads_keys in model_data[model_name], \
                f"Model {model_name} does not have {unstable_heads_keys} data"
            self.unstable_heads = model_data[model_name][unstable_heads_keys]

        assert len(self.unstable_heads) == self.num_unstable_heads, \
            f"Model {model_name} has {len(self.unstable_heads)} unstable heads, "\
            f"but num_unstable_heads is set to {self.num_unstable_heads}"

        self._build()

        self._populate_globals()

        global IS_INITIALIZED
        IS_INITIALIZED = True
    
    def _build(self):
        self.layer_to_unstable_heads = defaultdict(set)
        for layer, head in self.unstable_heads:
            self.layer_to_unstable_heads[layer].add(head)
        
        all_heads_set = set(range(self.num_kv_heads))
        self.layer_to_stable_heads = {
            l: (all_heads_set - self.layer_to_unstable_heads[l])
            for l in range(self.num_layers)
        }

        self.unstable_head_masks = np.zeros(self.num_layers, dtype=np.uint32)
        for layer, head in self.unstable_heads:
            self.unstable_head_masks[layer] |= (1 << head)
        self.unstable_head_masks = torch.tensor(self.unstable_head_masks, dtype=torch.uint32, device="cuda")

        self.layer_to_unstable_heads_tensor = {}
        self.layer_to_stable_heads_tensor   = {}
        for layer in range(self.num_layers):
            uh = sorted(self.layer_to_unstable_heads.get(layer, set()))
            sh = sorted(self.layer_to_stable_heads[layer])
            self.layer_to_unstable_heads_tensor[layer] = torch.tensor(uh, device='cuda', dtype=torch.int64)
            self.layer_to_stable_heads_tensor[layer]   = torch.tensor(sh, device='cuda', dtype=torch.int64)

    def _populate_globals(self):
        global MINMAX_KEY_CACHE_BLOCK_SIZE, NUM_LAYERS, LAYER_HAS_UNSTABLE_HEADS, UNSTABLE_HEAD_MASKS
        global LAYER_TO_UNSTABLE_HEADS_TENSOR, LAYER_TO_STABLE_HEADS_TENSOR, LAYER_TO_STABLE_HEADS
        global RERANK_FREQUENCY, LAYER_TO_UNSTABLE_HEADS

        MINMAX_KEY_CACHE_BLOCK_SIZE    = self.minmax_key_cache_block_size
        NUM_LAYERS                     = self.num_layers
        UNSTABLE_HEAD_MASKS            = self.unstable_head_masks
        LAYER_TO_UNSTABLE_HEADS_TENSOR = self.layer_to_unstable_heads_tensor
        LAYER_TO_STABLE_HEADS_TENSOR   = self.layer_to_stable_heads_tensor
        LAYER_TO_STABLE_HEADS          = self.layer_to_stable_heads
        RERANK_FREQUENCY               = self.rerank_frequency
        LAYER_TO_UNSTABLE_HEADS        = self.layer_to_unstable_heads

        has_unstable = [0] * self.num_layers
        for layer in range(self.num_layers):
            if layer in self.layer_to_unstable_heads and len(self.layer_to_unstable_heads[layer]) > 0:
                has_unstable[layer] = 1
        LAYER_HAS_UNSTABLE_HEADS = has_unstable

    def _get_layerwise_block_distribution_weight(self) -> tuple[list[float], list[float]]:
        gpu_weights = np.array([0] * self.num_layers, dtype=np.float32)
        cpu_weights = np.array([0] * self.num_layers, dtype=np.float32)
        mm_weights  = np.array([0] * self.num_layers, dtype=np.float32)

        for l in range(self.num_layers):
            unstable = self.layer_to_unstable_heads[l]
            for h in range(self.num_kv_heads):
                if h in unstable:
                    gpu_weights[l] += self.avg_num_tokens_per_req
                else:
                    gpu_weights[l] += self.top_k_token_budget
                    cpu_weights[l] += self.avg_num_tokens_per_req

        # mm_weights is the same for all layers
        mm_weights[:] = 1 / self.num_layers

        gpu_weights = gpu_weights / gpu_weights.sum()
        cpu_weights = cpu_weights / cpu_weights.sum()

        assert np.isclose(gpu_weights.sum(), 1.0)
        assert np.isclose(cpu_weights.sum(), 1.0)
        assert all(w > 0 for w in gpu_weights)
        assert all(w >= 0 for w in cpu_weights)

        return gpu_weights.tolist(), cpu_weights.tolist(), mm_weights.tolist()

    def _distribute_blocks_to_layers(self, total_blocks: int, weights: list[float],
                                     is_gpu: bool = False, is_cpu: bool = False) -> list[int]:
        assert len(weights) == self.num_layers
        sum_weights = sum(weights)
        assert abs(sum_weights - 1.0) < 1e-6, f'Invalid weights: sum={sum_weights}'
        if not is_cpu:
            assert all(w > 0 for w in weights), f'Invalid weights: weights={weights}'
        
        if is_gpu:
            required = self.min_blocks_per_gpu_layer * self.num_layers
            if total_blocks < required:
                raise ValueError(
                    f"Not enough GPU blocks to satisfy per-layer minimum: {total_blocks} < {required}"
                )

        if all(math.isclose(w, weights[0]) for w in weights):
            return [total_blocks // self.num_layers] * self.num_layers

        quota_frac = [total_blocks * (w / sum_weights) for w in weights]
        quota_blks = [int(q) for q in quota_frac]
        remaining  = total_blocks - sum(quota_blks)

        if remaining > 0:
            assert remaining <= self.num_layers
            # Distribute remaining to layers in order of largest fractional part.
            order = sorted(range(self.num_layers), key=lambda i: quota_frac[i] - quota_blks[i], reverse=True)
            for r in range(remaining):
                quota_blks[order[r]] += 1

        assert sum(quota_blks) == total_blocks

        if is_gpu:
            deficits = [(i, self.min_blocks_per_gpu_layer - quota_blks[i])
                        for i in range(self.num_layers) if quota_blks[i] < self.min_blocks_per_gpu_layer]
            if deficits:
                donors = [(i, quota_blks[i] - self.min_blocks_per_gpu_layer)
                          for i in range(self.num_layers) if quota_blks[i] > self.min_blocks_per_gpu_layer]
                total_deficit  = sum(d for _, d in deficits)
                total_capacity = sum(c for _, c in donors)
                assert total_capacity >= total_deficit

                # Proportional donation: each donor contributes roughly
                # cap_i / sum(cap) * total_deficit (integer-rounded).
                # 1) Plan donations per donor with largest-fraction rounding.
                donors.sort(key=lambda x: x[1], reverse=True)  # stable order
                frac_plan = []
                for (idx, cap) in donors:
                    share = (cap / total_capacity) * total_deficit if total_capacity > 0 else 0.0
                    give  = int(share)
                    frac_plan.append((idx, cap, share, give, share - give))

                planned_total = sum(g for *_, g, __ in frac_plan)
                remainder = total_deficit - planned_total
                if remainder > 0:
                    # Give the leftover one-by-one to donors with largest fractional parts.
                    order = sorted(range(len(frac_plan)),
                                   key=lambda k: frac_plan[k][4], reverse=True)
                    for k in order[:remainder]:
                        idx, cap, share, give, frac = frac_plan[k]
                        give += 1
                        frac_plan[k] = (idx, cap, share, give, frac)

                # Cap safety: do not exceed donor capacity due to rounding.
                donation_plan = []
                for (idx, cap, _share, give, _frac) in frac_plan:
                    donation_plan.append((idx, min(give, cap)))

                # If capping reduced total below total_deficit by a tiny amount (numerical edge),
                # top up from donors with remaining capacity in descending capacity order.
                donated_now = sum(g for _, g in donation_plan)
                shortfall = total_deficit - donated_now
                if shortfall > 0:
                    # Compute residual capacities.
                    residual = []
                    for (idx, cap), (_, planned) in zip(donors, donation_plan):
                        rem = cap - planned
                        if rem > 0:
                            residual.append((idx, rem))
                    residual.sort(key=lambda x: x[1], reverse=True)
                    ptr = 0
                    while shortfall > 0 and ptr < len(residual):
                        idx, rem = residual[ptr]
                        take = min(rem, shortfall)
                        # bump planned donation
                        for j, (didx, planned) in enumerate(donation_plan):
                            if didx == idx:
                                donation_plan[j] = (didx, planned + take)
                                break
                        shortfall -= take
                        ptr += 1
                # Apply donor deductions and route to receivers.
                # First deduct from donors' quotas according to the donation plan.
                donation_pool = []
                for donor_idx, give in donation_plan:
                    if give > 0:
                        quota_blks[donor_idx] -= give
                        donation_pool.append([donor_idx, give])

                # Then satisfy deficits (largest first) from the pooled donations.
                deficits.sort(key=lambda x: x[1], reverse=True)
                dp = 0
                for receiver_idx, need in deficits:
                    while need > 0:
                        # Advance to next donor if current one exhausted.
                        while dp < len(donation_pool) and donation_pool[dp][1] == 0:
                            dp += 1
                        assert dp < len(donation_pool), "Donation pool exhausted before filling deficits"
                        _, give_left = donation_pool[dp]
                        take = min(need, give_left)
                        quota_blks[receiver_idx] += take
                        donation_pool[dp][1] -= take
                        need -= take
                assert all(b >= self.min_blocks_per_gpu_layer for b in quota_blks), quota_blks

        return quota_blks
    
    @classmethod
    def mem_portion_for_minmax_key_cache(cls, block_size: int) -> float:
        if cls._inst is None:
            raise RuntimeError("FlexiCacheConfig.initialize(...) must be called first.")
        inst: FlexiCacheConfig = cls._inst
        kvc_mem_for_stable_heads   = inst.top_k_token_budget * (1 - inst.unstable_heads_portion)
        kvc_mem_for_unstable_heads = inst.avg_num_tokens_per_req * inst.unstable_heads_portion
        minmax_kc_mem              = inst.avg_num_tokens_per_req / block_size
        return minmax_kc_mem / (kvc_mem_for_stable_heads + kvc_mem_for_unstable_heads + minmax_kc_mem)

    @classmethod
    def get_blocks_per_layer(
        cls, total_gpu_blocks: int, total_cpu_blocks: int, total_minmax_blocks: int
    ) -> tuple[list[int], list[int], list[int]]:
        if cls._inst is None:
            raise RuntimeError("FlexiCacheConfig.initialize(...) must be called first.")
        inst: FlexiCacheConfig = cls._inst
        gpu_weights, cpu_weights, mm_weights = inst._get_layerwise_block_distribution_weight()
        gpu_blks = inst._distribute_blocks_to_layers(total_gpu_blocks, gpu_weights, is_gpu=True)
        cpu_blks = inst._distribute_blocks_to_layers(total_cpu_blocks, cpu_weights, is_cpu=True)
        mm_blks  = inst._distribute_blocks_to_layers(total_minmax_blocks, mm_weights)
        return gpu_blks, cpu_blks, mm_blks