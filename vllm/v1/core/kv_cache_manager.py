# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.block_pool import BlockPool, MinMaxBlockPool
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlock,
                                         hash_request_tokens)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus
import vllm.v1.flexicache.config as FCC
from vllm.v1.flexicache.config import FlexiCacheConfig

logger = init_logger(__name__)


class KVCacheManager:

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        max_model_len: int,
        num_kv_heads: int,
        num_attn_layers: int,  
        sliding_window: Optional[int] = None,
        enable_caching: bool = True,
        num_preallocate_tokens: int = 64,
        log_stats: bool = False,
        enable_flexicache: bool = False,
        num_minmax_blocks: int = 0,
        num_cpu_blocks: int = 0
    ) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.num_minmax_blocks = num_minmax_blocks
        self.max_model_len = max_model_len
        self.num_layers = num_attn_layers
        if enable_flexicache:
            self.max_num_blocks_per_req = cdiv(max_model_len, block_size) * num_kv_heads
        else:
            self.max_num_blocks_per_req = cdiv(max_model_len, block_size)
        self.sliding_window = sliding_window
        self.enable_caching = enable_caching
        # FIXME: make prefix cache stats conditional on log_stats
        self.log_stats = log_stats
        # NOTE(woosuk): To avoid frequent block allocation, we preallocate some
        # blocks for each request. For example, when a request reaches the end
        # of its block table, we preallocate N blocks in advance. This way, we
        # reduce the overhead of updating free_block_ids and ref_cnts for each
        # request every step (at the cost of some memory waste).
        # NOTE(woosuk): This is different from the "lookahead" slots since this
        # does not guarantee that the request always has N empty blocks. After
        # the request gets N empty blocks, it starts to use the blocks without
        # further allocation. When it uses up all the N empty blocks, it gets
        # N new empty blocks.
        self.num_preallocate_tokens = num_preallocate_tokens
        if enable_flexicache:
            self.num_preallocate_blocks = cdiv(num_preallocate_tokens, block_size) * num_kv_heads
        else:
            self.num_preallocate_blocks = cdiv(num_preallocate_tokens, block_size)

        if enable_flexicache:
            gpu_blocks_per_layer, cpu_blocks_per_layer, mm_blocks_per_layer = \
                FlexiCacheConfig.get_blocks_per_layer(num_gpu_blocks, num_cpu_blocks, num_minmax_blocks)
            
            self.block_pools: list[BlockPool] = [
                BlockPool(
                    gpu_blocks_per_layer[L], enable_caching, enable_flexicache, num_kv_heads, cpu_blocks_per_layer[L]
                )
                for L in range(self.num_layers)
            ]
            self.minmax_block_pools: list[MinMaxBlockPool] = [
                MinMaxBlockPool(mm_blocks_per_layer[L]) for L in range(self.num_layers)
            ]
            
            # per-layer request->blocks
            self.req_to_blocks_by_layer: list[
                defaultdict[str, list[KVCacheBlock]]
            ] = [defaultdict(list) for _ in range(self.num_layers)]

            self.req_to_cpu_blocks_by_layer: list[
                defaultdict[str, list[KVCacheBlock]]
            ] = [defaultdict(list) for _ in range(self.num_layers)]

            self.req_to_minmax_blocks_by_layer: list[
                defaultdict[str, list[KVCacheBlock]]
            ] = [defaultdict(list) for _ in range(self.num_layers)]

            # Free these GPU blocks after the prompt KV cache has been offloaded
            self.pending_free: dict[str, list[list[KVCacheBlock]]] = {}
            
            assert FCC.IS_INITIALIZED, "FlexiCacheConfig is not initialized!"
            self.minmax_key_cache_block_size = FCC.MINMAX_KEY_CACHE_BLOCK_SIZE
            self.layer_to_unstable_heads     = FCC.LAYER_TO_UNSTABLE_HEADS
            self.layer_to_stable_heads       = FCC.LAYER_TO_STABLE_HEADS
            self.rerank_freq                 = FCC.RERANK_FREQUENCY
        else:
            self.block_pool = BlockPool(num_gpu_blocks, enable_caching, enable_flexicache, num_kv_heads)

            # Mapping from request ID to blocks to track the blocks allocated
            # for each request, so that we can free the blocks when the request
            # is finished.
            self.req_to_blocks: defaultdict[str,
                                            list[KVCacheBlock]] = defaultdict(list)

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHashType]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for reempted ones.
        self.num_cached_block: dict[str, int] = {}
        self.prefix_cache_stats = PrefixCacheStats()

        self.enable_flexicache = enable_flexicache
        self.num_kv_heads = num_kv_heads

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        if self.enable_flexicache:
            usages = [bp.get_usage() for bp in self.block_pools]
            min_val = float('inf')
            max_val = float('-inf')
            min_idx = max_idx = -1

            for i, u in enumerate(usages):
                if u < min_val:
                    min_val = u
                    min_idx = i
                if u > max_val:
                    max_val = u
                    max_idx = i

            return min_val, max_val, min_idx, max_idx
        else:
            return self.block_pool.get_usage(), self.block_pool.get_usage(), 0, 0
        
    @property
    def cpu_usage(self) -> float:
        if self.enable_flexicache:
            return min([bp.get_cpu_usage() for bp in self.block_pools]), \
                   max([bp.get_cpu_usage() for bp in self.block_pools])
        else:
            return 0, 0

    def _min_free_blocks_across_layers(self) -> int:
        return min(bp.get_num_free_blocks() for bp in self.block_pools)
    
    def _min_free_cpu_blocks_across_layers(self) -> int:
        return min(bp.get_num_free_cpu_blocks() for bp in self.block_pools)

    def _min_free_minmax_across_layers(self) -> int:
        return min(mbp.get_num_free_blocks() for mbp in self.minmax_block_pools)

    def make_prefix_cache_stats(self) -> PrefixCacheStats:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats.
        """
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(
            self, request: Request) -> tuple[list[KVCacheBlock], int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return [], 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            block_hashes = hash_request_tokens(self.block_size, request)
            self.req_to_block_hashes[request.request_id] = block_hashes

        self.prefix_cache_stats.requests += 1
        if request.sampling_params.prompt_logprobs is None:
            # Check for cache hits
            computed_blocks = []
            for block_hash in block_hashes:
                # block_hashes is a chain of block hashes. If a block hash
                # is not in the cached_block_hash_to_id, the following
                # block hashes are not computed yet for sure.
                if cached_block := self.block_pool.get_cached_block(
                        block_hash):
                    computed_blocks.append(cached_block)
                else:
                    break

            self.prefix_cache_stats.queries += len(block_hashes)
            self.prefix_cache_stats.hits += len(computed_blocks)

            # NOTE(woosuk): Since incomplete blocks are not eligible for
            # sharing, `num_computed_tokens` is always a multiple of
            # `block_size`.
            num_computed_tokens = len(computed_blocks) * self.block_size
            return computed_blocks, num_computed_tokens
        else:
            # Skip cache hits for prompt logprobs
            return [], 0

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None
    ) -> tuple[
            Optional[list[list[KVCacheBlock]]],   # GPU KV
            Optional[list[list[KVCacheBlock]]],   # MinMax KV
            Optional[list[list[KVCacheBlock]]],   # CPU KV (with -1 for unstable)
        ]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            new_computed_blocks: A list of new computed blocks just hitting the
                prefix caching.

        Blocks layout:
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_tokens == 0:
            raise ValueError("num_tokens must be greater than 0")
        
        assert not (self.enable_flexicache and self.enable_caching)

        new_computed_blocks = new_computed_blocks or []
        
        if self.enable_flexicache:
            num_total_tokens            = request.num_computed_tokens + num_tokens
            num_required_logical_blocks = cdiv(num_total_tokens, self.block_size)
            
            # We are assuming that all layers will have the same number of blocks
            # This can be ensured despite freeing of blocks in the middle replacing
            # the freed block with the guard block. This also allows us to find a
            # block of a request given the logical block number and head number, making
            # freeing and reallocating easier
            num_req_blocks               = len(self.req_to_blocks_by_layer[0][request.request_id])
            num_allocated_logical_blocks = cdiv(num_req_blocks, self.num_kv_heads)
            
            num_new_logical_blocks = num_required_logical_blocks - num_allocated_logical_blocks
            num_new_blocks         = num_new_logical_blocks * self.num_kv_heads

            if num_new_blocks > 0:
                min_free_blocks     = self._min_free_blocks_across_layers()
                min_free_cpu_blocks = self._min_free_cpu_blocks_across_layers()
                if num_new_blocks > min_free_blocks:
                    if request.status == RequestStatus.RUNNING:
                        logger.info(
                            ("PREEMPTING RUNNING REQ %s: Need %d GPU blocks but only %d free. "),
                            request.request_id, num_new_blocks, min_free_blocks
                        )
                    else:
                        pass
                        # logger.info(
                        #     ("Cannot allocate REQ %s with status %s: Need %d GPU blocks but only %d free. "),
                        #     request.request_id, request.status, num_new_blocks, min_free_blocks
                        # )
                    return None, None, None
                if num_new_blocks > min_free_cpu_blocks:
                    if request.status == RequestStatus.RUNNING:
                        logger.info(
                            ("PREEMPTING RUNNING REQ %s: Need %d CPU blocks but only %d free. "),
                            request.request_id, num_new_blocks, min_free_cpu_blocks
                        )
                    else:
                        logger.info(
                            ("Cannot allocate REQ %s with status %s: Need %d CPU blocks but only %d free. "),
                            request.request_id, request.status, num_new_blocks, min_free_cpu_blocks
                        )
                    assert False, "DEBUG: Should I allocated larger CPU KV cache?"
                    return None, None, None

                num_new_blocks_w_prealloc = min(
                    num_new_blocks + self.num_preallocate_blocks,
                    min(min_free_blocks, min_free_cpu_blocks),
                    self.max_num_blocks_per_req - num_req_blocks
                )

                # Ensure we allocate a multiple of num_kv_heads
                num_new_blocks_w_prealloc = (num_new_blocks_w_prealloc // self.num_kv_heads) * self.num_kv_heads

                assert num_new_blocks_w_prealloc > 0

                num_allocated_mm_logical_blocks = \
                    cdiv(num_allocated_logical_blocks, self.minmax_key_cache_block_size)
                num_required_mm_logical_blocks = \
                    cdiv(num_allocated_logical_blocks + (num_new_blocks_w_prealloc // self.num_kv_heads), self.minmax_key_cache_block_size)
                num_new_mm_blocks = (num_required_mm_logical_blocks - num_allocated_mm_logical_blocks) * self.num_kv_heads
                if num_new_mm_blocks > self._min_free_minmax_across_layers():
                    if request.status == RequestStatus.RUNNING:
                        logger.info(
                            ("PREEMPTING RUNNING REQ %s: Need %d MinMax blocks but only %d free. "),
                            request.request_id, num_new_mm_blocks, self._min_free_minmax_across_layers()
                        )
                    else:
                        logger.info(
                            ("Cannot allocate REQ %s with status %s: Need %d MinMax blocks but only %d free. "),
                            request.request_id, request.status, num_new_mm_blocks, self._min_free_minmax_across_layers()
                        )
                    assert False, "DEBUG: Should I allocated larger MinMax KV cache?"
                    return None, None, None
            else:
                num_new_blocks_w_prealloc = 0
                num_new_mm_blocks         = 0

            new_blocks_by_layer: list[list[KVCacheBlock]]     = []
            new_minmax_by_layer: list[list[KVCacheBlock]]     = []
            new_cpu_blocks_by_layer: list[list[KVCacheBlock]] = []
            for L in range(self.num_layers):
                if num_new_blocks_w_prealloc > 0:
                    new_blocks_L = self.block_pools[L].get_new_blocks(num_new_blocks_w_prealloc)
                    self.req_to_blocks_by_layer[L][request.request_id].extend(new_blocks_L)

                    new_cpu_L = self.block_pools[L].get_new_blocks_cpu(
                        num_new_blocks_w_prealloc, self.layer_to_unstable_heads[L]
                    )
                    self.req_to_cpu_blocks_by_layer[L][request.request_id].extend(new_cpu_L)
                else:
                    new_blocks_L = []
                    new_cpu_L = []
                new_blocks_by_layer.append(new_blocks_L)
                new_cpu_blocks_by_layer.append(new_cpu_L)

                if num_new_mm_blocks > 0:
                    new_mm_L = self.minmax_block_pools[L].get_new_blocks(num_new_mm_blocks)
                    self.req_to_minmax_blocks_by_layer[L][request.request_id].extend(new_mm_L)
                else:
                    new_mm_L = []
                new_minmax_by_layer.append(new_mm_L)

            return new_blocks_by_layer, new_minmax_by_layer, new_cpu_blocks_by_layer
        
        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                                len(new_computed_blocks) * self.block_size)
        num_required_blocks = cdiv(num_computed_tokens + num_tokens,
                                    self.block_size)
        req_blocks = self.req_to_blocks[request.request_id]
        num_new_blocks = (num_required_blocks - len(req_blocks) -
                            len(new_computed_blocks))

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(1 for blk in new_computed_blocks
                                            if blk.ref_cnt == 0)
        if (num_new_blocks > self.block_pool.get_num_free_blocks() - num_evictable_computed_blocks):
            if request.status == RequestStatus.RUNNING:
                logger.info(
                    ("PREEMPTING RUNNING REQ %s: Need %d blocks but only %d free. "),
                    request.request_id, num_new_blocks, self.block_pool.get_num_free_blocks() - num_evictable_computed_blocks
                )
            return None, None, None
        
        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not new_computed_blocks, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        req_blocks.extend(new_computed_blocks)

        # Start to handle new blocks

        if num_new_blocks <= 0:
            # No new block is needed.
            new_blocks = []
        else:
            # Get new blocks from the free block pool considering
            # preallocated blocks.
            num_new_blocks = min(
                num_new_blocks + self.num_preallocate_blocks,
                self.block_pool.get_num_free_blocks(),
                # Should not exceed the maximum number of blocks per request.
                # This is especially because the block table has the shape
                # [..., max_num_blocks_per_req].
                self.max_num_blocks_per_req - len(req_blocks),
            )

            assert num_new_blocks > 0

            # Concatenate the computed block IDs and the new block IDs.
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)

        if not self.enable_caching:
            return [new_blocks], [[]], [[]]

        # Use `new_computed_blocks` for a new request, and `num_cached_block`
        # for a running request.
        num_cached_blocks = self.num_cached_block.get(request.request_id,
                                                      len(new_computed_blocks))
        # Speculated tokens might be rejected in the future, so we does
        # not cache any speculated tokens. We only cache blocks with
        # generated (accepted) tokens.
        num_full_blocks_after_append = (num_computed_tokens + num_tokens - len(
            request.spec_token_ids)) // self.block_size

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=req_blocks,
            block_hashes=self.req_to_block_hashes[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks_after_append,
            block_size=self.block_size,
        )

        self.num_cached_block[
            request.request_id] = num_full_blocks_after_append
        return [new_blocks], [[]], [[]]

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        if self.enable_flexicache:
            # free KV blocks per layer
            if request.request_id in self.pending_free:
                self.commit_pending_free({request.request_id})

            for L in range(self.num_layers):
                blocks = self.req_to_blocks_by_layer[L].pop(request.request_id, [])
                self.block_pools[L].free_blocks(blocks)

                cpu_blocks = self.req_to_cpu_blocks_by_layer[L].pop(request.request_id, [])
                self.block_pools[L].free_cpu_blocks(cpu_blocks)

                mm_blocks = self.req_to_minmax_blocks_by_layer[L].pop(request.request_id, [])
                self.minmax_block_pools[L].free_blocks(mm_blocks)
        else:
            # Default to [] in case a request is freed (aborted) before alloc.
            blocks = self.req_to_blocks.pop(request.request_id, [])
            ordered_blocks: Iterable[KVCacheBlock] = blocks
            if self.enable_caching:
                # Free blocks in reverse order so that the tail blocks are
                # freed first.
                ordered_blocks = reversed(blocks)

            self.block_pool.free_blocks(ordered_blocks)
            self.num_cached_block.pop(request.request_id, None)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if self.block_pool.reset_prefix_cache():
            self.prefix_cache_stats.reset = True
            return True
        return False

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> int:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            int: The number of common prefix blocks.
        """
        assert request.status == RequestStatus.RUNNING
        blocks = self.req_to_blocks[request.request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == num_running_requests:
                num_common_blocks += 1
            else:
                break
        return num_common_blocks

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_block_hashes.pop(request.request_id, None)

    def restrict_to_topk(
        self, req_id: str, topk_by_layer_head: list[list[list[int]]], n_full_logical: int,
    ) -> None:
        """
        For each layer and each *stable* head h, free all GPU blocks that lie in
        logical positions j in [0, n_full_logical) where j is NOT in the Top-K
        set for that layer and head. Freed slots in the per-request block list are replaced
        with a guard (null) block.
        """
        assert len(topk_by_layer_head) == self.num_layers, f"Unexpected number of layers: {len(topk_by_layer_head)} != {self.num_layers}"

        pending_free = []
        for L in range(self.num_layers):
            blocks_L = self.req_to_blocks_by_layer[L].get(req_id)
            if not blocks_L:
                continue

            guard = self.block_pools[L].free_block_queue.null_block

            stable_heads: set[int] = self.layer_to_stable_heads[L]
            if not stable_heads:
                continue

            to_free: list[KVCacheBlock] = []

            topk_heads = topk_by_layer_head[L]
            assert len(topk_heads) == self.num_kv_heads, f"Unexpected number of heads: {len(topk_heads)} != {self.num_kv_heads}"

            for h in range(len(topk_heads)):
                if h not in stable_heads:
                    continue
                # ASSUMPTION: topk_heads[h] is sorted ascending and unique.
                sel_list = topk_heads[h]
                H = self.num_kv_heads

                def _free_range(start_log: int, end_log: int) -> None:
                    # Free logical blocks in [start_log, end_log) for head h via strided slicing.
                    if start_log >= end_log:
                        return
                    i0 = start_log * H + h
                    i1 = end_log   * H + h
                    seg = blocks_L[i0:i1:H]          # blocks to free for this head in one shot
                    to_free.extend(seg)
                    blocks_L[i0:i1:H] = [guard] * len(seg)

                # Sweep complement of sel_list over [0, n_full_logical).
                cur = 0
                for keep_log in sel_list:
                    # free gap before the kept logical index
                    _free_range(cur, min(keep_log, n_full_logical))
                    # advance past the kept index
                    if keep_log < n_full_logical:
                        cur = keep_log + 1
                # tail gap
                _free_range(cur, n_full_logical)

            pending_free.append(to_free)
        
        if pending_free:
            self.pending_free[req_id] = pending_free

    def free_pages_decode_phase(self, req_id_seq_len: list[tuple[str, int]]) -> None:
        block_pools = self.block_pools
        req_to_blocks_by_layer = self.req_to_blocks_by_layer
        layer_to_stable_heads  = self.layer_to_stable_heads
        rerank_freq            = self.rerank_freq

        num_layers, H, B = self.num_layers, self.num_kv_heads, self.block_size

        to_free: list[list[KVCacheBlock]] = [[] for _ in range(num_layers)]
        for req_id, seq_len in req_id_seq_len:
            # Minus 1 because here in the decode phase, the last token has just been
            # generated and its KV cache is not generated yet
            seq_len -= 1
            start_log = cdiv(seq_len - rerank_freq, B)
            end_log   = cdiv(seq_len, B) - 1

            for L in range(num_layers):
                stable_heads: set[int] = layer_to_stable_heads[L]
                if not stable_heads:
                    continue

                blocks_L = req_to_blocks_by_layer[L].get(req_id)

                if blocks_L is None:
                    continue

                guard = block_pools[L].free_block_queue.null_block

                if start_log == end_log:
                    base = start_log * H
                    for h in stable_heads:
                        idx = base + h
                        if idx >= len(blocks_L):
                            print(f'SEQ LEN {seq_len} L {L} H {h} IDX {idx} BLEN {len(blocks_L)} START {start_log} END {end_log}')
                            assert False, "Index out of range"
                        blk = blocks_L[idx]
                        to_free[L].append(blk)
                        blocks_L[idx] = guard
                else:
                    for h in stable_heads:
                        i0 = start_log * H + h
                        stop = (end_log + 1) * H + h
                        seg_all = blocks_L[i0:stop:H]
                        to_free[L].extend(seg_all)
                        blocks_L[i0:stop:H] = [guard] * len(seg_all)

        for L in range(num_layers):
            if to_free[L]:
                block_pools[L].free_blocks(to_free[L], guard_check=True)


    def commit_pending_free(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            pending_free = self.pending_free.pop(req_id, None)
            if pending_free:
                for layer_idx, layer_blocks in enumerate(pending_free):
                    if layer_blocks:
                        self.block_pools[layer_idx].free_blocks(layer_blocks, guard_check=True)