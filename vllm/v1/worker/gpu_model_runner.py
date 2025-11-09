# SPDX-License-Identifier: Apache-2.0

import gc
import time
import math
import os
import importlib.util
import weakref
from typing import TYPE_CHECKING, Optional, Union
from collections import deque

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from torch.utils.cpp_extension import load as load_ext

from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.layer import Attention
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed.parallel_state import get_pp_group, graph_capture
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        LayerBlockType, LazyLoader, cdiv,
                        is_pin_memory_available)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.utils import is_spec_decode_supported
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch, PausedInputs
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
import vllm.v1.flexicache.config as FCC
from vllm.v1.flexicache.config import FlexiCacheConfig
from vllm.attention.ops.flexi_cache_triton_kernels import (
    store_minmax_key_cache_for_prefill, store_minmax_key_cache_for_decode, slot_map_kernel
)

if TYPE_CHECKING:
    import xgrammar as xgr

    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


class GPUModelRunner(LoRAModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        main_stream: torch.cuda.Stream,
        hi_priority: int,
        low_priority: int
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        # NOTE(woosuk): sliding_window is None for models with interleaved
        # attention. Use interleaved_sliding_window instead.
        self.sliding_window = model_config.get_sliding_window()
        self.interleaved_sliding_window = getattr(
            model_config.hf_text_config, "interleaved_sliding_window", None)
        self.window_size = (self.sliding_window
                            or self.interleaved_sliding_window)

        self.is_multimodal_model = model_config.is_multimodal_model
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_logical_blks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        self.enable_flexicache = cache_config.enable_flexicache

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()

        self.attn_backend = get_attn_backend(
            self.head_size,
            self.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
            use_mla=self.model_config.use_mla,
        )
        if self.attn_backend is None:
            error_msg = (
                f"Error with get_att_backend: {self.head_size=}, "
                f"{self.dtype=}, {self.kv_cache_dtype=}, {self.block_size=}, "
                f"{self.model_config.is_attention_free=}, "
                f"{self.model_config.use_mla=}")
            logger.error(error_msg)
            raise NotImplementedError(
                "Non-Attention backend is not supported by V1 GPUModelRunner.")

        self.attn_metadata_builder = self.attn_backend.get_builder_cls()(
            weakref.proxy(self))
        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn

        # Multi-modal data support
        self.input_registry = INPUT_REGISTRY
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
        )
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: list[torch.Tensor]     = []
        self.cpu_kv_caches: list[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        self.minmax_key_cache: torch.Tensor = None

        # Set up speculative decoding.
        self.use_spec_decode = False
        if self.speculative_config:
            self.use_spec_decode = True
            assert self.speculative_config.method == "ngram", \
                    "Currently, only ngram spec decode is supported in V1."
            if get_pp_group().is_last_rank:
                self.drafter = NgramProposer()
                # Trigger Numba JIT compilation for N-gram proposer.
                # This usually takes less than 1 second.
                self.drafter.propose(
                    np.zeros(1024, dtype=np.int32),
                    self.speculative_config.prompt_lookup_min,
                    self.speculative_config.prompt_lookup_max,
                    self.speculative_config.num_speculative_tokens,
                )
                self.rejection_sampler = RejectionSampler()

        if self.enable_flexicache:
            self.max_filtered_blocks = FlexiCacheConfig.top_k_page_budget
            self.rank_frequency      = FlexiCacheConfig.rerank_frequency
            slot_mapping_shape = (self.num_attn_layers, self.max_num_tokens, self.num_kv_heads)
            slot_mapping_dtype = torch.int64
        else:
            slot_mapping_shape = (self.max_num_tokens,)
            slot_mapping_dtype = torch.int32

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_logical_blks_per_req=self.max_logical_blks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=model_config.get_vocab_size(),
            enable_flexicache=self.enable_flexicache,
            num_kv_heads=self.num_kv_heads,
            num_query_heads=self.num_query_heads,
            max_filtered_blocks=self.max_filtered_blocks if self.enable_flexicache else -1,
            num_attn_layers=self.num_attn_layers,
            block_size=self.block_size,
        )

        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE
                               and not self.model_config.enforce_eager)
        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        # The convention is different.
        # self.cudagraph_batch_sizes sorts in ascending order.
        # The batch sizes in the config are in descending order.
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        # Cache the device properties.
        self.device_properties = torch.cuda.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

        # Persistent buffers for CUDA graphs.
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.logical_indices = torch.zeros(self.max_num_tokens,
                                           dtype=torch.int32,
                                           device=self.device)
        self.offsets = torch.zeros(self.max_num_tokens,
                                   dtype=torch.int32,
                                   device=self.device)
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                               dtype=torch.int64,
                                               device=self.device)
            self.mrope_positions_cpu = torch.zeros(
                (3, self.max_num_tokens + 1),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory)

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device)

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        self.arange_np = np.arange(max(self.max_num_reqs + 1,
                                       self.max_model_len,
                                       self.max_num_tokens),
                                   dtype=np.int32)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.input_ids_np = self.input_ids_cpu.numpy()
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_np = self.positions_cpu.numpy()
        self.slot_mapping_cpu = torch.zeros(slot_mapping_shape,
                                            dtype=slot_mapping_dtype,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.slot_mapping_gpu = torch.zeros(slot_mapping_shape,
                                            dtype=slot_mapping_dtype,
                                            device=self.device)
        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        if self.enable_flexicache:
            self.main_stream = main_stream
            self.background_stream: torch.cuda.Stream = torch.cuda.Stream(device=self.device, priority=low_priority)
            self.gpu_kv_cache_pool: Optional[torch.Tensor]   = None   # (2, total_blocks_across_layers, block_size, head_size)
            self.gpu_layer_first_blk: Optional[torch.Tensor] = None   # [L] int64
            self.gpu_layer_nblk: Optional[torch.Tensor]      = None   # [L] int32

            self.cpu_kv_cache_pool: Optional[torch.Tensor]   = None   # (2, cpu_total_blocks_across_layers, block_size, head_size)
            self.cpu_layer_first_blk: Optional[torch.Tensor] = None   # [L] int64
            self.cpu_layer_nblk: Optional[torch.Tensor]      = None   # [L] int32

            self.layer_gpu_offsets: Optional[torch.Tensor] = None   # [L+1] int64
            self.layer_cpu_offsets: Optional[torch.Tensor] = None   # [L+1] int64

            flexicache_cuda = importlib.util.find_spec("vllm.v1.flexicache.kernels.cuda")
            assert flexicache_cuda is not None, "vllm.v1.flexicache.kernels.cuda not found"
            base_path = os.path.dirname(flexicache_cuda.origin)

            bt_tx_path = os.path.join(base_path, "block_table_h2d_dirty.cu")
            self._bt_tx = load_ext(
                name="block_table_h2d_dirty",
                sources=[bt_tx_path],
                extra_cuda_cflags=["-O3"],
                extra_cflags=["-O3"],
                verbose=False,
            )

            kv_d2h_tx_path = os.path.join(base_path, "kv_d2h_tx_gpu_mapped.cu")
            self._kv_d2h_tx = load_ext(
                name="kv_d2h_tx_gpu_mapped",
                sources=[kv_d2h_tx_path],
                extra_cuda_cflags=["-O3"],
                extra_cflags=["-O3"],
                verbose=False,
            )

            kv_h2d_tx_path = os.path.join(base_path, "h2d_copy_blocks.cu")
            self._kv_h2d_tx = load_ext(
                name="h2d_copy_blocks",
                sources=[kv_h2d_tx_path],
                extra_cuda_cflags=["-O3"],
                extra_cflags=["-O3"],
                verbose=False,
            )

            topk_swap_map_path = os.path.join(base_path, "topk_swap_map.cu")
            self._topk_swap_map = load_ext(
                name="topk_swap_map",
                sources=[topk_swap_map_path],
                extra_cuda_cflags=["-O3"],
                extra_cflags=["-O3"],
                verbose=False,
            )

            # Small persistent scratch for the tiny dirty-range tensors
            self._dirty_gpu_kv  = torch.empty((self.max_num_reqs, 2), dtype=torch.int32, device=self.device)
            self._dirty_cpu_kv  = torch.empty((self.max_num_reqs, 2), dtype=torch.int32, device=self.device)
            self._dirty_gpu_mmb = torch.empty((self.max_num_reqs, 2), dtype=torch.int32, device=self.device)

            self._kv_reload_queue: deque[tuple[str, torch.cuda.Event]] = deque()

            self.needs_rerank: list[int]    = []
            self.is_first_decode: list[int] = []
            self.resumed_from_preemption: set[str] = set()

            self._idx_prefill = np.empty(self.max_num_reqs, dtype=np.int32)
            self._idx_decode  = np.empty(self.max_num_reqs, dtype=np.int32)
            self._idx_d2h     = np.empty(self.max_num_reqs, dtype=np.int32)
            self._idx_first   = np.empty(self.max_num_reqs, dtype=np.int32)
            self._idx_rerank  = np.empty(self.max_num_reqs, dtype=np.int32)

            self._prompt_buf  = np.empty(self.max_num_reqs, dtype=np.int32)  # per-selected prefill prompts
            self._page_buf    = np.empty(self.max_num_reqs, dtype=np.int32)  # per-selected decode new-page

            self.paused_inputs = PausedInputs(
                max_paused_reqs          = self.max_num_reqs,
                device                   = self.device,
                enable_flexicache        = self.enable_flexicache,
                num_kv_heads             = self.num_kv_heads,
                max_filtered_blocks      = self.max_filtered_blocks,
                num_attn_layers          = self.num_attn_layers,
                max_logical_blks_per_req = self.max_logical_blks_per_req,
                max_model_len            = self.max_model_len,
                block_size               = self.block_size
            )

            self._prompt_offload_queue: deque[tuple[str, torch.cuda.Event]] = deque()

            assert FCC.IS_INITIALIZED, "FlexiCacheConfig is not initialized properly."
            self.minmax_key_cache_block_size = FCC.MINMAX_KEY_CACHE_BLOCK_SIZE
            self.unstable_head_masks         = FCC.UNSTABLE_HEAD_MASKS

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        if self.enable_flexicache:
            need_to_discard = (
                (scheduler_output.preempted_req_ids | scheduler_output.finished_req_ids)
                & set(self.paused_inputs.req_to_slot.keys())
            )
            if need_to_discard:
                self.background_stream.synchronize()
                for req_id in need_to_discard:
                    self.paused_inputs.discard(req_id)
            self.resumed_from_preemption.clear()

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            if self.enable_flexicache and req_id not in scheduler_output.preempted_req_ids:
                self.paused_inputs.save(req_id, self.input_batch)
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt=new_req_data.prompt,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids_by_layer=new_req_data.block_ids_by_layer,
                cpu_block_ids_by_layer=new_req_data.cpu_block_ids_by_layer,
                minmax_block_ids_by_layer=new_req_data.minmax_block_ids_by_layer,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.extend(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.extend(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.extend(
                            mm_input["second_per_grid_ts"])

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec decode tokens.
            num_new_tokens = (num_computed_tokens +
                              len(req_data.new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(req_data.new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    req_data.new_token_ids[-num_new_tokens:])
            # Update the block IDs.
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                if self.enable_flexicache:
                    for l in range(self.num_attn_layers):
                        new_blocks     = req_data.new_block_ids_by_layer[l]
                        new_minmax     = req_data.new_minmax_block_ids_by_layer[l]
                        new_cpu_blocks = req_data.new_cpu_block_ids_by_layer[l]

                        if new_blocks:
                            req_state.block_ids_by_layer[l].extend(new_blocks)
                        if new_minmax:
                            req_state.minmax_block_ids_by_layer[l].extend(new_minmax)
                        if new_cpu_blocks:
                            req_state.cpu_block_ids_by_layer[l].extend(new_cpu_blocks)
                else:
                    req_state.block_ids_by_layer[0].extend(req_data.new_block_ids_by_layer[0])
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                if self.enable_flexicache:
                    self.resumed_from_preemption.add(req_id)
                    for l in range(self.num_attn_layers):
                        req_state.block_ids_by_layer[l]        = req_data.new_block_ids_by_layer[l]
                        req_state.minmax_block_ids_by_layer[l] = req_data.new_minmax_block_ids_by_layer[l]
                        req_state.cpu_block_ids_by_layer[l]    = req_data.new_cpu_block_ids_by_layer[l]
                else:
                    req_state.block_ids_by_layer[0] = req_data.new_block_ids_by_layer[0]

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            self.input_batch.block_table.append_row(
                req_data.new_block_ids_by_layer, req_index
            )
            if self.enable_flexicache:
                self.input_batch.minmax_block_table.append_row(
                    req_data.new_minmax_block_ids_by_layer, req_index
                )
                self.input_batch.cpu_block_table.append_row(
                    req_data.new_cpu_block_ids_by_layer, req_index
                )
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(req_data.new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index,
                start_token_index:end_token_index] = req_data.new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index
            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ())
            if spec_token_ids:
                start_index = end_token_index
                end_token_index += len(spec_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
            self.input_batch.num_tokens[req_index] = end_token_index

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            if self.enable_flexicache:
                is_paused = self.paused_inputs.is_paused(req_id)
                self.input_batch.add_request(req_state, req_index, is_paused)
                if is_paused:
                    self.paused_inputs.restore(req_id, self.input_batch, req_state)
            else:
                self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices, scheduler_output)

        if batch_changed:
            self.input_batch.refresh_sampling_metadata()

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[FlashAttentionMetadata, torch.Tensor,
               Optional[SpecDecodeMetadata]]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Some attention backends (namely MLA) may want to separate requests
        # based on if the attention computation will be compute-bound or
        # memory-bound. This gives them a hook to do that.
        modified_batch = self.attn_metadata_builder.reorder_batch(
            self.input_batch, scheduler_output)
        if modified_batch:
            self.input_batch.refresh_sampling_metadata()

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        if self.enable_flexicache:
            self.commit_block_table(num_reqs)
            self.input_batch.block_scores[:, :num_reqs, :, :].fill_(float('-inf'))
        else:
            self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
        max_num_scheduled_tokens = 0

        if self.enable_flexicache:
            self.needs_rerank.clear()
            self.is_first_decode.clear()

        for i, req_id in enumerate(self.input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens[i] = num_tokens
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)
            seq_len = self.input_batch.num_computed_tokens_cpu[i] + num_tokens
            self.seq_lens_np[i] = seq_len
            if self.enable_flexicache:
                num_decode_step = seq_len - self.input_batch.num_prompt_tokens[i]
                self.input_batch.num_decode_step_np[i] = num_decode_step
                if num_decode_step > 1 and (num_decode_step % self.rank_frequency == 0
                                            or req_id in self.resumed_from_preemption):
                    self.needs_rerank.append(i)
                if num_decode_step == 1:
                    self.input_batch.last_ranked_num_blks_np[i] = math.ceil(seq_len / self.block_size)
                    self.is_first_decode.append(i)

        if self.enable_flexicache:
            needs_rerank_gpu    = torch.tensor(self.needs_rerank, device=self.device, dtype=torch.int64)
            is_first_decode_gpu = torch.tensor(self.is_first_decode, device=self.device, dtype=torch.int64)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_scheduled_tokens])
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                    num_scheduled_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Calculate the slot mapping.
        if self.enable_flexicache:
            num_scheduled_tokens_t = torch.from_numpy(num_scheduled_tokens).to(
                device=self.device, dtype=torch.int32, non_blocking=True
            )
            self.positions[:total_num_scheduled_tokens].copy_(
                self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True
            )

            pos_gpu         = self.positions[:total_num_scheduled_tokens]
            logical_idx_gpu = self.logical_indices[:total_num_scheduled_tokens]
            offs_gpu        = self.offsets[:total_num_scheduled_tokens]

            req_indices_gpu = torch.repeat_interleave(
                torch.arange(self.input_batch.num_reqs, device=self.device, dtype=torch.int32),
                num_scheduled_tokens_t
            )                                                   # int32 [T]

            torch.div(pos_gpu, self.block_size, rounding_mode="floor", out=logical_idx_gpu)
            torch.remainder(pos_gpu, self.block_size, out=offs_gpu)

            bt = self.input_batch.block_table.get_device_tensor()    # [R, L, H, Lb], int32
            out = self.slot_mapping_gpu                              # [L, T_max, H], int64

            if total_num_scheduled_tokens <= self.max_num_reqs: # All decode
                phys = bt[req_indices_gpu.long(), :, :, logical_idx_gpu.long()]
                slots = (phys.to(torch.int64) * self.block_size
                        + offs_gpu.view(-1, 1, 1).to(torch.int64)).permute(1, 0, 2)
                out[:, :total_num_scheduled_tokens, :].copy_(slots, non_blocking=True)
            else:
                BLOCK_T = 256
                grid = (self.num_attn_layers,
                        self.num_kv_heads,
                        (total_num_scheduled_tokens + BLOCK_T - 1) // BLOCK_T)
                
                stride_bt_bs, stride_bt_ly, stride_bt_kh, stride_bt_bl = bt.stride()
                stride_out_ly, stride_out_tok, stride_out_kh           = out.stride()

                slot_map_kernel[grid](
                    bt, out,
                    req_indices_gpu, logical_idx_gpu, offs_gpu,
                    stride_bt_bs, stride_bt_ly, stride_bt_kh, stride_bt_bl,
                    stride_out_ly, stride_out_tok, stride_out_kh,
                    BLOCK_T=BLOCK_T,
                    NUM_TOKENS=total_num_scheduled_tokens,
                    BLOCK_SIZE=self.block_size,
                    num_warps=4, num_stages=1,
                )
        else:
            # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
            # where K is the max_logical_blks_per_req and the block size is 2.
            # NOTE(woosuk): We can't simply use `token_indices // block_size` here
            # because M (max_model_len) is not necessarily divisible by block_size.
            block_table_indices = (req_indices * self.max_logical_blks_per_req +
                                    positions_np // self.block_size)
            # NOTE(woosuk): We use torch.index_select instead of np.take here
            # because torch.index_select is much faster than np.take for large
            # tensors.
            block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
            block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
            block_offsets = positions_np % self.block_size
            np.add(block_numbers * self.block_size,
                    block_offsets,
                    out=self.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        # Copy the tensors to the GPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            if not self.enable_flexicache:
                # We already copied positions to GPU above for flexicache.
                self.positions[:total_num_scheduled_tokens].copy_(
                    self.positions_cpu[:total_num_scheduled_tokens],
                    non_blocking=True)

        # Prepare for cascade attention if enabled & beneficial.
        common_prefix_len = 0
        if self.cascade_attn_enabled:
            common_prefix_len = self._compute_cascade_attn_prefix_len(
                num_scheduled_tokens,
                scheduler_output.num_common_prefix_blocks,
            )

        if self.enable_flexicache:
            from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
            assert isinstance(self.attn_metadata_builder, FlashAttentionMetadataBuilder)

            self.input_batch.num_decode_step_gpu.copy_(
                self.input_batch.num_decode_step_cpu, non_blocking=True
            )
            self.input_batch.last_ranked_num_blks_gpu.copy_(
                self.input_batch.last_ranked_num_blks_cpu, non_blocking=True
            )

            attn_metadata = self.attn_metadata_builder.build(
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                common_prefix_len=common_prefix_len,
                enable_flexicache=True,
                block_scores=self.input_batch.block_scores,
                num_decode_step=self.input_batch.num_decode_step_gpu,
                max_filtered_blocks=self.max_filtered_blocks,
                rank_frequency=self.rank_frequency,
                top_k_blocks=self.input_batch.top_k_blocks,
                old_top_k_blocks=self.input_batch.old_top_k_blocks,
                needs_rerank_gpu=needs_rerank_gpu,
                is_first_decode_gpu=is_first_decode_gpu,
                last_ranked_num_blks=self.input_batch.last_ranked_num_blks_gpu,
            )
        else:
            attn_metadata = self.attn_metadata_builder.build(
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                common_prefix_len=common_prefix_len,
            )

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = attn_metadata.query_start_loc[1:] - 1
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        return attn_metadata, logits_indices, spec_decode_metadata

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_common_prefix_blocks: int,
    ) -> int:
        """Compute the length of the common prefix for cascade attention.

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """
        common_prefix_len = num_common_prefix_blocks * self.block_size
        if common_prefix_len == 0:
            # Common case.
            return 0

        # NOTE(woosuk): Cascade attention uses two attention kernels: one
        # for the common prefix and the other for the rest. For the first
        # kernel, we concatenate all the query tokens (possibly from
        # different requests) and treat them as if they are from the same
        # request. Then, we use bi-directional attention to process the
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.

        # Consider the following example:
        # Request 1's input query: [D, E, X]
        # Request 1's kv cache: [A, B, C, D, E, X]
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        # Request 2's input query: [E, Y]
        # Request 2's kv cache: [A, B, C, D, E, Y]
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        # If we use [A, B, C, D, E] as the common prefix, then the
        # first kernel will compute the bi-directional attention between
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        # However, this is wrong because D in Request 1 should not attend to
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.
        # That is, the common prefix should be capped by the minimum
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.

        # In practice, we use [A, B, C] as the common prefix, instead of
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).
        # This is because of an implementation detail: We want to always
        # use two kernels for cascade attention. Let's imagine:
        # Request 3's input query: [D]
        # Request 3's kv cache: [A, B, C, D]
        # Request 3's num_computed_tokens: 4 (i.e., [A, B, C, D])
        # If we use [A, B, C, D] as the common prefix for Request 1-3,
        # then Request 3 will be processed only by the first kernel,
        # and the second kernel will get an empty input. While this is not
        # a fundamental problem, our current implementation does not support
        # this case.
        num_reqs = len(num_scheduled_tokens)
        common_prefix_len = min(
            common_prefix_len,
            self.input_batch.num_computed_tokens_cpu[:num_reqs].min())
        # common_prefix_len should be a multiple of the block size.
        common_prefix_len = (common_prefix_len // self.block_size *
                             self.block_size)
        use_cascade = self.attn_backend.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=self.num_kv_heads,
            use_alibi=False,  # FIXME
            use_sliding_window=self.window_size is not None,
            num_sms=self.num_sms,
        )
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = \
                self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = \
                scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = len(req.prompt_token_ids)

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0,
                                      num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(
                    0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions_cpu[:, dst_start:dst_end] = \
                    req.mrope_positions[:,src_start:src_end]

                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                self.mrope_positions_cpu[:, dst_start:dst_end] = \
                    MRotaryEmbedding.get_next_input_positions_tensor(
                        req.mrope_position_delta,
                        context_len=num_computed_tokens +
                        prompt_part_len,
                        seq_len=num_computed_tokens +
                        prompt_part_len +
                        completion_part_len,
                    )

                mrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1
        # Step 1. [4, 5, 8, 9, 11]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        total_num_sampled_tokens = cu_num_sampled_tokens[-1]
        # Step 2. [0, 0, 0, 0, 4, 5, 5, 5, 8, 9, 9]
        cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens,
                                    num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets
        # Step 4. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 5. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # [3, 3, 5, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        total_num_draft_tokens = cu_num_draft_tokens[-1]
        # [0, 0, 0, 3, 3, 5]
        cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens,
                                    num_draft_tokens)
        # [0, 1, 2, 0, 1, 0]
        arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).to(self.device,
                                                             non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
        return metadata

    def _execute_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs: list[MultiModalKwargs] = []
        req_input_ids: list[tuple[str, int]] = []
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]
            for input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[input_id])
                req_input_ids.append((req_id, input_id))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        grouped_mm_inputs_list = group_mm_inputs_by_modality(mm_inputs)

        encoder_outputs = []
        for grouped_mm_inputs in grouped_mm_inputs_list:
            batched_mm_inputs = MultiModalKwargs.batch(grouped_mm_inputs)
            batched_mm_inputs = MultiModalKwargs.as_kwargs(batched_mm_inputs,
                                                           device=self.device)

            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = self.model.get_multimodal_embeddings(
                **batched_mm_inputs)

            for output in curr_group_outputs:
                encoder_outputs.append(output)

        # Cache the encoder outputs.
        for (req_id, input_id), output in zip(req_input_ids, encoder_outputs):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}
            self.encoder_cache[req_id][input_id] = output

    def _gather_encoder_outputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> list[torch.Tensor]:
        encoder_outputs: list[torch.Tensor] = []
        for req_id in self.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_positions = req_state.mm_positions
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info["offset"]
                num_encoder_tokens = pos_info["length"]

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens)
                assert start_idx < end_idx
                assert req_id in self.encoder_cache
                assert i in self.encoder_cache[req_id]
                encoder_output = self.encoder_cache[req_id][i]
                encoder_outputs.append(encoder_output[start_idx:end_idx])
        return encoder_outputs

    def get_model(self) -> nn.Module:
        return self.model

    def apply_grammar_bitmask(
        self,
        scheduler_output: "SchedulerOutput",
        logits: torch.Tensor,
    ):
        # Serialization of np.ndarray is much more efficient than a tensor,
        # so we receive it in that format.
        grammar_bitmask = scheduler_output.grammar_bitmask
        if grammar_bitmask is None:
            return

        # We receive the structured output bitmask from the scheduler, but the
        # indices of the requests in the batch may not match the indices of
        # the bitmask since the scheduler doesn't know how the gpu runner is
        # ordering the requests in the batch. We need to sort the bitmask to
        # match the order of the requests used here.
        struct_out_req_batch_indices: dict[str, int] = {}
        indices_match = True
        for req_id in self.input_batch.req_ids:
            mask_index = scheduler_output.structured_output_request_ids.get(
                req_id)
            if mask_index is None:
                # not a structured output request
                continue
            batch_index = self.input_batch.req_id_to_index[req_id]
            if batch_index != mask_index:
                indices_match = False
            struct_out_req_batch_indices[req_id] = batch_index

        if not indices_match:
            # Sort the bitmask to match the order of the requests
            sorted_bitmask = np.zeros_like(grammar_bitmask)
            for req_id, batch_index in struct_out_req_batch_indices.items():
                orig_index = scheduler_output.structured_output_request_ids[
                    req_id]
                sorted_bitmask[batch_index] = grammar_bitmask[orig_index]
            grammar_bitmask = sorted_bitmask

        grammar_bitmask = torch.from_numpy(grammar_bitmask)

        # TODO: compatibility with spec decode
        xgr.apply_token_bitmask_inplace(
            logits,
            grammar_bitmask.to(self.device, non_blocking=True),
            indices=list(struct_out_req_batch_indices.values()),
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, torch.Tensor]:
        
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOuptut if there's no work to do.
            if self.enable_flexicache:
                output = EMPTY_MODEL_RUNNER_OUTPUT
                output.kv_reload_finished_reqs = self.poll_finished_kv_cache_reloads()
                return output
            else:
                return EMPTY_MODEL_RUNNER_OUTPUT

        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_encoder(scheduler_output)
            encoder_outputs = self._gather_encoder_outputs(scheduler_output)
        else:
            encoder_outputs = []

        # Prepare the decoder inputs.
        attn_metadata, logits_indices, spec_decode_metadata = (
            self._prepare_inputs(scheduler_output))
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            num_input_tokens = num_scheduled_tokens
        attn_metadata.num_input_tokens = num_input_tokens

        if self.is_multimodal_model:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]
            if encoder_outputs:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, encoder_outputs)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[k][:num_input_tokens].copy_(
                    v[:num_input_tokens], non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k: v[:num_input_tokens]
                for k, v in self.intermediate_tensors.items()
            })

        # Run the decoder.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            return hidden_states

        hidden_states = hidden_states[:num_scheduled_tokens]
        sample_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.model.sample(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.model.sample(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        for i, generator in self.input_batch.generators.items():
            req_id = self.input_batch.req_ids[i]
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator.set_offset(generator.get_offset() - 4)

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states,
            scheduler_output,
        )

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids, self.input_batch.vocab_size)

        if not self.use_spec_decode:
            spec_token_ids = None
        else:
            spec_token_ids = self.generate_draft_token_ids(
                valid_sampled_token_ids, sampling_metadata)
            
        if self.enable_flexicache:
            blk = self.block_size
            rank_f = self.rank_frequency

            cnt_prefill = cnt_decode = cnt_d2h = cnt_first = cnt_rerank = 0

            # single pass across requests in current batch order
            for i, rid in enumerate(self.input_batch.req_ids):
                prev   = int(self.requests[rid].num_computed_tokens)
                sched  = int(scheduler_output.num_scheduled_tokens[rid])
                prompt = int(self.input_batch.num_prompt_tokens[i])
                new    = prev + sched

                # prefill just finished (and at least one full page)
                if (prev + sched == prompt) and prompt > 0 and (prompt // blk) > 0:
                    self._idx_prefill[cnt_prefill] = i
                    self._prompt_buf[cnt_prefill]  = prompt
                    cnt_prefill += 1

                # decode: a full page became complete now
                if (sched == 1) and (new != prompt) and (new % blk == 0):
                    self._idx_decode[cnt_decode] = i
                    self._page_buf[cnt_decode]   = (new // blk) - 1
                    cnt_decode += 1

                self.input_batch.last_tx_blk_np[i]    = (prev // blk)
                self.input_batch.total_full_blk_np[i] = (new  // blk)
                # offload: full-page count increased
                if self.input_batch.total_full_blk_np[i] > self.input_batch.last_tx_blk_np[i]:
                    self._idx_d2h[cnt_d2h] = i
                    cnt_d2h += 1

                # first decode / rerank bookkeeping
                dec_step = new - prompt
                if dec_step == 1:
                    self.input_batch.last_ranked_num_blks_np[i] = (new + blk - 1) // blk
                    self._idx_first[cnt_first] = i
                    cnt_first += 1
                elif dec_step > 1 and (dec_step % rank_f == 0):
                    self.input_batch.last_ranked_num_blks_np[i] = (new + blk - 1) // blk
                    self._idx_rerank[cnt_rerank] = i
                    cnt_rerank += 1

            # launch kernels with precomputed selections
            if cnt_prefill:
                self.launch_store_minmax_key_for_prefill_selected(
                    self._idx_prefill[:cnt_prefill], self._prompt_buf[:cnt_prefill]
                )
            if cnt_decode:
                self.launch_store_minmax_key_for_decode_selected(
                    self._idx_decode[:cnt_decode], self._page_buf[:cnt_decode]
                )

            prompt_offload_finished = set()
            # so stop at the first offload that isn't done yet.
            queue = self._prompt_offload_queue
            while queue and queue[0][1].query():
                rid, _ = queue.popleft()
                prompt_offload_finished.add(rid)
            if cnt_d2h:
                self.offload_kv_cache_d2h_selected(self._idx_d2h[:cnt_d2h], cnt_prefill)

            # scheduler payloads (first decode only)
            topk_blocks_by_req      = {}
            ranked_n_logical_by_req = {}
            for j in self._idx_first[:cnt_first].tolist():
                rid = self.input_batch.req_ids[j]
                ranked_n_logical_by_req[rid] = int(self.input_batch.last_ranked_num_blks_np[j])
                topk_blocks_by_req[rid] = torch.sort(
                    self.input_batch.top_k_blocks[:, j, :, :], dim=-1
                ).values.to(torch.int16).cpu().tolist()

            # reload bookkeeping
            self.needs_rerank    = self._idx_rerank[:cnt_rerank].tolist()
            self.is_first_decode = self._idx_first [:cnt_first ].tolist()
            kv_reload_finished = self.poll_finished_kv_cache_reloads()
            kv_reload_started  = self.reload_kv_cache_h2d()
        else:
            topk_blocks_by_req, ranked_n_logical_by_req = {}, {}
            kv_reload_started, kv_reload_finished = set(), set()
            prompt_offload_finished = set()
            
        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,

            topk_blocks_by_req      = topk_blocks_by_req,
            ranked_n_logical_by_req = ranked_n_logical_by_req,
            prompt_offload_finished = prompt_offload_finished,
            kv_reload_started_reqs  = kv_reload_started,
            kv_reload_finished_reqs = kv_reload_finished,
        )

    def generate_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
    ) -> list[list[int]]:
        # TODO(woosuk): Optimize.
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                draft_token_ids.append([])
                continue

            # Skip requests that require top-p, top-k, etc.
            req_id = self.input_batch.req_ids[i]
            if not is_spec_decode_supported(req_id, self.input_batch):
                draft_token_ids.append([])
                continue

            # Add sampled_token_ids to token_ids_cpu.
            start_idx = self.input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + num_sampled_ids
            self.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids
            drafter_output = self.drafter.propose(
                self.input_batch.token_ids_cpu[i, :end_idx],
                self.speculative_config.prompt_lookup_min,
                self.speculative_config.prompt_lookup_max,
                self.speculative_config.num_speculative_tokens,
            )
            if drafter_output is None or len(drafter_output) == 0:
                draft_token_ids.append([])
            else:
                draft_token_ids.append(drafter_output.tolist())
        return draft_token_ids

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            time_before_load = time.perf_counter()
            self.model = get_model(vllm_config=self.vllm_config)
            if self.lora_config:
                self.model = self.load_lora_model(self.model,
                                                  self.model_config,
                                                  self.scheduler_config,
                                                  self.lora_config,
                                                  self.device)
            time_after_load = time.perf_counter()
        self.model_memory_usage = m.consumed_memory
        logger.info("Model loading took %.4f GB and %.6f seconds",
                    self.model_memory_usage / float(2**30),
                    time_after_load - time_before_load)

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, Optional[LogprobsTensors]]:
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():

            num_tokens = scheduler_output.num_scheduled_tokens[req_id]

            # Get metadata for this request.
            request = self.requests[req_id]
            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True)

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1)
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc_np[req_idx].item()
            prompt_hidden_states = hidden_states[offset:offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states, None)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.model.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.model.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids)

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs,
                                                         non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True)

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            torch.cuda.synchronize()

        return prompt_logprobs_dict

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
    ) -> torch.Tensor:

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = max_num_reqs if num_tokens >= max_num_reqs else num_tokens
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens):
            model = self.model
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None
            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device))
                intermediate_tensors = IntermediateTensors({
                    k: v[:num_tokens]
                    for k, v in self.intermediate_tensors.items()
                })

            with set_forward_context(None,
                                     self.vllm_config,
                                     num_tokens=num_tokens):
                hidden_states = model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return hidden_states[logit_indices]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        logits = self.model.compute_logits(hidden_states, None)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full(
            (num_reqs, ), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            min_p=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            min_tokens={},
            logit_bias=[None for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
        )
        try:
            sampler_output = self.model.sample(
                logits=logits, sampling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e
        if self.use_spec_decode:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device)

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            # draft_probs = torch.randn(
            #     num_tokens, logits.shape[-1], device=self.device,
            #     dtype=logits.dtype)
            draft_probs = None
            target_logits = torch.randn(num_tokens,
                                        logits.shape[-1],
                                        device=self.device,
                                        dtype=logits.dtype)
            # NOTE(woosuk): Here, we should use int32 because the sampler uses
            # int32 for bonus_token_ids. If the dtype mismatches, re-compilation
            # will occur at runtime.
            bonus_token_ids = torch.zeros(num_reqs,
                                          device=self.device,
                                          dtype=torch.int32)
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                target_logits,
                bonus_token_ids,
                dummy_metadata,
            )
        return sampler_output

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        # TODO: handle encoder-decoder models once we support them.
        if (self.is_multimodal_model and self.max_num_encoder_input_tokens > 0
                and self.encoder_cache_size > 0):

            # NOTE: Currently model is profiled with a single non-text
            # modality with the max possible input tokens even when
            # it supports multiple.
            max_tokens_by_modality_dict = (
                MULTIMODAL_REGISTRY.
                get_max_tokens_per_item_by_nonzero_modality(self.model_config))
            dummy_data_modality, max_tokens_per_mm_item = max(
                max_tokens_by_modality_dict.items(), key=lambda item: item[1])

            # Check how many items of this modality can be supported by
            # the encoder budget.
            encoder_budget = min(self.max_num_encoder_input_tokens,
                                 self.encoder_cache_size)

            max_num_mm_items_encoder_budget = cdiv(encoder_budget,
                                                   max_tokens_per_mm_item)

            # Check how many items of this modality can be supported by
            # the decoder budget.
            max_mm_items_per_req = self.mm_registry.get_mm_limits_per_prompt(
                self.model_config)[dummy_data_modality]

            # NOTE: We do not consider max_num_batched_tokens on purpose
            # because the multimodal embeddings can be generated in advance
            # and chunked prefilled.
            max_num_mm_items_decoder_budget = self.max_num_reqs * \
                max_mm_items_per_req

            max_num_mm_items = min(max_num_mm_items_encoder_budget,
                                   max_num_mm_items_decoder_budget)

            logger.info(
                "Encoder cache will be initialized with a budget of %s tokens,"
                " and profiled with %s %s items of the maximum feature size.",
                encoder_budget, max_num_mm_items, dummy_data_modality)

            # Create dummy batch of multimodal inputs.
            dummy_request_data = self.input_registry.dummy_data_for_profiling(
                model_config=self.model_config,
                seq_len=self.max_num_tokens,
                mm_registry=self.mm_registry,
            )
            dummy_mm_data = dummy_request_data.multi_modal_data
            if not isinstance(dummy_mm_data, MultiModalKwargs):
                # TODO: Delete this check once input mapper is fully removed.
                raise RuntimeError(
                    "Legacy input mapper is not supported in V1")

            # Dummy data definition may contain multiple multimodal items
            # (e.g, multiple images) for a single request, therefore here we
            # always replicate first item by max_num_mm_items times since in V1
            # they are scheduled to be processed separately.
            dummy_mm_item = dummy_mm_data.get_item(
                modality=dummy_data_modality, item_index=0)
            dummy_mm_kwargs = MultiModalKwargs.from_items([dummy_mm_item])

            batched_dummy_mm_inputs = MultiModalKwargs.batch(
                [dummy_mm_kwargs] * max_num_mm_items)
            batched_dummy_mm_inputs = MultiModalKwargs.as_kwargs(
                batched_dummy_mm_inputs, device=self.device)

            # Run multimodal encoder.
            dummy_encoder_outputs = self.model.get_multimodal_embeddings(
                **batched_dummy_mm_inputs)
            assert len(dummy_encoder_outputs) == max_num_mm_items, (
                "Expected dimension 0 of encoder outputs to match the number "
                f"of multimodal data items: {max_num_mm_items}, got "
                f"{len(dummy_encoder_outputs)=} instead. This is most likely "
                "due to the 'get_multimodal_embeddings' method of the model "
                "not implemented correctly.")

            # Cache the dummy encoder outputs.
            self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

        hidden_states = self._dummy_run(self.max_num_tokens)
        if get_pp_group().is_last_rank:
            sampler_output = self._dummy_sampler_run(hidden_states)
        else:
            sampler_output = None
        torch.cuda.synchronize()
        del hidden_states, sampler_output
        self.encoder_cache.clear()
        gc.collect()

    def capture_model(self) -> None:
        if not self.use_cuda_graph:
            logger.warning(
                "Skipping CUDA graph capture. Please add "
                "-O %s to use CUDA graphs.", CompilationLevel.PIECEWISE)
            return

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with graph_capture(device=self.device):
            for num_tokens in reversed(self.cudagraph_batch_sizes):
                for _ in range(self.vllm_config.compilation_config.
                               cudagraph_num_of_warmups):
                    self._dummy_run(num_tokens)
                self._dummy_run(num_tokens)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_caches: dict[str, torch.Tensor]     = {}
        cpu_kv_caches: dict[str, torch.Tensor] = {}

        if self.enable_flexicache:
            assert len(kv_cache_config.kv_cache_groups) == 1
            kv_group      = kv_cache_config.kv_cache_groups[0]
            kv_cache_spec = kv_group.kv_cache_spec

            # TODO: (TAKBIR) It is a hacky solution to use FullAttentionSpec Ideally, should
            # define a new FlexiCacheAttentionSpec. Will work on refactoring this later
            assert isinstance(kv_cache_spec, FullAttentionSpec)
            
            layer_names    = kv_group.layer_names
            kv_tensors     = kv_cache_config.tensors
            cpu_kv_tensors = kv_cache_config.cpu_tensors
            minmax_tensors = kv_cache_config.minmax_tensors
            dtype          = kv_cache_spec.dtype

            # Derive per-layer block counts
            gpu_layer_nblk_list: list[int] = []
            cpu_layer_nblk_list: list[int] = []
            mm_layer_nblk: int | None      = None
            for layer_name in layer_names:
                gpu_tensor_config = kv_tensors[layer_name]
                cpu_tensor_config = cpu_kv_tensors[layer_name]
                mm_tensor_config  = minmax_tensors[layer_name]

                assert gpu_tensor_config.size % kv_cache_spec.page_size_bytes == 0
                assert cpu_tensor_config.size % kv_cache_spec.page_size_bytes == 0
                assert mm_tensor_config.size  % (FlexiCacheConfig.minmax_key_cache_block_size * 2 * kv_cache_spec.head_size) == 0

                gpu_nblk_L = gpu_tensor_config.size // kv_cache_spec.page_size_bytes
                cpu_nblk_L = cpu_tensor_config.size // kv_cache_spec.page_size_bytes
                mm_nblk_L  = mm_tensor_config.size  // kv_cache_spec.minmax_page_size_bytes

                gpu_layer_nblk_list.append(int(gpu_nblk_L))
                cpu_layer_nblk_list.append(int(cpu_nblk_L))

                if mm_layer_nblk is None:
                    mm_layer_nblk = int(mm_nblk_L)
                else:
                    assert mm_layer_nblk == int(mm_nblk_L), \
                        f"All layers should have the same number of minmax blocks, got {mm_layer_nblk} and {int(mm_nblk_L)}"

            assert len(gpu_layer_nblk_list) == len(cpu_layer_nblk_list) == self.num_attn_layers

            # Allocate KV cache pools
            gpu_layer_nblk      = np.asarray(gpu_layer_nblk_list, dtype=np.int64)           # [L]
            gpu_layer_first_blk = np.concatenate(([0], np.cumsum(gpu_layer_nblk)[:-1]))     # [L]
            gpu_total_blocks    = int(gpu_layer_nblk.sum())

            if FlexiCacheConfig.num_unstable_heads > 0:
                assert gpu_total_blocks == kv_cache_config.num_blocks, \
                    f'Expected total {kv_cache_config.num_blocks} GPU blocks, got {gpu_total_blocks}'
            else:
                expected_total = (kv_cache_config.num_blocks) // self.num_attn_layers * self.num_attn_layers
                assert gpu_total_blocks == expected_total, \
                    f'Expected total {expected_total} GPU blocks, got {gpu_total_blocks}'

            self.gpu_kv_cache_pool = torch.zeros(
                (2, gpu_total_blocks, self.block_size, self.head_size),
                dtype=dtype, device=self.device
            )

            cpu_layer_nblk      = np.asarray(cpu_layer_nblk_list, dtype=np.int64)        # [L]
            cpu_layer_first_blk = np.concatenate(([0], np.cumsum(cpu_layer_nblk)[:-1]))  # [L]
            cpu_total_blocks    = int(cpu_layer_nblk.sum())

            if FlexiCacheConfig.num_unstable_heads > 0:
                assert cpu_total_blocks == kv_cache_config.num_cpu_blocks, \
                    f'Expected total {kv_cache_config.num_cpu_blocks} CPU blocks, got {cpu_total_blocks}'
            else:
                expected_total = (kv_cache_config.num_cpu_blocks) // self.num_attn_layers * self.num_attn_layers
                assert cpu_total_blocks == expected_total, \
                    f'Expected total {expected_total} CPU blocks, got {cpu_total_blocks}'

            self.cpu_kv_cache_pool = torch.zeros(
                (2, cpu_total_blocks, self.block_size, self.head_size),
                dtype=dtype, device="cpu", pin_memory=True
            )

            assert mm_layer_nblk * self.num_attn_layers == kv_cache_config.num_minmax_blocks, \
                f'Expected total {kv_cache_config.num_minmax_blocks} minmax blocks, got {mm_layer_nblk * self.num_attn_layers}'

            self.minmax_key_cache = torch.zeros(
                (self.num_attn_layers, mm_layer_nblk,
                FlexiCacheConfig.minmax_key_cache_block_size, 2, kv_cache_spec.head_size),
                dtype=dtype, device=self.device
            )

            # Derive per-layer views
            for L, layer_name in enumerate(layer_names):
                b_start_gpu = int(gpu_layer_first_blk[L])
                b_end_gpu   = b_start_gpu + int(gpu_layer_nblk[L])
                view_L_gpu  = self.gpu_kv_cache_pool[:, b_start_gpu:b_end_gpu, :, :]
                kv_caches[layer_name] = view_L_gpu

                b_start_cpu = int(cpu_layer_first_blk[L])
                b_end_cpu   = b_start_cpu + int(cpu_layer_nblk[L])
                view_L_cpu  = self.cpu_kv_cache_pool[:, b_start_cpu:b_end_cpu, :, :]
                cpu_kv_caches[layer_name] = view_L_cpu

            # Set required persistent metadata
            self.gpu_layer_first_blk = torch.tensor(gpu_layer_first_blk, dtype=torch.int64, device=self.device)
            self.gpu_layer_nblk      = torch.tensor(gpu_layer_nblk, dtype=torch.int32, device=self.device)

            self.cpu_layer_first_blk = torch.tensor(cpu_layer_first_blk, dtype=torch.int64, device=self.device)
            self.cpu_layer_nblk      = torch.tensor(cpu_layer_nblk, dtype=torch.int32, device=self.device)

            self.layer_gpu_offsets = torch.cat([
                self.gpu_layer_first_blk,
                (self.gpu_layer_first_blk[-1] + self.gpu_layer_nblk[-1].to(torch.int64)).view(1)]
            )
            self.layer_cpu_offsets = torch.cat([
                self.cpu_layer_first_blk,
                (self.cpu_layer_first_blk[-1] + self.cpu_layer_nblk[-1].to(torch.int64)).view(1)]
            )

            # Register CPU KV cache pool with the KV-H2D transfer manager
            self._kv_h2d_tx.register_cpu_kv_cache_pinned(self.cpu_kv_cache_pool)
        else:
            for kv_cache_group in kv_cache_config.kv_cache_groups:
                kv_cache_spec = kv_cache_group.kv_cache_spec
                for layer_name in kv_cache_group.layer_names:
                    tensor_config = kv_cache_config.tensors[layer_name]
                    assert tensor_config.size % kv_cache_spec.page_size_bytes == 0
                    num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.
                    assert num_blocks >= kv_cache_config.num_blocks
                    if isinstance(kv_cache_spec, FullAttentionSpec):
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                        dtype = kv_cache_spec.dtype
                        kv_caches[layer_name] = torch.zeros(kv_cache_shape,
                                                            dtype=dtype,
                                                            device=self.device)
                    else:
                        # TODO: add new branches when introducing more types of
                        # KV cache specs.
                        raise ValueError("Unknown KV cache spec type.")
                    
            cpu_kv_caches = None

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
            cpu_kv_caches,
            self.cpu_kv_caches
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in forward_ctx.items():
            if isinstance(attn_module, FusedMoE):
                continue

            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    use_mla=use_mla,
                    enable_flexicache=self.enable_flexicache)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec
    
    def commit_block_table(self, num_reqs):
        # GPU KV-CACHE BLOCK TABLE
        bt_cpu = self.input_batch.block_table.get_cpu_tensor()
        bt_gpu = self.input_batch.block_table.get_device_tensor()
        dirty_gpu_kv = self.input_batch.block_table.build_dirty_ranges(num_reqs)

        self._dirty_gpu_kv[:num_reqs].copy_(dirty_gpu_kv[:num_reqs], non_blocking=True)

        self._bt_tx.block_table_h2d_dirty(
            bt_cpu, bt_gpu, self._dirty_gpu_kv[:num_reqs],
            self.max_num_reqs, self.num_attn_layers, self.num_kv_heads,
            self.input_batch.block_table.max_logical_blks_per_req, num_reqs
        )
        
        # GPU MIN-MAX KV CACHE BLOCK TABLE
        mmb_cpu = self.input_batch.minmax_block_table.get_cpu_tensor()
        mmb_gpu = self.input_batch.minmax_block_table.get_device_tensor()
        dirty_gpu_mmb = self.input_batch.minmax_block_table.build_dirty_ranges(num_reqs)

        self._dirty_gpu_mmb[:num_reqs].copy_(dirty_gpu_mmb[:num_reqs], non_blocking=True)

        self._bt_tx.block_table_h2d_dirty(
            mmb_cpu, mmb_gpu, self._dirty_gpu_mmb[:num_reqs],
            self.max_num_reqs, self.num_attn_layers,self.num_kv_heads,
            self.input_batch.minmax_block_table.max_logical_blks_per_req, num_reqs
        )

        # CPU KV CACHE BLOCK TABLE
        cpu_kv_cpu = self.input_batch.cpu_block_table.get_cpu_tensor()
        cpu_kv_gpu = self.input_batch.cpu_block_table.get_device_tensor()
        dirty_cpu_kv = self.input_batch.cpu_block_table.build_dirty_ranges(num_reqs)

        self._dirty_cpu_kv[:num_reqs].copy_(dirty_cpu_kv[:num_reqs], non_blocking=True)
        self._bt_tx.block_table_h2d_dirty(
            cpu_kv_cpu, cpu_kv_gpu, self._dirty_cpu_kv[:num_reqs],
            self.max_num_reqs, self.num_attn_layers, self.num_kv_heads,
            self.input_batch.cpu_block_table.max_logical_blks_per_req, num_reqs
        )

        self.input_batch.block_table.finalize_commit(num_reqs)
        self.input_batch.minmax_block_table.finalize_commit(num_reqs)
        self.input_batch.cpu_block_table.finalize_commit(num_reqs)

    @torch.inference_mode()
    def launch_store_minmax_key_for_prefill_selected(self, req_idx_np: np.ndarray, prompt_len_np: np.ndarray) -> None:
        if req_idx_np.size == 0: return
        max_full_pages = int((prompt_len_np.max() // self.block_size))
        if max_full_pages == 0:  return

        req_indices = torch.as_tensor(req_idx_np, dtype=torch.int32, device=self.device)
        prompt_lens = torch.as_tensor(prompt_len_np, dtype=torch.int32, device=self.device)

        cur_bt  = self.input_batch.block_table.get_device_tensor()[:, :self.input_batch.num_reqs]
        cur_mmb = self.input_batch.minmax_block_table.get_device_tensor()[:, :self.input_batch.num_reqs]

        x = 8
        k_pool = self.gpu_kv_cache_pool[0].view(-1, self.head_size // x, self.block_size, x)

        # strides
        s_k_nb, s_k_hx, s_k_bs, s_k_x      = k_pool.stride()
        s_bt_bs, s_bt_ly, s_bt_kh, s_bt_bl = cur_bt.stride()
        s_mb_bs, s_mb_ly, s_mb_kh, s_mb_bl = cur_mmb.stride()
        s_mk_l, s_mk_bl, s_mk_bs, s_mk_2, s_mk_hs = self.minmax_key_cache.stride()

        grid = (req_indices.numel() * max_full_pages, self.num_attn_layers, self.num_kv_heads)
        store_minmax_key_cache_for_prefill[grid](
            k_pool, self.minmax_key_cache, cur_bt, cur_mmb, self.gpu_layer_first_blk,
            s_k_nb, s_k_hx, s_k_bs, s_k_x,
            s_bt_bs, s_bt_ly, s_bt_kh, s_bt_bl,
            s_mb_bs, s_mb_ly, s_mb_kh, s_mb_bl,
            s_mk_l, s_mk_bl, s_mk_bs, s_mk_2, s_mk_hs,
            req_indices, prompt_lens, max_full_pages,
            HEAD_SIZE=self.head_size, x=x, BLOCK_SIZE=self.block_size,
            MINMAX_CACHE_BLOCK_SIZE=self.minmax_key_cache_block_size,
            num_warps=4, num_stages=1,
        )

    @torch.inference_mode()
    def launch_store_minmax_key_for_decode_selected(self, req_idx_np: np.ndarray, new_page_np: np.ndarray) -> None:
        if req_idx_np.size == 0: return
        req_indices  = torch.as_tensor(req_idx_np, dtype=torch.int32, device=self.device)
        new_pages    = torch.as_tensor(new_page_np, dtype=torch.int32, device=self.device)

        cur_bt  = self.input_batch.block_table.get_device_tensor()[:, :self.input_batch.num_reqs]
        cur_mmb = self.input_batch.minmax_block_table.get_device_tensor()[:, :self.input_batch.num_reqs]

        x = 8
        k_pool = self.gpu_kv_cache_pool[0].view(-1, self.head_size // x, self.block_size, x)

        s_k_nb, s_k_hx, s_k_bs, s_k_x      = k_pool.stride()
        s_bt_bs, s_bt_ly, s_bt_kh, s_bt_bl = cur_bt.stride()
        s_mb_bs, s_mb_ly, s_mb_kh, s_mb_bl = cur_mmb.stride()
        s_mk_l, s_mk_bl, s_mk_bs, s_mk_2, s_mk_hs = self.minmax_key_cache.stride()

        grid = (req_indices.numel(), self.num_attn_layers, self.num_kv_heads)
        store_minmax_key_cache_for_decode[grid](
            k_pool, self.minmax_key_cache, cur_bt, cur_mmb, self.gpu_layer_first_blk,
            s_k_nb, s_k_hx, s_k_bs, s_k_x,
            s_bt_bs, s_bt_ly, s_bt_kh, s_bt_bl,
            s_mb_bs, s_mb_ly, s_mb_kh, s_mb_bl,
            s_mk_l, s_mk_bl, s_mk_bs, s_mk_2, s_mk_hs,
            req_indices, new_pages,
            HEAD_SIZE=self.head_size, x=x, BLOCK_SIZE=self.block_size,
            MINMAX_CACHE_BLOCK_SIZE=self.minmax_key_cache_block_size,
            num_warps=4, num_stages=1,
        )

    @torch.inference_mode()
    def offload_kv_cache_d2h_selected(self, req_idx_np: np.ndarray, cnt_prefill: int) -> None:
        if req_idx_np.size == 0:
            return
        R = self.input_batch.num_reqs

        # Keep these small control arrays fresh on device (non-blocking).
        self.input_batch.last_tx_blk_gpu[:R].copy_(
            self.input_batch.last_tx_blk_cpu[:R], non_blocking=True)
        self.input_batch.total_full_blk_gpu[:R].copy_(
            self.input_batch.total_full_blk_cpu[:R], non_blocking=True)

        # Build (src,dst) mapping **on the main stream**, then launch the copy on the background stream.
        src_bt = self.input_batch.block_table.get_device_tensor()[:R]         # [R, L, H, Lb]
        dst_bt = self.input_batch.cpu_block_table.get_device_tensor()[:R]     # [R, L, H, Lb]
        sel_req_ids = torch.as_tensor(req_idx_np, dtype=torch.int32, device=self.device)

        # Precompute all pairs (global GPU block, global CPU block) to offload.
        # This mirrors the H2D path (kv_cache_h2d_transfer) style.
        src_pairs, dst_pairs, total_pairs = self._kv_d2h_tx.build_d2h_pairs(
            src_bt,
            dst_bt,
            sel_req_ids,
            self.input_batch.last_tx_blk_gpu[:R],
            self.input_batch.total_full_blk_gpu[:R],
            self.layer_gpu_offsets,   # GPU layer start offsets
            self.layer_cpu_offsets    # CPU layer start offsets
        )
        if int(total_pairs) == 0 or src_pairs.numel() == 0:
            return

        # Hand off to the background stream with a small grid (doesn't starve the model).
        assert self.main_stream == torch.cuda.current_stream(device=self.device)
        start_event = torch.cuda.Event(blocking=False, enable_timing=False, interprocess=False)
        start_event.record(self.main_stream)
        self.background_stream.wait_event(start_event)

        # Keep mapping tensors alive on the background stream.
        src_pairs.record_stream(self.background_stream)
        dst_pairs.record_stream(self.background_stream)

        if cnt_prefill > 0:
            finish_event = torch.cuda.Event(blocking=False, enable_timing=False, interprocess=False)

        with torch.cuda.stream(self.background_stream):
            CHUNK_PAIRS = 4096   # conservative chunking like H2D
            GRID_BLOCKS = 8     # small grid to avoid stealing too many SMs
            self._kv_d2h_tx.kv_cache_d2h_transfer(
                self.gpu_kv_cache_pool,      # src: device pool
                self.cpu_kv_cache_pool,      # dst: pinned host pool
                src_pairs,                   # [N] int32 global GPU block ids
                dst_pairs,                   # [N] int32 global CPU block ids
                CHUNK_PAIRS,
                GRID_BLOCKS
            )
            if cnt_prefill > 0:
                finish_event.record()

        if cnt_prefill > 0:
            for i in range(cnt_prefill):
                rid = self.input_batch.req_ids[self._idx_prefill[i]]
                self._prompt_offload_queue.append((rid, finish_event))

    def reload_kv_cache_h2d(self) -> set[str]:
        kv_reload_started: set[str] = set()
        if len(self.needs_rerank) == 0:
            return kv_reload_started
        
        needs_rerank = []
        for req_idx in self.needs_rerank:
            if self.seq_lens_np[req_idx] > self.max_filtered_blocks * self.block_size:
                needs_rerank.append(req_idx)

        if len(needs_rerank) == 0:
            return kv_reload_started

        sel_idx = torch.tensor(needs_rerank, device='cuda', dtype=torch.int64)
        # sel_idx = torch.tensor(self.needs_rerank, device='cuda', dtype=torch.int64)

        R = self.input_batch.num_reqs
        self.input_batch.total_full_blk_gpu[:R].copy_(self.input_batch.total_full_blk_cpu[:R], non_blocking=True)

        evicted, incoming, counts, src, dst, total_pairs, g2g_src, g2g_dst, total_g2g = \
            self._topk_swap_map.topk_swap_map_launcher_int64(
                self.input_batch.old_top_k_blocks,
                self.input_batch.top_k_blocks,
                sel_idx,
                self.input_batch.total_full_blk_gpu,
                self.input_batch.block_table.get_device_tensor(),
                self.input_batch.cpu_block_table.get_device_tensor(),
                self.layer_cpu_offsets,
                self.layer_gpu_offsets,
                self.unstable_head_masks,
                -1
            )

        assert self.main_stream == torch.cuda.current_stream(device=self.device)
        start_event = torch.cuda.Event(blocking=False, enable_timing=False, interprocess=False)
        start_event.record(self.main_stream)
        self.background_stream.wait_event(start_event)

        # Tie lifetime of src/dst to the background stream so their storage 
        # isn't recycled early by the caching allocator.
        src.record_stream(self.background_stream)
        dst.record_stream(self.background_stream)
        g2g_src.record_stream(self.background_stream)
        g2g_dst.record_stream(self.background_stream)

        finish_event = torch.cuda.Event(blocking=False, enable_timing=False, interprocess=False)

        with torch.cuda.stream(self.background_stream):
            CHUNK_PAIRS = 8192
            GRID_BLOCKS = 32
            self._kv_h2d_tx.kv_cache_h2d_transfer(
                self.cpu_kv_cache_pool,
                self.gpu_kv_cache_pool,
                src,
                dst,
                chunk_pairs=CHUNK_PAIRS,
                grid_blocks=GRID_BLOCKS,
            )
            if int(total_g2g) > 0:
                # Indices are global GPU block indices along dim=1
                self.gpu_kv_cache_pool[:, g2g_dst.long()] = \
                    self.gpu_kv_cache_pool[:, g2g_src.long()]
            finish_event.record()

        for req_idx in needs_rerank:
        # for req_idx in self.needs_rerank:
            req_id = self.input_batch.req_ids[req_idx]
            kv_reload_started.add(req_id)
            self._kv_reload_queue.append((req_id, finish_event))

        return kv_reload_started

    def poll_finished_kv_cache_reloads(self) -> set[str]:
        finished: set[str] = set()
        queue = self._kv_reload_queue
        while queue and queue[0][1].query():
            rid, _ = queue.popleft()
            finished.add(rid)
        return finished