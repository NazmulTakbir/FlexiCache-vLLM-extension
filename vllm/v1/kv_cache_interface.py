# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.utils import cdiv, get_dtype_size
from vllm.v1.flexicache.config import FlexiCacheConfig

logger = init_logger(__name__)


@dataclass
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    block_size: int

    @property
    def type_id(self) -> str:
        """
        The type identifier of this KV cache.
        Return different strings for layers with different KV cache type (e.g.,
        different number of tokens like full attention vs sliding window
        attention, different KV cache size per token like layers with different
        number of heads)

        Returns:
            The type identifier of this KV cache.
        """
        raise NotImplementedError

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError
    
    @property
    def minmax_page_size_bytes(self) -> int:
        """
        The size of a minmax page in bytes.

        Returns:
            The minmax page size
        """
        raise NotImplementedError

    def bytes_for_tokens(self, num_tokens: int) -> int:
        """
        The KV cache size for `num_tokens` tokens in bytes. Returns the real
        memory size after padding `num_tokens` to full blocks.

        Returns:
            The KV cache size
        """
        raise NotImplementedError


@dataclass
class FullAttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    use_mla: bool
    enable_flexicache: bool = False

    @property
    def type_id(self) -> str:
        prefix = "flexicache" if self.enable_flexicache else "full_attention"
        return f"{prefix}_{self.block_size}_{self.page_size_bytes}"
    
    @property
    def page_size_bytes(self) -> int:
        if self.enable_flexicache:
            return 2 * self.block_size * self.head_size * get_dtype_size(self.dtype)
        else:
            # For MLA we only store a single latent vector
            coef = 1 if self.use_mla else 2
            return coef * self.block_size * self.num_kv_heads * self.head_size \
                    * get_dtype_size(self.dtype)
        
    @property
    def minmax_page_size_bytes(self) -> int:
        assert self.enable_flexicache
        return FlexiCacheConfig.minmax_key_cache_block_size * 2 * self.head_size * get_dtype_size(self.dtype)

    def bytes_for_tokens(self, num_tokens: int) -> int:
        if self.enable_flexicache:
            return cdiv(num_tokens, self.block_size) * self.num_kv_heads * self.page_size_bytes
        else:
            return cdiv(num_tokens, self.block_size) * self.page_size_bytes


@dataclass
class KVCacheTensor:
    """
    A dataclass for specifying how the workers should initialize the KV cache
    for a layer. Only contains the size of KV cache for that layer for now. Will
    be extended to support multiple layers sharing the same memory pool.
    """
    size: int  # The size of KV cache Tensor in bytes


@dataclass
class KVCacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """
    # The names of model layers in this group
    layer_names: list[str]
    # The KV cache spec of this manager layer
    kv_cache_spec: KVCacheSpec


@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """
    """The number of KV cache blocks"""
    num_blocks: int
    tensors: dict[str, KVCacheTensor]
    """
    The kv cache groups of the model.
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times. 
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the 
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 groups, each of which
    contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer 
    in the group.
    For example:
    1. A model only uses full attention. The pattern is 
    (num_hidden_layers * full), so there is only one group and the block table 
    is shared by all layers.
    2. (WIP) A model with 10 full attention layers and 20 sliding window 
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so 
    there are 3 groups, each of which represents 10 layers in the model.
    """
    kv_cache_groups: list[KVCacheGroupSpec]

    """
    For FlexiCache, `num_blocks`, `num_cpu_blocks`, and `num_minmax_blocks` are the total number
    of gpu blocks, cpu blocks, and minmax blocks. The total number of blocks are not distributed
    evenly across all layers or heads.
    The size of each tensor determines the number of blocks allocated to each layer.
    Heads within the same layer are allocated blocks from the same pool of blocks for that layer.
    """
    """The number of KV cache CPU blocks"""
    num_cpu_blocks: int  = 0
    """The number of KV cache minmax blocks"""
    num_minmax_blocks: int = 0
    """layer_name -> how to initialize KV cache for that layer"""
    cpu_tensors: dict[str, KVCacheTensor] | None = None
    minmax_tensors: dict[str, KVCacheTensor] | None = None
