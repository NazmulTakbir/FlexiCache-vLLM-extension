# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


class BlockTable:

    def __init__(
        self,
        max_num_reqs: int,
        max_logical_blks_per_req: int,
        pin_memory: bool,
        device: torch.device,
        enable_flexicache: bool,
        num_kv_heads: int,
        num_layers: int = 1,
        name: str = "KV-Block-Table"
    ):
        self.max_num_reqs = max_num_reqs
        self.max_logical_blks_per_req = max_logical_blks_per_req
        self.pin_memory = pin_memory
        self.device = device
        self.enable_flexicache = enable_flexicache
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.name = name

        if self.enable_flexicache:
            shape = (max_num_reqs, num_layers, num_kv_heads, max_logical_blks_per_req)
            self.committed_num_logical_blocks = np.zeros(max_num_reqs, dtype=np.int32)
            self.dirty = torch.zeros((self.max_num_reqs, 2), dtype=torch.int32, device="cpu", pin_memory=True)
            self.dirty_np = self.dirty.numpy()
        else:
            shape = (max_num_reqs, max_logical_blks_per_req)

        self.block_table = torch.zeros(
            shape,
            device=self.device,
            dtype=torch.int32,
        )
        self.block_table_cpu = torch.zeros(
            shape,
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.block_table_np = self.block_table_cpu.numpy()
        self.num_logical_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

    def append_row(
        self,
        block_ids: list[list[int]],
        row_idx: int
    ) -> None:
        if not block_ids:
            return

        if self.enable_flexicache:
            num_physical_blocks = len(block_ids[0])
            if num_physical_blocks % self.num_kv_heads != 0:
                raise ValueError(
                    f"Number of physical blocks ({num_physical_blocks}) is not "
                    f"divisible by number of KV heads ({self.num_kv_heads})")
            num_logical_blocks = num_physical_blocks // self.num_kv_heads

            start = self.num_logical_blocks_per_row[row_idx]
            end   = start + num_logical_blocks
            self.num_logical_blocks_per_row[row_idx] = end

            assert end <= self.max_logical_blks_per_req, \
                f"{self.name} overflow: {end} > {self.max_logical_blks_per_req}"
            
            blk_array = np.asarray(block_ids, dtype=np.int32)
            blk_reshaped = \
                blk_array.reshape(self.num_layers, num_logical_blocks, self.num_kv_heads).swapaxes(1, 2)
            self.block_table_np[row_idx, :, :, start:end] = blk_reshaped
        else:
            block_ids = block_ids[0]
            num_blocks = len(block_ids)
            start = self.num_logical_blocks_per_row[row_idx]
            assert start + num_blocks <= self.max_logical_blks_per_req, \
                f"{self.name} overflow: {start + num_blocks} > {self.max_logical_blks_per_req}"
            self.num_logical_blocks_per_row[row_idx] += num_blocks
            self.block_table_np[row_idx, start:start + num_blocks] = block_ids

    def add_row(self, block_ids: list[list[int]], row_idx: int) -> None:
        self.num_logical_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)
        if self.enable_flexicache:
            self.committed_num_logical_blocks[row_idx] = 0

    def move_row(self, src: int, tgt: int, in_decode_phase: bool = False) -> None:
        if self.enable_flexicache:
            num_logical = self.num_logical_blocks_per_row[src]
            committed   = self.committed_num_logical_blocks[src]

            if in_decode_phase:
                assert committed > 0, "In decode phase, committed should be > 0"

                # D2D copy. uncommitted tail will be copied from CPU block table on next commit
                self.block_table[tgt, :, :, :committed].copy_(
                    self.block_table[src, :, :, :committed], non_blocking=True
                )

                # CPU tail should be updated for slot mapping. Rest of cpu row can be stale.
                if committed < num_logical:
                    self.block_table_cpu[tgt, :, :, committed:num_logical].copy_(
                    self.block_table_cpu[src, :, :, committed:num_logical], non_blocking=True
                )

                self.committed_num_logical_blocks[tgt] = committed
                self.num_logical_blocks_per_row[tgt]   = num_logical
            else:
                # H2H copy if in prefill phase
                self.block_table_cpu[tgt, :, :, :num_logical].copy_(
                    self.block_table_cpu[src, :, :, :num_logical], non_blocking=True
                )
                self.num_logical_blocks_per_row[tgt] = num_logical
                # full H2D for this row on next block table commit
                self.committed_num_logical_blocks[tgt] = 0 
        else:
            num_blocks = self.num_logical_blocks_per_row[src]
            self.block_table_np[tgt, :num_blocks] = self.block_table_np[
                src, :num_blocks]
            self.num_logical_blocks_per_row[tgt] = num_blocks
        
    def swap_row(self, src: int, tgt: int) -> None:
        num_blocks_src = self.num_logical_blocks_per_row[src]
        num_blocks_tgt = self.num_logical_blocks_per_row[tgt]
        self.num_logical_blocks_per_row[src] = num_blocks_tgt
        self.num_logical_blocks_per_row[tgt] = num_blocks_src

        self.block_table_np[[src, tgt]] = self.block_table_np[[tgt, src]]

    def commit(self, num_reqs: int) -> None:
        if self.enable_flexicache:
            assert False, "Handled in gpu_model_runner"
        else:
            self.block_table[:num_reqs].copy_(
                self.block_table_cpu[:num_reqs], non_blocking=True
            )

    def clear(self) -> None:
        self.block_table.fill_(0)
        self.block_table_cpu.fill_(0)

    def get_device_tensor(self) -> torch.Tensor:
        """Ruturns the device tensor of the block table."""
        return self.block_table

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table_cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table_np

    def build_dirty_ranges(self, num_reqs: int) -> torch.Tensor:
        prev = self.committed_num_logical_blocks[:num_reqs]
        cur  = self.num_logical_blocks_per_row[:num_reqs]

        bad = np.flatnonzero(cur < prev)
        if bad.size:
            i = int(bad[0])
            raise AssertionError(
                f"num_logical_blocks_per_row[{i}]={cur[i]} < "
                f"committed_num_logical_blocks[{i}]={prev[i]}"
            )

        self.dirty_np[:num_reqs, 0] = prev
        self.dirty_np[:num_reqs, 1] = cur

        return self.dirty[:num_reqs]

    def finalize_commit(self, num_reqs: int) -> None:
        self.committed_num_logical_blocks[:num_reqs] = self.num_logical_blocks_per_row[:num_reqs]