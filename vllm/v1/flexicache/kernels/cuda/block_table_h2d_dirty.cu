#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace py = pybind11;

static inline void check_block_table_inputs(const torch::Tensor& src,
                                            const torch::Tensor& dst) {
  TORCH_CHECK(src.device().is_cpu(), "src must be on CPU");
  TORCH_CHECK(src.is_pinned(), "src must be pinned (pin_memory=True)");
  TORCH_CHECK(dst.is_cuda(), "dst must be CUDA");
  TORCH_CHECK(src.dtype() == torch::kInt32 && dst.dtype() == torch::kInt32,
              "only int32 supported");
  TORCH_CHECK(src.dim() == 4 && dst.dim() == 4, "need 4D tensors");
  TORCH_CHECK(src.sizes() == dst.sizes(), "src/dst shape mismatch");
  TORCH_CHECK(src.stride(3) == 1 && dst.stride(3) == 1,
              "last dim (BLKS) must be contiguous");
}

static inline void check_dirty_ranges(const torch::Tensor& dirty,
                                      int64_t MAX_SEQ, int64_t TX_SEQ) {
  TORCH_CHECK(dirty.is_cuda(), "dirty ranges must be on CUDA");
  TORCH_CHECK(dirty.dtype() == torch::kInt32, "dirty ranges must be int32");
  TORCH_CHECK(dirty.dim() == 2 && dirty.size(1) == 2,
              "dirty ranges must be shape (N, 2)");
  TORCH_CHECK(TX_SEQ > 0 && TX_SEQ <= MAX_SEQ, "bad TX_SEQ");
  TORCH_CHECK(dirty.size(0) >= TX_SEQ,
              "dirty ranges first dimension must be >= TX_SEQ");
}

// Copy only [start,end) segment for each (s,l,h) row.
// - host_src_devptr: device-visible pointer into pinned host memory (UVA zero-copy)
// - dst_dev: real device pointer
// - dirty: (TX_SEQ, 2) int32 pairs of [start, end) per request s ∈ [0, TX_SEQ)
__global__ void copy_dirty_rows_int32(
    const char* __restrict__ host_src_devptr,
    char* __restrict__ dst_dev,
    const int32_t* __restrict__ dirty, // length >= TX_SEQ*2
    int MAX_SEQ, int L, int H, int MAX_BLKS, int TX_SEQ) {

  int row = blockIdx.x;
  const int rows_per_seq = L * H;
  const int total_rows   = TX_SEQ * rows_per_seq;
  if (row >= total_rows) return;

  // Decode (s, l, h)
  int s = row / rows_per_seq;
  int rem = row - s * rows_per_seq;
  int l = rem / H;
  int h = rem - l * H;

  // Dirty pair for this request s
  const int32_t start_raw = dirty[s * 2 + 0];
  const int32_t end_raw   = dirty[s * 2 + 1];

  // Clamp to [0, MAX_BLKS]
  int32_t start = max(0, min(start_raw, (int32_t)MAX_BLKS));
  int32_t end   = max(0, min(end_raw,   (int32_t)MAX_BLKS));
  int32_t count = end - start;
  if (count <= 0) return; // nothing to do

  const size_t elem_size = sizeof(int32_t);
  const size_t base_elem =
      ((((size_t)s * (size_t)L + (size_t)l) * (size_t)H + (size_t)h) *
       (size_t)MAX_BLKS);

  // Row base (beginning of BLKS dimension)
  const char* src_row_base = host_src_devptr + base_elem * elem_size;
  char*       dst_row_base = dst_dev        + base_elem * elem_size;

  // Offset by 'start'
  const char* src_ptr = src_row_base + (size_t)start * elem_size;
  char*       dst_ptr = dst_row_base + (size_t)start * elem_size;

  // Bytes to move for this row
  const size_t row_bytes = (size_t)count * elem_size;

  // Fast path: 16-byte (uint4) stripes if both pointers are 16B-aligned and
  // row_bytes has at least one full 16B chunk. Otherwise, fallback to int32 stripes.
  size_t tid = threadIdx.x;

  const uintptr_t sp = reinterpret_cast<uintptr_t>(src_ptr);
  const uintptr_t dp = reinterpret_cast<uintptr_t>(dst_ptr);
  const bool vec_ok = ((sp | dp) & (uintptr_t)0xF) == 0; // both 16B-aligned

  if (vec_ok && row_bytes >= 16) {
    size_t vec16 = row_bytes / 16;
    // 16B stripes
    for (size_t i = tid; i < vec16; i += blockDim.x) {
      uint4 v = reinterpret_cast<const uint4*>(src_ptr)[i];
      reinterpret_cast<uint4*>(dst_ptr)[i] = v;
    }
    // tail (remaining <16B), parallelized by byte
    size_t copied = vec16 * 16;
    for (size_t b = copied + tid; b < row_bytes; b += blockDim.x) {
      dst_ptr[b] = src_ptr[b];
    }
  } else {
    // Fallback: copy in 4B (int32) stripes + final byte tail if needed
    size_t words = row_bytes / 4;
    const int32_t* __restrict__ s32 = reinterpret_cast<const int32_t*>(src_ptr);
    int32_t* __restrict__ d32 = reinterpret_cast<int32_t*>(dst_ptr);
    for (size_t i = tid; i < words; i += blockDim.x) {
      d32[i] = s32[i];
    }
    size_t copied = words * 4;
    for (size_t b = copied + tid; b < row_bytes; b += blockDim.x) {
      dst_ptr[b] = src_ptr[b];
    }
  }
}

void block_table_h2d_dirty(torch::Tensor src,
                           torch::Tensor dst,
                           torch::Tensor dirty_ranges, // (>=TX_SEQ, 2) on CUDA
                           int64_t MAX_SEQ, int64_t L,
                           int64_t H, int64_t MAX_BLKS,
                           int64_t TX_SEQ) {
  check_block_table_inputs(src, dst);
  TORCH_CHECK(MAX_SEQ == src.size(0) && L == src.size(1) &&
              H == src.size(2) && MAX_BLKS == src.size(3), "dims mismatch");
  check_dirty_ranges(dirty_ranges, MAX_SEQ, TX_SEQ);

  c10::cuda::OptionalCUDAGuard guard(dst.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Map pinned host src to device address space (UVA zero-copy)
  const void* host_ptr = src.data_ptr();
  const size_t total_bytes = (size_t)src.numel() * src.element_size();

  const char* host_devptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer((void**)&host_devptr,
                                             (void*)host_ptr, 0);
  bool unreg_after = false;

  if (err != cudaSuccess) {
    // Fallback: (re)register as mapped
    err = cudaHostRegister((void*)host_ptr, total_bytes,
                           cudaHostRegisterPortable | cudaHostRegisterMapped);
    TORCH_CHECK(err == cudaSuccess, "cudaHostRegister(MAPPED) failed: ",
                cudaGetErrorString(err));
    unreg_after = true;
    err = cudaHostGetDevicePointer((void**)&host_devptr, (void*)host_ptr, 0);
    TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed: ",
                cudaGetErrorString(err));
  }

  const int64_t total_rows = TX_SEQ * L * H;
  if (total_rows > 0) {
    dim3 grid((unsigned)total_rows);
    dim3 block(256);

    copy_dirty_rows_int32<<<grid, block, 0, stream>>>(
        host_devptr,
        (char*)dst.data_ptr(),
        dirty_ranges.data_ptr<int32_t>(),
        (int)MAX_SEQ, (int)L, (int)H, (int)MAX_BLKS, (int)TX_SEQ);

    cudaError_t kerr = cudaGetLastError();
    TORCH_CHECK(kerr == cudaSuccess, "kernel launch failed: ",
                cudaGetErrorString(kerr));
  }

  if (unreg_after) cudaHostUnregister((void*)host_ptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_table_h2d_dirty", &block_table_h2d_dirty,
        "Zero-copy H2D for block-table using per-request dirty ranges",
        py::arg("src"), py::arg("dst"), py::arg("dirty_ranges"),
        py::arg("MAX_SEQ"), py::arg("L"), py::arg("H"), py::arg("MAX_BLKS"),
        py::arg("TX_SEQ"));
}
