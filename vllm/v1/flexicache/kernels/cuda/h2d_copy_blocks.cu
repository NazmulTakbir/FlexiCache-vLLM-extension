// h2d_copy_blocks.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

static const char* g_src_host_devptr = nullptr;
static size_t g_src_host_bytes = 0;

void register_cpu_kv_cache_pinned(at::Tensor cpu_kv_cache) {
  TORCH_CHECK(cpu_kv_cache.device().is_cpu() && cpu_kv_cache.is_pinned(),
              "cpu_kv_cache must be pinned CPU");
  const char* host_ptr = reinterpret_cast<const char*>(cpu_kv_cache.data_ptr());
  const size_t total_bytes =
      (size_t)cpu_kv_cache.numel() * (size_t)cpu_kv_cache.element_size();

  cudaError_t err = cudaHostGetDevicePointer((void**)&g_src_host_devptr,
                                             (void*)host_ptr, 0);
  if (err != cudaSuccess) {
    err = cudaHostRegister((void*)host_ptr, total_bytes,
                           cudaHostRegisterPortable | cudaHostRegisterMapped);
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostRegister(MAPPED) failed: ", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&g_src_host_devptr,
                                   (void*)host_ptr, 0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostGetDevicePointer failed after register: ",
                cudaGetErrorString(err));
  }
  g_src_host_bytes = total_bytes;
}

__global__ void h2d_copy_blocks_kernel(
    const char* __restrict__ src_host_devptr, // UVA pointer to pinned host
    char* __restrict__ dst_devptr,            // device pointer
    size_t subrow_bytes,                      // bytes in one [P,D] subrow
    size_t src_blk_stride_b, size_t dst_blk_stride_b,
    size_t src_chan_stride_b, size_t dst_chan_stride_b,
    const int32_t* __restrict__ src_blocks,   // [count]
    const int32_t* __restrict__ dst_blocks,   // [count]
    int64_t count)
{
    for (int64_t p = blockIdx.x; p < count; p += gridDim.x) {
        const int32_t sb = src_blocks[p];
        const int32_t db = dst_blocks[p];

        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const char* s = src_host_devptr + (size_t)c * src_chan_stride_b + (size_t)sb * src_blk_stride_b;
            char*       d = dst_devptr      + (size_t)c * dst_chan_stride_b + (size_t)db * dst_blk_stride_b;

            const size_t chunks16 = subrow_bytes / 16;
            for (size_t i = threadIdx.x; i < chunks16; i += blockDim.x) {
                uint4 v = reinterpret_cast<const uint4*>(s)[i];
                reinterpret_cast<uint4*>(d)[i] = v;
            }

            const size_t copied = chunks16 * 16;
            for (size_t off = copied + threadIdx.x; off < subrow_bytes; off += blockDim.x) {
                d[off] = s[off];
            }
        }
    }
}

void kv_cache_h2d_transfer(
    at::Tensor cpu_kv_cache,   // [2, B_cpu, P, D] float16 (CPU pinned)
    at::Tensor gpu_kv_cache,   // [2, B_gpu, P, D] float16 (CUDA)
    at::Tensor src_blocks,     // [N] int32 (CUDA)
    at::Tensor dst_blocks,     // [N] int32 (CUDA)
    int64_t chunk_pairs,       // e.g., 8192
    int64_t grid_blocks)       // e.g., 32
{
    TORCH_CHECK(cpu_kv_cache.device().is_cpu() && cpu_kv_cache.is_pinned(), "cpu_kv_cache must be pinned CPU");
    TORCH_CHECK(gpu_kv_cache.is_cuda(), "gpu_kv_cache must be CUDA");
    TORCH_CHECK(cpu_kv_cache.scalar_type() == gpu_kv_cache.scalar_type(), "dtype mismatch");
    TORCH_CHECK(cpu_kv_cache.dim() == 4 && gpu_kv_cache.dim() == 4 &&
                cpu_kv_cache.size(0) == 2 && gpu_kv_cache.size(0) == 2,
                "KV caches must be [2, B, P, D]");

    TORCH_CHECK(src_blocks.is_cuda() && dst_blocks.is_cuda(), "block lists must be CUDA");
    TORCH_CHECK(src_blocks.dtype() == at::kInt && dst_blocks.dtype() == at::kInt, "block lists must be int32");
    TORCH_CHECK(src_blocks.numel() == dst_blocks.numel(), "src/dst size mismatch");

    const int64_t N = src_blocks.numel();
    if (N == 0) return;

    c10::cuda::OptionalCUDAGuard guard(gpu_kv_cache.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(g_src_host_devptr != nullptr,
                "Call register_cpu_kv_cache_pinned() once before transfers.");
    const char* src_host_devptr = g_src_host_devptr;     // use cached mapping
    char* dst_devptr = reinterpret_cast<char*>(gpu_kv_cache.data_ptr());

    const size_t elem = (size_t)gpu_kv_cache.element_size();
    const size_t subrow_bytes = (size_t)gpu_kv_cache.size(2) * (size_t)gpu_kv_cache.size(3) * elem; // P*D*elem

    const size_t src_blk_stride_b   = (size_t)cpu_kv_cache.stride(1) * elem; // along B
    const size_t dst_blk_stride_b   = (size_t)gpu_kv_cache.stride(1) * elem;
    const size_t src_chan_stride_b  = (size_t)cpu_kv_cache.stride(0) * elem; // channel (K/V)
    const size_t dst_chan_stride_b  = (size_t)gpu_kv_cache.stride(0) * elem;

    const int threads = 128;
    const int64_t rows_per_chunk = std::max<int64_t>(1, chunk_pairs);

    for (int64_t start = 0; start < N; start += rows_per_chunk) {
    const int64_t cnt = std::min<int64_t>(rows_per_chunk, N - start);
    auto sb = src_blocks.narrow(0, start, cnt).contiguous();
    auto db = dst_blocks.narrow(0, start, cnt).contiguous();

    const int grid = (int)std::max<int64_t>(1, std::min<int64_t>(grid_blocks, cnt));
    h2d_copy_blocks_kernel<<<grid, threads, 0, stream>>>(
        src_host_devptr, dst_devptr, 
        subrow_bytes,
        src_blk_stride_b, dst_blk_stride_b,
        src_chan_stride_b, dst_chan_stride_b,
        sb.data_ptr<int32_t>(),
        db.data_ptr<int32_t>(),
        cnt);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_cpu_kv_cache_pinned", &register_cpu_kv_cache_pinned,
        "Register/map pinned host KV cache for UVA");
    m.def("kv_cache_h2d_transfer",
        &kv_cache_h2d_transfer,
        "Chunked pinned-host→GPU KV transfer",
        py::arg("cpu_kv_cache"),
        py::arg("gpu_kv_cache"),
        py::arg("src_blocks"),
        py::arg("dst_blocks"),
        py::arg("chunk_pairs") = 8192,
        py::arg("grid_blocks") = 32);
}
