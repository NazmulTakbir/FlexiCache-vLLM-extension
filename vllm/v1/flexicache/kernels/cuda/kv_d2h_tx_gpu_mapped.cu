#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

// ---------------------------------------------------------------------------
// Build (src,dst) mapping on MAIN stream
// ---------------------------------------------------------------------------

// For each (req, layer) enumerate heads x [last..total) and emit (gpu, cpu) pairs
__global__ void build_pairs_kernel(
    const int32_t* __restrict__ src_bt, // [R, L, Hn, Lbmax]
    const int32_t* __restrict__ dst_bt, // [R, L, Hn, Lbmax]
    int64_t L, int64_t Hn, int64_t Lbmax,
    const int32_t* __restrict__ req_ids, int64_t num_tx,
    const int32_t* __restrict__ last_tx_blk,      // [R]
    const int32_t* __restrict__ total_full_blks,  // [R]
    const int64_t* __restrict__ src_layer_offsets,// [L+1]
    const int64_t* __restrict__ dst_layer_offsets,// [L+1]
    int32_t* __restrict__ out_src_global,         // [<= total_pairs]
    int32_t* __restrict__ out_dst_global,         // [<= total_pairs]
    unsigned long long* __restrict__ write_ptr)   // global counter
{
  const int ridx = blockIdx.x; // 0..num_tx-1
  const int ly   = blockIdx.y; // 0..L-1
  if (ridx >= num_tx || ly >= L) return;

  const int s = req_ids[ridx];
  const int start_lb = last_tx_blk[s];
  const int end_lb   = total_full_blks[s];
  const int num_lb   = max(0, end_lb - start_lb);
  if (num_lb <= 0) return;

  // Strides for [R, L, Hn, Lbmax]
  const int64_t stride_R  = L * Hn * Lbmax;
  const int64_t stride_L  = Hn * Lbmax;
  const int64_t stride_Hn = Lbmax;

  const int64_t src_layer_base = src_layer_offsets[ly];
  const int64_t dst_layer_base = dst_layer_offsets[ly];

  // Parallelize across heads; loop blocks across logical pages
  for (int lb_rel = 0; lb_rel < num_lb; ++lb_rel) {
    const int lb = start_lb + lb_rel;
    for (int h = threadIdx.x; h < Hn; h += blockDim.x) {
      const int64_t bt_off = (int64_t)s * stride_R + (int64_t)ly * stride_L
                           + (int64_t)h * stride_Hn + (int64_t)lb;
      const int32_t dst_phys = dst_bt[bt_off];
      if (dst_phys == -1) continue; // skip unstable heads/pages
      const int32_t src_phys = src_bt[bt_off];

      const int32_t g_gpu = (int32_t)(src_layer_base + (int64_t)src_phys);
      const int32_t g_cpu = (int32_t)(dst_layer_base + (int64_t)dst_phys);

      const unsigned long long pos = atomicAdd(write_ptr, 1ULL);
      out_src_global[pos] = g_gpu;
      out_dst_global[pos] = g_cpu;
    }
    __syncthreads();
  }
}

// Host wrapper: returns (src_blocks[int32 cuda], dst_blocks[int32 cuda], total_pairs[int64 cpu])
static std::tuple<at::Tensor, at::Tensor, int64_t>
build_d2h_pairs_impl(
    at::Tensor src_bt, at::Tensor dst_bt,
    at::Tensor req_ids,
    at::Tensor last_tx_blk, at::Tensor total_full_blks,
    at::Tensor src_layer_offsets, at::Tensor dst_layer_offsets) {

  TORCH_CHECK(src_bt.is_cuda() && dst_bt.is_cuda(), "block tables must be CUDA");
  TORCH_CHECK(src_bt.dtype()==at::kInt && dst_bt.dtype()==at::kInt, "block tables must be int32");
  TORCH_CHECK(src_bt.sizes() == dst_bt.sizes() && src_bt.dim()==4, "block table shape [R,L,H,Lb]");
  TORCH_CHECK(last_tx_blk.is_cuda() && total_full_blks.is_cuda(), "last/total must be CUDA");
  TORCH_CHECK(last_tx_blk.dtype()==at::kInt && total_full_blks.dtype()==at::kInt, "last/total int32");
  TORCH_CHECK(req_ids.dtype()==at::kInt && req_ids.dim()==1, "req_ids int32 1-D");

  const int64_t L = src_bt.size(1);
  const int64_t Hn = src_bt.size(2);

  TORCH_CHECK(src_layer_offsets.dim()==1 && dst_layer_offsets.dim()==1, "offsets 1-D");
  TORCH_CHECK(src_layer_offsets.dtype()==at::kLong && dst_layer_offsets.dtype()==at::kLong, "offsets int64");
  TORCH_CHECK(src_layer_offsets.size(0)==L+1 && dst_layer_offsets.size(0)==L+1, "offsets len L+1");

  c10::cuda::CUDAGuard guard(src_bt.device());

  // exact upper bound on total pairs (CPU)
  at::Tensor req_cpu  = req_ids.is_cuda() ? req_ids.cpu() : req_ids;
  at::Tensor last_cpu = last_tx_blk.cpu();
  at::Tensor tot_cpu  = total_full_blks.cpu();
  const int64_t num_tx = req_cpu.size(0);

  auto rptr = req_cpu.data_ptr<int32_t>();
  auto lptr = last_cpu.data_ptr<int32_t>();
  auto tptr = tot_cpu.data_ptr<int32_t>();

  int64_t sum_pairs = 0;
  for (int64_t i = 0; i < num_tx; ++i) {
    const int s = rptr[i];
    const int diff = std::max(0, (int)tptr[s] - (int)lptr[s]);
    sum_pairs += (int64_t)diff * Hn;  // per layer
  }
  sum_pairs *= L;
  if (sum_pairs <= 0) {
    auto empty_i32 = at::empty({0}, src_bt.options().dtype(at::kInt));
    return {empty_i32, empty_i32, 0};
  }

  // outputs on CUDA + counter
  auto opts_i32 = src_bt.options().dtype(at::kInt);
  at::Tensor src_global = at::empty({sum_pairs}, opts_i32);
  at::Tensor dst_global = at::empty({sum_pairs}, opts_i32);
  at::Tensor counter    = at::zeros({1}, src_bt.options().dtype(at::kLong));

  // ensure device residency
  at::Tensor req_dev = req_ids.is_cuda() ? req_ids : req_ids.to(src_bt.device(), /*non_blocking=*/true);
  at::Tensor src_off = src_layer_offsets.is_cuda() ? src_layer_offsets
                      : src_layer_offsets.to(src_bt.device(), /*non_blocking=*/true);
  at::Tensor dst_off = dst_layer_offsets.is_cuda() ? dst_layer_offsets
                      : dst_layer_offsets.to(src_bt.device(), /*non_blocking=*/true);

  dim3 grid((unsigned)num_tx, (unsigned)L, 1);
  dim3 block(256);
  auto stream = at::cuda::getCurrentCUDAStream();
  build_pairs_kernel<<<grid, block, 0, stream>>>(
      src_bt.data_ptr<int32_t>(),
      dst_bt.data_ptr<int32_t>(),
      L, Hn, src_bt.size(3),
      req_dev.data_ptr<int32_t>(), num_tx,
      last_tx_blk.data_ptr<int32_t>(),
      total_full_blks.data_ptr<int32_t>(),
      src_off.data_ptr<int64_t>(),
      dst_off.data_ptr<int64_t>(),
      src_global.data_ptr<int32_t>(),
      dst_global.data_ptr<int32_t>(),
      reinterpret_cast<unsigned long long*>(counter.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t total_pairs = counter.to(at::kCPU).item<int64_t>();
  auto src_trim = src_global.narrow(0, 0, total_pairs);
  auto dst_trim = dst_global.narrow(0, 0, total_pairs);
  return {src_trim, dst_trim, total_pairs};
}

// ---------------------------------------------------------------------------
// Background D2H copy using prebuilt mappings (mirrors H2D chunking)
// ---------------------------------------------------------------------------

__global__ void d2h_copy_blocks_kernel(
    const char* __restrict__ src_devptr,  // device ptr
    char* __restrict__ dst_host_devptr,   // UVA to pinned host
    size_t subrow_bytes,
    size_t src_blk_stride_b, size_t dst_blk_stride_b,
    size_t src_chan_stride_b, size_t dst_chan_stride_b,
    const int32_t* __restrict__ src_blocks, // [count]
    const int32_t* __restrict__ dst_blocks, // [count]
    int64_t count)
{
  for (int64_t p = blockIdx.x; p < count; p += gridDim.x) {
    const int32_t sb = src_blocks[p];
    const int32_t db = dst_blocks[p];
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
      const char* s = src_devptr      + (size_t)c * src_chan_stride_b + (size_t)sb * src_blk_stride_b;
      char*       d = dst_host_devptr + (size_t)c * dst_chan_stride_b + (size_t)db * dst_blk_stride_b;
      const size_t chunks16 = subrow_bytes / 16;
      for (size_t i = threadIdx.x; i < chunks16; i += blockDim.x) {
        reinterpret_cast<uint4*>(d)[i] = reinterpret_cast<const uint4*>(s)[i];
      }
      const size_t copied = chunks16 * 16;
      for (size_t off = copied + threadIdx.x; off < subrow_bytes; off += blockDim.x) {
        d[off] = s[off];
      }
    }
  }
  __threadfence_system(); // make host writes visible
}

static void kv_cache_d2h_transfer_impl(
    at::Tensor gpu_kv_cache,   // [2, B_gpu, P, D] CUDA
    at::Tensor cpu_kv_cache,   // [2, B_cpu, P, D] CPU pinned
    at::Tensor src_blocks,     // [N] int32 CUDA (global GPU block ids)
    at::Tensor dst_blocks,     // [N] int32 CUDA (global CPU block ids)
    int64_t chunk_pairs,       // e.g., 8192
    int64_t grid_blocks)       // e.g., 32
{
  TORCH_CHECK(gpu_kv_cache.is_cuda(), "gpu_kv_cache must be CUDA");
  TORCH_CHECK(cpu_kv_cache.device().is_cpu() && cpu_kv_cache.is_pinned(),
              "cpu_kv_cache must be pinned CPU");
  TORCH_CHECK(gpu_kv_cache.scalar_type() == cpu_kv_cache.scalar_type(), "dtype mismatch");
  TORCH_CHECK(gpu_kv_cache.dim()==4 && cpu_kv_cache.dim()==4 &&
              gpu_kv_cache.size(0)==2 && cpu_kv_cache.size(0)==2,
              "KV caches must be [2,B,P,D]");
  TORCH_CHECK(src_blocks.is_cuda() && dst_blocks.is_cuda(), "src/dst lists must be CUDA");
  TORCH_CHECK(src_blocks.dtype()==at::kInt && dst_blocks.dtype()==at::kInt, "src/dst lists int32");
  TORCH_CHECK(src_blocks.numel()==dst_blocks.numel(), "src/dst size mismatch");

  const int64_t N = src_blocks.numel();
  if (N == 0) return;

  c10::cuda::OptionalCUDAGuard guard(gpu_kv_cache.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const char* src_devptr = reinterpret_cast<const char*>(gpu_kv_cache.data_ptr());
  char* dst_devptr = nullptr;

  // Resolve / register UVA alias for pinned host dst
  char* dst_host = (char*)cpu_kv_cache.data_ptr();
  size_t total_bytes = (size_t)cpu_kv_cache.numel() * (size_t)cpu_kv_cache.element_size();
  cudaError_t err = cudaHostGetDevicePointer((void**)&dst_devptr, (void*)dst_host, 0);
  if (err != cudaSuccess) {
    err = cudaHostRegister((void*)dst_host, total_bytes,
                           cudaHostRegisterPortable | cudaHostRegisterMapped);
    TORCH_CHECK(err == cudaSuccess, "cudaHostRegister(MAPPED) failed: ", cudaGetErrorString(err));
    err = cudaHostGetDevicePointer((void**)&dst_devptr, (void*)dst_host, 0);
    TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed after register: ",
                cudaGetErrorString(err));
  }

  const size_t elem = (size_t)gpu_kv_cache.element_size();
  const size_t subrow_bytes = (size_t)gpu_kv_cache.size(2) * (size_t)gpu_kv_cache.size(3) * elem; // P*D*elem

  const size_t src_blk_stride_b  = (size_t)gpu_kv_cache.stride(1) * elem;
  const size_t dst_blk_stride_b  = (size_t)cpu_kv_cache.stride(1) * elem;
  const size_t src_chan_stride_b = (size_t)gpu_kv_cache.stride(0) * elem;
  const size_t dst_chan_stride_b = (size_t)cpu_kv_cache.stride(0) * elem;

  const int threads = 128;
  const int64_t rows_per_chunk = std::max<int64_t>(1, chunk_pairs);
  for (int64_t start = 0; start < N; start += rows_per_chunk) {
    const int64_t cnt = std::min<int64_t>(rows_per_chunk, N - start);
    auto sb = src_blocks.narrow(0, start, cnt).contiguous();
    auto db = dst_blocks.narrow(0, start, cnt).contiguous();
    const int grid = (int)std::max<int64_t>(1, std::min<int64_t>(grid_blocks, cnt));
    d2h_copy_blocks_kernel<<<grid, threads, 0, stream>>>(
        src_devptr, dst_devptr,
        subrow_bytes,
        src_blk_stride_b, dst_blk_stride_b,
        src_chan_stride_b, dst_chan_stride_b,
        sb.data_ptr<int32_t>(),
        db.data_ptr<int32_t>(),
        cnt);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------- PyBind ---------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_d2h_pairs", &build_d2h_pairs_impl,
        "Build (src,dst) block pairs for D2H offload",
        py::arg("src_block_table"),
        py::arg("dst_block_table"),
        py::arg("req_ids"),
        py::arg("last_tx_blk"),
        py::arg("total_full_blks"),
        py::arg("src_layer_offsets"),
        py::arg("dst_layer_offsets"));

  m.def("kv_cache_d2h_transfer", &kv_cache_d2h_transfer_impl,
        "Chunked GPU→pinned-host KV transfer using prebuilt pairs",
        py::arg("gpu_kv_cache"),
        py::arg("cpu_kv_cache"),
        py::arg("src_blocks"),
        py::arg("dst_blocks"),
        py::arg("chunk_pairs") = 8192,
        py::arg("grid_blocks") = 32);
}
