// topk_swap_map.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <pybind11/pybind11.h>
#include <type_traits>
namespace py = pybind11;

// Scalar type mapping
template <typename T> struct ScalarTypeOf;
template <> struct ScalarTypeOf<int64_t> { static constexpr at::ScalarType value = at::kLong; };
template <> struct ScalarTypeOf<int32_t> { static constexpr at::ScalarType value = at::kInt;  };
template <> struct ScalarTypeOf<int16_t> { static constexpr at::ScalarType value = at::kShort; };

template <typename index_t, int THREADS>
__global__ void topk_swap_map_kernel(
    const int64_t* __restrict__ sel_idx,          // [R_sel]
    const index_t* __restrict__ old_topk,         // [R_all, L, H, K] (logical layout as in your caller)
    const index_t* __restrict__ new_topk,         // [R_all, L, H, K]
    const int32_t* __restrict__ total_full_per_req, // [R_all]
    index_t* __restrict__ evicted_out,            // [R_sel, L, H, K]
    index_t* __restrict__ incoming_out,           // [R_sel, L, H, K]
    int16_t* __restrict__ counts_out,             // [R_sel, L, H]
    int32_t* __restrict__ gpu_tbl,                // [R_all, L, H, MAX_LOG]
    const int32_t* __restrict__ cpu_tbl,          // [R_all, L, H, MAX_LOG]
    const int64_t* __restrict__ start_cpu,        // [L]
    const int64_t* __restrict__ start_gpu,        // [L]
    const uint32_t* __restrict__ unstable_masks,  // [L] or nullptr
    int R_all, int R_sel, int L, int H, int K, int MAX_LOG,
    index_t pad_val,
    int32_t* __restrict__ out_src_global,         // [<= R_sel*L*H*K]
    int32_t* __restrict__ out_dst_global,         // [<= ...]
    unsigned long long* __restrict__ global_write_ptr, // single counter
    int32_t* __restrict__ g2g_src_global,       // [<= R_sel*L*H] (at most one per (r,l,h))
    int32_t* __restrict__ g2g_dst_global,         // [<= ...]
    unsigned long long* __restrict__ g2g_write_ptr
) {
    const int r = blockIdx.x;
    const int l = blockIdx.y;
    const int h = blockIdx.z;
    if (r >= R_sel || l >= L || h >= H) return;

    if (unstable_masks) {
        const uint32_t m = unstable_masks[l];
        if ((m >> h) & 1u) {
            if (threadIdx.x == 0) {
                const int64_t base_c = ((int64_t)r * L + l) * H + h;
                counts_out[base_c] = 0;
            }
            if (threadIdx.x < K) {
                const int64_t base_ei = (((int64_t)r * L + l) * H + h) * (int64_t)K;
                evicted_out [base_ei + threadIdx.x] = pad_val;
                incoming_out[base_ei + threadIdx.x] = pad_val;
            }
            return;
        }
    }

    const int64_t ridx = sel_idx[r];

    const int total_full = (int) total_full_per_req[ridx];

    const int64_t base_in   = (((int64_t)r    * L + l) * H + h) * (int64_t)K;
    const int64_t base_cnt  =  ((int64_t)r    * L + l) * H + h;
    const int64_t base_tbl  = (((int64_t)ridx * L + l) * H + h) * (int64_t)MAX_LOG;

    const index_t* old_row = old_topk + (((int64_t)l * (int64_t)R_all + (int64_t)ridx) * (int64_t)H + (int64_t)h) * (int64_t)K;
    const index_t* new_row = new_topk + (((int64_t)l * (int64_t)R_all + (int64_t)ridx) * (int64_t)H + (int64_t)h) * (int64_t)K;

    extern __shared__ unsigned char smem_raw[];
    index_t* old_s = reinterpret_cast<index_t*>(smem_raw);
    index_t* new_s = old_s + K;
    index_t* ev_s  = new_s + K;
    index_t* in_s  = ev_s  + K;

    __shared__ int partial_needed;          // 0/1
    if (threadIdx.x == 0) partial_needed = 0;

    if (threadIdx.x < K) {
        evicted_out [base_in + threadIdx.x] = pad_val;
        incoming_out[base_in + threadIdx.x] = pad_val;
    }

    if (threadIdx.x < K) {
        old_s[threadIdx.x] = old_row[threadIdx.x];
        new_s[threadIdx.x] = new_row[threadIdx.x];
    }
    __syncthreads();

    int new_only_flag = 0;
    int old_only_flag = 0;

    // Detect if the "partial" logical block (index == total_full) is requested in new_topk
    // but was not present in old_topk. We do not include it in CPU->GPU path.
    if (threadIdx.x < K) {
        index_t v = new_s[threadIdx.x];
        if ((int)v == total_full) {
            bool in_old = false;
            #pragma unroll
            for (int j = 0; j < K; ++j) { if (v == old_s[j]) { in_old = true; break; } }
            if (!in_old) {
                // one thread may set this; at most one such v can exist
                atomicExch(&partial_needed, 1);
            }
        }
    }

    if (threadIdx.x < K) {
        index_t v = new_s[threadIdx.x];
        bool in_old = false;
        #pragma unroll
        for (int j = 0; j < K; ++j) { if (v == old_s[j]) { in_old = true; break; } }
        new_only_flag = (!in_old) && ((int)v < total_full);
    }
    if (threadIdx.x < K) {
        index_t v = old_s[threadIdx.x];
        bool in_new = false;
        #pragma unroll
        for (int j = 0; j < K; ++j) { if (v == new_s[j]) { in_new = true; break; } }
        old_only_flag = !in_new;
    }

    using BlockScan = cub::BlockScan<int, THREADS>;
    __shared__ typename BlockScan::TempStorage scan_tmp1;
    __shared__ typename BlockScan::TempStorage scan_tmp2;

    int new_pos = 0, new_total = 0;
    int old_pos = 0, old_total = 0;

    BlockScan(scan_tmp1).ExclusiveSum(new_only_flag, new_pos, new_total);
    BlockScan(scan_tmp2).ExclusiveSum(old_only_flag, old_pos, old_total);

    if (threadIdx.x < K) {
        if (new_only_flag) {
            in_s[new_pos] = new_s[threadIdx.x];
            incoming_out[base_in + new_pos] = new_s[threadIdx.x];
        }
        if (old_only_flag) {
            ev_s[old_pos] = old_s[threadIdx.x];
            evicted_out [base_in + old_pos] = old_s[threadIdx.x];
        }
    }

    __shared__ int cnt_in, cnt_old, cnt;
    if (threadIdx.x == THREADS - 1) {
        cnt_in  = new_pos + new_only_flag;   // filtered incoming
        cnt_old = old_pos + old_only_flag;   // outgoing
        cnt     = cnt_in < cnt_old ? cnt_in : cnt_old;
        if (cnt > K) cnt = K;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        counts_out[base_cnt] = static_cast<int16_t>(cnt);
    }

    if (cnt > 0) {
        for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
            const int ev_log = static_cast<int>(ev_s[i]);
            const int in_log = static_cast<int>(in_s[i]);
            if (ev_log >= 0 && ev_log < MAX_LOG && in_log >= 0 && in_log < MAX_LOG) {
                const int32_t phys = gpu_tbl[base_tbl + ev_log];
                gpu_tbl[base_tbl + in_log] = phys;
            }
        }
        __syncthreads();

        __shared__ unsigned long long out_base;
        if (threadIdx.x == 0) {
            out_base = atomicAdd(global_write_ptr, static_cast<unsigned long long>(cnt));
        }
        __syncthreads();

        const int64_t start_cpu_l = start_cpu[l];
        const int64_t start_gpu_l = start_gpu[l];

        for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
            const int in_log = static_cast<int>(in_s[i]);
            const int ev_log = static_cast<int>(ev_s[i]);
            if (in_log >= 0 && in_log < MAX_LOG && ev_log >= 0 && ev_log < MAX_LOG) {
                const int32_t cpu_phy = cpu_tbl[base_tbl + in_log];
                const int32_t gpu_phy = gpu_tbl[base_tbl + ev_log]; // not overwritten
                const int32_t g_cpu = static_cast<int32_t>(start_cpu_l + (int64_t)cpu_phy);
                const int32_t g_gpu = static_cast<int32_t>(start_gpu_l + (int64_t)gpu_phy);
                out_src_global[out_base + i] = g_cpu;
                out_dst_global[out_base + i] = g_gpu;
            }
        }
    }

    // === Handle the PARTIAL block via GPU->GPU ===
    // Use one leftover evicted slot (if any) after pairing the fulls.
    __syncthreads();
    if (threadIdx.x == 0) {
        if (partial_needed) {
            // old_total may be >= cnt; if strictly greater, we have leftovers
            if (cnt_old > cnt) {
                const int ev_log_partial_dst = static_cast<int>(ev_s[cnt]); // first leftover evicted
                // Source physical: current mapping of partial logical block (total_full)
                const int32_t src_phys = gpu_tbl[base_tbl + total_full];
                // Destination physical: phys formerly used by evicted logical
                const int32_t dst_phys = gpu_tbl[base_tbl + ev_log_partial_dst];

                // Update GPU block table: partial logical now uses dst_phys
                gpu_tbl[base_tbl + total_full] = dst_phys;

                // Enqueue a single g2g copy pair using global indices
                const int64_t start_gpu_l = start_gpu[l];
                const int32_t g_src = static_cast<int32_t>(start_gpu_l + (int64_t)src_phys);
                const int32_t g_dst = static_cast<int32_t>(start_gpu_l + (int64_t)dst_phys);

                const unsigned long long w = atomicAdd(g2g_write_ptr, 1ULL);
                g2g_src_global[w] = g_src;
                g2g_dst_global[w] = g_dst;

                // Optionally annotate outgoing/ incoming tensors for visibility (not required)
                // evicted_out[base_in + cnt]  = static_cast<index_t>(ev_log_partial_dst);
                // incoming_out[base_in + cnt] = static_cast<index_t>(total_full);
                // NOTE: counts_out remains the number of CPU->GPU pairs (cnt)
            }
            // else: no leftover evicted slot; this should be extremely rare since
            //       K_new = overlap + full_incoming + (partial?1:0) and
            //       old_total = K_old - overlap >= full_incoming (+1 if partial).
        }
    }
}

template <typename index_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, at::Tensor, at::Tensor, int64_t>
topk_swap_map_launcher(
    at::Tensor old_topk, at::Tensor new_topk, at::Tensor sel_idx,
    at::Tensor total_full_per_req,  // [R_all] int32, CUDA
    at::Tensor gpu_block_table, at::Tensor cpu_block_table,
    at::Tensor start_cpu, at::Tensor start_gpu,
    c10::optional<at::Tensor> unstable_masks_opt,
    index_t pad_val
) {
    TORCH_CHECK(old_topk.is_cuda() && new_topk.is_cuda(), "old/new must be CUDA");
    TORCH_CHECK(sel_idx.is_cuda() && sel_idx.dtype() == at::kLong, "sel_idx must be CUDA int64");
    TORCH_CHECK(gpu_block_table.is_cuda() && cpu_block_table.is_cuda(), "tables must be CUDA");
    TORCH_CHECK(start_cpu.is_cuda() && start_gpu.is_cuda(), "starts must be CUDA");
    TORCH_CHECK(old_topk.sizes() == new_topk.sizes(), "top-k shape mismatch");
    TORCH_CHECK(old_topk.scalar_type() == ScalarTypeOf<index_t>::value, "top-k dtype mismatch");
    TORCH_CHECK(total_full_per_req.is_cuda(), "total_full_per_req must be CUDA");

    c10::cuda::CUDAGuard guard(old_topk.device());

    const int64_t L       = old_topk.size(0);
    const int64_t R_all   = old_topk.size(1);
    const int64_t H       = old_topk.size(2);
    const int64_t K       = old_topk.size(3);
    TORCH_CHECK(K > 0 && K <= 256, "K must be in (0,256]");

    TORCH_CHECK(gpu_block_table.size(0) == R_all && gpu_block_table.size(1) == L &&
                gpu_block_table.size(2) == H, "gpu_table shape mismatch");
    TORCH_CHECK(cpu_block_table.sizes() == gpu_block_table.sizes(), "tables must match");
    const int64_t MAX_LOG = gpu_block_table.size(3);

    const int64_t R_sel = sel_idx.size(0);
    TORCH_CHECK(R_sel > 0, "R_sel must be > 0");

    auto opts_idx   = old_topk.options();
    auto opts_i16   = old_topk.options().dtype(at::kShort);
    auto opts_i32   = old_topk.options().dtype(at::kInt);

    at::Tensor evicted  = at::empty({R_sel, L, H, K}, opts_idx);
    at::Tensor incoming = at::empty({R_sel, L, H, K}, opts_idx);
    at::Tensor counts   = at::empty({R_sel, L, H},    opts_i16);

    const int64_t max_pairs = R_sel * L * H * K;
    at::Tensor src_global = at::empty({max_pairs}, opts_i32);
    at::Tensor dst_global = at::empty({max_pairs}, opts_i32);
    at::Tensor counter    = at::zeros({1}, sel_idx.options().dtype(at::kLong));

    // At most one g2g pair per (r,l,h), so upper-bound by R_sel*L*H
    const int64_t max_g2g = R_sel * L * H;
    at::Tensor g2g_src_global = at::empty({max_g2g}, opts_i32);
    at::Tensor g2g_dst_global = at::empty({max_g2g}, opts_i32);
    at::Tensor g2g_counter    = at::zeros({1}, sel_idx.options().dtype(at::kLong));

    const uint32_t* unstable_ptr = nullptr;
    at::Tensor unstable;
    if (unstable_masks_opt.has_value() && unstable_masks_opt->defined() && unstable_masks_opt->numel() > 0) {
        unstable = unstable_masks_opt->contiguous();
        TORCH_CHECK(unstable.is_cuda(), "unstable_masks must be CUDA");
        if (unstable.scalar_type() != at::kInt) unstable = unstable.to(at::kInt);
        TORCH_CHECK((int)unstable.size(0) == L, "unstable_masks expected [L]");
        unstable_ptr = reinterpret_cast<const uint32_t*>(unstable.data_ptr<int32_t>());
    }

    constexpr int THREADS = 256;
    dim3 grid((unsigned)R_sel, (unsigned)L, (unsigned)H);
    dim3 block(THREADS);
    size_t shmem = (size_t)(4 * K) * sizeof(index_t);

    auto stream = at::cuda::getCurrentCUDAStream();
    topk_swap_map_kernel<index_t, THREADS><<<grid, block, shmem, stream>>>(
        sel_idx.data_ptr<int64_t>(),
        old_topk.data_ptr<index_t>(),
        new_topk.data_ptr<index_t>(),
        total_full_per_req.data_ptr<int32_t>(),
        evicted.data_ptr<index_t>(),
        incoming.data_ptr<index_t>(),
        counts.data_ptr<int16_t>(),
        gpu_block_table.data_ptr<int32_t>(),
        cpu_block_table.data_ptr<int32_t>(),
        start_cpu.data_ptr<int64_t>(),
        start_gpu.data_ptr<int64_t>(),
        unstable_ptr,
        (int)R_all, (int)R_sel, (int)L, (int)H, (int)K, (int)MAX_LOG,
        pad_val,
        src_global.data_ptr<int32_t>(),
        dst_global.data_ptr<int32_t>(),
        reinterpret_cast<unsigned long long*>(counter.data_ptr<int64_t>()),
        g2g_src_global.data_ptr<int32_t>(),
        g2g_dst_global.data_ptr<int32_t>(),
        reinterpret_cast<unsigned long long*>(g2g_counter.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t total_pairs = counter.to(at::kCPU).item<int64_t>();
    auto src_trim = src_global.narrow(0, 0, total_pairs);
    auto dst_trim = dst_global.narrow(0, 0, total_pairs);
    const int64_t total_g2g = g2g_counter.to(at::kCPU).item<int64_t>();
    auto g2g_src_trim = g2g_src_global.narrow(0, 0, total_g2g);
    auto g2g_dst_trim = g2g_dst_global.narrow(0, 0, total_g2g);
    return {evicted, incoming, counts, src_trim, dst_trim, total_pairs,
            g2g_src_trim, g2g_dst_trim, total_g2g};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_swap_map_launcher_int64",
          &topk_swap_map_launcher<int64_t>,
          "Fused diff+update+build (int64 indices)");
    m.def("topk_swap_map_launcher_int32",
          &topk_swap_map_launcher<int32_t>,
          "Fused diff+update+build (int32 indices)");
    m.def("topk_swap_map_launcher_int16",
          &topk_swap_map_launcher<int16_t>,
          "Fused diff+update+build (int16 indices)");
}
