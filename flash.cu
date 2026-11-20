// FlashAttention forward pass from scratch (CUDA, FP32)
//
// Computes O = softmax(Q K^T / sqrt(d)) V without ever building the N x N score matrix.
//   * Tiling:        K and V are processed in small tiles held in shared memory.
//   * Online softmax: a running max (m) and running sum (l) are updated tile by tile,
//                     and the partial output is rescaled, so the result is exact.
//   * Memory:        only the output O is allocated -> extra memory is O(N), not O(N^2).
//
// Layout: Q, K, V, O are contiguous float32 tensors of shape (B, H, N, D).
// Supported head sizes D: 32, 64, 128.  Optional causal mask.
//
// Mapping: one thread owns one query row (its q[] and o[] live in registers).
// A block of BR threads handles BR query rows and loops over all K/V tiles.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>

template <int D, int BR, int BC>
__global__ void __launch_bounds__(BR)
flash_fwd_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                 const float* __restrict__ V, float* __restrict__ O,
                 int N, float scale, bool causal)
{
    __shared__ float Ks[BC][D];
    __shared__ float Vs[BC][D];

    const int    tid    = threadIdx.x;
    const int    qi     = blockIdx.x * BR + tid;          // query row owned by this thread
    const size_t base   = (size_t)blockIdx.y * N * D;     // start of this (batch, head)
    const bool   active = qi < N;

    float q[D], o[D];
    #pragma unroll
    for (int d = 0; d < D; d++) {
        o[d] = 0.f;
        q[d] = active ? Q[base + (size_t)qi * D + d] * scale : 0.f;   // fold 1/sqrt(D) into q
    }
    float m = -INFINITY;   // running row max
    float l = 0.f;         // running row sum of exp

    // With a causal mask, no key beyond the last query row of this block is ever needed.
    int kv_end = N;
    if (causal) kv_end = min(N, (int)((blockIdx.x + 1) * BR));

    for (int kv0 = 0; kv0 < kv_end; kv0 += BC) {
        __syncthreads();                                   // everyone finished with the old tile
        for (int idx = tid; idx < BC * D; idx += BR) {     // cooperative load of K and V tiles
            int r = idx / D, c = idx % D, row = kv0 + r;
            float kk = 0.f, vv = 0.f;
            if (row < N) { kk = K[base + (size_t)row * D + c]; vv = V[base + (size_t)row * D + c]; }
            Ks[r][c] = kk;
            Vs[r][c] = vv;
        }
        __syncthreads();                                   // tile is ready

        if (active) {
            // 1) scores of this query row against the BC keys in the tile
            float s[BC];
            float m_tile = -INFINITY;
            #pragma unroll
            for (int j = 0; j < BC; j++) {
                float acc = 0.f;
                #pragma unroll
                for (int d = 0; d < D; d++) acc += q[d] * Ks[j][d];
                int key = kv0 + j;
                bool masked = (key >= N) || (causal && key > qi);
                s[j] = masked ? -INFINITY : acc;
                m_tile = fmaxf(m_tile, s[j]);
            }

            // 2) online softmax update
            float m_new = fmaxf(m, m_tile);
            if (m_new > -INFINITY) {
                float alpha = expf(m - m_new);             // rescale what we had so far
                l *= alpha;
                #pragma unroll
                for (int d = 0; d < D; d++) o[d] *= alpha;
                #pragma unroll
                for (int j = 0; j < BC; j++) {
                    float p = expf(s[j] - m_new);          // masked entries give exactly 0
                    l += p;
                    #pragma unroll
                    for (int d = 0; d < D; d++) o[d] += p * Vs[j][d];
                }
                m = m_new;
            }
        }
    }

    if (active) {
        float inv = 1.f / l;
        #pragma unroll
        for (int d = 0; d < D; d++) O[base + (size_t)qi * D + d] = o[d] * inv;
    }
}

torch::Tensor flash_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat32 && K.dtype() == torch::kFloat32 &&
                V.dtype() == torch::kFloat32, "only float32 is supported");
    TORCH_CHECK(Q.dim() == 4, "expected shape (B, H, N, D)");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q, K, V must have the same shape");
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();

    const int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    auto O = torch::empty_like(Q);
    const float scale = 1.0f / sqrtf((float)D);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH(D_, BR_, BC_)                                                           \
    {                                                                                  \
        dim3 grid((N + BR_ - 1) / BR_, B * H);                                         \
        flash_fwd_kernel<D_, BR_, BC_><<<grid, BR_, 0, stream>>>(                      \
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),             \
            O.data_ptr<float>(), N, scale, causal);                                    \
    }
    if      (D == 32)  LAUNCH(32, 128, 32)
    else if (D == 64)  LAUNCH(64, 128, 32)
    else if (D == 128) LAUNCH(128, 64, 16)
    else TORCH_CHECK(false, "head size D must be 32, 64 or 128");
#undef LAUNCH

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "flash_fwd_kernel launch failed");
    return O;
}
