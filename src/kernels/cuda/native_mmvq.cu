// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/core/sm75_series.hpp"
#include "s26_tsum.cuh"
#include "slots.cuh"
#include "strata/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <unordered_map>
#include <cstdio>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;

struct Q5KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q20Block {
    half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};
struct IQ4XSBlock {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
struct Q2KBlock {
    uint8_t scales[16];
    uint8_t qs[64];
    half2 dm;
};
struct Q40Block {
    half d;
    uint8_t qs[16];
};
struct Q50Block {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    half d;
    uint8_t qs[16];
};
// MXFP4: a byte of exponent, then 32 four-bit codes.  No alignment: a run of them has blocks at every byte.
struct MXFP4Block {
    uint8_t e;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q2KBlock) == 84 && alignof(Q2KBlock) == 4 &&
              offsetof(Q2KBlock, qs) == 16 && offsetof(Q2KBlock, dm) == 80);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);
static_assert(sizeof(MXFP4Block) == 17 && alignof(MXFP4Block) == 1 && offsetof(MXFP4Block, qs) == 1);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP));
    }
    return x;
}

__launch_bounds__(QUANT_THREADS, 1)
__global__ void native_quantize_q8_1_kernel(const float* x_, Q81Block* y_, int n_in) {
    const float* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    Q81Block* STRATA_PDL_RESTRICT y = y_;
    pdl_trigger();   // PDL (pdl.hpp): the projection after this one may start loading its weights
    pdl_wait();
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

// native_add_rms_norm_q8_1: a block a row.  The sum of squares, the scale and the product are poly's rms_norm_kernel
// with its block size (SUM threads of the block) and its IEEE operations written out (this file is built with
// --use_fast_math).  Then every thread of the block takes values, a warp 32 of one Q8_1 block, and quantizes them as
// native_quantize_q8_1_kernel does: more threads than the sum's make the row's quantization fewer steps.
template<int SUM>
__launch_bounds__(1024, 1)
__global__ void native_add_rms_norm_q8_1_kernel(float* __restrict__ x, const float* __restrict__ add,
                                                const float* __restrict__ w, float* __restrict__ y,
                                                Q81Block* __restrict__ yq, int n, float eps) {
    __shared__ float part[SUM / WARP];
    float* xr = x + std::size_t(blockIdx.x) * n;
    const float* ar = add ? add + std::size_t(blockIdx.x) * n : nullptr;
    float* yr = y + std::size_t(blockIdx.x) * n;
    Q81Block* qr = yq + std::size_t(blockIdx.x) * (n / Q8K);
    if (threadIdx.x < SUM) {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < n; i += SUM) {
            float v = xr[i];
            if (ar) xr[i] = v = __fadd_rn(v, ar[i]);
            ss = __fmaf_rn(v, v, ss);
        }
        ss = warp_sum(ss);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    }
    __syncthreads();
    float sum = 0.0f;
    for (int k = 0; k < SUM / WARP; ++k) sum = __fadd_rn(sum, part[k]);
    const float scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(n)), eps)));
    for (int i = threadIdx.x; i < n; i += blockDim.x) {   // n is a multiple of 32: a warp's 32 values are one block
        const float xi = __fmul_rn(__fmul_rn(xr[i], scale), w[i]);
        yr[i] = xi;
        const float amax = warp_max(fabsf(xi));
        const float bsum = warp_sum(xi);
        const float d = q8_1_finite(amax / 127.0f);
        const int8_t q = q8_1_quant(xi, d, amax);
        qr[i / Q8K].qs[i % Q8K] = q;
        if (i % Q8K == 0) qr[i / Q8K].ds = q8_1_ds(d, bsum);
    }
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q5_q8_dot_impl(
    const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm5,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = STRATA_DP4A(v0i, u[2 * i], STRATA_DP4A(v1i, u[2 * i + 1], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i], STRATA_DP4A(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__device__ __forceinline__ float q5_q8_dot(const Q5KBlock* __restrict__ bq5,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, m, bq5->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q5_k_mmvq_kernel(const Q5KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.
// The weight block is only 2-byte aligned, so qs is intentionally loaded as
// int16_t, unlike the naturally 4-byte aligned activation codes.
__device__ __forceinline__ float q2_q8_dot(const Q20Block* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const float d2 = w->d;
    const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
    const Q81Block* chunk = x + iqs;
    const int* q8 = reinterpret_cast<const int*>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2];
        const int v = q8[j * 2 + 1];
        const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
        const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = STRATA_DP4A(u, qx, sumi);
        sumi = STRATA_DP4A(v, qy, sumi);
    }
    const float d8 = __low2float(chunk->ds);
    return d2 * d8 * sumi;
}

// Q2_0 generic MMVQ: QK=64, QI=2, VDR=1, 64 blocks per iteration.
// Preserve the same outer accumulation and cross-warp reduction as the oracle.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q2_0_mmvq_kernel(const Q20Block* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 2;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 64;
    float tmp[ROWS] = {};
    for (int kbx = tid / 2; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 2;
        const int kqs = tid % 2;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q2_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__device__ __forceinline__ float q3_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const uint8_t* __restrict__ scales,
                                              int scale_offset, float d3,
                                              const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

__device__ __forceinline__ float q3_q8_dot(const Q3KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = w->d;
    const int vl = load_int_b2(w->qs, iqs);
    const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + i].ds);
    }
    return q3_q8_dot_impl(vl, vh, u, w->scales, scale_offset, d, d8);
}

// Q3_K generic MMVQ: QK=256, QI=16, VDR=1, eight blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q3_k_mmvq_kernel(const Q3KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 16;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 16;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q3_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Q2_K: llama.cpp's vec_dot_q2_K_q8_1 in its mmvq form, same float accumulation order.
__device__ __forceinline__ float q2k_q8_dot(const Q2KBlock* __restrict__ w,
                                           const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const uint8_t* scales = w->scales + (iqs - iqs % 8 + (iqs % 8) / 4);
    const int v = reinterpret_cast<const int*>(w->qs)[iqs];
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int u = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        const float d8 = __low2float(x[bq8_offset + i].ds);
        const int sc = scales[2 * i];
        const int vi = (v >> (2 * i)) & 0x03030303;
        sumf_d += d8 * (STRATA_DP4A(vi, u, 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += d8 * STRATA_DP4A(m, u, 0);
    }
    const float2 dm = __half22float2(w->dm);
    return dm.x * sumf_d - dm.y * sumf_m;
}

// Q2_K generic MMVQ: QK=256, QI=16, VDR=1, eight blocks per iteration, as Q3_K.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q2_k_mmvq_kernel(const Q2KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 16;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 16;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q2k_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The pinned nonlinear IQ4 codebook and its CUDA two-stage byte lookup. The
// explicit alignment satisfies the four 32-bit table loads; values are unchanged.
__device__ __align__(4) int8_t iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// MXFP4's codes are E2M1 values doubled, so a block's scale is its power of two halved: llama.cpp's kvalues_mxfp4.
__device__ __align__(4) int8_t mxfp4_values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// Eight four-bit codes of q4 looked up in a codebook of 16: the even nibbles' values in .x, the odd ones' in .y.
__device__ __forceinline__ int2 table16_lookup(const int8_t* table, int q4) {
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(table);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

__device__ __forceinline__ int2 iq4_table_lookup(int q4) { return table16_lookup(iq4nl_values, q4); }

// Four bytes from any byte address, little-endian, as llama.cpp's get_int_b1.
__device__ __forceinline__ int load_int_b1(const void* ptr, int i32) {
    const auto* x = static_cast<const uint8_t*>(ptr);
    return x[4 * i32] | x[4 * i32 + 1] << 8 | x[4 * i32 + 2] << 16 | x[4 * i32 + 3] << 24;
}

// A block's power of two, halved, as llama.cpp's ggml_cuda_e8m0_to_fp32(e) * 0.5f.
__device__ __forceinline__ float mxfp4_scale(uint8_t e) {
    return __uint_as_float(e == 0 ? 0x00400000u : (uint32_t) e << 23) * 0.5f;
}

// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock,
// computes integer dot products, applies signed scale in the integer domain,
// then multiplies the two half scales and integer sum in the original order.
__device__ __forceinline__ float iq4_xs_q8_dot(const IQ4XSBlock* __restrict__ w,
                                              const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = reinterpret_cast<const int*>(w->qs)[iqs + j];
        const int2 v = iq4_table_lookup(aux_q4);
        const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
        const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        sumi = STRATA_DP4A(v.x, u0, sumi);
        sumi = STRATA_DP4A(v.y, u1, sumi);
    }
    const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
                   (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(w->d) * __low2float(x[iqs / 4].ds);
    return d * sumi;
}

// IQ4_XS generic MMVQ: QK=256, QI=32, VDR=4,16 blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_iq4_xs_mmvq_kernel(const IQ4XSBlock* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 4 * WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 8; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = 4 * (tid % 8);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += iq4_xs_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q4_q8_dot_impl(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm4,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = STRATA_DP4A(v1i, u[2 * i + 1], STRATA_DP4A(v0i, u[2 * i], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i + 1], STRATA_DP4A(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float q4_q8_dot(const Q4KBlock* __restrict__ bq4,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, m, bq4->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q4_k_mmvq_kernel(const Q4KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q4_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

__device__ __forceinline__ float q6_q8_dot(const Q6KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + 2 * i].ds);
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}

// Q6_K generic MMVQ: QK=256, QI=32, VDR=1, four blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q6_k_mmvq_kernel(const Q6KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q6_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The four 32-element formats use native two-byte loads and VDR=2. The affine
// Q4_0/Q5_0 correction consumes the original-input sum stored in Q8_1, exactly
// as the pinned CUDA dot does; a signed-integer code substitution would differ.
__device__ __forceinline__ float small_q8_dot(const Q40Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = STRATA_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        sumi = STRATA_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 4 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q50Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = STRATA_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = STRATA_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 8 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q80Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = STRATA_DP4A(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = __low2float(x->ds);
    return d0 * d1 * float(sumi);
}

__device__ __forceinline__ float small_q8_dot(const IQ4NLBlock* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 v = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        sumi = STRATA_DP4A(v.x, q8[i], sumi);
        sumi = STRATA_DP4A(v.y, q8[i + 4], sumi);
    }
    const float d = __half2float(w->d) * __low2float(x->ds);
    return d * sumi;
}

// llama.cpp's vec_dot_mxfp4_q8_1 with VDR 2.
__device__ __forceinline__ float small_q8_dot(const MXFP4Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 v = table16_lookup(mxfp4_values, load_int_b1(w->qs, iqs + i));
        sumi = STRATA_DP4A(v.x, q8[i], sumi);
        sumi = STRATA_DP4A(v.y, q8[i + 4], sumi);
    }
    const float d = mxfp4_scale(w->e) * __low2float(x->ds);
    return d * sumi;
}

// QI=4 for Q4_0/Q5_0/IQ4_NL/MXFP4 and QI=8 for Q8_0. With VDR=2 this preserves
// the pinned 64/32-block iteration and 2048/1024-element small-K thresholds.
template<typename Weight, int Qi, bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_small_mmvq_kernel(const Weight* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// ============================ plan v0.3 P3: ncols = 2..8 (speculative verify, small batches) ============================
//
// One generic kernel for every format, parameterized by the format's iteration traits below, which are
// transcribed from the ncols = 1 kernels above (same thread-to-block mapping, same blocks per iteration, same
// small-K rule). Column j reads activation blocks x + j * (n_in / 32) and writes y + j * n_out. Each (column,
// row) value is accumulated over kbx in the same order, summed across warps in the same order and reduced with
// the same warp tree as the ncols = 1 kernel, so every column is BITWISE equal to a single-column call on that
// column (checked by bench/micro/native_mmvq_multi.cpp). The ncols = 1 kernels are untouched.
constexpr int MAX_NCOLS = 8;

// Each format splits its dot product into `load` (everything that depends only on the weight block: codes,
// unpacked scales, block scale) and `apply` (the activation loads and the original *_impl expression). `load` runs
// once per (row, block) and `apply` once per column, so adding columns adds only activation work. `apply` calls
// the same impl functions, in the same order, with the same values as the ncols = 1 dot, which is what keeps
// every column bitwise equal to it.
struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int vl[2], vh[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    __device__ static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4], qy[4]; float d2; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.d2 = w->d;
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
            const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
            r.qx[j] = __byte_perm(qe, qo, 0x5140);
            r.qy[j] = __byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const Q81Block* chunk = x + iqs;
        const int* q8 = reinterpret_cast<const int*>(chunk->qs);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = STRATA_DP4A(q8[j * 2], r.qx[j], sumi);
            sumi = STRATA_DP4A(q8[j * 2 + 1], r.qy[j], sumi);
        }
        const float d8 = __low2float(chunk->ds);
        return r.d2 * d8 * sumi;
    }
};
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    __device__ static int kqs(int tid) { return tid % 16; }
    struct W { int vl, vh; float d; const uint8_t* scales; int scale_offset, bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d;
        r.vl = load_int_b2(w->qs, iqs);
        r.vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
        r.scales = w->scales;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + i].ds);
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { int2 v[4]; int ls; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(reinterpret_cast<const int*>(w->qs)[iqs + j]);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
            sumi = STRATA_DP4A(r.v[j].x, u0, sumi);
            sumi = STRATA_DP4A(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * __low2float(x[iqs / 4].ds);
        return d * sumi;
    }
};
// Q2_K keeps the block pointer: `apply` is the q2k_q8_dot of native_q2_k_mmvq_kernel, which has no multi-column
// form, so the launch of several matrices (native_mmvq_slots) gives that kernel's bits.
struct Q2KTraits {
    using Block = Q2KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    __device__ static int kqs(int tid) { return tid % 16; }
    struct W { const Block* w; };
    __device__ static W load(const Block* __restrict__ w, int) { return W{w}; }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) { return q2k_q8_dot(r.w, x, iqs); }
};
// The four 32-element formats: `load` decodes the two 32-bit weight chunks and block scale once per (row, block),
// and `apply` runs the exact same STRATA_DP4A and scale arithmetic per activation column as `small_q8_dot`.
template<typename Weight, int Qi>
struct SmallTraits;

template<>
struct SmallTraits<Q40Block, 4> {
    using Block = Q40Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int vi0[2], vi1[2]; float d; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = load_int_b2(w->qs, iqs + i);
            r.vi0[i] = (v >> 0) & 0x0f0f0f0f;
            r.vi1[i] = (v >> 4) & 0x0f0f0f0f;
        }
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.vi0[i], reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
            sumi = STRATA_DP4A(r.vi1[i], reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
        }
        const float2 ds = __half22float2(x->ds);
        return r.d * (sumi * ds.x - 4 * ds.y);
    }
};

template<>
struct SmallTraits<Q50Block, 4> {
    using Block = Q50Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int vi0[2], vi1[2]; float d; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        const int qh = load_int_b2(w->qh, 0);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vl = load_int_b2(w->qs, iqs + i);
            const int vh = qh >> (4 * (iqs + i));
            int vi0 = (vl >> 0) & 0x0f0f0f0f;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            r.vi0[i] = vi0;
            int vi1 = (vl >> 4) & 0x0f0f0f0f;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            r.vi1[i] = vi1;
        }
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.vi0[i], reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
            sumi = STRATA_DP4A(r.vi1[i], reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
        }
        const float2 ds = __half22float2(x->ds);
        return r.d * (sumi * ds.x - 8 * ds.y);
    }
};

template<>
struct SmallTraits<Q80Block, 8> {
    using Block = Q80Block;
    static constexpr int DIV = 32, T = 4, KBY = 1, BPI = 2 * WARPS * WARP / 8;
    __device__ static int kqs(int tid) { return 2 * (tid % 4); }
    struct W { int v[2]; float d0; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = load_int_b2(w->qs, iqs + i);
        r.d0 = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
            sumi = STRATA_DP4A(r.v[i], u, sumi);
        }
        const float d1 = __low2float(x->ds);
        return r.d0 * d1 * float(sumi);
    }
};

template<>
struct SmallTraits<IQ4NLBlock, 4> {
    using Block = IQ4NLBlock;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int2 v[2]; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.v[i].x, q8[i], sumi);
            sumi = STRATA_DP4A(r.v[i].y, q8[i + 4], sumi);
        }
        const float d = r.dw * __low2float(x->ds);
        return d * sumi;
    }
};

template<>
struct SmallTraits<MXFP4Block, 4> {
    using Block = MXFP4Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int2 v[2]; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = table16_lookup(mxfp4_values, load_int_b1(w->qs, iqs + i));
        r.dw = mxfp4_scale(w->e);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.v[i].x, q8[i], sumi);
            sumi = STRATA_DP4A(r.v[i].y, q8[i + 4], sumi);
        }
        const float d = r.dw * __low2float(x->ds);
        return d * sumi;
    }
};

// NW warps per block and ROWS rows per block. The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is the
// ncols = 1 layout and keeps every column bitwise equal to a single-column call. The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
// equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
bool g_multi_exact = true;   // until the upstream layout is timed on an idle GPU (plan rule: default only what is measured)
// B6 STRATA_B6_MMVQ_ROWS (lab, default 0 = off; 1 = 2 rows per block, 2 / 4 / 8 = that many rows)
int g_b6_rows = [] { const char* v = std::getenv("STRATA_B6_MMVQ_ROWS"); return v ? std::atoi(v) : 0; }();

// S26 STRATA_TSUM=1 (TS): warp 0's NCOLS x ROWS sums as one transposed butterfly (s26_tsum.cuh), bitwise the same
static bool s26_tsum_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    return on;
}

// S26 STRATA_LFUSE (PAIR): blocks n_out.. compute the same rows of w2 into y2 (two matrices of one shape on one
// input in one launch; each output's code is the single matrix's)
template<typename F, int NCOLS, int NW, int ROWS, bool TS = false, bool PAIR = false>
__launch_bounds__(NW * WARP, (ROWS <= 2 ? 4 : 1))
__global__ void native_mmvq_multi_kernel(const typename F::Block* __restrict__ w, const Q81Block* x_, float* y_,
                                         int n_in, int n_out,
                                         const typename F::Block* __restrict__ w2 = nullptr, float* __restrict__ y2 = nullptr) {
    const Q81Block* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    float* STRATA_PDL_RESTRICT y = y_;
    constexpr int BPI = F::BPI * NW / WARPS;           // blocks per iteration scale with the warp count
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    int bxi = int(blockIdx.x);
    if constexpr (PAIR) {
        if (bxi >= n_out / ROWS) { bxi -= n_out / ROWS; w = w2; y = y2; }
    }
    const int row0 = ROWS * bxi;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
    float tmp[NCOLS][ROWS] = {};
    int kbx = tid / F::T;
    if constexpr (kPdlPrefetch) {
        // PDL (pdl.hpp): the first block's weights are loaded before waiting for the activations, then that block
        // is applied exactly as the loop below would apply it (the same rows, columns and order)
        typename F::W w0[ROWS];
        const int kqs = F::kqs(tid);
        if (kbx < blocks_per_row) {
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                if (row0 + i < n_out) w0[i] = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
        }
        pdl_trigger();
        pdl_wait();
        if (kbx < blocks_per_row) {
            const int kby = kbx * F::KBY;
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                if (row0 + i < n_out) {
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j)
                        tmp[j][i] += F::apply(w0[i], x + std::size_t(j) * x_stride + kby, kqs);
                }
            }
        }
        kbx += BPI;
    }
    for (; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                const typename F::W wv = F::load(w + block, kqs);      // once per (row, block)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)                         // then per column
                    tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
    __shared__ float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
    if constexpr (TS) {
        constexpr int V = NCOLS * ROWS, P = s26ts::pow2_ceil(V);
        float v[P];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
#pragma unroll
                for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
                v[j * ROWS + i] = tmp[j][i];
            }
#pragma unroll
        for (int k = V; k < P; ++k) v[k] = 0.0f;
        const float sum = s26ts::tsum<P>(v, int(threadIdx.x));
        const int k = s26ts::tsum_token<P>(int(threadIdx.x));
        const int j = k / ROWS, i = k % ROWS;
        if (int(threadIdx.x) == s26ts::tsum_lane<P>(k) && k < V && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = sum;
        return;
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

// MODEL-UPDATE: the dense Q8_0 projections of Unsloth's files (all of attention, SSM and the shared expert are Q8_0
// there) take one row per block where that was measured faster - the same per-row sums and reduction as two rows per
// block (each row its own partials), so every output is the same bits.  q8_gemv_bench, 2-4 columns:
//   gfx1201 (R9700): n_in <= 4096 with n_out >= 4096 (qkv 2560->10240 73-79% -> 83-84% of the 644 GB/s, gate 2560->6144
//     51-68% -> 76-78%, q 2560->12288 74-80% -> 85-86%); 6144->2560 is slower there (79% -> 55-59%) and keeps two rows.
//   sm_86 (RTX A4000): every n_out >= 2048 (+1-4% at 2-4 columns, 6144->2560 at 6 columns 70% -> 76%).
// Other cards keep two rows until measured (STRATA_NO_MMVQ_ROWS2=1 forces one row everywhere; =0 keeps two everywhere).
int dev_class() {   // 1 = gfx12, 2 = sm_86, 0 = the rest; per device
    static int cls[16];   // 0 unknown, else class + 1
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 0;
    if (cls[dev] == 0) {
        int c = 0;
#if defined(__HIPCC__)
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, dev) == cudaSuccess && strata::kernels::gfx_arch_is_gfx12(p.gcnArchName)) c = 1;
#else
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess && major == 8 && minor == 6)
            c = 2;
#endif
        (void) cudaGetLastError();
        cls[dev] = c + 1;
    }
    return cls[dev] - 1;
}
template<typename F>
bool q8_0_rows1(int n_in, int n_out) {
    if constexpr (!std::is_same_v<F, SmallTraits<Q80Block, 8>>) {
        (void) n_in; (void) n_out;
        return false;
    } else {
        static const int env = [] {
            const char* v = std::getenv("STRATA_NO_MMVQ_ROWS2");
            return v == nullptr || v[0] == '\0' ? -1 : v[0] != '0';
        }();
        if (env == 0) return false;
        const int c = dev_class();
        return (c == 1 && n_in <= 4096 && n_out >= 4096) || (c == 2 && n_out >= 2048);
    }
}

// ---- B6 STRATA_B6_MMVQ_ROWS (lab, opt-in): the exact multi-column layout with each column's activation fragment
// loaded ONCE per weight block and reused across ROWS rows of the block (the default kernel reloads it for every row
// it applies it to, and the K-quant scales once per column).  A thread keeps the same blocks (kbx = tid / T, stride
// BPI with WARPS warps), each (row, column) sum adds the same per-block values in the same order, and the cross-warp
// and warp reductions are the default kernel's: every output is bitwise the default exact layout's, so bitwise a
// single-column call.  Only the formats with a B6X below; the others keep the default kernel.
template<typename F> struct B6X { static constexpr bool ok = false; };
template<> struct B6X<Q6KTraits> {
    static constexpr bool ok = true;
    struct X { int u[2]; float d8[2]; };
    struct W { int vl, vh, sc[2]; float d; };
    __device__ static X xload(const Q81Block* x, int iqs) {
        X r;
        const int off = 4 * (iqs / 16) + (iqs % 16) / 8;   // Q6KTraits::load's bq8_offset
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            r.u[i] = reinterpret_cast<const int*>(x[off + 2 * i].qs)[iqs % 8];
            r.d8[i] = __low2float(x[off + 2 * i].ds);
        }
        return r;
    }
    __device__ static W wload(const Q6KBlock* __restrict__ w, int iqs) {
        const Q6KTraits::W t = Q6KTraits::load(w, iqs);
        W r;
        r.vl = t.vl;
        r.vh = t.vh;
        r.sc[0] = t.scales[0];
        r.sc[1] = t.scales[4];
        r.d = t.d;
        return r;
    }
    __device__ static float dot(const W& w, const X& x) {   // q6_q8_dot_impl, the scales already in registers
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vil = (w.vl >> (4 * i)) & 0x0f0f0f0f;
            const int vih = ((w.vh >> (4 * i)) << 4) & 0x30303030;
            const int vi = __vsubss4(vil | vih, 0x20202020);
            sumf += x.d8[i] * (STRATA_DP4A(vi, x.u[i], 0) * w.sc[i]);
        }
        return w.d * sumf;
    }
};
template<typename T, int KIND> struct B6XK {   // Q4_K (KIND 4) and Q5_K (KIND 5): the same activation fragment
    static constexpr bool ok = true;
    struct X { int u[4]; float d8[2]; };
    using W = typename T::W;
    __device__ static X xload(const Q81Block* bq8, int iqs) {
        X r;
        const int off = 2 * ((iqs / 2) / 4);   // the traits' bq8_offset
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + off + i;
            r.d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            r.u[2 * i] = q8[0];
            r.u[2 * i + 1] = q8[4];
        }
        return r;
    }
    __device__ static W wload(const typename T::Block* __restrict__ w, int iqs) { return T::load(w, iqs); }
    __device__ static float dot(const W& w, const X& x) {
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(w.aux);
        if constexpr (KIND == 4) return q4_q8_dot_impl(w.v, x.u, sc, sc + 2, w.dm, x.d8);
        else return q5_q8_dot_impl(w.vl, w.vh, x.u, sc, sc + 2, w.dm, x.d8);
    }
};
template<> struct B6X<Q4KTraits> : B6XK<Q4KTraits, 4> {};
template<> struct B6X<Q5KTraits> : B6XK<Q5KTraits, 5> {};
template<> struct B6X<IQ4XSTraits> {
    static constexpr bool ok = true;
    struct X { int u0[4], u1[4]; float dx; };
    using W = IQ4XSTraits::W;
    __device__ static X xload(const Q81Block* x, int iqs) {
        X r;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            r.u0[j] = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            r.u1[j] = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        }
        r.dx = __low2float(x[iqs / 4].ds);
        return r;
    }
    __device__ static W wload(const IQ4XSBlock* __restrict__ w, int iqs) { return IQ4XSTraits::load(w, iqs); }
    __device__ static float dot(const W& w, const X& x) {   // IQ4XSTraits::apply
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = STRATA_DP4A(w.v[j].x, x.u0[j], sumi);
            sumi = STRATA_DP4A(w.v[j].y, x.u1[j], sumi);
        }
        sumi *= w.ls - 32;
        const float d = w.dw * x.dx;
        return d * sumi;
    }
};

template<typename F, int NCOLS, int ROWS>
__launch_bounds__(WARPS * WARP, 1)
__global__ void b6_mmvq_rows_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                    float* __restrict__ y, int n_in, int n_out) {
    using XB = B6X<F>;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    const int kqs = F::kqs(tid);
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
        const int kby = kbx * F::KBY;
        typename XB::X xv[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) xv[j] = XB::xload(x + std::size_t(j) * x_stride + kby, kqs);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const typename XB::W wv = XB::wload(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) tmp[j][i] += XB::dot(wv, xv[j]);
            }
        }
    }
    __shared__ float partial[WARPS - 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < WARPS - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

// the B6 launch for a format that has a B6X: true when it ran.  g_b6_rows: 1 = 2 rows per block (the fastest on both
// cards in b6_mmvq_bench), else 2 / 4 / 8 rows.
template<typename F, int NCOLS>
bool b6_launch(const typename F::Block* w, const Q81Block* x, float* y, int n_in, int n_out, cudaStream_t s) {
    if constexpr (!B6X<F>::ok) {
        return false;
    } else {
        if (g_b6_rows <= 0 || !g_multi_exact || s26_tsum_on() || n_in / F::DIV < 1) return false;
        const int rows = g_b6_rows == 1 ? 2 : g_b6_rows;
        const dim3 threads(WARP, WARPS);
        if (rows == 2)
            b6_mmvq_rows_kernel<F, NCOLS, 2><<<unsigned((n_out + 1) / 2), threads, 0, s>>>(w, x, y, n_in, n_out);
        else if (rows == 8)
            b6_mmvq_rows_kernel<F, NCOLS, 8><<<unsigned((n_out + 7) / 8), threads, 0, s>>>(w, x, y, n_in, n_out);
        else
            b6_mmvq_rows_kernel<F, NCOLS, 4><<<unsigned((n_out + 3) / 4), threads, 0, s>>>(w, x, y, n_in, n_out);
        return true;
    }
}

template<typename F, int NCOLS>
void launch_multi_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    if (b6_launch<F, NCOLS>(w, x, y, n_in, n_out, s)) return;   // STRATA_B6_MMVQ_ROWS
    if (!g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + 1) / 2);
        launch_pdl(native_mmvq_multi_kernel<F, NCOLS, NW, 2, false, false>, dim3(blocks), dim3(WARP, NW), 0, s, w, x, y, n_in, n_out,
                   (const typename F::Block*) nullptr, (float*) nullptr);
        return;
    }
    const dim3 threads(WARP, WARPS);
    // the PAIR arguments, unused here (launch_pdl passes every parameter)
    const typename F::Block* const no_w2 = nullptr;
    float* const no_y2 = nullptr;
    // #783 PR-h (stuchapin909): two rows per block for every K, not only small K - each row keeps its own partial sums and
    // reduction, so a row's result does not depend on its neighbour; STRATA_NO_MMVQ_ROWS2=1 keeps one row for large K
    static const bool rows1 = [] {
        const char* v = std::getenv("STRATA_NO_MMVQ_ROWS2");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if ((rows1 || q8_0_rows1<F>(n_in, n_out)) && n_in / F::DIV >= F::BPI) {
        if (s26_tsum_on()) launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, true, false>, dim3(unsigned(n_out)), threads, 0, s, w, x, y, n_in, n_out, no_w2, no_y2);
        else launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, false, false>, dim3(unsigned(n_out)), threads, 0, s, w, x, y, n_in, n_out, no_w2, no_y2);
        return;
    }
    constexpr int ROWS = 2;
    const unsigned blocks = unsigned((std::size_t(n_out) + ROWS - 1) / ROWS);
    if (s26_tsum_on()) launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, ROWS, true, false>, dim3(blocks), threads, 0, s, w, x, y, n_in, n_out, no_w2, no_y2);
    else launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, ROWS, false, false>, dim3(blocks), threads, 0, s, w, x, y, n_in, n_out, no_w2, no_y2);
}

template<typename F>
void launch_multi(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                  void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 2: launch_multi_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: launch_multi_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: launch_multi_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: launch_multi_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: launch_multi_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: launch_multi_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: launch_multi_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ multi-column launch requires 2 <= ncols <= 8");
    }
}

// Several matrices of one format and shape in one launch (native_mmvq_slots, native_mmvq_ids): blockIdx.y is the
// matrix, and the rest is native_mmvq_multi_kernel for one column in its EXACT layout.
template<typename F, int ROWS, typename S>
__launch_bounds__(WARPS * WARP, (ROWS <= 2 ? 4 : 1))
__global__ void native_mmvq_slots_kernel(S slots, int n_in, int n_out) {
    const SlotPtrs at = slot_ptrs(slots, int(blockIdx.y));
    const auto* w = static_cast<const typename F::Block*>(at.w);
    const auto* x = static_cast<const Q81Block*>(at.x);
    float* y = at.y;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / F::DIV;
    float tmp[ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += F::apply(F::load(w + block, kqs), x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// native_mmvq_slots_kernel when a row's blocks all fall to the first warp's lanes (blocks_per_row * T <= WARP, e.g. an
// expert's IQ4_XS down projection of 512 values: 2 blocks, 16 lanes): there the other warps only add zeros, so each
// warp takes rows of its own instead - the same lanes' terms, the same three zero partials and the same warp_sum,
// bit for bit, with a quarter of the blocks and no shared memory or barrier.  Qwen3.6 UD-Q3_K_M on an RX 7800 XT:
// the down products were 32% of a decode's GPU time at ~75 GB/s.  Taken where native_mmvq_slots_wide() says so.
template<typename F, int ROWS, typename S, int G = 1>
__launch_bounds__(WARPS * WARP)
__global__ void native_mmvq_slots_warp_kernel(S slots, int n_in, int n_out) {
    // G row groups a warp, WARP / G lanes each (G = 2 when a row needs 16 lanes or fewer: the IQ4_XS down projection's
    // 512 values are 16): the full warp_sum's steps between the groups added the zeros of the lanes that had none of
    // the row, so a group adds those zeros and sums its own lanes - the same bits again
    constexpr int L = WARP / G;
    const SlotPtrs at = slot_ptrs(slots, int(blockIdx.y));
    const auto* w = static_cast<const typename F::Block*>(at.w);
    const auto* x = static_cast<const Q81Block*>(at.x);
    float* y = at.y;
    const int lane = int(threadIdx.x), grp = lane / L, local = lane - grp * L;
    const int warp_row0 = ROWS * (int(blockIdx.x) * WARPS + int(threadIdx.y)) * G;
    if (warp_row0 >= n_out) return;   // whole warps: no barrier below
    const int row0 = warp_row0 + ROWS * grp;
    const int blocks_per_row = n_in / F::DIV;
    float tmp[ROWS] = {};
    if (row0 < n_out) {
        for (int kbx = local / F::T; kbx < blocks_per_row; kbx += F::BPI) {
            const int kby = kbx * F::KBY;
            const int kqs = F::kqs(local);
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                if (row0 + i < n_out) {
                    const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                    tmp[i] += F::apply(F::load(w + block, kqs), x + kby, kqs);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += 0.0f;   // the block kernel's partials of the other warps
        if constexpr (G == 1) {
            tmp[i] = warp_sum(tmp[i]);
        } else {
#pragma unroll
            for (int offset = WARP / 2; offset >= L; offset >>= 1) tmp[i] += 0.0f;   // the other group's lanes: zeros
#pragma unroll
            for (int offset = L / 2; offset > 0; offset >>= 1) tmp[i] += __shfl_xor_sync(0xffffffff, tmp[i], offset, WARP);
        }
        if (local == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

template<typename F, typename S>
void launch_slots(const S& slots, int count, int n_in, int n_out, cudaStream_t s) {
    const dim3 threads(WARP, WARPS);
    if (native_mmvq_slots_wide() && n_in / F::DIV * F::T <= WARP && n_in % F::DIV == 0) {
        constexpr int ROWS = 2;
        if (n_in / F::DIV * F::T <= WARP / 2) {   // a row in half a warp: two rows' groups a warp
            const dim3 blocks(unsigned((std::size_t(n_out) + 2 * ROWS * WARPS - 1) / (2 * ROWS * WARPS)), unsigned(count));
            native_mmvq_slots_warp_kernel<F, ROWS, S, 2><<<blocks, threads, 0, s>>>(slots, n_in, n_out);
        } else {
            const dim3 blocks(unsigned((std::size_t(n_out) + ROWS * WARPS - 1) / (ROWS * WARPS)), unsigned(count));
            native_mmvq_slots_warp_kernel<F, ROWS, S, 1><<<blocks, threads, 0, s>>>(slots, n_in, n_out);
        }
        return;
    }
    if (n_in / F::DIV < F::BPI) {
        constexpr int ROWS = 2;
        const dim3 blocks(unsigned((std::size_t(n_out) + ROWS - 1) / ROWS), unsigned(count));
        native_mmvq_slots_kernel<F, ROWS, S><<<blocks, threads, 0, s>>>(slots, n_in, n_out);
    } else {
        native_mmvq_slots_kernel<F, 1, S><<<dim3(unsigned(n_out), unsigned(count)), threads, 0, s>>>(slots, n_in, n_out);
    }
}

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("native MMVQ launch: ") + cudaGetErrorString(error));
    }
}


template<typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y,
                int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<SmallTraits<Weight, Qi>>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Weight*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 32 < 2 * WARPS * WARP / Qi) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_small_mmvq_kernel<Weight, Qi, true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_small_mmvq_kernel<Weight, Qi, false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

} // namespace

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    // one flat vector whatever ncols is (below): up to kMmvqSlotsMax columns, a slots launch's activations
    validate_shape(n_in, 1);
    if (ncols < 1 || ncols > kMmvqSlotsMax) throw std::invalid_argument("native_quantize_q8_1 requires 1 <= ncols <= 32");
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    launch_pdl(native_quantize_q8_1_kernel, dim3(blocks), dim3(QUANT_THREADS), 0, static_cast<cudaStream_t>(stream), x,
               static_cast<Q81Block*>(x_q8_1), n_total);
    launch_check();
}

void native_add_rms_norm_q8_1(float* x, const float* add, const float* w, float* y, void* y_q8_1, int n, int rows,
                              float eps, int threads, void* stream) {
    validate_shape(n, 1);
    validate_pointer(x);
    validate_pointer(w);
    validate_pointer(y);
    validate_pointer(y_q8_1);
    validate_stream(stream);
    if (rows < 1) return;
    const auto s = static_cast<cudaStream_t>(stream);
    auto* q = static_cast<Q81Block*>(y_q8_1);
    const int block = std::max(threads, std::min(1024, n));   // n is a multiple of 32, and so is this
    switch (threads) {
    case 128: native_add_rms_norm_q8_1_kernel<128><<<rows, block, 0, s>>>(x, add, w, y, q, n, eps); break;
    case 256: native_add_rms_norm_q8_1_kernel<256><<<rows, block, 0, s>>>(x, add, w, y, q, n, eps); break;
    case 512: native_add_rms_norm_q8_1_kernel<512><<<rows, block, 0, s>>>(x, add, w, y, q, n, eps); break;
    case 1024: native_add_rms_norm_q8_1_kernel<1024><<<rows, block, 0, s>>>(x, add, w, y, q, n, eps); break;
    default: throw std::invalid_argument("native_add_rms_norm_q8_1: 128, 256, 512 or 1024 threads");
    }
    launch_check();
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q5KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q5KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / QK < VDR * WARPS * WARP / QI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q5_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q5_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

namespace {
// Turing: Q2_0 one to four columns with one warp per RPW rows.  The exact layout gives each row four warps, a chunk of
// a 64-weight block per thread and a cross-warp reduction; at the engine's 2560 inputs (40 blocks a row) 48 of the 128
// threads have no chunk and the reduction is most of the time (QSA attn_q 2560x12288 at 3 columns: 80 us, 110 GB/s on
// an RTX 2080 Ti).  Here lane l runs the chunks of the exact kernel's threads l, 32+l, 64+l and 96+l in their loop
// order, adds the four sums as warp 0 adds the other warps' partials, then the same warp_sum: every output is bit for
// bit native_q2_0_mmvq_kernel's (one column) and launch_multi's exact layout (two to four).  80 -> 31 us at three
// columns, 37 -> 25 us at one.  STRATA_Q2_WARP=0 keeps the old kernels; =1 takes this one on any CUDA card.
constexpr int kQ20WarpRows = 4;
template <int NCOLS>
__global__ void __launch_bounds__(WARPS * WARP) q2_0_warp_rows_kernel(const Q20Block* __restrict__ w,
                                                                      const Q81Block* __restrict__ x,
                                                                      float* __restrict__ y, int n_in, int n_out) {
    using F = Q20Traits;
    constexpr int RPW = kQ20WarpRows;
    static_assert(F::T == 2 && F::BPI == WARPS * WARP / 2, "the lanes play the exact layout's threads");
    const int lane = int(threadIdx.x) & (WARP - 1);
    const int row0 = (int(blockIdx.x) * WARPS + int(threadIdx.x) / WARP) * RPW;
    if (row0 >= n_out) return;
    const int bpr = n_in / F::DIV, x_stride = n_in / Q8K;
    float acc[NCOLS][RPW];
#pragma unroll
    for (int v = 0; v < WARPS; ++v) {   // the exact kernel's thread WARP * v + lane
        const int tid = WARP * v + lane;
        const int kqs = F::kqs(tid);
        float tmp[NCOLS][RPW] = {};
        for (int kbx = tid / F::T; kbx < bpr; kbx += F::BPI) {
            const int kby = kbx * F::KBY;
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                if (row0 + i < n_out) {
                    const F::W wv = F::load(w + std::size_t(row0 + i) * bpr + kbx, kqs);
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j) tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < RPW; ++i) acc[j][i] = v == 0 ? tmp[j][i] : acc[j][i] + tmp[j][i];
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const float sum = warp_sum(acc[j][i]);
            if (lane == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = sum;
        }
}
bool q2_0_warp_rows_on() {
    static int on[16] = {};   // by device ordinal: 0 unknown, 1 yes, -1 no
    static const int env = [] { const char* v = std::getenv("STRATA_Q2_WARP"); return v ? (v[0] == '0' ? 0 : 1) : -1; }();
    if (env >= 0) return env == 1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    if (on[dev] == 0) {
        int maj = 0, mnr = 0;
        const bool ok = cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        if (!ok) cudaGetLastError();
        on[dev] = ok && strata::sm75_default(maj, mnr) ? 1 : -1;   // Turing only, with STRATA_SM75_PERF=1
    }
    return on[dev] > 0;
}
}  // namespace

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
#if !defined(__HIPCC__)
    // one column always (bitwise the single-column kernel); two to four only where the exact layout is the one in use
    if (ncols <= 4 && (ncols == 1 || g_multi_exact) && q2_0_warp_rows_on()) {
        const auto* w = static_cast<const Q20Block*>(weights);
        const auto* x = static_cast<const Q81Block*>(x_q8_1);
        const auto s = static_cast<cudaStream_t>(stream);
        const int warps = (n_out + kQ20WarpRows - 1) / kQ20WarpRows;
        const unsigned blocks = unsigned((warps + WARPS - 1) / WARPS);
        switch (ncols) {
            case 1: q2_0_warp_rows_kernel<1><<<blocks, WARPS * WARP, 0, s>>>(w, x, y, n_in, n_out); break;
            case 2: q2_0_warp_rows_kernel<2><<<blocks, WARPS * WARP, 0, s>>>(w, x, y, n_in, n_out); break;
            case 3: q2_0_warp_rows_kernel<3><<<blocks, WARPS * WARP, 0, s>>>(w, x, y, n_in, n_out); break;
            default: q2_0_warp_rows_kernel<4><<<blocks, WARPS * WARP, 0, s>>>(w, x, y, n_in, n_out); break;
        }
        launch_check();
        return;
    }
#endif
    if (ncols > 1) {
        launch_multi<Q20Traits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q20Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 64 < WARPS * WARP / 2) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q2_0_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q2_0_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q3KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q3KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q3_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q3_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

// Q2_K has no multi-column kernel yet: each column is its own single-column launch.
void native_q2_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* w = static_cast<const Q2KBlock*>(weights);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    for (int j = 0; j < ncols; ++j) {
        const auto* x = static_cast<const Q81Block*>(x_q8_1) + std::size_t(j) * (n_in / Q8K);
        float* yj = y + std::size_t(j) * n_out;
        if (n_in / 256 < WARPS * WARP / 16) {
            const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
            native_q2_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, yj, n_in, n_out);
        } else {
            native_q2_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, yj, n_in, n_out);
        }
    }
    launch_check();
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<IQ4XSTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < 4 * WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_iq4_xs_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_iq4_xs_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q4KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q4KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q4_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q4_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q6_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q6_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q40Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q50Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q80Block, 8>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<IQ4NLBlock, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 10 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29 || ggml_type == 39 || ggml_type == 40;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 7: block_elems = 32; block_bytes = 24; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 39: block_elems = 32; block_bytes = 17; break;
    case 10: block_elems = 256; block_bytes = 84; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 40: block_elems = 64; block_bytes = 36; break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 7: case 40: iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 39: small_mmvq<MXFP4Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 10: native_q2_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

namespace {

// The formats a slots launch has a kernel for: 1 this file's, 2 iq_kernels.cu's, 0 none.
int slots_kind(int ggml_type) {
    switch (ggml_type) {
    case 2: case 6: case 8: case 20: case 39: case 10: case 11: case 12: case 13: case 14: case 23: case 42: return 1;
    case 7: case 16: case 17: case 18: case 21: case 22: case 29: case 40: return 2;   // the ones native_mmvq() hands to iq_mmvq()
    default: return 0;
    }
}

template<typename S>
void launch_slots_of(int ggml_type, const S& slots, int count, int n_in, int n_out, void* stream) {
    const bool small = ggml_type == 2 || ggml_type == 6 || ggml_type == 8 || ggml_type == 20 || ggml_type == 39;
    validate_shape(n_in, 1, small || ggml_type == 7 ? 32 : ggml_type == 42 || ggml_type == 40 ? 64 : 256);
    validate_stream(stream);
    if (slots_kind(ggml_type) == 2) {
        if constexpr (std::is_same_v<S, MmvqSlots>) iq_mmvq_slots(ggml_type, slots, count, n_in, n_out, stream);
        else iq_mmvq_ids(ggml_type, slots, count, n_in, n_out, stream);
        return;
    }
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ggml_type) {
    case 2: launch_slots<SmallTraits<Q40Block, 4>>(slots, count, n_in, n_out, s); break;
    case 6: launch_slots<SmallTraits<Q50Block, 4>>(slots, count, n_in, n_out, s); break;
    case 8: launch_slots<SmallTraits<Q80Block, 8>>(slots, count, n_in, n_out, s); break;
    case 20: launch_slots<SmallTraits<IQ4NLBlock, 4>>(slots, count, n_in, n_out, s); break;
    case 39: launch_slots<SmallTraits<MXFP4Block, 4>>(slots, count, n_in, n_out, s); break;
    case 10: launch_slots<Q2KTraits>(slots, count, n_in, n_out, s); break;
    case 11: launch_slots<Q3KTraits>(slots, count, n_in, n_out, s); break;
    case 12: launch_slots<Q4KTraits>(slots, count, n_in, n_out, s); break;
    case 13: launch_slots<Q5KTraits>(slots, count, n_in, n_out, s); break;
    case 14: launch_slots<Q6KTraits>(slots, count, n_in, n_out, s); break;
    case 23: launch_slots<IQ4XSTraits>(slots, count, n_in, n_out, s); break;
    default: launch_slots<Q20Traits>(slots, count, n_in, n_out, s); break;
    }
    launch_check();
}

}  // namespace

bool native_mmvq_slots_wide() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_SLOTS_WARP");
#if defined(__HIPCC__)
        return v == nullptr || v[0] != '0';
#else
        return v != nullptr && v[0] == '1';
#endif
    }();
    return on;
}

bool native_mmvq_slots(int ggml_type, const MmvqSlots& slots, int count, int n_in, int n_out, void* stream) {
    if (!slots_kind(ggml_type)) return false;
    if (n_out <= 0 || count < 1 || count > kMmvqSlotsMax)
        throw std::invalid_argument("native MMVQ requires n_out > 0 and 1 to 32 matrices");
    for (int i = 0; i < count; ++i) {
        validate_pointer(slots.w[i]);
        validate_pointer(slots.x[i]);
        validate_pointer(slots.y[i]);
    }
    launch_slots_of(ggml_type, slots, count, n_in, n_out, stream);
    return true;
}

bool native_mmvq_ids_supported(int ggml_type) { return slots_kind(ggml_type) != 0; }

bool native_mmvq_ids(int ggml_type, const MmvqIds& ids, int count, int n_in, int n_out, void* stream) {
    if (!slots_kind(ggml_type)) return false;
    if (n_out <= 0 || count < 1 || count > 65535 || ids.x_every < 1)
        throw std::invalid_argument("native_mmvq_ids requires n_out > 0, 1 to 65535 matrices and x_every > 0");
    validate_pointer(ids.table);
    validate_pointer(ids.ids);
    validate_pointer(ids.x);
    validate_pointer(ids.y);
    launch_slots_of(ggml_type, ids, count, n_in, n_out, stream);
    return true;
}

#if !defined(__HIPCC__)

namespace {
}  // namespace
namespace {
}  // namespace

namespace {
}  // namespace

#else  // HIP: no interleaved path
#endif  // !__HIPCC__

} // namespace strata::kernels
