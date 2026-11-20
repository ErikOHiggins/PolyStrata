#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

// Native GGUF Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL
// and IQ4_XS / CUDA Q8_1 adapters, pinned to llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, sm_120 generic MMVQ.
// All pointers are device pointers, at least 4-byte aligned, with no overlap.
// All calls enqueue on the explicit non-null CUDA stream; no allocation or wait.
// The translation unit must use --use_fast_math, as the pinned CUDA oracle does.
//
// Shapes use GGUF order: n_in is the contiguous reduction dimension, n_out is
// the weight row count, and ncols is the activation column/token count, 1..8
// (plan v0.3 P3). Columns are contiguous: activation column j is x + j * n_in
// (its Q8_1 blocks at j * n_in / 32) and output column j is y + j * n_out. Every
// column of an ncols > 1 call is bitwise what an ncols == 1 call gives for it.
// n_in must be a positive multiple of 32 for quantization/Q4_0/Q5_0/Q8_0/IQ4_NL,
// 64 for Q2_0, or 256 for the other formats. n_out must be positive. Weights remain
// unmodified row-major GGUF blocks: Q3_K=110, Q4_K=144, Q5_K=176, Q6_K=210 and
// IQ4_XS=136 bytes per 256 elements; Q2_0 is 18 bytes per 64 elements.
// Q4_0/IQ4_NL=18, Q5_0=22, Q8_0=34 bytes per 32 elements.
// Q8_1 scratch has 36 bytes per 32 elements, with no extra row padding here.
// Input floats must be finite, and their block scales/sums representable in FP16.
std::size_t native_q8_1_bytes(int n_in, int ncols = 1);

// One quantization may serve multiple weight matrices sharing the same input.
// Q8_1 stores FP16 scale and FP16 warp sum of the ORIGINAL float inputs; it does
// not reconstruct that sum from the quantized integers.
void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols,
                          void* stream);

/// An RMS norm of `rows` rows of n values and native_quantize_q8_1() of its output in one launch, bit for bit what
/// the two launches give: y = x / rms(x) * w as poly's rms_norm() writes it with `threads` a block (its
/// sum_threads(n)), and y_q8_1 its Q8_1 blocks.  With `add` (null: none), x += add first, as axpy() adds, and x
/// keeps the sum.  n is a multiple of 32.
void native_add_rms_norm_q8_1(float* x, const float* add, const float* w, float* y, void* y_q8_1, int n, int rows,
                              float eps, int threads, void* stream);

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q2_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

// Up to kMmvqSlotsMax (32) matrices of one format and shape in one launch: matrix i times its own Q8_1 column x[i]
// into y[i], as native_mmvq(type, w[i], x[i], y[i], n_in, n_out, 1, stream) would each.  A launch costs the calling thread more
// than a small product costs the card, which is what this saves.  False, with nothing launched: there is no
// kernel for it (a format native_mmvq() does not take, or a HIP build), and the caller launches them one by one.
constexpr int kMmvqSlotsMax = 32;   // 8 before: strata-poly's cached experts go 32 a launch where the wide path is on
struct MmvqSlots {
    const void* w[kMmvqSlotsMax];
    const void* x[kMmvqSlotsMax];
    float* y[kMmvqSlotsMax];
};
bool native_mmvq_slots(int ggml_type, const MmvqSlots& slots, int count, int n_in, int n_out, void* stream);
// Whether the slots launches take a warp per row group where a row's blocks fit one warp (an expert's IQ4_XS down
// projection of 512 values), and strata-poly gives them its cached experts 32 at a time instead of 8.  The same bits
// either way.  On by default on HIP (measured on an RX 7800 XT), off by default on CUDA (not measured there);
// STRATA_SLOTS_WARP=0 turns it off, =1 on.
bool native_mmvq_slots_wide();
// native_mmvq_slots with the matrices the card picked, so that the launch needs nothing of the host (a decode
// step's routed experts when every one of them is in VRAM): matrix j is table[ids[j]] + offset bytes, times the
// Q8_1 column x + (j / x_every) * x_stride bytes, into y + j * y_stride floats.  count may pass 8.  False, with
// nothing launched: there is no kernel for the format (native_mmvq_ids_supported() says so beforehand).
struct MmvqIds {
    const void* const* table;
    const int32_t* ids;
    std::size_t offset;
    const void* x;
    std::size_t x_stride;
    int x_every;
    float* y;
    std::size_t y_stride;
};
bool native_mmvq_ids_supported(int ggml_type);
bool native_mmvq_ids(int ggml_type, const MmvqIds& ids, int count, int n_in, int n_out, void* stream);

// Storage/dispatch helpers take stable GGML type IDs, avoiding a ggml runtime
// dependency in the engine: Q4_0=2, Q5_0=6, Q8_0=8, Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13,
// Q6_K=14, IQ4_NL=20, IQ4_XS=23, MXFP4=39, NVFP4=40, Q2_0=42. Unsupported IDs throw in the byte-count
// and launch helpers; only the
// capability query returns false.
bool native_mmvq_supported(int ggml_type) noexcept;
std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out);
void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream);


} // namespace strata::kernels
