// include/strata/kernels/iq_kernels.hpp - the GGUF formats native_mmvq() hands over (the i-quants, Q2_0, NVFP4 and
// the rest of STRATA_MMVQ_FMTS in iq_kernels.cu) as decode matrix-vector products on the GPU.
//
// The block layouts, codebook grids and dot products are llama.cpp's (ggml-common.h, ggml-cuda/vecdotq.cuh; MIT, see
// third_party/ggml/LICENSE and VERSION.txt), so a weight means exactly what it means in llama.cpp.  Activations are
// q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// Bytes of one row of `n` values of `ggml_type` (n a multiple of the type's block).
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;

/// y[c][r] = W[r] . x[c] for `ncols` columns of q8_1 activations (x stride n_in/32 blocks per column).
void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

struct MmvqSlots;   // native_mmvq.hpp
struct MmvqIds;
/// iq_mmvq for `count` matrices of one type and shape in one launch, each with one column of its own: what
/// iq_mmvq(..., ncols = 1) gives for each; iq_mmvq_ids the same for MmvqIds.  native_mmvq_slots() and
/// native_mmvq_ids() are the entry points.
void iq_mmvq_slots(int ggml_type, const MmvqSlots& slots, int count, int n_in, int n_out, void* stream);
void iq_mmvq_ids(int ggml_type, const MmvqIds& ids, int count, int n_in, int n_out, void* stream);

}  // namespace strata::kernels
