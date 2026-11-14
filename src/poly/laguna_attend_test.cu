// src/poly/laguna_attend_test.cu - attend_group() for Laguna's full layers (heads of 128 values, 6 a group, over an
// fp16 and a q8_0 cache) against attend(), and against itself: a token alone, with the tokens of a call beside it,
// and with the position on the card, which must be the same bits.  Without the room in score it must be attend()'s
// bits exactly.  Then both functions' time for a token over 70,000 rows, attend_group() the faster.
//
// usage: poly_laguna_attend_test   (exit 0: all passed; 77: no CUDA card)
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kDim = 128, kHeads = 48, kKvHeads = 8, kChunk = 4, kMaxRows = 70016;
constexpr int kRow = kKvHeads * kDim;   // a cache row's values

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> download(const float* d, size_t n) {
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

// ggml's quantize_row_q8_0: a block of 32 values as codes of its largest magnitude / 127, the scale in fp16
void quantize(const std::vector<float>& x, std::vector<int8_t>& codes, std::vector<uint16_t>& scales) {
    codes.resize(x.size());
    scales.resize(x.size() / 32);
    for (size_t b = 0; b < scales.size(); ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        scales[b] = __half_as_ushort(__float2half(d));
        for (int j = 0; j < 32; ++j) codes[b * 32 + j] = (int8_t) std::lround(x[b * 32 + j] * id);
    }
}

struct Buffers {
    float *q, *score, *out;
    uint16_t *k, *v;
    Kvq8 kq, vq;
    long long* at;
    size_t score_n;
};

// attend() or attend_group() over one format: the fp16 caches, or the q8_0 ones with the fp16 pointers null
void run(const Buffers& b, bool group, bool q8, const float* q, long long pos, int rows, size_t score_floats,
         const long long* at = nullptr, int room = 0) {
    const float scale = 1.f / std::sqrt((float) kDim);
    const uint16_t* k = q8 ? nullptr : b.k;
    const uint16_t* v = q8 ? nullptr : b.v;
    const Kvq8 kq = q8 ? b.kq : Kvq8{}, vq = q8 ? b.vq : Kvq8{};
    if (group)
        attend_group(q, k, v, pos, rows, b.score, kChunk, b.out, kDim, kHeads, kKvHeads, scale, nullptr, 0, 0, false,
                     at, room, score_floats, kq, vq);
    else
        attend(q, k, v, pos, rows, b.score, kChunk, b.out, kDim, kHeads, kKvHeads, scale, nullptr, 0, 0, false, at,
               room, score_floats, kq, vq);
}

// rows tokens from pos: attend(), attend_group(), each token alone, with `at`, and without the room in score
int check(const Buffers& b, bool q8, long long pos, int rows) {
    const size_t n = size_t(rows) * kHeads * kDim;
    run(b, false, q8, b.q, pos, rows, 0);
    const std::vector<float> other = download(b.out, n);
    run(b, true, q8, b.q, pos, rows, b.score_n);
    const std::vector<float> group = download(b.out, n);
    std::vector<float> alone(n);
    for (int t = 0; t < rows; ++t) {
        const size_t o = size_t(t) * kHeads * kDim;
        run(b, true, q8, b.q + o, pos + t, 1, b.score_n);
        cudaMemcpy(alone.data() + o, b.out, kHeads * kDim * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(b.at, &pos, sizeof pos, cudaMemcpyHostToDevice);
    const int room = int(std::min<long long>(pos + rows + 100, kMaxRows));
    run(b, true, q8, b.q, 0, rows, b.score_n, b.at, room);
    const std::vector<float> card = download(b.out, n);
    run(b, true, q8, b.q, pos, rows, 0);
    const std::vector<float> fallback = download(b.out, n);
    float most = 0.f, big = 0.f;
    for (size_t i = 0; i < n; ++i) {
        most = std::max(most, std::fabs(group[i] - other[i]));
        big = std::max(big, std::fabs(other[i]));
    }
    const bool same_alone = !std::memcmp(group.data(), alone.data(), n * sizeof(float)),
               same_card = !std::memcmp(group.data(), card.data(), n * sizeof(float)),
               same_fallback = !std::memcmp(other.data(), fallback.data(), n * sizeof(float)),
               ok = most <= 2e-5f && same_alone && same_card && same_fallback;
    std::printf("%s %s pos %5lld rows %d: %.2e from attend() (results up to %.2f), alone %s, on the card %s, "
                "without the room %s\n", ok ? "ok  " : "FAIL", q8 ? "q8_0" : "fp16", pos, rows, most, big,
                same_alone ? "same" : "DIFF", same_card ? "same" : "DIFF", same_fallback ? "attend()" : "DIFF");
    return ok ? 0 : 1;
}

// a token over pos + 1 rows, `reps` times: the milliseconds a call
float time_of(const Buffers& b, bool group, bool q8, long long pos, int reps) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    run(b, group, q8, b.q, pos, 1, b.score_n);   // a warm-up
    cudaEventRecord(e0);
    for (int i = 0; i < reps; ++i) run(b, group, q8, b.q, pos, 1, b.score_n);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    return ms / reps;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no CUDA card: skipped\n");
        return 77;
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(size_t(kChunk) * kHeads * kDim);
    for (float& x : q) x = 2.f * nd(rng);   // spread scores, so the softmax has a few large weights
    std::vector<float> kf(size_t(kMaxRows) * kRow), vf(kf.size());
    for (float& x : kf) x = nd(rng);
    for (float& x : vf) x = nd(rng);
    std::vector<uint16_t> k(kf.size()), v(vf.size());
    for (size_t i = 0; i < kf.size(); ++i) {
        k[i] = __half_as_ushort(__float2half(kf[i]));
        v[i] = __half_as_ushort(__float2half(vf[i]));
    }
    std::vector<int8_t> kc, vc;
    std::vector<uint16_t> ks, vs;
    quantize(kf, kc, ks);
    quantize(vf, vc, vs);
    Buffers b{};
    b.q = upload(q), b.k = upload(k), b.v = upload(v);
    b.kq = Kvq8{upload(kc), upload(ks)};
    b.vq = Kvq8{upload(vc), upload(vs)};
    b.score_n = std::max(size_t(kChunk) * kHeads * kMaxRows, attend_group_floats(kChunk, kHeads, kMaxRows, kDim));
    if (cudaMalloc(&b.score, b.score_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&b.out, q.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&b.at, sizeof(long long)) != cudaSuccess) {
        std::printf("CUDA: %s (about 750 MB of VRAM is needed)\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    int bad = 0;
    for (bool q8 : {false, true}) {
        for (long long pos : {0LL, 5LL, 63LL, 64LL, 255LL, 256LL, 1000LL, 4095LL, 30000LL, kMaxRows - 8LL})
            for (int rows : {1, 2, 4}) bad += check(b, q8, pos, rows);
        bad += check(b, q8, 2000, 3);   // a call of tokens that is not the chunk's size
    }
    // the time of a token over 70,000 rows: Laguna's full layer, once (12 of them at 70K make a token's attention)
    const long long far = kMaxRows - 1;
    for (bool q8 : {false, true}) {
        const float other = time_of(b, false, q8, far, 20), group = time_of(b, true, q8, far, 20);
        const double bytes = double(far + 1) * kRow * (q8 ? 2.0 * (1.0 + 2.0 / 32.0) : 4.0);   // K and V
        const bool ok = group * 1.5f < other;
        std::printf("%s %s, a token over %lld rows: attend() %.3f ms, attend_group() %.3f ms (%.1fx, %.0f GB/s of "
                    "the cache)\n", ok ? "ok  " : "FAIL", q8 ? "q8_0" : "fp16", far + 1, other, group, other / group,
                    bytes / (group * 1e-3) / 1e9);
        bad += ok ? 0 : 1;
    }
    cudaFree(b.q), cudaFree(b.k), cudaFree(b.v), cudaFree(b.score), cudaFree(b.out), cudaFree(b.at);
    cudaFree((void*) b.kq.codes), cudaFree((void*) b.kq.scales);
    cudaFree((void*) b.vq.codes), cudaFree((void*) b.vq.scales);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
