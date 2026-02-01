#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lsm_vec_db.h"

#if defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#endif

namespace lsm_vec::distance
{
namespace detail {
#if defined(__AVX2__)
inline float horizontal_add(__m256 value)
{
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, value);
    float sum = 0.0f;
    for (float v : tmp) {
        sum += v;
    }
    return sum;
}
#endif

#if defined(__SSE2__)
inline float horizontal_add(__m128 value)
{
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, value);
    float sum = 0.0f;
    for (float v : tmp) {
        sum += v;
    }
    return sum;
}
#endif

inline bool is_aligned(const void* ptr, std::size_t alignment)
{
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}
} // namespace detail

inline float DotProduct(const float* __restrict a,
                        const float* __restrict b,
                        std::size_t size)
{
    std::size_t i = 0;
    float sum = 0.0f;

#if defined(__AVX2__)
    if (size >= 8) {
        __m256 acc = _mm256_setzero_ps();
        const std::size_t bound = size - (size % 8);
        const bool aligned = detail::is_aligned(a, 32) && detail::is_aligned(b, 32);
        const float* aligned_a = aligned ? static_cast<const float*>(__builtin_assume_aligned(a, 32)) : a;
        const float* aligned_b = aligned ? static_cast<const float*>(__builtin_assume_aligned(b, 32)) : b;
        for (; i < bound; i += 8) {
            const __m256 va = aligned ? _mm256_load_ps(aligned_a + i)
                                      : _mm256_loadu_ps(a + i);
            const __m256 vb = aligned ? _mm256_load_ps(aligned_b + i)
                                      : _mm256_loadu_ps(b + i);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
        }
        sum += detail::horizontal_add(acc);
    }
#endif

#if defined(__SSE2__)
    if (size - i >= 4) {
        __m128 acc = _mm_setzero_ps();
        const std::size_t bound = i + ((size - i) / 4) * 4;
        const float* a_ptr = a + i;
        const float* b_ptr = b + i;
        const bool aligned = detail::is_aligned(a_ptr, 16) && detail::is_aligned(b_ptr, 16);
        const float* aligned_a = aligned ? static_cast<const float*>(__builtin_assume_aligned(a_ptr, 16)) : a_ptr;
        const float* aligned_b = aligned ? static_cast<const float*>(__builtin_assume_aligned(b_ptr, 16)) : b_ptr;
        const std::size_t base = static_cast<std::size_t>(a_ptr - a);
        for (; i < bound; i += 4) {
            const std::size_t offset = i - base;
            const __m128 va = aligned ? _mm_load_ps(aligned_a + offset)
                                      : _mm_loadu_ps(a + i);
            const __m128 vb = aligned ? _mm_load_ps(aligned_b + offset)
                                      : _mm_loadu_ps(b + i);
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
        }
        sum += detail::horizontal_add(acc);
    }
#endif

    for (; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float L2Norm(const float* __restrict a, std::size_t size)
{
    std::size_t i = 0;
    float sum = 0.0f;

#if defined(__AVX2__)
    if (size >= 8) {
        __m256 acc = _mm256_setzero_ps();
        const std::size_t bound = size - (size % 8);
        const bool aligned = detail::is_aligned(a, 32);
        const float* aligned_a = aligned ? static_cast<const float*>(__builtin_assume_aligned(a, 32)) : a;
        for (; i < bound; i += 8) {
            const __m256 va = aligned ? _mm256_load_ps(aligned_a + i)
                                      : _mm256_loadu_ps(a + i);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(va, va));
        }
        sum += detail::horizontal_add(acc);
    }
#endif

#if defined(__SSE2__)
    if (size - i >= 4) {
        __m128 acc = _mm_setzero_ps();
        const std::size_t bound = i + ((size - i) / 4) * 4;
        const float* a_ptr = a + i;
        const bool aligned = detail::is_aligned(a_ptr, 16);
        const float* aligned_a = aligned ? static_cast<const float*>(__builtin_assume_aligned(a_ptr, 16)) : a_ptr;
        const std::size_t base = static_cast<std::size_t>(a_ptr - a);
        for (; i < bound; i += 4) {
            const std::size_t offset = i - base;
            const __m128 va = aligned ? _mm_load_ps(aligned_a + offset)
                                      : _mm_loadu_ps(a + i);
            acc = _mm_add_ps(acc, _mm_mul_ps(va, va));
        }
        sum += detail::horizontal_add(acc);
    }
#endif

    for (; i < size; ++i) {
        sum += a[i] * a[i];
    }
    return std::sqrt(sum);
}

inline float L2Distance(const float* __restrict a,
                        const float* __restrict b,
                        std::size_t size)
{
    std::size_t i = 0;
    float sum = 0.0f;

#if defined(__AVX2__)
    if (size >= 8) {
        __m256 acc = _mm256_setzero_ps();
        const std::size_t bound = size - (size % 8);
        const bool aligned = detail::is_aligned(a, 32) && detail::is_aligned(b, 32);
        const float* aligned_a = aligned ? static_cast<const float*>(__builtin_assume_aligned(a, 32)) : a;
        const float* aligned_b = aligned ? static_cast<const float*>(__builtin_assume_aligned(b, 32)) : b;
        for (; i < bound; i += 8) {
            const __m256 va = aligned ? _mm256_load_ps(aligned_a + i)
                                      : _mm256_loadu_ps(a + i);
            const __m256 vb = aligned ? _mm256_load_ps(aligned_b + i)
                                      : _mm256_loadu_ps(b + i);
            const __m256 diff = _mm256_sub_ps(va, vb);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
        }
        sum += detail::horizontal_add(acc);
    }
#endif

#if defined(__SSE2__)
    if (size - i >= 4) {
        __m128 acc = _mm_setzero_ps();
        const std::size_t bound = i + ((size - i) / 4) * 4;
        const float* a_ptr = a + i;
        const float* b_ptr = b + i;
        const bool aligned = detail::is_aligned(a_ptr, 16) && detail::is_aligned(b_ptr, 16);
        const float* aligned_a = aligned ? static_cast<const float*>(__builtin_assume_aligned(a_ptr, 16)) : a_ptr;
        const float* aligned_b = aligned ? static_cast<const float*>(__builtin_assume_aligned(b_ptr, 16)) : b_ptr;
        const std::size_t base = static_cast<std::size_t>(a_ptr - a);
        for (; i < bound; i += 4) {
            const std::size_t offset = i - base;
            const __m128 va = aligned ? _mm_load_ps(aligned_a + offset)
                                      : _mm_loadu_ps(a + i);
            const __m128 vb = aligned ? _mm_load_ps(aligned_b + offset)
                                      : _mm_loadu_ps(b + i);
            const __m128 diff = _mm_sub_ps(va, vb);
            acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
        }
        sum += detail::horizontal_add(acc);
    }
#endif

    for (; i < size; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline float CosineDistance(const float* __restrict a,
                            const float* __restrict b,
                            std::size_t size,
                            float norm_a,
                            float norm_b)
{
    if (norm_a <= 0.0f) {
        norm_a = L2Norm(a, size);
    }
    if (norm_b <= 0.0f) {
        norm_b = L2Norm(b, size);
    }
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return std::numeric_limits<float>::infinity();
    }
    float dot = DotProduct(a, b, size);
    return 1.0f - (dot / (norm_a * norm_b));
}

inline float ComputeDistance(DistanceMetric metric,
                             const float* __restrict a,
                             const float* __restrict b,
                             std::size_t size,
                             float norm_a = -1.0f,
                             float norm_b = -1.0f)
{
    switch (metric) {
    case DistanceMetric::kL2:
        return L2Distance(a, b, size);
    case DistanceMetric::kCosine:
        return CosineDistance(a, b, size, norm_a, norm_b);
    default:
        break;
    }
    return L2Distance(a, b, size);
}

inline float ComputeDistance(DistanceMetric metric,
                             Span<const float> a,
                             Span<const float> b,
                             float norm_a = -1.0f,
                             float norm_b = -1.0f)
{
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::infinity();
    }
    return ComputeDistance(metric, a.data(), b.data(), a.size(), norm_a, norm_b);
}
} // namespace lsm_vec::distance
