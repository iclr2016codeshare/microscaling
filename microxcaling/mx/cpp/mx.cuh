/*
 * Microsoft Confidential
 */

#ifndef PYT_MX_MX_CUH
#define PYT_MX_MX_CUH

#include "common.cuh"
#include "shared_exp.cuh"
#include "quantize.cuh"
#include <math.h>

//-----------------------------------------------------------------------
// quantize_mx_cuda_kernel
//-----------------------------------------------------------------------
template<typename T>
__global__ void quantize_mx_cuda_kernel(
    const T* __restrict__ input,
    const int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    const float* __restrict__ max_values,
    const float* __restrict__ pos_values,
    const float* __restrict__ neg_values,
    const float* __restrict__ std_values,
    const long total_size,
    const int axis_size,
    const int post_axis_size,
    const bool flush_fp32_subnorms,
    const RoundingMode rounding_mode,
    const int scale_mode,
    const float scale_min,
    T* __restrict__ output
) {
    const long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= total_size) return;

    // Compute index of the max value for this element
    const long post_axis_i = offset % post_axis_size;
    const long pre_axis_i = offset / (post_axis_size * axis_size);

    // Get shared exponent
    const long m_i = pre_axis_i * post_axis_size + post_axis_i;
    int shared_exp = (int) get_biased_exponent(max_values[m_i]);
    bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);

    // Compute the shared scale
    if (scale_mode==1) {
        shared_exp += 1;
    }
    if (scale_mode==3) { // Ceil if necessary
        int threshold = 0x7FFFFF;
        threshold >>= (24 - elem_mbits);
        threshold <<= (24 - elem_mbits);
        int mantissa = (*(int*)&max_values[m_i] & 0x7FFFFF);
        if (mantissa >= threshold) {
            shared_exp += 1;
        }
    }
    float scale = 1;
    if (scale_mode==2) { // get fp max-scale from given tensor
        scale = max_values[m_i] / elem_max_norm;
        scale = (scale==0) ? 1 : scale;
    } else if (scale_mode==143) { // get fp max-scale from given tensor
        scale = max_values[m_i] / elem_max_norm;
        scale = quantize_elemwise(
                scale, 5, 4, 480,
                rounding_mode, true, true);
        scale = (scale==0) ? 1 : scale;
    } else if (scale_mode==152) { // get fp max-scale from given tensor
        scale = max_values[m_i] / elem_max_norm;
        scale = quantize_elemwise(
                scale, 4, 5, 57344.0,
                rounding_mode, true, true);
        scale = (scale==0) ? 1 : scale;
    } else if (scale_mode==53) { // get fp max-scale from given tensor
        scale = max_values[m_i] / elem_max_norm;
        scale = quantize_elemwise(
                scale, 5, 5, 61440.0,
                rounding_mode, true, true);
        scale = (scale==0) ? 1 : scale;
    } else if (scale_mode==16) {
        scale = max_values[m_i] / elem_max_norm;
        scale = (scale==0) ? 1 : scale;
    } else{
        scale = mx_get_shared_scale(
              shared_exp, scale_bits, elem_max_norm);
    }

    if ((scale_min > 0) && (scale < scale_min)) {
        scale = max_values[m_i] / elem_max_norm;  // replace scale with unquantized BF16 value
    }

    T scaled_in = (flush_tile) ? 0 : input[offset] / scale;

    T scaled_out = quantize_elemwise(
            scaled_in, elem_mbits, elem_ebits, elem_max_norm,
            rounding_mode, true, true);

    output[offset] = scaled_out * scale;
}

//-----------------------------------------------------------------------
// quantize_innermost, fast MX quantization for axis=[-1]
// input requirements:
//  - the axis is dim-1 (the innermost dim),
//  - tile_size divides axis_size evenly
//  - tile_size is a power of 2
//  - tile_size <= WARP_SIZE
//-----------------------------------------------------------------------
template<typename T>
__global__ void quantize_mx_innermost_cuda_kernel (
    const T* __restrict__ in,
    const int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    const long total_size,
    const int tile_size,
    const bool flush_fp32_subnorms,
    const RoundingMode rounding_mode,
    const int scale_mode,
    const float scale_min,
    T* __restrict__ out
) {
    const long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= total_size) return;
    const T elem = in[offset];

    // allreduce to get the max value in each tile
    int shared_exp = get_biased_exponent(elem);
    float max_val = fabsf(elem); // absolute max
    for (int mask = tile_size/2; mask > 0; mask /= 2) {
        int _tmp = __shfl_xor_sync(0xFFFFFFFF, shared_exp, mask);
        shared_exp = (_tmp > shared_exp) ? _tmp : shared_exp;
        // Compare value from other thread in the warp
        float _tmp_elem = __shfl_xor_sync(0xFFFFFFFF, max_val, mask);
        max_val = (_tmp_elem > max_val) ? _tmp_elem : max_val;
    }
    bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);

    // Compute the shared scale
    if (scale_mode==1) { // Ceil instead of floor
        int threshold = 0;
        int mantissa = (*(int*)&max_val & 0x7FFFFF);
        if (mantissa > threshold) {
            shared_exp += 1;
        }
    }
    if (scale_mode==3) { // Ceil if necessary
        int threshold = 0x7FFFFF;
        threshold >>= (24 - elem_mbits);
        threshold <<= (24 - elem_mbits);
        int mantissa = (*(int*)&max_val & 0x7FFFFF);
        if (mantissa >= threshold) {
            shared_exp += 1;
        }
    }
    float scale = 1;
    if (scale_mode==2) {
        scale = max_val / elem_max_norm;
        scale = (scale==0) ? 1 : scale;
    } else if (scale_mode==143) {
        scale = max_val / elem_max_norm;
        scale = quantize_elemwise(
                scale, 5, 4, 480,
                rounding_mode, true, true);
        scale = (scale==0) ? 1 : scale;
    } else if (scale_mode==152) {
        scale = max_val / elem_max_norm;
        scale = quantize_elemwise(
                scale, 4, 5, 57344.0,
                rounding_mode, true, true);
    } else if (scale_mode==53) {
        scale = max_val / elem_max_norm;
        scale = quantize_elemwise(
                scale, 5, 5, 61440.0,
                rounding_mode, true, true);
    } else if (scale_mode==16) {
        scale = max_val / elem_max_norm;
    } else{
        scale = mx_get_shared_scale(
              shared_exp, scale_bits, elem_max_norm);
    }

    if ((scale_min > 0) && (scale < scale_min)) {
        scale = max_val / elem_max_norm;  // replace scale with unquantized BF16 value
    }

    T scaled_in = (flush_tile) ? 0 : elem / scale;

    T scaled_out = quantize_elemwise(
            scaled_in, elem_mbits, elem_ebits, elem_max_norm,
            rounding_mode, true, true);

    out[offset] = scaled_out * scale;
}

//-----------------------------------------------------------------------
// quantize_mx_by_tile kernel
// Each thread loops across the tile to get the max exponent, then
// loops across it again to perform quantization.
//-----------------------------------------------------------------------
template<typename T>
__global__ void quantize_mx_by_tile_cuda_kernel (
    const T* __restrict__ in,
    const int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    const int total_tiles,
    const int tile_size,
    const int num_tiles,
    const int axis_size,
    const int post_axis_size,
    const bool flush_fp32_subnorms,
    const RoundingMode rounding_mode,
    const int scale_mode,
    const float scale_min,
    T* __restrict__ out
) {
    const long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= total_tiles) return;

    // Calculate indices on different dimensions
    const long post_axis_i = offset % post_axis_size;
    const long num_tiles_i = (offset / post_axis_size) % num_tiles;
    const long pre_axis_i = offset / (num_tiles * post_axis_size);

    // Handle non-full bounding box/tile
    int adjusted_tile_size;
    if ((num_tiles_i + 1) * tile_size > axis_size) {
        adjusted_tile_size = axis_size % tile_size;
    } else {
        adjusted_tile_size = tile_size;
    }

    // Find biased shared_exp
    int shared_exp = 0; // biased exp must be >= 0
    float max_val = 0;
    for (int i = 0; i < adjusted_tile_size; i++) {  // sweep every element in the block / tile
        long in_i = pre_axis_i * axis_size * post_axis_size +
            (num_tiles_i * tile_size + i) * post_axis_size +
            post_axis_i;

        int exp = get_biased_exponent(in[in_i]);
        shared_exp = (exp > shared_exp) ? exp : shared_exp;  // get largest exp in block
        max_val = (fabsf(in[in_i]) > max_val) ? fabsf(in[in_i]) : max_val;  // get largest abs-val in block
    }

    bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);

    // Compute the shared scale
    if (scale_mode==1) {
        int threshold = 0;
        int mantissa = (*(int*)&max_val & 0x7FFFFF);
        if (mantissa > threshold) {
            shared_exp += 1;
        }
        shared_exp += 1;
    }
    if (scale_mode==3) { // Ceil if necessary
        int threshold = 0x7FFFFF;
        threshold >>= (24 - elem_mbits);
        threshold <<= (24 - elem_mbits);
        int mantissa = (*(int*)&max_val & 0x7FFFFF);
        if (mantissa >= threshold) {
            shared_exp += 1;
        }
    }
    float scale = 1;
    if (scale_mode==2) {
        scale = max_val / elem_max_norm;
        scale = (scale==0) ? 1 : scale;
    } else if (scale_mode==143) {
        scale = max_val / elem_max_norm;
        scale = quantize_elemwise(
                scale, 5, 4, 480,
                rounding_mode, true, true);
        scale = (scale==0) ? 1 : scale;
    } else if (scale_mode==152) {
        scale = max_val / elem_max_norm;
        scale = quantize_elemwise(
                scale, 4, 5, 57344.0,
                rounding_mode, true, true);
    } else if (scale_mode==53) {
        scale = max_val / elem_max_norm;
        scale = quantize_elemwise(
                scale, 5, 5, 61440.0,
                rounding_mode, true, true);
    } else{
        scale = mx_get_shared_scale(  // shared_exp is max biased exp in block
              shared_exp, scale_bits, elem_max_norm);
    }

    // Loop over bounding box to quantize
    for (int i = 0; i < adjusted_tile_size; i++) {
        long in_i = pre_axis_i * axis_size * post_axis_size +
            (num_tiles_i * tile_size + i) * post_axis_size +
            post_axis_i;

        T scaled_in = (flush_tile) ? 0 : in[in_i] / scale;

        T scaled_out = quantize_elemwise(
                scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                rounding_mode, true, true);

        out[in_i] = scaled_out * scale;
    }
}

#endif
