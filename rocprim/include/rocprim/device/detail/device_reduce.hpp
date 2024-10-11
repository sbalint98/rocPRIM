// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_HPP_

#include <cstddef>
#include <iterator>
#include <type_traits>

#include "../../config.hpp"
#include "../../detail/temp_storage.hpp"
#include "../../detail/various.hpp"
#include "../config_types.hpp"
#include "../device_reduce_config.hpp"

#include "../../functional.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_reduce.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Helper functions for reducing final value with
// initial value.
template<bool WithInitialValue, class T, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto reduce_with_initial(T output, T initial_value, BinaryFunction reduce_op) ->
    typename std::enable_if<WithInitialValue, T>::type
{
    return reduce_op(initial_value, output);
}

template<bool WithInitialValue, class T, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto reduce_with_initial(T output, T initial_value, BinaryFunction reduce_op) ->
    typename std::enable_if<!WithInitialValue, T>::type
{
    (void)initial_value;
    (void)reduce_op;
    return output;
}

template<
    class Config,
    class InputIterator,
    class ResultType,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void load_block_reduce(InputIterator input, ResultType& output_value, const size_t input_size, BinaryFunction reduce_op, const unsigned int flat_block_id)
{
    static constexpr reduce_config_params params = device_params<Config>();

    constexpr unsigned int block_size = params.reduce_config.block_size;
    constexpr unsigned int items_per_thread = params.reduce_config.items_per_thread;

    using block_reduce_type
        = ::rocprim::block_reduce<ResultType, block_size, params.block_reduce_method>;

    constexpr unsigned int items_per_block     = block_size * items_per_thread;
    const unsigned int     flat_id             = ::rocprim::detail::block_thread_id<0>();
    const size_t           block_offset        = flat_block_id * items_per_block;
    const unsigned int     valid_in_last_block = input_size - block_offset;

    ResultType values[items_per_thread];

    // last incomplete block
    if(flat_block_id == (input_size / items_per_block))
    {
        block_load_direct_striped<block_size>(flat_id,
                                              input + block_offset,
                                              values,
                                              valid_in_last_block);

        output_value = values[0];
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < items_per_thread; i++)
        {
            unsigned int offset = i * block_size;
            if(flat_id + offset < valid_in_last_block)
            {
                output_value = reduce_op(output_value, values[i]);
            }
        }

        block_reduce_type().reduce(output_value, // input
                                   output_value, // output
                                   std::min(valid_in_last_block, block_size),
                                   reduce_op);
    }
    else
    {
        block_load_direct_striped<block_size>(flat_id, input + block_offset, values);

        // load input values into values
        block_reduce_type().reduce(values, // input
                                   output_value, // output
                                   reduce_op);
    }
}

template<
    bool WithInitialValue,
    class Config,
    class ResultType,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction,
    class OutputType
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void block_reduce_kernel_impl(InputIterator input,
                              const size_t input_size,
                              OutputIterator output,
                              InitValueType initial_value,
                              BinaryFunction reduce_op,
                              unsigned int* block_complete,
                              OutputType* block_tmp)
{
    static constexpr reduce_config_params params = device_params<Config>();

    constexpr unsigned int block_size       = params.reduce_config.block_size;
    constexpr unsigned int items_per_thread = params.reduce_config.items_per_thread;

    using result_type = ResultType;

    // using warp_reduce_type
    //     = warp_reduce_crosslane<result_type, device_warp_size(), false>;

    constexpr unsigned int items_per_block = block_size * items_per_thread;

    const unsigned int number_of_blocks    = ::rocprim::detail::grid_size<0>();
    const unsigned int flat_id             = ::rocprim::detail::block_thread_id<0>();
    // const unsigned int warp_id             = ::rocprim::warp_id();
    // const unsigned int lane_id             = ::rocprim::lane_id();
    const unsigned int flat_block_id       = ::rocprim::detail::block_id<0>();
    const bool         is_last_block       = flat_block_id + 1 == number_of_blocks;

    
    result_type output_value;
    
    load_block_reduce<Config>(input, output_value, input_size, reduce_op, flat_block_id);

    if(number_of_blocks > items_per_block)
    {
        // Save value into output
        if(flat_id == 0)
        {
            output[flat_block_id]
                = reduce_with_initial<WithInitialValue>(output_value,
                                                        static_cast<result_type>(initial_value),
                                                        reduce_op);
        }
    }
    else
    {
        if(flat_id == 0)
        {
            block_tmp[flat_block_id] = output_value;
            detail::memory_fence_device();
            atomic_add(block_complete, 1);
        }
        __syncthreads();

        if(is_last_block)
        {

            unsigned int amt = atomic_load(block_complete);
            while(amt != number_of_blocks)
            {
                amt = atomic_load(block_complete);
            }
            detail::memory_fence_device();

            load_block_reduce<Config>(block_tmp, output_value, number_of_blocks, reduce_op, 0);

            // for (unsigned i = lane_id; i < number_of_blocks; i += device_warp_size()) {
            //     auto value = block_tmp[i];
            //     warp_reduce_type().reduce(value, reduction, reduce_op);
            // }

            if(flat_id == 0)
            {
                output[0] = reduce_with_initial<WithInitialValue>(
                                                        output_value,
                                                        static_cast<result_type>(initial_value),
                                                        reduce_op);
            }
        }
    }
}
} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_HPP_
