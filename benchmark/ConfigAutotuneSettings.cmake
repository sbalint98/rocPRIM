# MIT License
#
# Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# All default fallback types as listed in scripts/autotune/fallback_config.json
set(TUNING_TYPES "int64_t int short int8_t double float rocprim::half")
# If config selection happens based on two types, the second type has limited fallbacks. The selection is based
# on the size and it is ignored whether the type is floating-point or integral. The autotuning script uses the
# benchmarks for the integral types as fallback, hence tuning for the floating-point types is not needed.
set(LIMITED_TUNING_TYPES "int64_t int short int8_t")

function(read_config_autotune_settings file list_across_names list_across output_pattern_suffix)
  if(file STREQUAL "benchmark_device_adjacent_difference")
    set(list_across_names "DataType;Left;InPlace;BlockSize" PARENT_SCOPE)
    set(list_across "${TUNING_TYPES};\
true;false true;32 64 128 256 512 1024" PARENT_SCOPE)
    set(output_pattern_suffix "@DataType@_@Left@_@InPlace@_@BlockSize@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_adjacent_find")
    set(list_across_names "InputType;BlockSize" PARENT_SCOPE)
    set(list_across "${TUNING_TYPES};64 128 256 512 1024" PARENT_SCOPE)
    set(output_pattern_suffix "@InputType@_@BlockSize@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_histogram")
    set(list_across_names "DataType;BlockSize" PARENT_SCOPE)
    set(list_across "${TUNING_TYPES};64 128 256" PARENT_SCOPE)
    set(output_pattern_suffix "@DataType@_@BlockSize@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_merge_sort_block_merge")
    set(list_across_names "KeyType;ValueType;BlockSize;UseMergePath" PARENT_SCOPE)
    set(list_across "\
${TUNING_TYPES};rocprim::empty_type ${LIMITED_TUNING_TYPES} custom_type<char,double>;\
128 256 512 1024;true" PARENT_SCOPE)
    set(output_pattern_suffix "@KeyType@_@ValueType@_@BlockSize@_@UseMergePath@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_merge_sort_block_sort")
    set(list_across_names "KeyType;ValueType;BlockSize;BlockSortMethod" PARENT_SCOPE)
    set(list_across "\
${TUNING_TYPES};rocprim::empty_type ${LIMITED_TUNING_TYPES} custom_type<char,double>;\
256 512 1024;rocprim::block_sort_algorithm::stable_merge_sort" PARENT_SCOPE)
    set(output_pattern_suffix "@KeyType@_@ValueType@_@BlockSize@_@BlockSortMethod@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_radix_sort_block_sort")
    set(list_across_names "KeyType;ValueType;BlockSize" PARENT_SCOPE)
    set(list_across "\
${TUNING_TYPES};rocprim::empty_type ${LIMITED_TUNING_TYPES};\
64 128 256 512 1024" PARENT_SCOPE)
    set(output_pattern_suffix "@KeyType@_@ValueType@_@BlockSize@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_radix_sort_onesweep")
    set(list_across_names "KeyType;ValueType;BlockSize;RadixBits" PARENT_SCOPE)
    set(list_across "\
${TUNING_TYPES};rocprim::empty_type ${LIMITED_TUNING_TYPES};\
128 256 512 1024;4 5 6 7 8" PARENT_SCOPE)
    set(output_pattern_suffix "@KeyType@_@ValueType@_@BlockSize@_@RadixBits@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_reduce")
    set(list_across_names "DataType;BlockSize;ItemsPerThread" PARENT_SCOPE)
    set(list_across "\
float;64 128 192 256 320 384 448 512 576 640 704 768 832 896 960;1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31" PARENT_SCOPE)
    set(output_pattern_suffix "@DataType@_@BlockSize@_@ItemsPerThread@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_scan")
    set(list_across_names "DataType;Algo" PARENT_SCOPE)
    set(list_across "\
${TUNING_TYPES};using_warp_scan reduce_then_scan" PARENT_SCOPE)
    set(output_pattern_suffix "@DataType@_@Algo@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_scan_by_key")
    set(list_across_names "KeyType;ValueType;Algo" PARENT_SCOPE)
    set(list_across "\
${TUNING_TYPES};${LIMITED_TUNING_TYPES};using_warp_scan reduce_then_scan" PARENT_SCOPE)
    set(output_pattern_suffix "@KeyType@_@ValueType@_@Algo@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_binary_search")
    set(list_across_names "SubAlgorithm;ValueType;OutputType;BlockSize;ItemsPerThread" PARENT_SCOPE)
    set(list_across "\
binary_search upper_bound lower_bound;${TUNING_TYPES};${LIMITED_TUNING_TYPES};64 128 256;1 2 4 8 16" PARENT_SCOPE)
    set(output_pattern_suffix "@SubAlgorithm@_@ValueType@_@OutputType@_@BlockSize@_@ItemsPerThread@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_segmented_radix_sort_keys")
    set(list_across_names "\
KeyType;BlockSize;ItemsPerThread;PartitionAllowed" PARENT_SCOPE)
    set(list_across "${TUNING_TYPES};128 256;4 8 16;false" PARENT_SCOPE)
    set(output_pattern_suffix "\
@KeyType@_@BlockSize@_@ItemsPerThread@_@PartitionAllowed@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_segmented_radix_sort_pairs")
    set(list_across_names "\
KeyType;ValueType;BlockSize;ItemsPerThread;PartitionAllowed" PARENT_SCOPE)
    set(list_across "${TUNING_TYPES};int8_t;64;4 8 16;true false" PARENT_SCOPE)
    set(output_pattern_suffix "\
@KeyType@_@ValueType@_@BlockSize@_@ItemsPerThread@_@PartitionAllowed@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_transform")
    set(list_across_names "\
DataType;BlockSize;" PARENT_SCOPE)
    set(list_across "${TUNING_TYPES};64 128 256 512 1024" PARENT_SCOPE)
    set(output_pattern_suffix "\
@DataType@_@BlockSize@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_partition")
    set(list_across_names "DataType;BlockSize" PARENT_SCOPE)
    set(list_across "${TUNING_TYPES};128 192 256 384 512" PARENT_SCOPE)
    set(output_pattern_suffix "@DataType@_@BlockSize@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_select")
    set(list_across_names "KeyType;ValueType;BlockSize" PARENT_SCOPE)
    set(list_across "${TUNING_TYPES};rocprim::empty_type ${LIMITED_TUNING_TYPES};128 192 256 384 512" PARENT_SCOPE)
    set(output_pattern_suffix "@KeyType@_@ValueType@_@BlockSize@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_reduce_by_key")
    set(list_across_names "KeyType;ValueType;BlockSize;TilesPerBlock" PARENT_SCOPE)
    set(list_across "${LIMITED_TUNING_TYPES};${TUNING_TYPES};128 192 256 384 512;1 2" PARENT_SCOPE)
    set(output_pattern_suffix "@KeyType@_@ValueType@_@BlockSize@_@TilesPerBlock@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_find_first_of")
    set(list_across_names "DataType;BlockSize" PARENT_SCOPE)
    set(list_across "${LIMITED_TUNING_TYPES};32 64 128 256 512 1024" PARENT_SCOPE)
    set(output_pattern_suffix "@DataType@_@BlockSize@" PARENT_SCOPE)
  elseif(file STREQUAL "benchmark_device_run_length_encode")
    set(list_across_names "KeyType;BlockSize;BlockLoadMethod" PARENT_SCOPE)
    set(list_across "${TUNING_TYPES};64 128 256 512 1024;block_load_vectorize block_load_warp_transpose" PARENT_SCOPE)
    set(output_pattern_suffix "@KeyType@_@BlockSize@_@BlockLoadMethod@" PARENT_SCOPE)
  endif()
endfunction()
