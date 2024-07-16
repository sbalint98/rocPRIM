// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_RUN_LENGTH_ENCODE_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_RUN_LENGTH_ENCODE_HPP_

#include "device_partition.hpp"
#include "lookback_scan_state.hpp"
#include "ordered_block_id.hpp"

#include "../../detail/binary_op_wrappers.hpp"
#include "../../detail/various.hpp"
#include "../../functional.hpp"
#include "../../intrinsics/thread.hpp"
#include "../../thread/thread_reduce.hpp"
#include "../../thread/thread_scan.hpp"
#include "../../type_traits.hpp"
#include "../../warp/warp_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
namespace run_length_encode
{

template<typename OffsetType, typename CountType>
using offset_count_pair_type_t = ::rocprim::tuple<OffsetType, CountType>;

template<typename InputType,
         unsigned int      BlockSize,
         unsigned int      ItemsPerThread,
         block_load_method load_input_method>
struct load_helper
{
    using block_load_input = block_load<InputType, BlockSize, ItemsPerThread, load_input_method>;
    union storage_type
    {
        typename block_load_input::storage_type input;
    };

    template<typename InputIterator>
    ROCPRIM_DEVICE
    void load_input_values(InputIterator      tile_input,
                           const bool         is_last_tile,
                           const unsigned int valid_in_last_tile,
                           InputType (&input)[ItemsPerThread],
                           storage_type& storage)
    {
        if(!is_last_tile)
        {
            block_load_input{}.load(tile_input, input, storage.input);
        }
        else
        {
            block_load_input{}.load(tile_input, input, valid_in_last_tile, storage.input);
        }
        ::rocprim::syncthreads();
    }
};

template<typename InputType, typename CompareFunction, unsigned int BlockSize>
struct discontinuity_helper
{
    using block_discontinuity_type = block_discontinuity<InputType, BlockSize>;
    using storage_type             = typename block_discontinuity_type::storage_type;

    template<typename InputIterator, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE
    void flag_heads_and_tails(InputIterator tile_input,
                              const InputType (&input)[ItemsPerThread],
                              unsigned int (&head_flags)[ItemsPerThread],
                              unsigned int (&tail_flags)[ItemsPerThread],
                              const bool    is_first_tile,
                              const bool    is_last_tile,
                              const size_t  valid_in_last_tile,
                              storage_type& storage)
    {
        if(is_last_tile)
        {
            // If it's the last tile globally, the out-of-bound items should not be flagged.
            auto guarded_not_equal
                = ::rocprim::detail::guarded_inequality_wrapper<CompareFunction,
                                                                1 /*Ret for out-of-bounds*/>(
                    CompareFunction(),
                    valid_in_last_tile);

            if(is_first_tile)
            {
                block_discontinuity_type{}.flag_heads_and_tails(head_flags,
                                                                tail_flags,
                                                                input,
                                                                guarded_not_equal,
                                                                storage);
            }
            else
            {
                const InputType tile_predecessor = tile_input[-1];
                block_discontinuity_type{}.flag_heads_and_tails(head_flags,
                                                                tile_predecessor,
                                                                tail_flags,
                                                                input,
                                                                guarded_not_equal,
                                                                storage);
            }
        }
        else
        {
            auto not_equal
                = ::rocprim::detail::inequality_wrapper<CompareFunction>(CompareFunction());

            constexpr unsigned int tile_size      = BlockSize * ItemsPerThread;
            const InputType        tile_successor = tile_input[tile_size];

            if(is_first_tile)
            {
                block_discontinuity_type{}.flag_heads_and_tails(head_flags,
                                                                tail_flags,
                                                                tile_successor,
                                                                input,
                                                                not_equal,
                                                                storage);
            }
            else
            {
                const InputType tile_predecessor = tile_input[-1];
                block_discontinuity_type{}.flag_heads_and_tails(head_flags,
                                                                tile_predecessor,
                                                                tail_flags,
                                                                tile_successor,
                                                                input,
                                                                not_equal,
                                                                storage);
            }
        }
    }
};

/// warp_exchange class with extra check in scatter_to_striped for out-of-bound accesses.
template<class T,
         unsigned int ItemsPerThread,
         unsigned int WarpSize = ::rocprim::device_warp_size()>
class custom_warp_exchange
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    ROCPRIM_DETAIL_DEVICE_STATIC_ASSERT(
        WarpSize <= ::rocprim::device_warp_size(),
        "Logical warp size cannot be larger than physical warp size.");

    static constexpr unsigned int warp_items = WarpSize * ItemsPerThread;

    // Struct used for creating a raw_storage object for this primitive's temporary storage.
    struct storage_type_
    {
        T buffer[warp_items];
    };

public:
    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by the related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords <tt>__shared__</tt>. It can be aliased to
    /// an externally allocated memory, or be a part of a union type with other storage types
    /// to increase shared memory reusability.
#ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type = detail::raw_storage<storage_type_>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP
#else
    using storage_type = storage_type_; // only for Doxygen
#endif

    /// \brief Orders \p input values according to ranks using temporary storage,
    /// then writes the values to \p output in a striped manner.
    /// Values in \p ranks that exceed \p WarpSize*ItemsPerThread-1 are not orderer nor written
    /// to output.
    /// \tparam U - [inferred] the output type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [in] ranks - array containing the positions.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reuse
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    template<class U, class OffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scatter_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const OffsetT (&ranks)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id  = ::rocprim::detail::logical_lane_id<WarpSize>();
        storage_type_&     storage_ = storage.get();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if(ranks[i] < warp_items)
            {
                storage_.buffer[ranks[i]] = input[i];
            }
        }
        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            unsigned int item_offset = (i * WarpSize) + flat_id;
            output[i]                = storage_.buffer[item_offset];
        }
    }
};

template<typename WarpExchangeOffsetType,
         typename WarpExchangeCountType,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int WarpSize>
struct scatter_helper
{
    using offsets_storage_type = typename WarpExchangeOffsetType::storage_type;
    using counts_storage_type  = typename WarpExchangeCountType::storage_type;

    template<typename OffsetCountPairType,
             typename OffsetsOutputIterator,
             typename CountsOutputIterator>
    ROCPRIM_DEVICE
    void scatter(const OffsetCountPairType (&offsets_and_counts)[ItemsPerThread],
                 OffsetsOutputIterator offsets_output,
                 CountsOutputIterator  counts_output,
                 size_t                tile_num_runs_aggregate,
                 size_t                tile_num_runs_exclusive_in_global,
                 size_t                warp_num_runs_aggregate,
                 size_t                warp_num_runs_exclusive_in_tile,
                 const size_t (&thread_num_runs_exclusive_in_warp)[ItemsPerThread],
                 offsets_storage_type& offsets_storage,
                 counts_storage_type&  counts_storage)
    {
        // Direct scatter
        if((ItemsPerThread == 1) || (tile_num_runs_aggregate < BlockSize))
        {
            // If the warp has any non-trivial run start
            if(warp_num_runs_aggregate)
            {
                for(unsigned int i = 0; i < ItemsPerThread; ++i)
                {
                    if(thread_num_runs_exclusive_in_warp[i] < warp_num_runs_aggregate)
                    {
                        size_t item_offset = tile_num_runs_exclusive_in_global
                                             + warp_num_runs_exclusive_in_tile
                                             + thread_num_runs_exclusive_in_warp[i];

                        // Scatter offset
                        offsets_output[item_offset] = ::rocprim::get<0>(offsets_and_counts[i]);

                        // Scatter count if not the first (global) length
                        if(item_offset > 0)
                        {
                            counts_output[item_offset - 1]
                                = ::rocprim::get<1>(offsets_and_counts[i]);
                        }
                    }
                }
            }
        }
        else // Two-phase scatter
        {
            using offset_type = unsigned int;
            using count_type  = unsigned int;

            unsigned int lane_id = ::rocprim::detail::logical_lane_id<WarpSize>();

            // Unzip
            offset_type run_offsets[ItemsPerThread];
            count_type  run_counts[ItemsPerThread];

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                run_offsets[i] = ::rocprim::get<0>(offsets_and_counts[i]);
                run_counts[i]  = ::rocprim::get<1>(offsets_and_counts[i]);
            }

            WarpExchangeOffsetType().scatter_to_striped(run_offsets,
                                                        run_offsets,
                                                        thread_num_runs_exclusive_in_warp,
                                                        offsets_storage);

            ::rocprim::syncthreads();

            WarpExchangeCountType().scatter_to_striped(run_counts,
                                                       run_counts,
                                                       thread_num_runs_exclusive_in_warp,
                                                       counts_storage);

            ::rocprim::syncthreads();

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                // warp_num_runs_aggregate - number of non-trivial runs starts in current warp
                if((i * WarpSize) + lane_id < warp_num_runs_aggregate)
                {
                    size_t item_offset = tile_num_runs_exclusive_in_global
                                         + warp_num_runs_exclusive_in_tile + (i * WarpSize)
                                         + lane_id;

                    // Scatter offset
                    offsets_output[item_offset] = run_offsets[i];

                    // Scatter length if not the first (global) length
                    if((i != 0) || (item_offset > 0))
                    {
                        counts_output[item_offset - 1] = run_counts[i];
                    }
                }
            }
        }
    }
};

template<typename OffsetCountPairType, unsigned int WarpSize, unsigned int WarpsNo>
struct scan_helper
{
    using warp_scan_pairs_type   = ::rocprim::warp_scan<OffsetCountPairType, WarpSize>;
    using warp_scan_storage_type = typename warp_scan_pairs_type::storage_type;

    template<unsigned int ItemsPerThread, typename ScanOp>
    ROCPRIM_DEVICE
    void scan(OffsetCountPairType (&offsets_and_run_items)[ItemsPerThread],
              OffsetCountPairType& tile_aggregate,
              OffsetCountPairType& warp_aggregate,
              OffsetCountPairType& warp_exclusive_in_tile,
              OffsetCountPairType& thread_exclusive_in_warp,
              ScanOp               scan_op,
              warp_scan_storage_type (&warp_scan_storage)[WarpsNo],
              OffsetCountPairType* warp_aggregates_storage)
    {
        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int lane_id = ::rocprim::detail::logical_lane_id<WarpSize>();

        // should be the partial state of last launch for the first tile of the second launch
        OffsetCountPairType init;
        ::rocprim::get<0>(init) = 0;
        ::rocprim::get<1>(init) = 0;

        // [0]: number of non-trivial run starts in this and previous threads of the warp
        // [1]: number of items in the last non-trivial run of this and previous threads of the warp
        OffsetCountPairType thread_inclusive;

        // [0]: number of non-trivial run starts in this thread
        // [1]: number of items in the last non-trivial run of this thread
        OffsetCountPairType thread_aggregate
            = ::rocprim::thread_reduce<ItemsPerThread>(&offsets_and_run_items[0], scan_op);

        // Warp scan results for thread i:
        //  - thread_inclusive = scan_op(thread_aggregate_0, thread_aggregate_1, ...,
        //                               thread_aggregate_i)
        //  - thread_exclusive = scan_op(init, thread_aggregate_0, thread_aggregate_1, ...
        //                               thread_aggregate_{i-1})
        warp_scan_pairs_type().scan(thread_aggregate,
                                    thread_inclusive,
                                    thread_exclusive_in_warp,
                                    init,
                                    warp_scan_storage[warp_id],
                                    scan_op);

        // Last thread of the warp sets the warp-aggregate.
        if(lane_id == WarpSize - 1)
        {
            warp_aggregates_storage[warp_id] = thread_inclusive;
        }

        ::rocprim::syncthreads();

        warp_exclusive_in_tile = init;
        warp_aggregate         = warp_aggregates_storage[warp_id];

        tile_aggregate = warp_aggregates_storage[0];
        for(unsigned int i = 1; i < WarpsNo; ++i)
        {
            // The aggregate from previous warps is the partial value of tile_aggregate.
            if(warp_id == i)
            {
                warp_exclusive_in_tile = tile_aggregate;
            }
            // Update tile_aggregate by adding up the warp_aggregate of warp i.
            tile_aggregate = scan_op(tile_aggregate, warp_aggregates_storage[i]);
        }

        // Ensure all threads have read warp aggregates before the storage is repurposed in the
        // subsequent scatter stage.
        ::rocprim::syncthreads();
    }
};

template<typename InputType,
         typename OffsetType,
         typename CountType,
         typename OffsetCountPairType,
         unsigned int         BlockSize,
         unsigned int         ItemsPerThread,
         block_load_method    load_input_method,
         block_scan_algorithm scan_algorithm>
class tile_helper
{
private:
    using equal_op          = ::rocprim::equal_to<InputType>;
    using prefix_op_factory = detail::offset_lookback_scan_factory<OffsetCountPairType>;

    // Helper class for loading input values.
    using load_type
        = run_length_encode::load_helper<InputType, BlockSize, ItemsPerThread, load_input_method>;
    // Helper class for flagging the heads and tails of the input values.
    using discontinuity_type
        = run_length_encode::discontinuity_helper<InputType, equal_op, BlockSize>;

    // Warp size.
    static constexpr unsigned int warp_size
        = detail::get_min_warp_size(BlockSize, ::rocprim::device_warp_size());
    // Number of warps in block.
    static constexpr unsigned int warps_no = (BlockSize + warp_size - 1) / warp_size;

    // warp_scan primitive that will be used to perform warp-level exclusive scans
    // on <offset, count> pairs.
    using warp_scan_pairs_type = ::rocprim::warp_scan<OffsetCountPairType, warp_size>;
    // Helper class for warp-wide scanning of <offset, count> pairs.
    using warp_scan_type = run_length_encode::scan_helper<OffsetCountPairType, warp_size, warps_no>;

    // warp_exchange primitives that will be used to perform warp-level scatter_to_striped
    // on offsets and counts.
    using warp_exchange_offsets_type
        = custom_warp_exchange<OffsetType, ItemsPerThread, /*logical_*/ warp_size>;
    using warp_exchange_counts_type
        = custom_warp_exchange<CountType, ItemsPerThread, /*logical_*/ warp_size>;
    // Helper class for scattering offsets and counts.
    using warp_scatter_type = run_length_encode::scatter_helper<warp_exchange_offsets_type,
                                                                warp_exchange_counts_type,
                                                                BlockSize,
                                                                ItemsPerThread,
                                                                warp_size>;

public:
    union storage_type
    {
        typename load_type::storage_type load;

        OffsetCountPairType tile_exclusive;

        struct
        {
            typename discontinuity_type::storage_type   flags;
            typename prefix_op_factory::storage_type    prefix;
            typename warp_scan_pairs_type::storage_type warp_scan[warps_no];
            ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
            typename detail::raw_storage<OffsetCountPairType[warps_no]> warp_aggregates;
            ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP
        } scan;

        union ScatterStorage
        {
            typename warp_exchange_offsets_type::storage_type scatter_offsets[warps_no];
            typename warp_exchange_counts_type::storage_type  scatter_counts[warps_no];
        } scatter;
    };

    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type_ = detail::raw_storage<storage_type>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP

    template<typename InputIterator,
             typename OffsetsOutputIterator,
             typename CountsOutputIterator,
             typename LookbackScanState>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    OffsetCountPairType process_tile(const InputIterator   tile_input,
                                     OffsetsOutputIterator offsets_output,
                                     CountsOutputIterator  counts_output,
                                     LookbackScanState     scan_state,
                                     const unsigned int    tile_id,
                                     const std::size_t     number_of_tiles,
                                     const std::size_t     size,
                                     storage_type_&        storage_)
    {
        storage_type& storage = storage_.get();

        static constexpr unsigned int items_per_tile = BlockSize * ItemsPerThread;
        const std::size_t             tile_offset    = tile_id * items_per_tile;

        // First and last tiles
        const bool is_first_tile = tile_id == 0;
        const bool is_last_tile  = tile_id == number_of_tiles - 1;

        // Input items remaining in last tile
        const unsigned int valid_in_last_tile
            = static_cast<unsigned int>(size - ((number_of_tiles - 1) * items_per_tile));

        const unsigned int flat_thread_id = threadIdx.x;
        unsigned int       warp_id        = ::rocprim::warp_id();

        InputType input[ItemsPerThread];

        // Load items.
        load_type{}.load_input_values(tile_input,
                                      is_last_tile,
                                      valid_in_last_tile,
                                      input,
                                      storage.load);
        ::rocprim::syncthreads(); // TODO: probably not necessary, depends on the load method

        // Flag items.
        unsigned int head_flags[ItemsPerThread];
        unsigned int tail_flags[ItemsPerThread];
        discontinuity_type{}.flag_heads_and_tails(tile_input,
                                                  input,
                                                  head_flags,
                                                  tail_flags,
                                                  is_first_tile,
                                                  is_last_tile,
                                                  valid_in_last_tile,
                                                  storage.scan.flags);

        // Heads and tails are flagged, so we can identify which runs are non-trivial:
        // input  [1, 1, 1, 2, 10, 10, 10, 88]
        // heads  [1, 0, 0, 1,  1,  0,  0,  1]
        // tails  [0, 0, 1, 1,  0,  0,  1,  1]
        // offset [1, 0, 0, 0,  1,  0,  0,  0] = head && !tail  (first items of non-trivial runs)
        // count  [1, 1, 1, 0,  1,  1,  1,  0] = !head || !tail (items of non-trivial runs)
        OffsetCountPairType offsets_and_run_items[ItemsPerThread];
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            ::rocprim::get<0>(offsets_and_run_items[i]) = head_flags[i] && !tail_flags[i];
            ::rocprim::get<1>(offsets_and_run_items[i]) = !head_flags[i] || !tail_flags[i];
        }

        // Exclusive scan of offsets and counts

        // [0]: number of non-trivial run starts in this block
        // [1]: number of items in the last non-trivial run of this block
        OffsetCountPairType tile_aggregate;

        // [0]: number of non-trivial run starts in this warp
        // [1]: number of items in the last non-trivial run of this warp
        OffsetCountPairType warp_aggregate;

        // [0]: number of non-trivial run starts in previous warps from this block
        // [1]: number of items in the last non-trivial run in previous warps from this block
        OffsetCountPairType warp_exclusive_in_tile;

        // [0]: number of non-trivial run starts in previous threads from this warp
        // [1]: number of items in the last non-trivial run in previous threads from this warp
        OffsetCountPairType thread_exclusive_in_warp;

        // Scan function:
        //  - always adds the number of non-trivial-run starts (first component) and
        //  - always keeps updated the number of items in the last non-trivial run found. Two
        //    cases can happen
        //      1. when the first component of rhs is 0, it is either a trivial run or a non-head
        //         item of the latest non-trivial run found. In this case, we always add the
        //         second component of rhs to the one of lhs because for the former it is 0
        //         (so nothing changes) and for the latter it is 1 so we add 1 to the length of
        //         the latest non-trivial run found).
        //      2. when the first component of rhs is 1, a new non-trivial run start
        //         has been found, so we start counting from 1, which is the second component
        //         of rhs.
        auto scan_op = [&](const OffsetCountPairType& lhs, const OffsetCountPairType& rhs)
        {
            return OffsetCountPairType{rocprim::get<0>(lhs) + rocprim::get<0>(rhs),
                                       rocprim::get<0>(rhs) == 0
                                           ? rocprim::get<1>(lhs) + rocprim::get<1>(rhs)
                                           : rocprim::get<1>(rhs)};
        };

        // Warp scan.
        warp_scan_type{}.scan(offsets_and_run_items,
                              tile_aggregate,
                              warp_aggregate,
                              warp_exclusive_in_tile,
                              thread_exclusive_in_warp,
                              scan_op,
                              storage.scan.warp_scan,
                              storage.scan.warp_aggregates.get());

        // At this point, we have computed:
        // - the pairs (offsets_and_run_items) with the heads and elements of non-trivial runs
        // - the pair (thread_exclusive_in_warp) with
        //      1. the number of non-trivial runs starts in previous threads of the warp
        //      2. the length of the last non-trivial run in previous threads of the warp
        // - the pair (warp_exclusive_in_tile) with
        //      1. the number of non-trivial runs starts in previous warps of the tile
        //      2. the length of the last non-trivial run in previous warps of the tile
        // - the pair (warp_aggregate) with
        //      1. the number of non-trivial runs starts in this warp
        //      2. the length of the last non-trivial run in this warp
        // - the pair (tile_aggregate) with
        //      1. the number of non-trivial runs starts in this tile
        //      2. the length of the last non-trivial run in this tile

        OffsetCountPairType offsets_and_run_items_output[ItemsPerThread];
        OffsetCountPairType offsets_and_counts[ItemsPerThread];
        OffsetCountPairType reduction;

        // Number of non-trivial run starts in previous threads from this warp
        size_t thread_num_runs_exclusive_in_warp[ItemsPerThread];

        size_t tile_num_runs_aggregate;
        size_t tile_num_runs_exclusive_in_global;
        size_t warp_num_runs_aggregate;
        size_t warp_num_runs_exclusive_in_tile;

        if(is_first_tile)
        {
            // Update tile status if this is not the last tile
            if(!is_last_tile && (flat_thread_id == 0))
            {
                scan_state.set_complete(0, tile_aggregate);
            }

            // If there are no non-trivial runs starts in the previous warp threads, then
            // `thread_exclusive_in_warp<1>` denotes the number of items in the last
            // non-trivial run of the previous tile threads
            if(::rocprim::get<0>(thread_exclusive_in_warp) == 0)
            {
                ::rocprim::get<1>(thread_exclusive_in_warp)
                    += ::rocprim::get<1>(warp_exclusive_in_tile);
            }

            // Update offsets_and_run_items with the non-trivial runs from previous threads
            // of the warp
            thread_scan_exclusive(offsets_and_run_items,
                                  offsets_and_run_items_output,
                                  scan_op,
                                  thread_exclusive_in_warp /*initial_value*/);

            // Compute the offsets of the non-trivial runs.
            // If there was a run that started in a previous tile, there will be no run start
            // flagged for it. Instead, either its length will be scattered when scattering the
            // offset of the next non-trivial run or it is the last non-trivial run and its
            // length will be stored when finishing processing the last tile.
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                ::rocprim::get<0>(offsets_and_counts[i])
                    = tile_offset + (flat_thread_id * ItemsPerThread) + i; // offset of run
                ::rocprim::get<1>(offsets_and_counts[i])
                    = ::rocprim::get<1>(offsets_and_run_items_output[i]); // length of run
                thread_num_runs_exclusive_in_warp[i]
                    = (::rocprim::get<0>(
                          offsets_and_run_items[i])) // if there was a non-trivial run start
                          ? ::rocprim::get<0>(offsets_and_run_items_output[i]) // keep offset
                          : warp_size * ItemsPerThread; // else, discard offset
            }

            tile_num_runs_aggregate           = ::rocprim::get<0>(tile_aggregate);
            tile_num_runs_exclusive_in_global = 0;
            warp_num_runs_aggregate           = ::rocprim::get<0>(warp_aggregate);
            warp_num_runs_exclusive_in_tile   = ::rocprim::get<0>(warp_exclusive_in_tile);

            // Return running total inclusive of this tile (this is the first tile so there is no)
            reduction = tile_aggregate;
        }
        else
        {
            auto lookback_op = detail::lookback_scan_prefix_op<OffsetCountPairType,
                                                               decltype(scan_op),
                                                               decltype(scan_state)>{tile_id,
                                                                                     scan_op,
                                                                                     scan_state};

            auto offset_lookback_op = prefix_op_factory::create(lookback_op, storage.scan.prefix);

            // First warp of the tile computes the tile prefix in lane 0
            if(warp_id == 0)
            {
                // 1. Set tile_aggregate as partial prefix for next block
                // 2. Get prefix from previous block
                // 3. Set scan_op(prefix, tile_aggregate) as complete (inclusive) prefix for next block
                // 4. Store block_reduction (tile_aggregate) and prefix (exclusive prefix)
                offset_lookback_op(tile_aggregate);

                if(flat_thread_id == 0)
                {
                    storage.tile_exclusive = prefix_op_factory::get_prefix(storage.scan.prefix);
                }
            }

            ::rocprim::syncthreads();

            OffsetCountPairType tile_exclusive_in_global = storage.tile_exclusive;
            OffsetCountPairType thread_exclusive
                = scan_op(tile_exclusive_in_global, warp_exclusive_in_tile);

            // If there are no non-trivial runs starts in the previous warp threads, then
            // `thread_exclusive_in_warp<1>` denotes the number of items in the last
            // non-trivial run of the previous grids threads
            if(::rocprim::get<0>(thread_exclusive_in_warp) == 0)
            {
                ::rocprim::get<1>(thread_exclusive_in_warp) += ::rocprim::get<1>(thread_exclusive);
            }

            thread_scan_exclusive(offsets_and_run_items,
                                  offsets_and_run_items_output,
                                  scan_op,
                                  thread_exclusive_in_warp /*initial_value*/);

            // Compute the offsets of the non-trivial runs.
            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                ::rocprim::get<0>(offsets_and_counts[i])
                    = tile_offset + (flat_thread_id * ItemsPerThread) + i; // offset of run
                ::rocprim::get<1>(offsets_and_counts[i])
                    = ::rocprim::get<1>(offsets_and_run_items_output[i]); // length of last run
                thread_num_runs_exclusive_in_warp[i]
                    = (::rocprim::get<0>(
                          offsets_and_run_items[i])) // if there was a non-trivial run start
                          ? ::rocprim::get<0>(offsets_and_run_items_output[i]) // keep offset
                          : warp_size * ItemsPerThread; // else, discard offset
            }

            tile_num_runs_aggregate           = ::rocprim::get<0>(tile_aggregate);
            tile_num_runs_exclusive_in_global = ::rocprim::get<0>(tile_exclusive_in_global);
            warp_num_runs_aggregate           = ::rocprim::get<0>(warp_aggregate);
            warp_num_runs_exclusive_in_tile   = ::rocprim::get<0>(warp_exclusive_in_tile);

            // Return running total (inclusive of this tile)
            reduction = scan_state.get_complete_value(tile_id);
        }

        // Scatter
        warp_scatter_type{}.scatter(offsets_and_counts,
                                    offsets_output,
                                    counts_output,
                                    tile_num_runs_aggregate,
                                    tile_num_runs_exclusive_in_global,
                                    warp_num_runs_aggregate,
                                    warp_num_runs_exclusive_in_tile,
                                    thread_num_runs_exclusive_in_warp,
                                    storage.scatter.scatter_offsets[warp_id],
                                    storage.scatter.scatter_counts[warp_id]);

        // Return running total (inclusive of this tile)
        return reduction;
    }
};

template<typename Config,
         typename OffsetCountPairType,
         typename InputIterator,
         typename OffsetsOutputIterator,
         typename CountsOutputIterator,
         typename RunsCountOutputIterator,
         typename LookbackScanState>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void non_trivial_kernel_impl(InputIterator                  input,
                             const OffsetsOutputIterator    offsets_output,
                             const CountsOutputIterator     counts_output,
                             const RunsCountOutputIterator  runs_count_output,
                             const LookbackScanState        scan_state,
                             ordered_block_id<unsigned int> ordered_tile_id,
                             const std::size_t              number_of_tiles,
                             const std::size_t              size)
{
    static constexpr unsigned int         block_size        = Config::block_size;
    static constexpr unsigned int         items_per_thread  = Config::items_per_thread;
    static constexpr unsigned int         tiles_per_block   = Config::tiles_per_block;
    static constexpr block_load_method    load_input_method = Config::load_keys_method;
    static constexpr block_scan_algorithm scan_algorithm    = Config::scan_algorithm;
    static constexpr unsigned int         items_per_tile    = block_size * items_per_thread;

    using input_type  = ::rocprim::detail::value_type_t<InputIterator>;
    using offset_type = unsigned int;
    using count_type  = unsigned int;

    using tile_processor = tile_helper<input_type,
                                       offset_type,
                                       count_type,
                                       OffsetCountPairType,
                                       block_size,
                                       items_per_thread,
                                       load_input_method,
                                       scan_algorithm>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename decltype(ordered_tile_id)::storage_type tile_id;
        typename tile_processor::storage_type_           tile;
    } storage;

    for(unsigned int i = 0; i < tiles_per_block; ++i)
    {
        ::rocprim::syncthreads();
        const std::size_t tile_id = ordered_tile_id.get(threadIdx.x, storage.tile_id);
        if(tile_id >= number_of_tiles)
        {
            return;
        }

        const std::size_t   tile_offset = tile_id * items_per_tile;
        const InputIterator tile_input  = input + tile_offset;

        const size_t valid_in_last_tile
            = static_cast<size_t>(size - (size_t{number_of_tiles - 1} * items_per_tile));

        ::rocprim::syncthreads();
        if(tile_id < number_of_tiles - 1)
        {
            tile_processor{}.process_tile(tile_input,
                                          offsets_output,
                                          counts_output,
                                          scan_state,
                                          tile_id,
                                          number_of_tiles,
                                          size,
                                          storage.tile);
        }
        else if(valid_in_last_tile > 0)
        {
            OffsetCountPairType total = tile_processor{}.process_tile(tile_input,
                                                                      offsets_output,
                                                                      counts_output,
                                                                      scan_state,
                                                                      tile_id,
                                                                      number_of_tiles,
                                                                      size,
                                                                      storage.tile);
            // First thread of last tile sets the total number of non-trivial runs found and updates
            // the counts with the last run's length if necessary.
            if(threadIdx.x == 0)
            {
                *runs_count_output = ::rocprim::get<0>(total);

                if(::rocprim::get<0>(total) > 0)
                {
                    counts_output[::rocprim::get<0>(total) - 1] = ::rocprim::get<1>(total);
                }
            }
        }
    }
}

} // namespace run_length_encode
} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_RUN_LENGTH_ENCODE_HPP_
