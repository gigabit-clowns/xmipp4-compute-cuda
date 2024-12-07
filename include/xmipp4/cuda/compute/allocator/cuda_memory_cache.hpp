#pragma once

/***************************************************************************
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

/**
 * @file cuda_memory_cache.hpp
 * @author Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
 * @brief Defines data structure representing a memory cache.
 * @date 2024-11-06
 * 
 */

#include "cuda_memory_block.hpp"
#include "cuda_memory_block_pool.hpp"

namespace xmipp4 
{
namespace compute
{

/**
 * @brief Cache and re-use memory allocations.
 * 
 * Based on:
 * https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html
 * 
 */
class cuda_memory_cache
{
public:
    /**
     * @brief Construct a new cuda memory allocation cache.
     * 
     * @param small_large_threshold Threshold in bytes to consider a block as 
     * big or small.
     * @param size_step Quantization step for block sizes.
     * @param request_step Quantization step for memory allocations.
     * 
     */
    explicit 
    cuda_memory_cache(std::size_t small_large_threshold = (1U << 20),
                      std::size_t size_step = 512,
                      std::size_t request_step = (2U << 20));
    cuda_memory_cache(const cuda_memory_cache &other) = default;
    cuda_memory_cache(cuda_memory_cache &&other) = default;
    ~cuda_memory_cache() = default;

    cuda_memory_cache& operator=(const cuda_memory_cache &other) = default;
    cuda_memory_cache& operator=(cuda_memory_cache &&other) = default;

    /**
     * @brief Return free blocks to the allocator when possible.
     * 
     * @tparam Allocator Class that implements allocate() and deallocate()
     * @param allocator Allocator used for deallocating free blocks.
     * Must be compatible with the allocator used in allocate()
     * 
     */
    template <typename Allocator>
    void release(Allocator &allocator);

    /**
     * @brief Allocate a new block.
     * 
     * @tparam Allocator Class that implements allocate() and deallocate()
     * @param allocator Allocator object. Used when there are no suitable blocks
     * in cache.
     * @param size Size of the requested block.
     * @param queue_id Queue if for the requested block.
     * @return const cuda_memory_block* Suitable block. nullptr if allocation
     * fails.
     * 
     */
    template <typename Allocator>
    const cuda_memory_block* allocate(Allocator &allocator, 
                                      std::size_t size, 
                                      std::size_t queue_id );

    /**
     * @brief Deallocate a block.
     * 
     * @param block Block to be deallocated.
     * 
     * @note This operation does not return the block to the allocator.
     * Instead, it caches it for potential re-use.
     * 
     */
    void deallocate(const cuda_memory_block &block);

private:
    cuda_memory_block_pool m_small_block_pool;
    cuda_memory_block_pool m_large_block_pool;
    std::size_t m_small_large_threshold;
    std::size_t m_size_step;
    std::size_t m_request_size_step;

    template <typename Allocator>
    const cuda_memory_block* allocate_from_pool(cuda_memory_block_pool &blocks, 
                                                Allocator &allocator, 
                                                std::size_t size,
                                                std::size_t queue_id,
                                                std::size_t min_size );

    bool is_small(std::size_t size) const noexcept;

}; 

} // namespace compute
} // namespace xmipp4

#include "cuda_memory_cache.inl"
