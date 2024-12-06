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
 * @file cuda_memory_block_cache.hpp
 * @author Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
 * @brief Defines data structure representing a memory cache.
 * @date 2024-11-28
 * 
 */

#include "cuda_memory_block.hpp"
#include "cuda_memory_block_pool.hpp"
#include "cuda_deferred_memory_block_release.hpp"

namespace xmipp4 
{
namespace compute
{

/**
 * @brief Manages a set of cuda_memory_block-s to efficiently
 * re-use them when possible.
 * 
 */
class cuda_memory_block_cache
{
public:
    cuda_memory_block_cache(std::size_t minimum_size, 
                            std::size_t request_size_step );
    cuda_memory_block_cache(const cuda_memory_block_cache &other) = delete;
    cuda_memory_block_cache(cuda_memory_block_cache &&other) = default;
    ~cuda_memory_block_cache() = default;

    cuda_memory_block_cache&
    operator=(const cuda_memory_block_cache &other) = delete;
    cuda_memory_block_cache&
    operator=(cuda_memory_block_cache &&other) = default;

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
     * @param alignment Alignment requirement for the requested block.
     * @param queue Queue of the requested block.
     * @param usage_tracker Output parameter to register alien queues. May be 
     * nullptr. Ownership is managed by the allocator and the caller shall not
     * call any delete/free on it.
     * @return const uda_memory_block* Suitable block. nullptr if allocation
     * fails.
     * 
     */
    template <typename Allocator>
    const cuda_memory_block* 
    allocate(Allocator &allocator, 
             std::size_t size, 
             std::size_t alignment,
             const cuda_device_queue *queue,
             cuda_memory_block_usage_tracker **usage_tracker );

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
    cuda_memory_block_pool m_block_pool;
    cuda_deferred_memory_block_release m_deferred_blocks;
    std::size_t m_minimum_size;
    std::size_t m_request_size_step;

}; 

} // namespace compute
} // namespace xmipp4

#include "cuda_memory_block_cache.inl"
