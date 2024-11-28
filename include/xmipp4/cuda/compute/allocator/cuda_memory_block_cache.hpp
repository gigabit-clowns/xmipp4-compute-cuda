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
     * @param other_queues Set of queues (other than the one used for 
     * allocation) where this block was used. 
     * 
     * @note This operation does not return the block to the allocator.
     * Instead, it caches it for potential re-use.
     * 
     */
    void deallocate(const cuda_memory_block &block, 
                    span<cuda_device_queue*> other_queues );

private:
    cuda_memory_block_pool m_block_pool;
    cuda_deferred_memory_block_release m_deferred_blocks;
    std::size_t m_minimum_size;
    std::size_t m_request_size_step;

}; 

} // namespace compute
} // namespace xmipp4

#include "cuda_memory_block_cache.inl"