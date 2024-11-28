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
 * @file cuda_caching_memory_allocator.hpp
 * @author Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
 * @brief Defines cuda_caching_memory_allocator
 * @date 2024-11-28
 * 
 */

#include "cuda_memory_block_cache.hpp"

#include <xmipp4/core/platform/attributes.hpp>

namespace xmipp4 
{
namespace compute
{

/**
 * @brief High performance memory allocator to allow stream-synchronous
 * allocations.
 * 
 */
template <typename Allocator>
class cuda_caching_memory_allocator
{
public:
    using allocator_type = Allocator;

    cuda_caching_memory_allocator(allocator_type allocator,
                                  std::size_t minimum_size, 
                                  std::size_t request_size_step );
    cuda_caching_memory_allocator(const cuda_caching_memory_allocator &other) = delete;
    cuda_caching_memory_allocator(cuda_caching_memory_allocator &&other) = default;
    ~cuda_caching_memory_allocator() = default;

    cuda_caching_memory_allocator&
    operator=(const cuda_caching_memory_allocator &other) = delete;
    cuda_caching_memory_allocator&
    operator=(cuda_caching_memory_allocator &&other) = default;
    
    /**
     * @brief Return free blocks to the allocator when possible.
     * 
     */
    void release();

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
    const cuda_memory_block* allocate(std::size_t size, std::size_t queue_d);

    /**
     * @brief Deallocate a block.
     * 
     * @param block Block to be deallocated.
     * 
     * @note This operation does not return the block to the allocator.
     * Instead, it caches it for potential re-use.
     * 
     */
    void deallocate(const cuda_memory_block &block,
                    span<cuda_device_queue*> other_queues );

private:
    XMIPP4_NO_UNIQUE_ADDRESS allocator_type m_allocator;
    cuda_memory_block_cache m_cache;

}; 

} // namespace compute
} // namespace xmipp4

#include "cuda_caching_memory_allocator.inl"