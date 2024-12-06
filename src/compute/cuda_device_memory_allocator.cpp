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
 * @file cuda_device_memory_allocator.cpp
 * @author Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
 * @brief Implementation of cuda_device_memory_allocator.hpp
 * @date 2024-11-06
 * 
 */

#include <xmipp4/cuda/compute/cuda_device_memory_allocator.hpp>

#include "default_cuda_device_buffer.hpp"

#include <xmipp4/cuda/compute/cuda_device.hpp>
#include <xmipp4/cuda/compute/cuda_device_queue.hpp>
#include <xmipp4/core/platform/constexpr.hpp>

namespace xmipp4
{
namespace compute
{

XMIPP4_CONST_CONSTEXPR
std::size_t XMIPP4_CUDA_DEVICE_MEMORY_REQUEST_ROUND_STEP = 512;
XMIPP4_CONST_CONSTEXPR
std::size_t XMIPP4_CUDA_DEVICE_MEMORY_ALLOCATE_ROUND_STEP = 2<<20; // 2MB

cuda_device_memory_allocator::cuda_device_memory_allocator(cuda_device &device)
    : m_allocator(
        cuda_device_malloc(device.get_index()), 
        XMIPP4_CUDA_DEVICE_MEMORY_REQUEST_ROUND_STEP, 
        XMIPP4_CUDA_DEVICE_MEMORY_ALLOCATE_ROUND_STEP
    )
{
}

std::unique_ptr<device_buffer> 
cuda_device_memory_allocator::create_device_buffer(std::size_t size, 
                                                   std::size_t alignment,
                                                   device_queue &queue )
{
    return create_device_buffer(
        size, 
        alignment,
        dynamic_cast<cuda_device_queue&>(queue)
    );
}

std::unique_ptr<cuda_device_buffer> 
cuda_device_memory_allocator::create_device_buffer(std::size_t size, 
                                                   std::size_t alignment,
                                                   cuda_device_queue &queue )
{
    return std::make_unique<default_cuda_device_buffer>(
        size,
        alignment,
        &queue,
        *this
    );
}

std::shared_ptr<device_buffer> 
cuda_device_memory_allocator::create_device_buffer_shared(std::size_t size, 
                                                          std::size_t alignment,
                                                          device_queue &queue )
{
    return create_device_buffer_shared(
        size, 
        alignment,
        dynamic_cast<cuda_device_queue&>(queue)
    );
}

std::shared_ptr<cuda_device_buffer> 
cuda_device_memory_allocator::create_device_buffer_shared(std::size_t size, 
                                                          std::size_t alignment,
                                                          cuda_device_queue &queue )
{
    return std::make_shared<default_cuda_device_buffer>(
        size,
        alignment,
        &queue,
        *this
    );
}

const cuda_memory_block&
cuda_device_memory_allocator::allocate(std::size_t size,
                                       std::size_t alignment,
                                       cuda_device_queue *queue,
                                       cuda_memory_block_usage_tracker **usage_tracker)
{
    return m_allocator.allocate(size, alignment, queue, usage_tracker);
}

void cuda_device_memory_allocator::deallocate(const cuda_memory_block &block)
{
    m_allocator.deallocate(block);
}

} // namespace compute
} // namespace xmipp4
