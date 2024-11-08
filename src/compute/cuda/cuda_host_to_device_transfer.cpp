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
 * @file cuda_host_to_device_transfer.cpp
 * @author Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
 * @brief Implementation of cuda_host_to_device_transfer.hpp
 * @date 2024-11-06
 * 
 */

#include "cuda_host_to_device_transfer.hpp"

#include "cuda_device_queue.hpp"
#include "cuda_device_buffer.hpp"
#include "cuda_device_memory_allocator.hpp"

#include <xmipp4/core/compute/host_buffer.hpp>

#include <cuda_runtime.h>

namespace xmipp4
{
namespace compute
{

void cuda_host_to_device_transfer::transfer(const std::shared_ptr<const host_buffer> &src_buffer, 
                                            device_buffer &dst_buffer, 
                                            device_queue &queue )
{
    if (!src_buffer)
    {
        throw std::invalid_argument("src_buffer cannot be nullptr");
    }

    if (src_buffer->get_type() != dst_buffer.get_type())
    {
        throw std::invalid_argument("Both buffers must have the same numerical type");
    }
    
    if (src_buffer->get_count() != dst_buffer.get_count())
    {
        throw std::invalid_argument("Both buffers must have the same element count");
    }

    auto &cuda_queue = dynamic_cast<cuda_device_queue&>(queue);
    const auto element_size = get_size(src_buffer->get_type());

    // TODO check return
    cudaMemcpyAsync(
        dynamic_cast<cuda_device_buffer&>(dst_buffer).get_data(),
        src_buffer->get_data(),
        element_size*src_buffer->get_count(),
        cudaMemcpyHostToDevice,
        cuda_queue.get_handle()
    );

    wait();
    m_current = src_buffer;
    m_event.record(cuda_queue);
}

std::shared_ptr<device_buffer> 
cuda_host_to_device_transfer::transfer_nocopy(const std::shared_ptr<host_buffer> &buffer, 
                                              device_memory_allocator &allocator,
                                              device_queue &queue )
{
    std::shared_ptr<device_buffer> result;

    if (buffer)
    {
        auto result = allocator.create_buffer_shared(
            buffer->get_type(), buffer->get_count(), queue
        );

        transfer(buffer, *result, queue);
    }

    return result;
}

std::shared_ptr<const device_buffer> 
cuda_host_to_device_transfer::transfer_nocopy(const std::shared_ptr<const host_buffer> &buffer, 
                                              device_memory_allocator &allocator,
                                              device_queue &queue )
{
    std::shared_ptr<const device_buffer> result;

    if (buffer)
    {
        auto result = allocator.create_buffer_shared(
            buffer->get_type(), buffer->get_count(), queue
        );

        transfer(buffer, *result, queue);
    }

    return result;
}

void cuda_host_to_device_transfer::wait()
{
    if (m_current)
    {
        m_event.synchronize();
        m_current = nullptr;
    }
}

void cuda_host_to_device_transfer::wait(device_queue &queue)
{
    if (m_current)
    {
        m_event.wait(queue);
    }   
}

} // namespace compute
} // namespace xmipp4