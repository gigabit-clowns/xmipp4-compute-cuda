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

#include <xmipp4/cuda/compute/cuda_host_to_device_transfer.hpp>

#include <xmipp4/cuda/compute/cuda_device_queue.hpp>
#include <xmipp4/cuda/compute/cuda_device_buffer.hpp>
#include <xmipp4/cuda/compute/cuda_device_memory_allocator.hpp>

#include <xmipp4/core/compute/host_buffer.hpp>

#include "cuda_buffer_memcpy.hpp"

namespace xmipp4
{
namespace compute
{

void cuda_host_to_device_transfer::transfer_copy(const host_buffer &src_buffer, 
                                                 device_buffer &dst_buffer, 
                                                 device_queue &queue )
{
    transfer_copy(
        src_buffer,
        dynamic_cast<cuda_device_buffer&>(dst_buffer),
        dynamic_cast<cuda_device_queue&>(queue)
    );
}

void cuda_host_to_device_transfer::transfer_copy(const host_buffer &src_buffer, 
                                                      cuda_device_buffer &dst_buffer, 
                                                      cuda_device_queue &queue )
{
    cuda_memcpy(src_buffer, dst_buffer, queue);
}

void cuda_host_to_device_transfer::transfer_copy(const host_buffer &src_buffer, 
                                                 device_buffer &dst_buffer, 
                                                 span<const copy_region> regions,
                                                 device_queue &queue )
{
    transfer_copy(
        src_buffer,
        dynamic_cast<cuda_device_buffer&>(dst_buffer),
        regions,
        dynamic_cast<cuda_device_queue&>(queue)
    );
}

void cuda_host_to_device_transfer::transfer_copy(const host_buffer &src_buffer, 
                                                      cuda_device_buffer &dst_buffer, 
                                                      span<const copy_region> regions,
                                                      cuda_device_queue &queue )
{
    cuda_memcpy(src_buffer, dst_buffer, regions, queue);
}

std::shared_ptr<device_buffer> 
cuda_host_to_device_transfer::transfer(const std::shared_ptr<host_buffer> &buffer, 
                                       device_memory_allocator &allocator,
                                       std::size_t alignment,
                                       device_queue &queue )
{
    return transfer(buffer.get(), allocator, alignment, queue);
}

std::shared_ptr<const device_buffer> 
cuda_host_to_device_transfer::transfer(const std::shared_ptr<const host_buffer> &buffer, 
                                       device_memory_allocator &allocator,
                                       std::size_t alignment,
                                       device_queue &queue )
{
    return transfer(buffer.get(), allocator, alignment, queue);
}

std::shared_ptr<cuda_device_buffer> 
cuda_host_to_device_transfer::transfer(const host_buffer *buffer, 
                                       device_memory_allocator &allocator,
                                       std::size_t alignment,
                                       device_queue &queue )
{
    std::shared_ptr<cuda_device_buffer> result;

    if(buffer)
    {
        result = transfer(
            *buffer,
            dynamic_cast<cuda_device_memory_allocator&>(allocator),
            alignment,
            dynamic_cast<cuda_device_queue&>(queue)
        );
    }

    return result;
}

std::shared_ptr<cuda_device_buffer> 
cuda_host_to_device_transfer::transfer(const host_buffer &buffer, 
                                       cuda_device_memory_allocator &allocator,
                                       std::size_t alignment,
                                       cuda_device_queue &queue )
{
    std::shared_ptr<cuda_device_buffer> result;

    result = allocator.create_device_buffer(
        buffer.get_size(),
        alignment,
        queue
    );

    transfer_copy(buffer, *result, queue);

    return result;
}

} // namespace compute
} // namespace xmipp4
