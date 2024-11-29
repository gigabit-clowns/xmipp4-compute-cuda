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
 * @file cuda_device_to_host_transfer.cpp
 * @author Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
 * @brief Implementation of cuda_device_to_host_transfer.hpp
 * @date 2024-11-06
 * 
 */

#include <xmipp4/cuda/compute/cuda_device_to_host_transfer.hpp>

#include <xmipp4/cuda/compute/cuda_error.hpp>
#include <xmipp4/cuda/compute/cuda_device_queue.hpp>
#include <xmipp4/cuda/compute/cuda_device_buffer.hpp>
#include <xmipp4/cuda/compute/cuda_host_memory_allocator.hpp>

#include <xmipp4/core/memory/align.hpp>
#include <xmipp4/core/compute/host_buffer.hpp>
#include <xmipp4/core/compute/checks.hpp>


namespace xmipp4
{
namespace compute
{

void cuda_device_to_host_transfer::transfer_copy(const device_buffer &src_buffer, 
                                                 host_buffer &dst_buffer, 
                                                 device_queue &queue )
{
    transfer_copy_impl(
        dynamic_cast<const cuda_device_buffer&>(src_buffer),
        dst_buffer,
        dynamic_cast<cuda_device_queue&>(queue)
    );
}

void cuda_device_to_host_transfer::transfer_copy_impl(const cuda_device_buffer &src_buffer, 
                                                      host_buffer &dst_buffer, 
                                                      cuda_device_queue &queue )
{
    const auto type = require_same_type(
        src_buffer.get_type(), dst_buffer.get_type()
    );
    const auto count = require_same_count(
        src_buffer.get_count(), dst_buffer.get_count()
    );
    const auto element_size = get_size(type);

    XMIPP4_CUDA_CHECK(
        cudaMemcpyAsync(
            dst_buffer.get_data(),
            src_buffer.get_data(),
            element_size*count,
            cudaMemcpyDeviceToHost,
            queue.get_handle()
        )
    );
}

void cuda_device_to_host_transfer::transfer_copy(const device_buffer &src_buffer,
                                                 host_buffer &dst_buffer,
                                                 span<const copy_region> regions,
                                                 device_queue &queue )
{
    transfer_copy_impl(
        dynamic_cast<const cuda_device_buffer&>(src_buffer),
        dst_buffer,
        regions,
        dynamic_cast<cuda_device_queue&>(queue)
    );
}

void cuda_device_to_host_transfer::transfer_copy_impl(const cuda_device_buffer &src_buffer,
                                                      host_buffer &dst_buffer,
                                                      span<const copy_region> regions,
                                                      cuda_device_queue &queue )
{
    const auto* src_data = src_buffer.get_data();
    auto* dst_data = dst_buffer.get_data();
    const auto src_count = src_buffer.get_count();
    const auto dst_count = dst_buffer.get_count();
    const auto type = require_same_type(
        src_buffer.get_type(), dst_buffer.get_type()
    );
    const auto element_size = get_size(type);

    for (const copy_region &region : regions)
    {
        require_valid_region(region, src_count, dst_count);
        const auto region_bytes = as_bytes(region, element_size);

        XMIPP4_CUDA_CHECK(
            cudaMemcpyAsync(
                memory::offset_bytes(dst_data, region_bytes.get_destination_offset()),
                memory::offset_bytes(src_data, region_bytes.get_source_offset()),
                region_bytes.get_count(),
                cudaMemcpyDeviceToHost,
                queue.get_handle()
            )
        );
    }
}

std::shared_ptr<host_buffer> 
cuda_device_to_host_transfer::transfer(const std::shared_ptr<device_buffer> &buffer, 
                                       host_memory_allocator &allocator,
                                       device_queue &queue )
{
    return transfer_impl(
        std::dynamic_pointer_cast<cuda_device_buffer>(buffer),
        dynamic_cast<cuda_host_memory_allocator&>(allocator),
        dynamic_cast<cuda_device_queue&>(queue)
    );
}

std::shared_ptr<host_buffer> 
cuda_device_to_host_transfer::transfer_impl(const std::shared_ptr<cuda_device_buffer> &buffer, 
                                            cuda_host_memory_allocator &allocator,
                                            cuda_device_queue &queue )
{
    std::shared_ptr<host_buffer> result;

    if (buffer)
    {
        result = allocator.create_host_buffer_shared_impl(
            buffer->get_type(), buffer->get_count(), queue
        );

        transfer_copy_impl(*buffer, *result, queue);
    }

    return result;
}

std::shared_ptr<const host_buffer> 
cuda_device_to_host_transfer::transfer(const std::shared_ptr<const device_buffer> &buffer, 
                                       host_memory_allocator &allocator,
                                       device_queue &queue )
{
    return transfer_impl(
        std::dynamic_pointer_cast<const cuda_device_buffer>(buffer),
        dynamic_cast<cuda_host_memory_allocator&>(allocator),
        dynamic_cast<cuda_device_queue&>(queue)
    );
}

std::shared_ptr<const host_buffer> 
cuda_device_to_host_transfer::transfer_impl(const std::shared_ptr<const cuda_device_buffer> &buffer, 
                                            cuda_host_memory_allocator &allocator,
                                            cuda_device_queue &queue )
{
    std::shared_ptr<const host_buffer> result;

    if (buffer)
    {
        auto tmp = allocator.create_host_buffer_shared_impl(
            buffer->get_type(), buffer->get_count(), queue
        );

        transfer_copy_impl(*buffer, *tmp, queue);
        result = std::move(tmp);
    }

    return result;
}

} // namespace compute
} // namespace xmipp4
