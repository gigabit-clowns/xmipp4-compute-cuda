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
 * @file cuda_event.cpp
 * @author Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
 * @brief Implementation of cuda_event.hpp
 * @date 2024-11-07
 * 
 */

#include <xmipp4/cuda/compute/cuda_event.hpp>

#include <xmipp4/cuda/compute/cuda_error.hpp>
#include <xmipp4/cuda/compute/cuda_device_queue.hpp>

#include <utility>

namespace xmipp4
{
namespace compute
{

cuda_event::cuda_event()
{
    XMIPP4_CUDA_CHECK( cudaEventCreate(&m_event) );
}

cuda_event::cuda_event(cuda_event &&other) noexcept
    : m_event(other.m_event)
{
    other.m_event = nullptr;
}

cuda_event::~cuda_event()
{
    reset();
}

cuda_event& 
cuda_event::operator=(cuda_event &&other) noexcept
{
    swap(other);
    other.reset();
    return *this;
}

void cuda_event::swap(cuda_event &other) noexcept
{
    std::swap(m_event, other.m_event);
}

void cuda_event::reset() noexcept
{
    if (m_event)
    {
        XMIPP4_CUDA_CHECK( cudaEventDestroy(m_event) );
    }
}

cuda_event::handle cuda_event::get_handle() noexcept
{
    return m_event;
}



void cuda_event::signal(device_queue &queue)
{
    signal(dynamic_cast<cuda_device_queue&>(queue));
}

void cuda_event::signal(cuda_device_queue &queue)
{
    XMIPP4_CUDA_CHECK( cudaEventRecord(m_event, queue.get_handle()) );
}

void cuda_event::wait() const
{
    XMIPP4_CUDA_CHECK( cudaEventSynchronize(m_event) );
}

void cuda_event::wait(device_queue &queue) const
{
    wait(dynamic_cast<cuda_device_queue&>(queue));
}

void cuda_event::wait(cuda_device_queue &queue) const
{
    XMIPP4_CUDA_CHECK(
        cudaStreamWaitEvent(queue.get_handle(), m_event, cudaEventWaitDefault)
    );
}

bool cuda_event::is_signaled() const
{
    const auto code = cudaEventQuery(m_event);

    bool result;
    switch (code)
    {
    case cudaSuccess:
        result = true;
        break;

    case cudaErrorNotReady:
        result = false;
        break;
    
    default:
        XMIPP4_CUDA_CHECK(code);
        break;
    }
    return result;
}

} // namespace compute
} // namespace xmipp4