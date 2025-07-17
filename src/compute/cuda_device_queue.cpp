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

#include <xmipp4/cuda/compute/cuda_device_queue.hpp>

#include <xmipp4/cuda/compute/cuda_error.hpp>
#include <xmipp4/cuda/compute/cuda_device.hpp>

#include <utility>

namespace xmipp4
{
namespace compute
{

cuda_device_queue::cuda_device_queue()
{
    XMIPP4_CUDA_CHECK( cudaStreamCreate(&m_stream) );
}

cuda_device_queue::cuda_device_queue(cuda_device &device)
{
    XMIPP4_CUDA_CHECK( cudaSetDevice(device.get_index()) );
    XMIPP4_CUDA_CHECK( cudaStreamCreate(&m_stream) );
}

cuda_device_queue::cuda_device_queue(cuda_device_queue &&other) noexcept
    : m_stream(other.m_stream)
{
    other.m_stream = nullptr;
}

cuda_device_queue::~cuda_device_queue()
{
    reset();
}

cuda_device_queue& 
cuda_device_queue::operator=(cuda_device_queue &&other) noexcept
{
    swap(other);
    other.reset();
    return *this;
}

void cuda_device_queue::swap(cuda_device_queue &other) noexcept
{
    std::swap(m_stream, other.m_stream);
}

void cuda_device_queue::reset() noexcept
{
    if (m_stream)
    {
        XMIPP4_CUDA_CHECK( cudaStreamDestroy(m_stream) );
    }
}


cuda_device_queue::handle cuda_device_queue::get_handle() noexcept
{
    return m_stream;
}

void cuda_device_queue::wait_until_completed() const
{
    XMIPP4_CUDA_CHECK( cudaStreamSynchronize(m_stream) );
}

bool cuda_device_queue::is_idle() const noexcept
{
    const auto code = cudaStreamQuery(m_stream);

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
        result = false; // To avoid warnings. The above line should throw.
        break;
    }
    return result;
}

} // namespace compute
} // namespace xmipp4
