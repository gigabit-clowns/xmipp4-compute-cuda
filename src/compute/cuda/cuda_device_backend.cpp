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
 * @file cuda_device_backend.cpp
 * @author Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
 * @brief Implementation of cuda_device_backend.hpp
 * @date 2024-10-30
 * 
 */

#include "cuda_device_backend.hpp"

#include <xmipp4/core/compute/device_manager.hpp>
#include <xmipp4/core/compute/device.hpp> // TODO replace with cuda dev

#include <numeric>
#include <cstdlib>

#include <cuda_runtime.h>

namespace xmipp4
{
namespace compute
{

const std::string cuda_device_backend::m_name = "cuda";



const std::string& cuda_device_backend::get_name() const noexcept
{
    return m_name;
}

version cuda_device_backend::get_version() const noexcept
{
    int cuda_version;
    cudaRuntimeGetVersion(&cuda_version);

    const auto major_div = std::div(cuda_version, 1000);
    const auto minor_div = std::div(major_div.rem, 10);

    return version(
        major_div.quot,
        minor_div.quot,
        minor_div.rem
    );
}

bool cuda_device_backend::is_available() const noexcept
{
    return true;
}

void cuda_device_backend::enumerate_devices(std::vector<std::size_t> &ids) const
{
    int count;
    cudaGetDeviceCount(&count);
    
    ids.resize(count);
    std::iota(
        ids.begin(), ids.end(),
        count
    );
}

bool cuda_device_backend::get_device_properties(std::size_t id, 
                                                device_properties &desc ) const
{
    int count;
    cudaGetDeviceCount(&count);

    const auto device = static_cast<int>(id);
    const auto result = device < count;
    if (result)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        // Write
        desc.set_name(std::string(prop.name));
        desc.set_physical_location("TODO");
        desc.set_type(device_type::gpu); // Maybe not?
        desc.set_total_memory_bytes(prop.totalGlobalMem);
    }

    return result;
}

std::unique_ptr<device> 
cuda_device_backend::create_device(std::size_t id)
{
    return nullptr; // TODO
}

std::shared_ptr<device> 
cuda_device_backend::create_device_shared(std::size_t id)
{
    return nullptr; // TODO
}

bool cuda_device_backend::register_at(device_manager &manager)
{
    return manager.register_backend(std::make_unique<cuda_device_backend>());
}

} // namespace system
} // namespace xmipp4
