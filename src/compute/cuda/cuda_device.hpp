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
 * @file cuda_device.hpp
 * @author Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
 * @brief Defines cuda_device class.
 * @date 2024-10-30
 * 
 */

#include <xmipp4/core/compute/device.hpp>

namespace xmipp4 
{
namespace compute
{

class cuda_device_backend;

class cuda_device final
    : public device
{
public:
    cuda_device(int device);
    cuda_device(const cuda_device &other) = default;
    cuda_device(cuda_device &&other) = default;
    virtual ~cuda_device() = default;

    cuda_device& operator=(const cuda_device &other) = default;
    cuda_device& operator=(cuda_device &&other) = default;

    std::unique_ptr<device_queue> create_queue() final;
    std::shared_ptr<device_queue> create_queue_shared() final;

    std::unique_ptr<device_buffer> 
    create_buffer(numerical_type type, std::size_t count) final;
    std::shared_ptr<device_buffer> 
    create_buffer_shared(numerical_type type, std::size_t count) final;

private:
    int m_device;

}; 

} // namespace compute
} // namespace xmipp4
