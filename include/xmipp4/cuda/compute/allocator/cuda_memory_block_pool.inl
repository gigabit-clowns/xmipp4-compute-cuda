// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_memory_block_pool.hpp"

#include <xmipp4/core/memory/align.hpp>
#include <xmipp4/core/platform/assert.hpp>

#include <algorithm>
#include <functional>
#include <tuple>

namespace xmipp4
{
namespace hardware
{

inline
cuda_memory_block_context::cuda_memory_block_context(iterator prev, 
                                                     iterator next, 
                                                     bool free ) noexcept
    : m_prev(prev)
    , m_next(next)
    , m_free(free)
{
}

inline
cuda_memory_block_usage_tracker&
cuda_memory_block_context::get_usage_tracker() noexcept
{
    return m_usage_tracker;
}

inline
void cuda_memory_block_context::set_previous_block(iterator prev) noexcept
{
    m_prev = prev;
}

inline
cuda_memory_block_context::iterator 
cuda_memory_block_context::get_previous_block() const noexcept
{
    return m_prev;
}

inline
void cuda_memory_block_context::set_next_block(iterator next) noexcept
{
    m_next = next;
}

inline
cuda_memory_block_context::iterator
cuda_memory_block_context::get_next_block() const noexcept
{
    return m_next;
}

inline
void cuda_memory_block_context::set_free(bool free)
{
    m_free = free;
}

inline
bool cuda_memory_block_context::is_free() const noexcept
{
    return m_free;
}





inline
bool is_partition(const cuda_memory_block_context &block) noexcept
{
    const cuda_memory_block_pool::iterator null;
    return block.get_previous_block() != null ||
           block.get_next_block() != null ;
}

inline
bool is_mergeable(cuda_memory_block_pool::iterator ite) noexcept
{
    bool result;

    if (ite != cuda_memory_block_pool::iterator())
    {
        result = ite->second.is_free();
    }
    else
    {
        result = false;
    }

    return result;
}

inline
void update_forward_link(cuda_memory_block_pool::iterator ite) noexcept
{
    const auto next = ite->second.get_next_block();
    if (next != cuda_memory_block_pool::iterator())
    {
        next->second.set_previous_block(ite);
    }
}

inline
void update_backward_link(cuda_memory_block_pool::iterator ite) noexcept
{
    const auto prev = ite->second.get_previous_block();
    if (prev != cuda_memory_block_pool::iterator())
    {
        prev->second.set_next_block(ite);
    }
}

inline
void update_links(cuda_memory_block_pool::iterator ite) noexcept
{
    update_backward_link(ite);
    update_forward_link(ite);
}

inline
bool check_forward_link(cuda_memory_block_pool::iterator ite)  noexcept
{
    bool result;

    const auto next = ite->second.get_next_block();
    if (next != cuda_memory_block_pool::iterator())
    {
        result = next->second.get_previous_block() == ite;
    }
    else
    {
        result = true;
    }

    return result;
}

inline
bool check_backward_link(cuda_memory_block_pool::iterator ite) noexcept
{
    bool result;

    const auto prev = ite->second.get_previous_block();
    if (prev != cuda_memory_block_pool::iterator())
    {
        result = prev->second.get_next_block() == ite;
    }
    else
    {
        result = true;
    }

    return result;
}

inline
bool check_links(cuda_memory_block_pool::iterator ite) noexcept
{
    return check_backward_link(ite) && check_forward_link(ite);
}

inline
cuda_memory_block_pool::iterator 
find_suitable_block(cuda_memory_block_pool &blocks, 
                    std::size_t size,
                    std::size_t alignment,
                    const cuda_device_queue *queue )
{
    // Assuming that the blocks are ordered according to their queue reference
    // first and then their sizes, the best fit is achieved iterating from
    // the first suitable block.
    const cuda_memory_block key(nullptr, alignment, size, queue);
    auto ite = blocks.lower_bound(key);

    while (ite != blocks.end())
    {
        if(ite->first.get_queue() != queue)
        {
            // Reached the end of the allowed range.
            ite = blocks.end();
            break;
        }
        else if (ite->second.is_free())
        {
            // Found a suitable block
            break;
        }
        else
        {
            // Occupied. Try with the next.
            ++ite;
        }
    }

    return ite;
}

inline
cuda_memory_block_pool::iterator 
consider_partitioning_block(cuda_memory_block_pool &blocks,
                            cuda_memory_block_pool::iterator ite,
                            std::size_t size,
                            std::size_t threshold )
{
    const auto remaining = ite->first.get_size() - size;
    if (remaining >= threshold)
    {
        ite = partition_block(blocks, ite, size, remaining);
    }

    return ite;
}

inline
cuda_memory_block_pool::iterator 
partition_block(cuda_memory_block_pool &blocks,
                cuda_memory_block_pool::iterator ite,
                std::size_t size,
                std::size_t remaining )
{
    const auto queue = ite->first.get_queue();
    const auto prev = ite->second.get_previous_block();
    const auto next = ite->second.get_next_block();

    cuda_memory_block_pool::iterator first;
    cuda_memory_block_pool::iterator second;
    bool inserted;
    std::tie(first, inserted) = blocks.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(
            ite->first.get_data(), 
            size, 
            queue
        ),
        std::forward_as_tuple(
            prev,
            cuda_memory_block_pool::iterator(), // To be set
            ite->second.is_free()
        )
    );
    XMIPP4_ASSERT(inserted);
    std::tie(second, inserted) = blocks.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(
            memory::offset_bytes(ite->first.get_data(), size), 
            remaining, 
            queue
        ),
        std::forward_as_tuple(
            first,
            next,
            ite->second.is_free()
        )
    );
    XMIPP4_ASSERT(inserted);
    
    first->second.set_next_block(second);
    update_backward_link(first);
    update_forward_link(second);

    // Remove old block
    blocks.erase(ite);

    XMIPP4_ASSERT( check_links(first) );
    XMIPP4_ASSERT( check_links(second) );

    return first;
}

inline
cuda_memory_block_pool::iterator
consider_merging_block(cuda_memory_block_pool &blocks, 
                       cuda_memory_block_pool::iterator ite)
{
    const auto prev = ite->second.get_previous_block();
    const auto merge_prev = is_mergeable(prev);
    const auto next = ite->second.get_next_block();
    const auto merge_next = is_mergeable(next);

    if (merge_prev && merge_next)
    {
        ite = merge_blocks(blocks, prev, ite, next);
    }
    else if (merge_prev)
    {
        ite = merge_blocks(blocks, prev, ite);
    }
    else if (merge_next)
    {   
        ite = merge_blocks(blocks, ite, next);
    }

    return ite;
}

inline
cuda_memory_block_pool::iterator
merge_blocks(cuda_memory_block_pool &blocks,
             cuda_memory_block_pool::iterator first,
             cuda_memory_block_pool::iterator second )
{
    const auto data = first->first.get_data();
    const auto size = first->first.get_size() +
                      second->first.get_size() ;
    const auto queue = first->first.get_queue();
    const auto prev = first->second.get_previous_block();
    const auto next = second->second.get_next_block();

    cuda_memory_block_pool::iterator ite;
    bool inserted;
    std::tie(ite, inserted) = blocks.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(data, size, queue),
        std::forward_as_tuple(prev, next, true)
    );
    XMIPP4_ASSERT(inserted);

    update_links(ite);

    blocks.erase(first);
    blocks.erase(second);

    XMIPP4_ASSERT( check_links(ite) );
    return ite;
}

inline
cuda_memory_block_pool::iterator
merge_blocks(cuda_memory_block_pool &blocks,
             cuda_memory_block_pool::iterator first,
             cuda_memory_block_pool::iterator second,
             cuda_memory_block_pool::iterator third )
{
    const auto data = first->first.get_data();
    const auto size = first->first.get_size() +
                      second->first.get_size() +
                      third->first.get_size() ;
    const auto queue = first->first.get_queue();
    const auto prev = first->second.get_previous_block();
    const auto next = third->second.get_next_block();

    cuda_memory_block_pool::iterator ite;
    bool inserted;
    std::tie(ite, inserted) = blocks.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(data, size, queue),
        std::forward_as_tuple(prev, next, true)
    );
    XMIPP4_ASSERT(inserted);

    update_links(ite);

    blocks.erase(first);
    blocks.erase(second);
    blocks.erase(third);

    XMIPP4_ASSERT( check_links(ite) );
    return ite;
}

template <typename Allocator>
inline
cuda_memory_block_pool::iterator create_block(cuda_memory_block_pool &blocks,
                                              Allocator& allocator,
                                              std::size_t size,
                                              const cuda_device_queue *queue )
{
    cuda_memory_block_pool::iterator result;

    // Try to allocate
    void* data = allocator.allocate(size);
    if (data)
    {
        const cuda_memory_block_pool::iterator null;

        // Add block
        bool inserted;
        std::tie(result, inserted) = blocks.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(data, size, queue),
            std::forward_as_tuple(null, null, true)
        );
        XMIPP4_ASSERT(inserted);
    }
    else
    {
        result = blocks.end();
    }

    return result;
}

template <typename Allocator>
inline
cuda_memory_block_pool::iterator 
allocate_block(cuda_memory_block_pool &blocks, 
               const Allocator &allocator, 
               std::size_t size,
               std::size_t alignment,
               const cuda_device_queue *queue,
               std::size_t partition_min_size,
               std::size_t create_size_step )
{
    auto ite = find_suitable_block(blocks, size, alignment, queue);
    if (ite == blocks.end())
    {
        const auto create_size = memory::align_ceil(size, create_size_step);
        ite = create_block(blocks, allocator, create_size, queue);
    }

    if (ite != blocks.end())
    {
        ite = consider_partitioning_block(blocks, ite, size, partition_min_size);
        ite->second.set_free(false);
    }

    return ite;
}

inline
void deallocate_block(cuda_memory_block_pool &blocks, 
                      cuda_memory_block_pool::iterator ite )
{
    ite->second.set_free(true);
    ite->second.get_usage_tracker().reset();
    ite = consider_merging_block(blocks, ite);
}

template <typename Allocator>
inline
void release_blocks(cuda_memory_block_pool &blocks, Allocator &allocator)
{
    auto ite = blocks.begin();
    while (ite != blocks.cend())
    {
        if(ite->second.is_free() && !is_partition(ite->second))
        {
            allocator.deallocate(
                ite->first.get_data(),
                ite->first.get_size()
            );
            ite = blocks.erase(ite);
        }
        else
        {
            ++ite;
        }
    }
}

} // namespace hardware
} // namespace xmipp4
