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
 * @file test_cuda_memory_block_pool.cpp
 * @author Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
 * @brief Tests for cuda_memory_block_pool.hpp
 * @date 2024-11-19
 * 
 */

#include <xmipp4/cuda/compute/allocator/cuda_memory_block_pool.hpp>

#include <cstddef>

#include <catch2/catch_test_macros.hpp>
#include <trompeloeil.hpp>

using namespace xmipp4::compute;

class mock_allocator
{
public:
    MAKE_MOCK1(allocate, void* (std::size_t), const);
    MAKE_MOCK2(deallocate, void (void*, std::size_t), const);

};

TEST_CASE( "is_partition should return false when not partitioned", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, false)
    );

    REQUIRE( is_partition(ite->second) == false );
}

TEST_CASE( "is_partition should return true when partitioned in two", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, false)
    );
    ite = partition_block(pool, ite, 512, 512);

    // Check for both halves that the condition is satisfied
    REQUIRE( is_partition(ite->second) == true );
    ite = ite->second.get_next_block();
    REQUIRE( is_partition(ite->second) == true );
}

TEST_CASE( "is_partition should return true when partitioned in three", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, false)
    );
    ite = partition_block(pool, ite, 512, 512);
    ite = partition_block(pool, ite, 256, 256);

    // Check for all thirds that the condition is satisfied
    REQUIRE( is_partition(ite->second) == true );
    ite = ite->second.get_next_block();
    REQUIRE( is_partition(ite->second) == true );
    ite = ite->second.get_next_block();
    REQUIRE( is_partition(ite->second) == true );
}

TEST_CASE( "is_mergeable should return false when null iterator", "[cuda_memory_block_pool]" )
{
    const cuda_memory_block_pool::iterator null;
    REQUIRE( is_mergeable(null) == false );
}

TEST_CASE( "is_mergeable should return false when occupied", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, false)
    );
    REQUIRE( is_mergeable(ite) == false );
}

TEST_CASE( "is_mergeable should return true when free", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, true)
    );
    REQUIRE( is_mergeable(ite) == true );
}

TEST_CASE( "update_forward_link should produce valid link", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, true)
    );
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(1024)), 1024, 0),
        std::forward_as_tuple(null, ite, true)
    );

    REQUIRE( check_forward_link(ite) == false );
    update_forward_link(ite);
    REQUIRE( check_forward_link(ite) == true );
}

TEST_CASE( "update_backward_link should produce valid link", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, true)
    );
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(1024)), 1024, 0),
        std::forward_as_tuple(ite, null, true)
    );

    REQUIRE( check_backward_link(ite) == false );
    update_backward_link(ite);
    REQUIRE( check_backward_link(ite) == true );
}

TEST_CASE( "update_links should produce valid links", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite, prev, next;
    std::tie(prev, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, true)
    );
    std::tie(next, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(2048)), 1024, 0),
        std::forward_as_tuple(null, null, true)
    );
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(1024)), 1024, 0),
        std::forward_as_tuple(prev, next, true)
    );

    REQUIRE( check_forward_link(ite) == false );
    REQUIRE( check_backward_link(ite) == false );
    update_links(ite);
    REQUIRE( check_forward_link(ite) == true );
    REQUIRE( check_backward_link(ite) == true );
}

TEST_CASE( "check_forward_link should return false with invalid link", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, true)
    );
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(1024)), 1024, 0),
        std::forward_as_tuple(null, ite, true)
    );

    REQUIRE( check_forward_link(ite) == false );
}

TEST_CASE( "check_forward_link should return true with valid link", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator prev, next;
    std::tie(prev, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, true)
    );
    std::tie(next, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(1024)), 1024, 0),
        std::forward_as_tuple(prev, null, true)
    );

    prev->second.set_next_block(next);

    REQUIRE( check_forward_link(prev) == true );
}

TEST_CASE( "check_forward_link should return true with empty link", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, true)
    );

    REQUIRE( check_forward_link(ite) == true );
}

TEST_CASE( "check_backward_link should return false with invalid link", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, true)
    );
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(1024)), 1024, 0),
        std::forward_as_tuple(ite, null, true)
    );

    REQUIRE( check_backward_link(ite) == false );
}

TEST_CASE( "check_backward_link should return true with valid link", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator prev, next;
    std::tie(prev, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, true)
    );
    std::tie(next, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(1024)), 1024, 0),
        std::forward_as_tuple(prev, null, true)
    );
    
    prev->second.set_next_block(next);

    REQUIRE( check_backward_link(next) == true );
}

TEST_CASE( "find_suitable_block should return the smallest free block larger than the requested size", "[cuda_memory_block_pool]" )
{
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool pool = {
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(1)), 32, 1), 
            cuda_memory_block_context(null, null, true)
        ), // Too small, not matching queue
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(2)), 64, 1), 
            cuda_memory_block_context(null, null, true)
        ), // Not matching queue
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(3)), 64, 1), 
            cuda_memory_block_context(null, null, false)
        ), // Not matching queue, occupied
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(4)), 32, 2), 
            cuda_memory_block_context(null, null, true)
        ), // Too small
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(5)), 64, 2), 
            cuda_memory_block_context(null, null, false)
        ), // Occupied
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(6)), 64, 2), 
            cuda_memory_block_context(null, null, true)
        ), // Valid
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(7)), 128, 2), 
            cuda_memory_block_context(null, null, true)
        ), // Larger than necessary
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(8)), 64, 3), 
            cuda_memory_block_context(null, null, true)
        ), // Non valid queue
    };

    const auto ite = find_suitable_block(pool, 64, 2);
    REQUIRE( reinterpret_cast<std::uintptr_t>(ite->first.get_data()) == 6 );
}

TEST_CASE( "find_suitable_block should return end when no suitable block is found", "[cuda_memory_block_pool]" )
{
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool pool = {
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(1)), 32, 1), 
            cuda_memory_block_context(null, null, true)
        ), // Too small, not matching queue
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(2)), 64, 1), 
            cuda_memory_block_context(null, null, true)
        ), // Not matching queue
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(3)), 64, 1), 
            cuda_memory_block_context(null, null, false)
        ), // Not matching queue, occupied
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(4)), 32, 2), 
            cuda_memory_block_context(null, null, true)
        ), // Too small
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(5)), 64, 2), 
            cuda_memory_block_context(null, null, false)
        ), // Occupied
        std::make_pair(
            cuda_memory_block(reinterpret_cast<void*>(std::uintptr_t(8)), 64, 3), 
            cuda_memory_block_context(null, null, true)
        ), // Non valid queue
    };

    REQUIRE( find_suitable_block(pool, 64, 2) == pool.end() );
    REQUIRE( find_suitable_block(pool, 32, 4) == pool.end() );
}

TEST_CASE( "consider partitioning block should not partition with reminder is smaller than threshold", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, true)
    );

    consider_partitioning_block(pool, ite, 768, 512);

    REQUIRE( pool.size() == 1 );
}

TEST_CASE( "consider partitioning block should partition with reminder is greater or equal than threshold", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(nullptr, 1024, 0),
        std::forward_as_tuple(null, null, true)
    );

    SECTION("equal")
    {
        consider_partitioning_block(pool, ite, 768, 256);
        REQUIRE( pool.size() == 2 );
    }
    SECTION("greater")
    {
        consider_partitioning_block(pool, ite, 768, 64);
        REQUIRE( pool.size() == 2 );
    }
}

TEST_CASE( "partition_block should output two valid partitions", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(4096)), 1024, 8),
        std::forward_as_tuple(null, null, true)
    );

    ite = partition_block(pool, ite, 768, 256);
    REQUIRE( pool.size() == 2 );

    REQUIRE( reinterpret_cast<std::uintptr_t>(ite->first.get_data()) == 4096 );
    REQUIRE( ite->first.get_size() == 768 );
    REQUIRE( ite->first.get_queue_id() == 8 );
    REQUIRE( ite->second.get_previous_block() == null );
    REQUIRE( ite->second.is_free() == true );
    REQUIRE( check_forward_link(ite) == true );

    ite = ite->second.get_next_block();
    REQUIRE( reinterpret_cast<std::uintptr_t>(ite->first.get_data()) == 4864 );
    REQUIRE( ite->first.get_size() == 256 );
    REQUIRE( ite->first.get_queue_id() == 8 );
    REQUIRE( ite->second.get_next_block() == null );
    REQUIRE( ite->second.is_free() == true );
    REQUIRE( check_backward_link(ite) == true );
}

TEST_CASE( "consider_merging_block should not merge when is not partition", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(0xDEADBEEF)), 1024, 8),
        std::forward_as_tuple(null, null, true)
    );

    consider_merging_block(pool, ite);
    REQUIRE( pool.size() == 1 );
}

TEST_CASE( "consider_merging_block should not merge when prev is occupied", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(0xDEADBEEF)), 1024, 8),
        std::forward_as_tuple(null, null, true)
    );
    ite = partition_block(pool, ite, 512, 512);
    ite->second.set_free(false);
    ite = ite->second.get_next_block();

    const auto old_ite = ite;
    consider_merging_block(pool, ite);
    REQUIRE( ite == old_ite );
    REQUIRE( pool.size() == 2 );
}

TEST_CASE( "consider_merging_block should not merge when next is occupied", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(0xDEADBEEF)), 1024, 8),
        std::forward_as_tuple(null, null, true)
    );
    ite = partition_block(pool, ite, 512, 512);
    ite = ite->second.get_next_block();
    ite->second.set_free(false);
    ite = ite->second.get_previous_block();
    
    const auto old_ite = ite;
    ite = consider_merging_block(pool, ite);
    REQUIRE( ite == old_ite );
    REQUIRE( pool.size() == 2 );
}

TEST_CASE( "consider_merging_block should merge when prev is free", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(0xDEADBEEF)), 1024, 8),
        std::forward_as_tuple(null, null, true)
    );
    ite = partition_block(pool, ite, 512, 512);
    ite = ite->second.get_next_block();

    consider_merging_block(pool, ite);
    REQUIRE( pool.size() == 1 );
}

TEST_CASE( "consider_merging_block should merge when next is free", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(0xDEADBEEF)), 1024, 8),
        std::forward_as_tuple(null, null, true)
    );
    ite = partition_block(pool, ite, 512, 512);
    
    ite = consider_merging_block(pool, ite);
    REQUIRE( pool.size() == 1 );
}

TEST_CASE( "consider_merging_block should merge when prev and next is free", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(0xDEADBEEF)), 1024, 8),
        std::forward_as_tuple(null, null, true)
    );
    ite = partition_block(pool, ite, 512, 512);
    ite = partition_block(pool, ite, 256, 256);
    ite = ite->second.get_next_block();

    ite = consider_merging_block(pool, ite);
    REQUIRE( pool.size() == 1 );
}

TEST_CASE( "merge_blocks (2) produce valid blocks", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(0xDEADBEEF)), 1024, 8),
        std::forward_as_tuple(null, null, true)
    );
    ite = partition_block(pool, ite, 768, 256);
    ite = partition_block(pool, ite, 512, 256);
    ite = partition_block(pool, ite, 256, 256);

    const auto keep_left = ite;
    const auto merge1 = keep_left->second.get_next_block();
    const auto merge2 = merge1->second.get_next_block();
    const auto keep_right = merge2->second.get_next_block();

    const auto merged = merge_blocks(pool, merge1, merge2);
    REQUIRE( reinterpret_cast<std::uintptr_t>(merged->first.get_data()) == (0xDEADBEEF + 256));
    REQUIRE( merged->first.get_size() == 512 );
    REQUIRE( merged->first.get_queue_id() == 8 );
    REQUIRE( merged->second.get_previous_block() == keep_left );
    REQUIRE( merged->second.get_next_block() == keep_right );
    REQUIRE( merged->second.is_free() == true );
    REQUIRE( check_links(merged) == true );
    REQUIRE( pool.size() == 3 );
}

TEST_CASE( "merge_blocks (3) produce valid blocks", "[cuda_memory_block_pool]" )
{
    cuda_memory_block_pool pool;
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(reinterpret_cast<void*>(std::uintptr_t(0xDEADBEEF)), 1024, 8),
        std::forward_as_tuple(null, null, true)
    );
    ite = partition_block(pool, ite, 768, 256);
    ite = partition_block(pool, ite, 512, 256);
    ite = partition_block(pool, ite, 256, 256);
    ite = partition_block(pool, ite, 128, 128);

    const auto keep_left = ite;
    const auto merge1 = keep_left->second.get_next_block();
    const auto merge2 = merge1->second.get_next_block();
    const auto merge3 = merge2->second.get_next_block();
    const auto keep_right = merge3->second.get_next_block();

    const auto merged = merge_blocks(pool, merge1, merge2, merge3);
    REQUIRE( reinterpret_cast<std::uintptr_t>(merged->first.get_data()) == (0xDEADBEEF + 128));
    REQUIRE( merged->first.get_size() == 640 );
    REQUIRE( merged->first.get_queue_id() == 8 );
    REQUIRE( merged->second.get_previous_block() == keep_left );
    REQUIRE( merged->second.get_next_block() == keep_right );
    REQUIRE( merged->second.is_free() == true );
    REQUIRE( check_links(merged) == true );
    REQUIRE( pool.size() == 3 );
}

TEST_CASE( "create_block should call the allocator", "[cuda_memory_block_pool]" )
{
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool pool;
    mock_allocator allocator;
    auto* address = reinterpret_cast<void*>(std::uintptr_t(0xDEADBEEF));
    const auto size = 768;
    REQUIRE_CALL(allocator, allocate(size))
        .RETURN(address)
        .TIMES(1);

    const auto ite = create_block(pool, allocator, size, 8);
    REQUIRE( ite->first.get_data() == address );
    REQUIRE( ite->first.get_size() == size);
    REQUIRE( ite->first.get_queue_id() == 8 );
    REQUIRE( ite->second.get_previous_block() == null );
    REQUIRE( ite->second.get_next_block() == null );
    REQUIRE( ite->second.is_free() == true );
    REQUIRE( pool.size() == 1 );
}

TEST_CASE( "deallocate_block should call the deallocator with free blocks", "[cuda_memory_block_pool]" )
{
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool pool;
    mock_allocator allocator;
    auto* address = reinterpret_cast<void*>(std::uintptr_t(0xDEADBEEF));
    const auto size = 768;
    REQUIRE_CALL(allocator, deallocate(address, size))
        .TIMES(1);
    
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(address, size, 8),
        std::forward_as_tuple(null, null, true)
    );

    release_blocks(pool, allocator);
    REQUIRE( pool.size() == 0 );
}   

TEST_CASE( "deallocate_block should not call the deallocator with free blocks", "[cuda_memory_block_pool]" )
{
    const cuda_memory_block_pool::iterator null;
    cuda_memory_block_pool pool;
    mock_allocator allocator;
    auto* address = reinterpret_cast<void*>(std::uintptr_t(0xDEADBEEF));
    const auto size = 768;
    
    cuda_memory_block_pool::iterator ite;
    std::tie(ite, std::ignore) = pool.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(address, size, 8),
        std::forward_as_tuple(null, null, false)
    );

    release_blocks(pool, allocator);
    REQUIRE( pool.size() == 1 );
}