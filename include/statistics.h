#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ostream>

namespace lsm_vec {

// Thread-safe stats counters.
//
// Counters use std::atomic<size_t>; time accumulators use
// std::atomic<uint64_t> measured in nanoseconds. Nanoseconds-as-uint64
// avoids the C++17 limitation that std::atomic<double>::fetch_add is
// C++20-only, while still supporting 584 years of monotonic
// accumulation per metric.
//
// `enabled_` is set at construction via setEnabledOnConstruction and
// must not be toggled at runtime — the assertion makes the convention
// mechanical.
class HNSWStats {
public:
    using Clock = std::chrono::high_resolution_clock;

    struct TimerToken {
        bool active  = false;     // false => everything is a no-op
        bool stopped = true;      // true => duration already computed
        Clock::time_point start;
        uint64_t duration_ns = 0; // cached duration in nanoseconds
    };

    explicit HNSWStats(bool enabled = false)
        : enabled_(enabled) {}

    // The name forbids runtime mutation; setEnabledOnConstruction is the
    // documented entry point. The assertion catches accidental second
    // calls.
    void setEnabledOnConstruction(bool enabled) {
        assert(!enabled_set_ && "HNSWStats::setEnabledOnConstruction "
                                 "must only be called once, from the "
                                 "owning object's constructor");
        enabled_ = enabled;
        enabled_set_ = true;
    }
    bool enabled() const { return enabled_; }

    void reset() {
        io_time.store(0, std::memory_order_relaxed);
        indexing_time.store(0, std::memory_order_relaxed);
        search_time.store(0, std::memory_order_relaxed);
        insert_count.store(0, std::memory_order_relaxed);
        search_count.store(0, std::memory_order_relaxed);

        io_count.store(0, std::memory_order_relaxed);
        read_io_count.store(0, std::memory_order_relaxed);
        write_node_io_count.store(0, std::memory_order_relaxed);
        add_edge_io_count.store(0, std::memory_order_relaxed);
        delete_edge_io_count.store(0, std::memory_order_relaxed);

        read_io_time.store(0, std::memory_order_relaxed);
        write_node_io_time.store(0, std::memory_order_relaxed);
        add_edge_io_time.store(0, std::memory_order_relaxed);
        delete_edge_io_time.store(0, std::memory_order_relaxed);

        read_vertex_property_count.store(0, std::memory_order_relaxed);
        read_edges_count.store(0, std::memory_order_relaxed);
        read_vertex_property_time.store(0, std::memory_order_relaxed);
        read_edges_time.store(0, std::memory_order_relaxed);

        vec_read_count.store(0, std::memory_order_relaxed);
        vec_read_time.store(0, std::memory_order_relaxed);
        vec_write_count.store(0, std::memory_order_relaxed);
        vec_write_time.store(0, std::memory_order_relaxed);

        page_cache_hits.store(0, std::memory_order_relaxed);
        page_cache_misses.store(0, std::memory_order_relaxed);

        metadata_gets.store(0, std::memory_order_relaxed);
        metadata_cache_hits.store(0, std::memory_order_relaxed);
        filter_evaluations.store(0, std::memory_order_relaxed);
        filter_matches.store(0, std::memory_order_relaxed);
        filter_scanned.store(0, std::memory_order_relaxed);
        filter_cap_hits.store(0, std::memory_order_relaxed);
    }

    // ------------------------------------------------------------
    // Timing API
    // ------------------------------------------------------------

    // Start a timer. If stats are disabled, this will NOT call Clock::now().
    inline TimerToken startTimer() const {
        TimerToken token;
        if (!enabled_) {
            // inactive token; no time accounting will happen
            return token;
        }
        token.active  = true;
        token.stopped = false;
        token.start   = Clock::now();
        return token;
    }

    // Accumulate the timer's elapsed nanoseconds into time_metric_ns.
    // Multiple calls on the same token reuse the cached duration.
    inline void accumulateTime(TimerToken& token,
                               std::atomic<uint64_t>& time_metric_ns) const {
        if (!enabled_ || !token.active) return;

        if (!token.stopped) {
            auto end = Clock::now();
            token.duration_ns =
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end - token.start).count());
            token.stopped  = true;
        }
        time_metric_ns.fetch_add(token.duration_ns,
                                  std::memory_order_relaxed);
    }

    // ------------------------------------------------------------
    // Counting API
    // ------------------------------------------------------------

    inline void addCount(std::size_t count,
                         std::atomic<std::size_t>& counter) const {
        if (!enabled_) return;
        counter.fetch_add(count, std::memory_order_relaxed);
    }

    // ------------------------------------------------------------
    // Public metrics. Time fields are nanoseconds (uint64). Counters
    // are size_t. All atomic with relaxed memory order.
    // ------------------------------------------------------------

    // High-level times
    std::atomic<uint64_t> io_time{0};
    std::atomic<uint64_t> indexing_time{0};
    std::atomic<uint64_t> search_time{0};

    std::atomic<std::size_t> insert_count{0};
    std::atomic<std::size_t> search_count{0};

    // AsterDB I/O counters and times
    std::atomic<std::size_t> io_count{0};
    std::atomic<std::size_t> read_io_count{0};
    std::atomic<std::size_t> write_node_io_count{0};
    std::atomic<std::size_t> add_edge_io_count{0};
    std::atomic<std::size_t> delete_edge_io_count{0};

    std::atomic<uint64_t> read_io_time{0};
    std::atomic<uint64_t> write_node_io_time{0};
    std::atomic<uint64_t> add_edge_io_time{0};
    std::atomic<uint64_t> delete_edge_io_time{0};

    std::atomic<std::size_t> read_vertex_property_count{0};
    std::atomic<std::size_t> read_edges_count{0};
    std::atomic<uint64_t>    read_vertex_property_time{0};
    std::atomic<uint64_t>    read_edges_time{0};

    // Vector I/O
    std::atomic<std::size_t> vec_read_count{0};
    std::atomic<uint64_t>    vec_read_time{0};
    std::atomic<std::size_t> vec_write_count{0};
    std::atomic<uint64_t>    vec_write_time{0};

    // Page-based vector storage cache stats
    std::atomic<std::size_t> page_cache_hits{0};
    std::atomic<std::size_t> page_cache_misses{0};

    // Metadata filtering counters
    std::atomic<std::size_t> metadata_gets{0};
    std::atomic<std::size_t> metadata_cache_hits{0};   // reserved; 0 in v1
    std::atomic<std::size_t> filter_evaluations{0};
    std::atomic<std::size_t> filter_matches{0};
    std::atomic<std::size_t> filter_scanned{0};
    std::atomic<std::size_t> filter_cap_hits{0};

    // ------------------------------------------------------------
    // Print helper. Converts ns → seconds for display.
    // ------------------------------------------------------------

    void print(std::ostream& os) const {
        if (!enabled_) {
            os << "Detailed statistics are disabled." << std::endl;
            return;
        }

        auto seconds = [](const std::atomic<uint64_t>& ns) {
            return static_cast<double>(ns.load(std::memory_order_relaxed))
                   / 1e9;
        };
        auto count = [](const std::atomic<std::size_t>& c) {
            return c.load(std::memory_order_relaxed);
        };

        os << "Indexing Time: " << seconds(indexing_time) << " seconds\n";
        os << "Search Time: "   << seconds(search_time)   << " seconds\n";
        os << "Insert Operations: " << count(insert_count) << "\n";
        os << "Search Operations: " << count(search_count) << "\n";

        os << "-------graph part------\n";
        os << "Total Aster I/O Operations: " << count(io_count) << "\n";
        os << "Total Aster I/O Time: " << seconds(io_time) << " seconds\n";
        os << "Read Operations: " << count(read_io_count)
           << ", Time: " << seconds(read_io_time) << " seconds\n";
        os << "Write Node Operations: " << count(write_node_io_count)
           << ", Time: " << seconds(write_node_io_time) << " seconds\n";
        os << "Add Edge Operations: " << count(add_edge_io_count)
           << ", Time: " << seconds(add_edge_io_time) << " seconds\n";
        os << "Delete Edge Operations: " << count(delete_edge_io_count)
           << ", Time: " << seconds(delete_edge_io_time) << " seconds\n";
        os << "ReadVertexProperty Count: " << count(read_vertex_property_count)
           << ", Time: " << seconds(read_vertex_property_time) << " seconds\n";
        os << "ReadEdges Count: " << count(read_edges_count)
           << ", Time: " << seconds(read_edges_time) << " seconds\n";

        os << "-------vector part------\n";
        os << "Total Vector I/O Time: "
           << (seconds(vec_read_time) + seconds(vec_write_time))
           << " seconds\n";
        os << "Vector Read Operations: " << count(vec_read_count)
           << ", Time: " << seconds(vec_read_time) << " seconds\n";
        os << "Vector Write Operations: " << count(vec_write_count)
           << ", Time: " << seconds(vec_write_time) << " seconds\n";
        std::size_t hits = count(page_cache_hits);
        std::size_t misses = count(page_cache_misses);
        if (hits + misses > 0) {
            std::size_t total = hits + misses;
            double hit_rate = 100.0 * static_cast<double>(hits) /
                              static_cast<double>(total);
            os << "Page Cache Avoided I/O: " << hits << "\n";
            os << "Page Cache Hit Rate: " << hit_rate << "%\n";
        }

        std::size_t mg = count(metadata_gets);
        std::size_t fe = count(filter_evaluations);
        std::size_t fs = count(filter_scanned);
        if (mg + fe + fs > 0) {
            os << "-------metadata filter part------\n";
            os << "Metadata Gets: " << mg << "\n";
            os << "Metadata Cache Hits: " << count(metadata_cache_hits) << "\n";
            os << "Filter Evaluations: " << fe << "\n";
            os << "Filter Matches: " << count(filter_matches) << "\n";
            os << "Filter Scanned: " << fs << "\n";
            os << "Filter Cap Hits: " << count(filter_cap_hits) << "\n";
        }
        os << std::endl;
    }

private:
    bool enabled_ = false;
    bool enabled_set_ = false;
};

} // namespace lsm_vec
