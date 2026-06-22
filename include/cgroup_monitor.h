#pragma once
#ifndef LSM_VEC_CGROUP_MONITOR_H_
#define LSM_VEC_CGROUP_MONITOR_H_

// CgroupMemoryMonitor — lightweight cgroup v2 memory-pressure watcher.
//
// Background thread samples a few small sysfs files every `poll_ms` and sets a
// one-way "escalate" latch when the container is genuinely thrashing the OS
// page cache under its memory cap. Consumed by AsterVec to escalate the
// RocksGraph store from buffered to Direct I/O during an incremental build.
//
// Trigger (both must hold, sustained for `debounce_s`):
//   1. high-water pre-gate:  memory.current > high_fraction * memory.max
//   2. thrash confirm:       d(workingset_refault_file)/dt > refault_min_rate
// The pre-gate is cheap; the refault rate is the real "evicted pages are being
// faulted back in" signal that distinguishes near-cap-but-fits / compaction
// bursts (no switch) from near-cap-and-thrashing (switch).
//
// Read-only: never writes cgroup control files. Fail-safe: if the cgroup v2
// files can't be located/parsed (e.g. macOS dev, cgroup v1, unlimited cap) the
// monitor disables itself and the engine stays buffered.

#include <atomic>
#include <thread>
#include <string>
#include <functional>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cstdint>

#include "logger.h"

namespace astervec {

class CgroupMemoryMonitor {
public:
    struct Config {
        double   high_fraction   = 0.90;    // pre-gate fraction of memory.max
        double   refault_min_rate = 1000.0; // pages/sec floor for thrash confirm
        int      debounce_s      = 4;       // sustained seconds before latching
        int      poll_ms         = 1000;    // sample interval
        bool     enabled         = true;
    };

    // `inserts_getter` (optional) returns a monotonic total-inserts counter so
    // the monitor can log instantaneous insert throughput alongside memory —
    // this is the before/after-reopen perf time series.
    explicit CgroupMemoryMonitor(Config cfg,
                                 std::function<uint64_t()> inserts_getter = {})
        : cfg_(cfg), inserts_getter_(std::move(inserts_getter)) {}

    ~CgroupMemoryMonitor() { stop(); }

    CgroupMemoryMonitor(const CgroupMemoryMonitor&) = delete;
    CgroupMemoryMonitor& operator=(const CgroupMemoryMonitor&) = delete;

    bool should_escalate() const {
        return escalate_.load(std::memory_order_acquire);
    }
    uint64_t last_current() const {
        return last_current_.load(std::memory_order_relaxed);
    }

    void start() {
        if (!cfg_.enabled) {
            LOG(INFO) << "[adaptive-dio] monitor disabled by config";
            return;
        }
        if (!resolve_paths()) {
            LOG(WARN) << "[adaptive-dio] cgroup v2 memory files not found; "
                         "monitor disabled (staying buffered)";
            return;
        }
        uint64_t mx = read_u64(max_path_);
        if (mx == 0 || mx == kUnlimited) {
            LOG(INFO) << "[adaptive-dio] memory.max unlimited/unknown; monitor disabled";
            return;
        }
        mem_max_ = mx;
        running_.store(true, std::memory_order_release);
        thread_ = std::thread([this] { run(); });
        LOG(WARN) << "[adaptive-dio] monitor started: max=" << (mem_max_ >> 20)
                  << "MiB high=" << static_cast<uint64_t>(cfg_.high_fraction * 100)
                  << "% refault_min=" << static_cast<uint64_t>(cfg_.refault_min_rate)
                  << "/s debounce=" << cfg_.debounce_s << "s poll=" << cfg_.poll_ms << "ms";
    }

    void stop() {
        if (running_.exchange(false)) {
            if (thread_.joinable()) thread_.join();
        }
    }

private:
    static constexpr uint64_t kUnlimited = ~0ULL;

    static bool file_exists(const std::string& p) {
        std::ifstream f(p);
        return f.good();
    }

    static uint64_t read_u64(const std::string& path) {
        if (path.empty()) return 0;
        std::ifstream f(path);
        if (!f.good()) return 0;
        std::string tok;
        f >> tok;
        if (tok == "max") return kUnlimited;
        if (tok.empty()) return 0;
        try { return std::stoull(tok); } catch (...) { return 0; }
    }

    // Parse "key value" lines of memory.stat; 0 if the key is absent.
    static uint64_t read_stat_field(const std::string& path, const char* key) {
        if (path.empty()) return 0;
        std::ifstream f(path);
        if (!f.good()) return 0;
        std::string k;
        uint64_t v = 0;
        while (f >> k >> v) {
            if (k == key) return v;
        }
        return 0;
    }

    // file refaults: prefer the v2-split field, fall back to the combined one.
    uint64_t read_refault() const {
        uint64_t v = read_stat_field(stat_path_, "workingset_refault_file");
        if (v == 0) v = read_stat_field(stat_path_, "workingset_refault");
        return v;
    }

    // Resolve the cgroup v2 path. Docker (cgroup namespace) → /sys/fs/cgroup;
    // systemd-run on host → /sys/fs/cgroup/<unit-path> from /proc/self/cgroup.
    bool resolve_paths() {
        const std::string mount = "/sys/fs/cgroup";
        std::string suffix;
        std::ifstream pc("/proc/self/cgroup");
        if (pc.good()) {
            std::string line;
            while (std::getline(pc, line)) {
                // v2 unified line: "0::<path>"
                if (line.rfind("0::", 0) == 0) {
                    suffix = line.substr(3);
                    break;
                }
            }
        }
        auto try_dir = [&](const std::string& dir) -> bool {
            std::string cur = dir + "/memory.current";
            std::string mx  = dir + "/memory.max";
            if (file_exists(cur) && file_exists(mx)) {
                current_path_ = cur;
                max_path_     = mx;
                stat_path_    = dir + "/memory.stat";
                return true;
            }
            return false;
        };
        if (!suffix.empty() && suffix != "/") {
            if (try_dir(mount + suffix)) return true;
        }
        // namespace case (suffix "/" or empty), or fallback.
        return try_dir(mount);
    }

    void sleep_poll() {
        int remaining = cfg_.poll_ms;
        while (remaining > 0 && running_.load(std::memory_order_acquire)) {
            int chunk = std::min(remaining, 100);
            std::this_thread::sleep_for(std::chrono::milliseconds(chunk));
            remaining -= chunk;
        }
    }

    void run() {
        using clock = std::chrono::steady_clock;
        uint64_t prev_refault = read_refault();
        uint64_t prev_inserts = inserts_getter_ ? inserts_getter_() : 0;
        auto prev_t = clock::now();
        double sustained_s = 0.0;

        while (running_.load(std::memory_order_acquire)) {
            sleep_poll();
            if (!running_.load(std::memory_order_acquire)) break;

            auto now = clock::now();
            double dt = std::chrono::duration<double>(now - prev_t).count();
            prev_t = now;
            if (dt <= 0) dt = 1e-6;

            uint64_t cur = read_u64(current_path_);
            last_current_.store(cur, std::memory_order_relaxed);

            uint64_t refault = read_refault();
            double refault_rate = static_cast<double>(refault - prev_refault) / dt;
            prev_refault = refault;

            uint64_t inserts = inserts_getter_ ? inserts_getter_() : 0;
            double insert_rate = static_cast<double>(inserts - prev_inserts) / dt;
            prev_inserts = inserts;

            double frac = mem_max_ ? static_cast<double>(cur) / static_cast<double>(mem_max_) : 0.0;
            bool already = escalate_.load(std::memory_order_acquire);

            LOG(INFO) << "[adaptive-dio] mem.current=" << (cur >> 20) << "MiB ("
                      << static_cast<int>(frac * 100) << "% of max) refault="
                      << static_cast<uint64_t>(refault_rate < 0 ? 0 : refault_rate)
                      << "/s inserts=" << static_cast<uint64_t>(insert_rate < 0 ? 0 : insert_rate)
                      << "/s" << (already ? " [direct]" : " [buffered]");

            if (already) continue;

            bool pregate = frac >= cfg_.high_fraction;
            bool thrash  = refault_rate >= cfg_.refault_min_rate;
            if (pregate && thrash) {
                sustained_s += dt;
                if (sustained_s >= static_cast<double>(cfg_.debounce_s)) {
                    escalate_.store(true, std::memory_order_release);
                    LOG(WARN) << "[adaptive-dio] sustained pressure (" << static_cast<int>(frac * 100)
                              << "% of max, refault=" << static_cast<uint64_t>(refault_rate)
                              << "/s for " << static_cast<uint64_t>(sustained_s)
                              << "s) -> escalate latch set";
                }
            } else {
                sustained_s = 0.0;
            }
        }
    }

    Config cfg_;
    std::function<uint64_t()> inserts_getter_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> escalate_{false};
    std::atomic<uint64_t> last_current_{0};
    std::string current_path_, max_path_, stat_path_;
    uint64_t mem_max_ = 0;
};

} // namespace astervec

#endif // LSM_VEC_CGROUP_MONITOR_H_
