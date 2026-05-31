// =============================================================================
// test_join_baseline.cpp  AJB-adapted triangle join baseline benchmark
//
// Origin: upstream/joinrenum/testjoin.cpp (90 lines)
// AJB adaptation (~20%): structured timing for each phase (load, index,
//   join, filter), memory snapshots, progress counter for large joins,
//   and throughput metrics.
//
// Build: g++ -O3 test_join_baseline.cpp -o test_join_bl
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
using namespace std;

struct PairHash {
    size_t operator()(const pair<int,int>& p) const noexcept {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

// upstream: flush CPU cache
void flush_cache() {
    const size_t size = 100 * 1024 * 1024;
    vector<char> buffer(size);
    for (size_t i = 0; i < size; i++) buffer[i] = i % 256;
    volatile char sink = 0;
    for (size_t i = 0; i < size; i++) sink ^= buffer[i];
}

// AJB: memory helper
void ajbMem(const char* label) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("[AJB_MEM] %-24s maxRSS=%ld MB\n", label, ru.ru_maxrss/1024);
}

int main() {
    printf("[AJB] ============================================\n");
    printf("[AJB] test_join_baseline  triangle join benchmark\n");
    printf("[AJB] ============================================\n");

    flush_cache();
    printf("[AJB_TRACE] CPU cache flushed\n");

    // --- Phase 1: Load data ---
    auto t_load0 = chrono::high_resolution_clock::now();

    string filename = "db/Ra.tbl";
    ifstream infile(filename);
    if (!infile.is_open()) {
        fprintf(stderr, "[AJB_FAIL] Cannot open %s\n", filename.c_str());
        return 1;
    }

    vector<pair<int,int>> data;
    string line;
    int parse_errors = 0;

    while (getline(infile, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string x_str, y_str;
        if (getline(ss, x_str, '|') && getline(ss, y_str)) {
            try {
                data.emplace_back(stoi(x_str), stoi(y_str));
            } catch (const exception& e) {
                parse_errors++;
            }
        }
    }
    infile.close();

    auto t_load1 = chrono::high_resolution_clock::now();
    double load_ms = chrono::duration<double,milli>(t_load1 - t_load0).count();

    printf("[AJB_TIMER] data load: %.3f ms\n", load_ms);
    printf("[AJB_STATE] Loaded %zu edges from %s (parse_errors=%d)\n",
           data.size(), filename.c_str(), parse_errors);
    ajbMem("after_load");

    // --- Phase 2: Build index ---
    auto t_idx0 = chrono::high_resolution_clock::now();

    set<pair<int,int>> R(data.begin(), data.end());
    map<int, vector<int>> index;
    for (auto& [y, z] : data) {
        index[y].push_back(z);
    }

    auto t_idx1 = chrono::high_resolution_clock::now();
    double idx_ms = chrono::duration<double,milli>(t_idx1 - t_idx0).count();

    printf("[AJB_TIMER] index build: %.3f ms\n", idx_ms);
    printf("[AJB_STATE] R set: %zu unique edges, index: %zu keys\n",
           R.size(), index.size());

    // AJB: index distribution stats
    size_t max_fan = 0, total_fan = 0;
    for (auto& [k, v] : index) {
        max_fan = max(max_fan, v.size());
        total_fan += v.size();
    }
    printf("[AJB_STATE] Index fanout: avg=%.1f max=%zu\n",
           index.empty() ? 0.0 : (double)total_fan / index.size(), max_fan);

    ajbMem("after_index");

    // --- Phase 3: Triangle join ---
    auto t_join0 = chrono::high_resolution_clock::now();

    set<vector<int>> res;
    long long count = 0, total = 0;
    long long progress_interval = max((long long)data.size() / 10, 1LL);

    for (size_t ei = 0; ei < data.size(); ei++) {
        auto& [x, y] = data[ei];
        auto it = index.find(y);
        if (it != index.end()) {
            for (int z : it->second) {
                total++;
                if (R.find({x, z}) != R.end()) {
                    res.insert({x, y, z});
                }
            }
        }
        // AJB: progress reporting
        if ((long long)(ei + 1) % progress_interval == 0) {
            printf("[AJB_TRACE] progress: %zu/%zu edges (%.0f%%)"
                   " results_so_far=%zu\n",
                   ei + 1, data.size(),
                   100.0 * (ei + 1) / data.size(), res.size());
        }
    }

    auto t_join1 = chrono::high_resolution_clock::now();
    double join_ms = chrono::duration<double,milli>(t_join1 - t_join0).count();

    printf("[AJB_TIMER] triangle join: %.3f ms\n", join_ms);
    printf("[AJB_STATE] Total probes: %lld, Triangles found: %zu\n",
           total, res.size());
    printf("[AJB_STATE] Selectivity: %.6f\n",
           total > 0 ? (double)res.size() / total : 0.0);

    ajbMem("after_join");

    // --- Summary ---
    double total_ms = load_ms + idx_ms + join_ms;
    printf("[AJB_STATE] --- Timing breakdown ---\n");
    printf("[AJB_STATE]   Load:  %7.3f ms (%5.1f%%)\n",
           load_ms, 100*load_ms/total_ms);
    printf("[AJB_STATE]   Index: %7.3f ms (%5.1f%%)\n",
           idx_ms, 100*idx_ms/total_ms);
    printf("[AJB_STATE]   Join:  %7.3f ms (%5.1f%%)\n",
           join_ms, 100*join_ms/total_ms);
    printf("[AJB_STATE]   Total: %7.3f ms\n", total_ms);
    printf("[AJB_STATE] Throughput: %.0f edges/sec\n",
           data.size() * 1000.0 / total_ms);

    printf("[AJB] VERDICT: test_join_baseline PASSED\n");
    return 0;
}
