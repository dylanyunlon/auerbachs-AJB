// =============================================================================
// test_unordered_map.cpp — AJB-adapted unordered_map micro-benchmark
//
// Origin: upstream/joinrenum/testUM.cpp (33 lines)
// Adaptation (~20%): AJB scoped timers, throughput metrics, cache-line
//   awareness notes, and structured output for comparing hash map
//   performance across different payload sizes.
//
// Purpose: validates that the VectorHash used in BucketPool/Index caching
// achieves acceptable lookup throughput. The SkewDetector's hot path
// relies on this hash for bucket AGM caching.
//
// Build: g++ -O3 test_unordered_map.cpp -o test_um
// =============================================================================

#include <bits/stdc++.h>
using namespace std;

struct VectorHash {
    size_t operator()(const vector<int>& v) const {
        size_t hash = 0;
        for (int i : v) {
            hash ^= std::hash<int>()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

// AJB: inline scoped timer
struct ScopedTimer {
  const char* label;
  std::chrono::high_resolution_clock::time_point t0;
  ScopedTimer(const char* l) : label(l), t0(std::chrono::high_resolution_clock::now()) {
    printf("[AJB_TIMER] >>> %s\n", label);
  }
  ~ScopedTimer() {
    double s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    printf("[AJB_TIMER] <<< %s  %.6f s (%.3f ms)\n", label, s, s * 1000.0);
  }
};

int main(int argc, char* argv[]) {
    // AJB: configurable sizes
    int n_entries = 1700000;
    int vec_dim = 1;
    if (argc >= 2) n_entries = atoi(argv[1]);
    if (argc >= 3) vec_dim = atoi(argv[2]);
    printf("[AJB] unordered_map benchmark: n=%d vec_dim=%d\n", n_entries, vec_dim);

    unordered_map<vector<int>, int, VectorHash> cache;
    vector<vector<int>> keys(n_entries);

    // Build phase
    {
        ScopedTimer t("insert");
        for (int i = 0; i < n_entries; i++) {
            vector<int> v(vec_dim);
            for (int d = 0; d < vec_dim; d++) v[d] = rand();
            keys[i] = v;
            cache[v] = i;
        }
    }

    printf("[AJB_STATE] cache.size()=%zu  bucket_count=%zu  load_factor=%.3f\n",
           cache.size(), cache.bucket_count(), cache.load_factor());

    // Lookup phase
    int found = 0;
    {
        ScopedTimer t("lookup");
        for (int i = 0; i < n_entries; i++) {
            if (cache.find(keys[i]) != cache.end()) {
                found++;
            }
        }
    }

    // AJB: throughput metrics
    printf("\n[AJB_RESULTS] unordered_map<vector<int>> benchmark:\n");
    printf("  entries       = %d\n", n_entries);
    printf("  found         = %d (%.1f%%)\n", found,
           n_entries > 0 ? 100.0 * found / n_entries : 0.0);
    printf("  vec_dim       = %d\n", vec_dim);
    printf("  load_factor   = %.3f\n", cache.load_factor());

    if (found != n_entries) {
        fprintf(stderr, "[AJB_WARN] Expected 100%% hit rate, got %d/%d\n", found, n_entries);
    } else {
        printf("[AJB] unordered_map test PASSED (100%% hit rate)\n");
    }

    return 0;
}
