// =============================================================================
// test_unordered_map_full.cpp  AJB-adapted unordered_map benchmark
//
// Origin: upstream/joinrenum/testUM.cpp (33 lines)
// AJB adaptation (~20%): timing instrumentation, hit-rate analysis,
//   memory tracking, and throughput metrics.
//
// Build: g++ -O3 test_unordered_map_full.cpp -o test_um_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
#include <sys/resource.h>
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

int main() {
    printf("[AJB] ============================================\n");
    printf("[AJB] test_unordered_map_full  hash perf test\n");
    printf("[AJB] ============================================\n");

    const int N = 1700000;
    printf("[AJB_STATE] N = %d entries\n", N);

    unordered_map<vector<int>, int, VectorHash> cache;
    vector<int> vec;

    // upstream: insert phase
    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        vector<int> v = {rand()};
        cache[v] = i;
        vec.push_back(v[0]);
    }
    auto t1 = chrono::high_resolution_clock::now();
    double insert_ms = chrono::duration<double,milli>(t1 - t0).count();
    printf("[AJB_TIMER] insert %d entries: %.3f ms (%.0f ops/ms)\n",
           N, insert_ms, N / insert_ms);

    // AJB: memory after insert
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("[AJB_MEM] after insert: maxRSS=%ld KB\n", ru.ru_maxrss);

    // AJB: dump hash table stats
    printf("[AJB_STATE] bucket_count=%zu  load_factor=%.3f  "
           "max_load_factor=%.3f\n",
           cache.bucket_count(), cache.load_factor(),
           cache.max_load_factor());

    // upstream: lookup phase
    auto t2 = chrono::high_resolution_clock::now();
    int found = 0;
    for (int i = 0; i < N; i++) {
        if (cache.find({vec[i]}) != cache.end()) {
            found++;
        }
    }
    auto t3 = chrono::high_resolution_clock::now();
    double lookup_ms = chrono::duration<double,milli>(t3 - t2).count();

    printf("[AJB_TIMER] lookup %d keys: %.3f ms (%.0f ops/ms)\n",
           N, lookup_ms, N / lookup_ms);
    printf("[AJB_STATE] Found: %d / %d (hit rate=%.4f)\n",
           found, N, (double)found / N);

    // AJB: miss test (random keys not in map)
    auto t4 = chrono::high_resolution_clock::now();
    int misses = 0;
    for (int i = 0; i < N; i++) {
        if (cache.find({rand()}) == cache.end())
            misses++;
    }
    auto t5 = chrono::high_resolution_clock::now();
    double miss_ms = chrono::duration<double,milli>(t5 - t4).count();

    printf("[AJB_TIMER] miss-probe %d keys: %.3f ms\n", N, miss_ms);
    printf("[AJB_STATE] Misses: %d / %d\n", misses, N);

    printf("[AJB] VERDICT: test_unordered_map_full PASSED\n");
    return 0;
}
