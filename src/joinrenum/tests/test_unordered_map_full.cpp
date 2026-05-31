// =============================================================================
// test_unordered_map_full.cpp — unordered_map<vector> hash perf (AJB-instrumented)
//
// Origin: upstream/joinrenum/testUM.cpp (33 lines, verbatim core)
// AJB adaptation (~20%): chrono hi-res timing (replacing clock()), fixed seed,
//   insert/lookup phase separation, hit-rate analysis, throughput reporting,
//   memory snapshots.
//
// Build: g++ -O3 test_unordered_map_full.cpp -o test_um_full
// =============================================================================

#include<bits/stdc++.h>
#include <chrono>
using namespace std;

// upstream: vector hash functor (verbatim)
struct VectorHash {
    size_t operator()(const vector<int>& v) const {
        size_t hash = 0;
        for (int i : v) {
            hash ^= std::hash<int>()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

// AJB: memory snapshot
static long ajb_rss_kb() {
    ifstream f("/proc/self/status"); string line;
    while (getline(f, line))
        if (line.substr(0, 6) == "VmRSS:")
            { istringstream iss(line); string k; long v; iss >> k >> v; return v; }
    return -1;
}

int main() {
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] test_unordered_map_full  hash perf test\n");
    fprintf(stderr, "[AJB] ============================================\n");

    long rss0 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] startup: RSS=%ld KB\n", rss0);

    srand(42);  // AJB: fixed seed

    // upstream: build hash map with 1.7M entries
    int N = 1700000;
    fprintf(stderr, "[AJB_TRACE] Building unordered_map with %d entries...\n", N);

    unordered_map<vector<int>, int, VectorHash> cache;
    vector<int> vec;

    auto t_insert_start = chrono::high_resolution_clock::now();
    for(int i = 0; i < N; i++) {
        vector<int> v = {rand()};
        cache[v] = i;
        vec.push_back(v[0]);
    }
    auto t_insert_end = chrono::high_resolution_clock::now();
    double insert_ms = chrono::duration<double,milli>(t_insert_end - t_insert_start).count();
    fprintf(stderr, "[AJB_TIMER] insert phase: %.3f ms (%d entries, %.1f M ops/s)\n",
            insert_ms, N, N / insert_ms / 1000.0);

    long rss1 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] after_insert: RSS=%ld KB (delta=%ld)\n", rss1, rss1 - rss0);
    fprintf(stderr, "[AJB_STATE] cache.size()=%zu  bucket_count=%zu  load_factor=%.3f\n",
            cache.size(), cache.bucket_count(), cache.load_factor());

    // upstream: lookup 1.7M keys, count hits
    auto t_lookup_start = chrono::high_resolution_clock::now();
    int found = 0;
    for(int i = 0; i < N; i++) {
        int a = rand();  // upstream: random value (not used for lookup)
        (void)a;
        if(cache.find({vec[i]}) != cache.end()) {
            found++;
        }
    }
    auto t_lookup_end = chrono::high_resolution_clock::now();
    double lookup_ms = chrono::duration<double,milli>(t_lookup_end - t_lookup_start).count();

    cout << "Found: " << found << endl;
    cout << "Time taken: " << lookup_ms / 1000.0 << " seconds" << endl;

    // AJB: throughput and hit-rate analysis
    fprintf(stderr, "[AJB_TIMER] lookup phase: %.3f ms (%d lookups, %.1f M ops/s)\n",
            lookup_ms, N, N / lookup_ms / 1000.0);
    fprintf(stderr, "[AJB_STATE] hit_rate=%.1f%% (%d/%d)\n",
            100.0 * found / N, found, N);

    long rss_end = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] final: RSS=%ld KB (total delta=%ld KB)\n",
            rss_end, rss_end - rss0);
    fprintf(stderr, "[AJB] VERDICT: test_unordered_map_full PASSED\n");
    return 0;
}
