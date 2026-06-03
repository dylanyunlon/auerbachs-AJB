// =============================================================================
// test_unordered_map_full.cpp — Hash table perf (AJB-instrumented)
//
// Origin: upstream/joinrenum/testUM.cpp (33 lines, verbatim core)
// AJB adaptation (~20%): bucket count + load factor tracking at
//   5 growth points during insertion, per-bucket occupancy histogram
//   (0/1/2/3+), lookup hit-rate analysis, insert vs lookup phase split
//   with separate timing, max_load_factor / rehash diagnostics.
//
// Build: g++ -O3 test_unordered_map_full.cpp -o test_um_full
// =============================================================================

#include<bits/stdc++.h>
using namespace std;

// upstream: VectorHash
struct VectorHash {
    size_t operator()(const vector<int>& v) const {
        size_t hash = 0;
        for (int i : v) {
            hash ^= std::hash<int>()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

int main(int argc, char* argv[]) {
    int N = 1700000;
    if (argc >= 2) N = atoi(argv[1]);
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] test_unordered_map_full  hash perf (N=%d)\n", N);
    fprintf(stderr, "[AJB] ============================================\n");

    unordered_map<vector<int>, int, VectorHash> cache;
    vector<int> vec;

    // === Insert phase with growth tracking ===
    auto t_ins_start = chrono::high_resolution_clock::now();
    int report_interval = N / 5;
    for(int i = 0; i < N; i++) {
        vector<int> v = {rand()};
        cache[v] = i;
        vec.push_back(v[0]);

        // AJB: dump growth stats at 20%/40%/60%/80%/100%
        if (report_interval > 0 && (i+1) % report_interval == 0) {
            fprintf(stderr, "[AJB_STATE] insert %d/%d: buckets=%zu load=%.3f max_load=%.3f\n",
                    i+1, N, cache.bucket_count(), cache.load_factor(),
                    cache.max_load_factor());
        }
    }
    auto t_ins_end = chrono::high_resolution_clock::now();
    double ins_ms = chrono::duration<double,milli>(t_ins_end - t_ins_start).count();
    fprintf(stderr, "[AJB_TIMER] insert phase: %.3f ms (%d entries)\n", ins_ms, N);

    // AJB_STATE: bucket occupancy histogram
    size_t nbuckets = cache.bucket_count();
    int occ_0 = 0, occ_1 = 0, occ_2 = 0, occ_3plus = 0;
    size_t max_chain = 0;
    for (size_t b = 0; b < nbuckets; b++) {
        size_t sz = cache.bucket_size(b);
        if (sz == 0) occ_0++;
        else if (sz == 1) occ_1++;
        else if (sz == 2) occ_2++;
        else occ_3plus++;
        max_chain = max(max_chain, sz);
    }
    fprintf(stderr, "[AJB_STATE] bucket histogram: empty=%d single=%d double=%d chain3+=%d max_chain=%zu\n",
            occ_0, occ_1, occ_2, occ_3plus, max_chain);
    fprintf(stderr, "[AJB_STATE] utilization=%.1f%% (non-empty/total)\n",
            100.0 * (nbuckets - occ_0) / nbuckets);

    // === Lookup phase ===
    auto t_lk_start = chrono::high_resolution_clock::now();
    int found = 0;
    for(int i = 0; i < N; i++) {
        int a = rand();
        if(cache.find({vec[i]}) != cache.end()) {
            found++;
        }
    }
    auto t_lk_end = chrono::high_resolution_clock::now();
    double lk_ms = chrono::duration<double,milli>(t_lk_end - t_lk_start).count();

    cout << "Found: " << found << endl;
    fprintf(stderr, "[AJB_TIMER] lookup phase: %.3f ms (%d queries)\n", lk_ms, N);
    fprintf(stderr, "[AJB_STATE] hit_rate=%.2f%% (found=%d / %d)\n",
            100.0 * found / N, found, N);
    fprintf(stderr, "[AJB_STATE] lookup throughput: %.1f Mops/s\n",
            N / lk_ms / 1000.0);

    clock_t end = clock();
    double elapsed = ins_ms + lk_ms;
    cout << "Time taken: " << elapsed / 1000.0 << " seconds" << endl;

    fprintf(stderr, "[AJB] VERDICT: test_unordered_map_full PASSED\n");
    return 0;
}
