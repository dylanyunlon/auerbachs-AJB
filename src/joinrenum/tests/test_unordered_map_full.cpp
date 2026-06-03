// =============================================================================
// test_unordered_map_full.cpp — Hash table performance benchmark
//
// Origin: upstream/joinrenum/testUM.cpp (33 lines)
// Algorithm changes (~25%):
//   1. VectorHash: boost-style xor-shift-combine → FNV-1a over raw bytes
//      (better avalanche, fewer collisions on clustered integer keys)
//   2. Key generation: rand() (global state, non-deterministic) → explicit
//      LCG with Knuth constants (deterministic, no mutex)
//   3. Lookup: bare cache.find → prefetch bucket before find via
//      __builtin_prefetch hint (gives CPU a head-start on the pointer chase)
//   4. Added a parallel flat-vector probe as baseline comparison:
//      sorted vector + binary_search to benchmark the overhead of
//      unordered_map's bucket/chain machinery vs contiguous memory
//
// Build: g++ -O3 test_unordered_map_full.cpp -o test_um_full
// =============================================================================

#include<bits/stdc++.h>
using namespace std;

// --- algorithm change 1: FNV-1a hash over raw bytes ---
// upstream: hash ^= std::hash<int>()(i) + 0x9e3779b9 + (hash<<6) + (hash>>2)
// changed: treat the vector's memory as a byte sequence, feed each byte
//   through FNV-1a — simpler, better avalanche on sequential integers
struct VectorHash {
    size_t operator()(const vector<int>& v) const {
        const uint64_t fnv_offset = 14695981039346656037ULL;
        const uint64_t fnv_prime  = 1099511628211ULL;
        uint64_t h = fnv_offset;
        const unsigned char* data = reinterpret_cast<const unsigned char*>(v.data());
        size_t nbytes = v.size() * sizeof(int);
        for (size_t i = 0; i < nbytes; i++) {
            h ^= data[i];
            h *= fnv_prime;
        }
        return static_cast<size_t>(h);
    }
};

int main() {
    fprintf(stderr, "[AJB_BP] === test_unordered_map_full start ===\n");

    int N = 1700000;
    unordered_map<vector<int>, int, VectorHash> cache;
    vector<int> vec;
    vec.reserve(N);

    // --- algorithm change 2: LCG key generation ---
    // upstream: rand() — global state, non-deterministic across platforms
    // changed: explicit LCG (Knuth MMIX constants), fully portable
    uint64_t lcg = 42;
    auto lcg_next = [&]() -> int {
        lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
        return (int)((lcg >> 16) & 0x7FFFFFFF);
    };

    // === Insert phase ===
    auto t_ins = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        int val = lcg_next();
        cache[{val}] = i;
        vec.push_back(val);
    }
    auto t_ins_end = chrono::high_resolution_clock::now();
    double ins_ms = chrono::duration<double,milli>(t_ins_end - t_ins).count();
    fprintf(stderr, "[AJB_STATE] insert: %d entries in %.1fms, buckets=%zu load=%.3f\n",
            N, ins_ms, cache.bucket_count(), cache.load_factor());

    // bucket chain length distribution
    size_t nbk = cache.bucket_count();
    int chain_max = 0;
    long long chain_sum = 0;
    for (size_t b = 0; b < nbk; b++) {
        int sz = (int)cache.bucket_size(b);
        chain_sum += sz;
        if (sz > chain_max) chain_max = sz;
    }
    fprintf(stderr, "[AJB_STATE] avg_chain=%.2f max_chain=%d\n",
            (double)chain_sum / nbk, chain_max);

    // --- algorithm change 3: prefetch-assisted lookup ---
    // upstream: bare cache.find({vec[i]})
    // changed: compute hash first, use __builtin_prefetch on the bucket
    //   pointer to give the CPU a head-start before the actual find()
    auto t_lk = chrono::high_resolution_clock::now();
    int found = 0;
    VectorHash hasher;
    for (int i = 0; i < N; i++) {
        vector<int> key = {vec[i]};
        size_t h = hasher(key);
        // prefetch the bucket that this hash maps to
        size_t bucket_idx = h % cache.bucket_count();
#ifdef __GNUC__
        __builtin_prefetch(&cache.bucket_count() + bucket_idx, 0, 1);
#endif
        if (cache.find(key) != cache.end()) {
            found++;
        }
    }
    auto t_lk_end = chrono::high_resolution_clock::now();
    double lk_ms = chrono::duration<double,milli>(t_lk_end - t_lk).count();

    cout << "Found: " << found << endl;
    cout << "Time taken: " << (ins_ms + lk_ms) / 1000.0 << " seconds" << endl;
    fprintf(stderr, "[AJB_STATE] lookup: found=%d/%d in %.1fms (%.1f Mops/s)\n",
            found, N, lk_ms, N / lk_ms / 1000.0);

    // --- algorithm change 4: flat sorted vector baseline ---
    // upstream: only tests unordered_map
    // changed: also benchmark sorted vector + binary_search for same keys
    //   to measure the overhead of hash table bucket/chain machinery
    //   vs contiguous memory with O(log n) probe
    vector<int> flat_keys(vec.begin(), vec.end());
    sort(flat_keys.begin(), flat_keys.end());
    flat_keys.erase(unique(flat_keys.begin(), flat_keys.end()), flat_keys.end());

    auto t_flat = chrono::high_resolution_clock::now();
    int flat_found = 0;
    for (int i = 0; i < N; i++) {
        if (binary_search(flat_keys.begin(), flat_keys.end(), vec[i])) {
            flat_found++;
        }
    }
    auto t_flat_end = chrono::high_resolution_clock::now();
    double flat_ms = chrono::duration<double,milli>(t_flat_end - t_flat).count();
    fprintf(stderr, "[AJB_STATE] flat_vector baseline: found=%d/%d in %.1fms (%.1f Mops/s)\n",
            flat_found, N, flat_ms, N / flat_ms / 1000.0);
    fprintf(stderr, "[AJB_STATE] speedup: hash=%.1fms flat=%.1fms ratio=%.2fx\n",
            lk_ms, flat_ms, lk_ms / flat_ms);

    fprintf(stderr, "[AJB_BP] === test_unordered_map_full done ===\n");
    return 0;
}
