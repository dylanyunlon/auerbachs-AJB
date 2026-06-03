// =============================================================================
// test_bucket_pool_full.cpp — BucketPool alloc/free/copy stress test
//
// Origin: upstream/joinrenum/testBucketPool.cpp (21 lines)
// Algorithm changes (~25%):
//   1. Bucket generation: fixed 3 hardcoded buckets → N random-sized buckets
//      via LCG, exercising the pool allocator under realistic load
//   2. splitDim verification: upstream calls getSplitDim() and prints
//      → we independently compute expected splitDim by scanning all
//      dimensions for max span (upper[d]-lower[d]) and comparing
//   3. Free-list stress: upstream frees slot 1 once → we do alloc/free
//      interleaving in a pattern that forces the free-list to chain,
//      then verify every reused slot via a seen-id set
//   4. Copy-on-write test: after newCopy, mutate original and verify
//      the copy is independent (data isolation check)
//
// Build: g++ -O3 test_bucket_pool_full.cpp -lglpk -o test_bp_full
// =============================================================================

#include<bits/stdc++.h>
#include <chrono>
#include "CountOracle.hpp"
#include "Bucket.hpp"
#include "BucketPool.hpp"
using namespace std;

// --- algorithm change 2: independent splitDim computation ---
// upstream: just calls getSplitDim() and trusts it
// changed: compute from scratch — find dimension with max span
static int compute_expected_split_dim(const vector<int>& lo, const vector<int>& hi) {
    int best_dim = 0;
    int best_span = -1;
    for (size_t d = 0; d < lo.size(); d++) {
        int span = hi[d] - lo[d];
        if (span > best_span) {
            best_span = span;
            best_dim = (int)d;
        }
    }
    return best_dim;
}

int main() {
    fprintf(stderr, "[AJB_BP] === test_bucket_pool_full start ===\n");
    auto t0 = chrono::high_resolution_clock::now();

    BucketPool pool;

    // --- upstream core: 3 fixed allocations ---
    int id0 = pool.newBucket({0,0}, {1,1});
    int id1 = pool.newBucket({0,0}, {2,2});
    int id2 = pool.newBucket({0,0}, {3,3});
    cout << id0 << endl;
    cout << id1 << endl;
    cout << id2 << endl;

    // --- upstream: free(1), two copies of pool[0] ---
    pool.free(1);
    int id3 = pool.newCopy(pool[0]);
    int id4 = pool.newCopy(pool[0]);
    cout << id3 << endl;
    cout << id4 << endl;

    // --- algorithm change 3: free-list stress test ---
    // upstream: single free(1) + observe reuse
    // changed: alloc N more, free every other one, re-alloc and verify
    //   that freed slots are reused (tracked via set of seen IDs)
    const int NSTRESS = 20;
    vector<int> stress_ids;
    stress_ids.reserve(NSTRESS);

    // LCG for deterministic random bounds
    uint32_t lcg = 7919;
    auto lcg_next = [&]() -> int {
        lcg = lcg * 1103515245u + 12345u;
        return (int)((lcg >> 8) % 50);
    };

    for (int i = 0; i < NSTRESS; i++) {
        int lo0 = lcg_next(), lo1 = lcg_next();
        int hi0 = lo0 + 1 + lcg_next(), hi1 = lo1 + 1 + lcg_next();
        int sid = pool.newBucket({lo0, lo1}, {hi0, hi1});
        stress_ids.push_back(sid);
    }
    fprintf(stderr, "[AJB_STATE] allocated %d stress buckets, last_id=%d\n",
            NSTRESS, stress_ids.back());

    // free every other slot
    set<int> freed_ids;
    for (int i = 0; i < NSTRESS; i += 2) {
        pool.free(stress_ids[i]);
        freed_ids.insert(stress_ids[i]);
    }
    fprintf(stderr, "[AJB_STATE] freed %zu slots\n", freed_ids.size());

    // re-allocate and check which IDs come back (free-list reuse)
    int reused = 0;
    for (int i = 0; i < (int)freed_ids.size(); i++) {
        int lo0 = lcg_next(), lo1 = lcg_next();
        int hi0 = lo0 + 1 + lcg_next(), hi1 = lo1 + 1 + lcg_next();
        int rid = pool.newBucket({lo0, lo1}, {hi0, hi1});
        if (freed_ids.count(rid)) reused++;
    }
    fprintf(stderr, "[AJB_STATE] free-list reuse: %d/%zu (%.0f%%)\n",
            reused, freed_ids.size(),
            freed_ids.empty() ? 0.0 : 100.0 * reused / freed_ids.size());

    // --- upstream: modify pool[0] and check splitDim ---
    pool[0].reset({1,1}, {4,4});
    pool[0].upperBound[0] = 1;
    pool[0].updateSplitDim();

    // --- algorithm change 2: verify splitDim for first 4 buckets ---
    int mismatches = 0;
    for (int i = 0; i < 4; i++) {
        pool[i].print();
        int reported = pool[i].getSplitDim();
        cout << reported << endl;

        // independently compute expected splitDim
        auto lo_vec = pool[i].getLowerBound();
        auto hi_vec = pool[i].getUpperBound();
        int expected = compute_expected_split_dim(lo_vec, hi_vec);
        if (reported != expected) {
            fprintf(stderr, "[AJB_WARN] pool[%d] splitDim mismatch: got=%d expected=%d\n",
                    i, reported, expected);
            mismatches++;
        }
    }
    fprintf(stderr, "[AJB_STATE] splitDim verification: %d mismatches in 4 buckets\n",
            mismatches);

    // --- algorithm change 4: copy independence test ---
    // upstream: newCopy then never checks if copies are independent
    // changed: copy a bucket, mutate the original, verify the copy
    //   retains its old values (i.e. deep copy, not shallow alias)
    int src_id = pool.newBucket({10, 20}, {50, 80});
    int cpy_id = pool.newCopy(pool[src_id]);
    // save copy's upper bound before mutation
    vector<int> copy_upper_before = pool[cpy_id].getUpperBound();
    // mutate original
    pool[src_id].reset({0, 0}, {999, 999});
    // check copy is unchanged
    vector<int> copy_upper_after = pool[cpy_id].getUpperBound();
    bool copy_independent = (copy_upper_before == copy_upper_after);
    fprintf(stderr, "[AJB_STATE] copy independence: %s\n",
            copy_independent ? "PASS (deep copy)" : "FAIL (shallow alias!)");

    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] total: %.1fus\n",
            chrono::duration<double,micro>(t1 - t0).count());
    fprintf(stderr, "[AJB_BP] === test_bucket_pool_full done ===\n");
    return 0;
}
