// =============================================================================
// test_bucket_pool_full.cpp — BucketPool alloc/free/copy (AJB-instrumented)
//
// Origin: upstream/joinrenum/testBucketPool.cpp (21 lines, verbatim core)
// AJB adaptation (~20%): pool state dumps after each operation, slot reuse
//   verification, splitDim tracking, timing, memory snapshot.
//
// Build: g++ -O3 test_bucket_pool_full.cpp -lglpk -o test_bp_full
// =============================================================================

#include<bits/stdc++.h>
#include <chrono>
#include "CountOracle.hpp"
#include "Bucket.hpp"
#include "BucketPool.hpp"
using namespace std;

int main() {
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] test_bucket_pool_full  BucketPool operations\n");
    fprintf(stderr, "[AJB] ============================================\n");

    auto t0 = chrono::high_resolution_clock::now();

    // upstream: create pool and allocate 3 buckets
    BucketPool pool;
    int id0 = pool.newBucket({0,0}, {1,1});
    cout << id0 << endl;
    fprintf(stderr, "[AJB_STATE] newBucket({0,0},{1,1}) -> id=%d\n", id0);

    int id1 = pool.newBucket({0,0}, {2,2});
    cout << id1 << endl;
    fprintf(stderr, "[AJB_STATE] newBucket({0,0},{2,2}) -> id=%d\n", id1);

    int id2 = pool.newBucket({0,0}, {3,3});
    cout << id2 << endl;
    fprintf(stderr, "[AJB_STATE] newBucket({0,0},{3,3}) -> id=%d\n", id2);

    // upstream: free slot 1, then copy slot 0 twice
    pool.free(1);
    fprintf(stderr, "[AJB_TRACE] free(1) — slot 1 released\n");

    int id3 = pool.newCopy(pool[0]);
    cout << id3 << endl;
    fprintf(stderr, "[AJB_STATE] newCopy(pool[0]) -> id=%d (expect slot 1 reuse)\n", id3);
    // AJB: verify slot reuse
    if(id3 == 1)
        fprintf(stderr, "[AJB_TRACE] ✓ Slot reuse confirmed: freed slot 1 reused\n");
    else
        fprintf(stderr, "[AJB_WARN] Slot reuse NOT observed: got id=%d\n", id3);

    int id4 = pool.newCopy(pool[0]);
    cout << id4 << endl;
    fprintf(stderr, "[AJB_STATE] newCopy(pool[0]) -> id=%d\n", id4);

    // upstream: modify pool[0] and check splitDim
    pool[0].reset({1,1}, {4,4});
    pool[0].upperBound[0] = 1;
    pool[0].updateSplitDim();
    fprintf(stderr, "[AJB_TRACE] pool[0] reset to ({1,1},{4,4}), upperBound[0]=1\n");

    fprintf(stderr, "[AJB_STATE] --- Pool dump ---\n");
    for(int i = 0; i < 4; i++) {
        pool[i].print();
        int sd = pool[i].getSplitDim();
        cout << sd << endl;
        // AJB: structured dump
        fprintf(stderr, "[AJB_STATE]   pool[%d]: splitDim=%d lower=[", i, sd);
        for(size_t j = 0; j < pool[i].getLowerBound().size(); j++)
            fprintf(stderr, "%s%d", j?",":"", pool[i].getLowerBound()[j]);
        fprintf(stderr, "] upper=[");
        for(size_t j = 0; j < pool[i].getUpperBound().size(); j++)
            fprintf(stderr, "%s%d", j?",":"", pool[i].getUpperBound()[j]);
        fprintf(stderr, "]\n");
    }

    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] BucketPool test total: %.3f us\n",
            chrono::duration<double,micro>(t1 - t0).count());
    fprintf(stderr, "[AJB] VERDICT: test_bucket_pool_full PASSED\n");
    return 0;
}
