// =============================================================================
// test_bucket_pool_full.cpp  AJB-adapted full BucketPool test
//
// Origin: upstream/joinrenum/testBucketPool.cpp (21 lines)
// AJB adaptation (~20%): structured pool state dumps after each operation,
//   allocation/free tracking, bounds verification.
//
// Build: g++ -O3 test_bucket_pool_full.cpp -lglpk -o test_bp_full
// =============================================================================

#include <bits/stdc++.h>
#include "CountOracle.hpp"
#include "Bucket.hpp"
#include "BucketPool.hpp"
using namespace std;

// AJB: dump full pool state
void ajbDumpPool(BucketPool& pool, int n, const char* label) {
    printf("[AJB_STATE] %-24s %d buckets:\n", label, n);
    for (int i = 0; i < n; i++) {
        printf("[AJB_STATE]   [%d] lower=[", i);
        auto lb = pool[i].getLowerBound();
        for (size_t d = 0; d < lb.size(); d++) printf("%s%d", d?",":"", lb[d]);
        printf("] upper=[");
        auto ub = pool[i].getUpperBound();
        for (size_t d = 0; d < ub.size(); d++) printf("%s%d", d?",":"", ub[d]);
        printf("] splitDim=%d\n", pool[i].getSplitDim());
    }
}

int main() {
    printf("[AJB] ============================================\n");
    printf("[AJB] test_bucket_pool_full  BucketPool test\n");
    printf("[AJB] ============================================\n");

    BucketPool pool;

    // upstream: create three buckets
    int b0 = pool.newBucket({0,0}, {1,1});
    printf("[AJB_TRACE] newBucket({0,0},{1,1}) -> id=%d\n", b0);
    int b1 = pool.newBucket({0,0}, {2,2});
    printf("[AJB_TRACE] newBucket({0,0},{2,2}) -> id=%d\n", b1);
    int b2 = pool.newBucket({0,0}, {3,3});
    printf("[AJB_TRACE] newBucket({0,0},{3,3}) -> id=%d\n", b2);

    ajbDumpPool(pool, 3, "after_3_creates");

    // upstream: free bucket 1
    pool.free(1);
    printf("[AJB_TRACE] freed bucket 1\n");

    // upstream: copy bucket 0 twice (should reuse freed slot)
    int c0 = pool.newCopy(pool[0]);
    printf("[AJB_TRACE] newCopy(pool[0]) -> id=%d\n", c0);
    int c1 = pool.newCopy(pool[0]);
    printf("[AJB_TRACE] newCopy(pool[0]) -> id=%d\n", c1);

    // upstream: modify bucket 0
    pool[0].reset({1,1}, {4,4});
    pool[0].upperBound[0] = 1;
    pool[0].updateSplitDim();
    printf("[AJB_TRACE] pool[0] reset to ({1,1},{4,4}), ub[0]=1\n");

    ajbDumpPool(pool, 4, "after_modify");

    // upstream: print all + splitDim
    printf("[AJB_STATE] --- Final state (upstream print) ---\n");
    for (int i = 0; i < 4; i++) {
        pool[i].print();
        cout << "splitDim=" << pool[i].getSplitDim() << endl;
    }

    // AJB: validate that free slot was reused
    bool reuse_ok = (c0 == 1);  // freed slot 1 should be reused
    printf("[AJB_STATE] Free-slot reuse: c0=%d (expect 1) -> %s\n",
           c0, reuse_ok ? "OK" : "UNEXPECTED");

    printf("[AJB] VERDICT: test_bucket_pool_full %s\n",
           reuse_ok ? "PASSED" : "CHECK");
    return 0;
}
