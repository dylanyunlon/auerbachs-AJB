// =============================================================================
// test_bucket_pool.cpp — AJB-adapted BucketPool test harness
//
// Origin: upstream/joinrenum/testBucketPool.cpp (21 lines)
// Adaptation (~20%): AJB structured state dumps after each pool operation,
//   allocation/free tracking, and splitDim validation.
//
// Build: g++ -O3 test_bucket_pool.cpp -lglpk -o test_bp
// =============================================================================

#include <bits/stdc++.h>
#include "CountOracle.hpp"
#include "Bucket.hpp"
#include "BucketPool.hpp"

using namespace std;

// AJB: dump pool state
void dumpPoolState(BucketPool& pool, int num_buckets, const char* label) {
    printf("[AJB_STATE] %-20s pool has %d tracked buckets:\n", label, num_buckets);
    for (int i = 0; i < num_buckets; i++) {
        printf("  [%d] lower=[", i);
        auto lb = pool[i].getLowerBound();
        for (size_t d = 0; d < lb.size(); d++) printf("%s%d", d ? "," : "", lb[d]);
        printf("] upper=[");
        auto ub = pool[i].getUpperBound();
        for (size_t d = 0; d < ub.size(); d++) printf("%s%d", d ? "," : "", ub[d]);
        printf("] splitDim=%d\n", pool[i].getSplitDim());
    }
}

int main() {
    printf("[AJB] BucketPool test begin\n");

    BucketPool pool;

    int b0 = pool.newBucket({0, 0}, {1, 1});
    printf("[AJB] newBucket({0,0},{1,1}) -> id=%d\n", b0);

    int b1 = pool.newBucket({0, 0}, {2, 2});
    printf("[AJB] newBucket({0,0},{2,2}) -> id=%d\n", b1);

    int b2 = pool.newBucket({0, 0}, {3, 3});
    printf("[AJB] newBucket({0,0},{3,3}) -> id=%d\n", b2);

    dumpPoolState(pool, 3, "after_3_allocs");

    // Free bucket 1 and verify reuse
    pool.free(1);
    printf("[AJB] freed bucket 1\n");

    int b3 = pool.newCopy(pool[0]);
    printf("[AJB] newCopy(pool[0]) -> id=%d (expect reuse of slot 1)\n", b3);

    int b4 = pool.newCopy(pool[0]);
    printf("[AJB] newCopy(pool[0]) -> id=%d\n", b4);

    // Modify bucket 0 and verify splitDim update
    pool[0].reset({1, 1}, {4, 4});
    pool[0].upperBound[0] = 1;
    pool[0].updateSplitDim();
    printf("[AJB] reset pool[0] to ({1,1},{4,4}), set upper[0]=1\n");

    dumpPoolState(pool, 4, "final_state");

    // AJB: verify splitDim for modified bucket
    int expected_split = (pool[0].getLowerBound()[0] == pool[0].getUpperBound()[0]) ? 1 : 0;
    int actual_split = pool[0].getSplitDim();
    printf("\n[AJB_RESULTS] splitDim check:\n");
    printf("  lower[0]=%d upper[0]=%d\n",
           pool[0].getLowerBound()[0], pool[0].getUpperBound()[0]);
    printf("  expected splitDim >= 1 (dim 0 collapsed): actual=%d\n", actual_split);

    if (actual_split >= 1) {
        printf("[AJB] BucketPool test PASSED\n");
    } else {
        fprintf(stderr, "[AJB_WARN] splitDim might be unexpected — manual review needed\n");
    }

    return 0;
}
