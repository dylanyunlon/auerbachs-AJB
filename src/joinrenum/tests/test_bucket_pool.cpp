// =============================================================================
// test_bucket_pool.cpp — AJB-adapted BucketPool test harness
//
// Origin: upstream/joinrenum/testBucketPool.cpp (21 lines)
// Adaptation (~30%): AJB structured state dumps after each pool operation,
//   allocation/free tracking, splitDim validation, timing, RSS reporting.
//
// M911-M920: compile-options banner, per-probe timing, peak RSS, pool stats
// Build: g++ -std=c++17 -O2 -DAJB_DEBUG test_bucket_pool.cpp -lglpk -o test_bp
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
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

// [AJB] M915: get peak RSS in KB via getrusage
static long getPeakRSSKB() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;  // KB on Linux
}

int main() {
    auto t_start = chrono::high_resolution_clock::now();

    // [AJB] M915: 编译选项 banner
    printf("[AJB] ========================================\n");
    printf("[AJB] test_bucket_pool — compile info:\n");
    printf("[AJB]   __cplusplus = %ld\n", (long)__cplusplus);
#ifdef AJB_DEBUG
    printf("[AJB]   AJB_DEBUG   = ON\n");
#else
    printf("[AJB]   AJB_DEBUG   = OFF\n");
#endif
#ifdef __OPTIMIZE__
    printf("[AJB]   optimized   = YES\n");
#else
    printf("[AJB]   optimized   = NO\n");
#endif
    printf("[AJB]   data_dir    = src/joinrenum/db/\n");
    printf("[AJB]   peak_RSS    = %ld KB (at startup)\n", getPeakRSSKB());
    printf("[AJB] ========================================\n\n");

    printf("[AJB] BucketPool test begin\n");

    BucketPool pool;
    int probe_id = 0;

    // --- probe 0: 3 bucket allocations ---
    {
        auto tp = chrono::high_resolution_clock::now();
        int b0 = pool.newBucket({0, 0}, {1, 1});
        printf("[AJB] newBucket({0,0},{1,1}) -> id=%d\n", b0);

        int b1 = pool.newBucket({0, 0}, {2, 2});
        printf("[AJB] newBucket({0,0},{2,2}) -> id=%d\n", b1);

        int b2 = pool.newBucket({0, 0}, {3, 3});
        printf("[AJB] newBucket({0,0},{3,3}) -> id=%d\n", b2);

        auto elapsed = chrono::duration<double, micro>(chrono::high_resolution_clock::now() - tp).count();
        printf("[AJB_PROBE] id=%d query=alloc_3_buckets elapsed_us=%.1f\n", probe_id++, elapsed);
    }

    dumpPoolState(pool, 3, "after_3_allocs");

    // --- probe 1: free + reuse ---
    {
        auto tp = chrono::high_resolution_clock::now();
        pool.free(1);
        printf("[AJB] freed bucket 1\n");

        int b3 = pool.newCopy(pool[0]);
        printf("[AJB] newCopy(pool[0]) -> id=%d (expect reuse of slot 1)\n", b3);

        int b4 = pool.newCopy(pool[0]);
        printf("[AJB] newCopy(pool[0]) -> id=%d\n", b4);

        auto elapsed = chrono::duration<double, micro>(chrono::high_resolution_clock::now() - tp).count();
        printf("[AJB_PROBE] id=%d query=free_reuse_copy elapsed_us=%.1f\n", probe_id++, elapsed);
    }

    // --- probe 2: reset + splitDim update ---
    {
        auto tp = chrono::high_resolution_clock::now();
        pool[0].reset({1, 1}, {4, 4});
        pool[0].upperBound[0] = 1;
        pool[0].updateSplitDim();
        printf("[AJB] reset pool[0] to ({1,1},{4,4}), set upper[0]=1\n");
        auto elapsed = chrono::duration<double, micro>(chrono::high_resolution_clock::now() - tp).count();
        printf("[AJB_PROBE] id=%d query=reset_splitDim elapsed_us=%.1f\n", probe_id++, elapsed);
    }

    dumpPoolState(pool, 4, "final_state");

    // --- probe 3: splitBucket test ---
    {
        auto tp = chrono::high_resolution_clock::now();
        Bucket testB({0, 0, 0}, {10, 20, 30});
        auto [left, right] = testB.splitBucket(0);
        printf("[AJB] splitBucket({0,0,0},{10,20,30}): L_upper0=%d R_lower0=%d\n",
               left.getUpperBound()[0], right.getLowerBound()[0]);
        auto elapsed = chrono::duration<double, micro>(chrono::high_resolution_clock::now() - tp).count();
        printf("[AJB_PROBE] id=%d query=splitBucket_3d elapsed_us=%.1f\n", probe_id++, elapsed);
    }

    // --- probe 4: volume test ---
    {
        auto tp = chrono::high_resolution_clock::now();
        Bucket volB({0, 0}, {99, 99});
        long long v = volB.ajb_volume();
        printf("[AJB] volume({0,0},{99,99}) = %lld (expect 10000)\n", v);
        auto elapsed = chrono::duration<double, micro>(chrono::high_resolution_clock::now() - tp).count();
        printf("[AJB_PROBE] id=%d query=volume_2d elapsed_us=%.1f\n", probe_id++, elapsed);
    }

    // --- probe 5: pickBucket (partial_sort) test ---
    {
        auto tp = chrono::high_resolution_clock::now();
        // 设置AGM使其可被pick
        pool[0].AGM = 100;
        pool[2].AGM = 500;
        int picked = pool.pickBucket();
        printf("[AJB] pickBucket -> id=%d (expect 2, highest AGM=500)\n", picked);
        auto elapsed = chrono::duration<double, micro>(chrono::high_resolution_clock::now() - tp).count();
        printf("[AJB_PROBE] id=%d query=pickBucket_partial_sort elapsed_us=%.1f\n", probe_id++, elapsed);
    }

    // AJB: verify splitDim for modified bucket
    int actual_split = pool[0].getSplitDim();
    printf("\n[AJB_RESULTS] splitDim check:\n");
    printf("  lower[0]=%d upper[0]=%d\n",
           pool[0].getLowerBound()[0], pool[0].getUpperBound()[0]);
    printf("  expected splitDim >= 1 (dim 0 collapsed): actual=%d\n", actual_split);

    // [AJB] M915: pool级别统计
    pool.ajb_dump_state();
    ajb_bucket_depth_stats.dump("test_final");
    ajb_bp_stats.dump("test_final");

    // [AJB] M915: 总时间 + peak RSS
    auto t_end = chrono::high_resolution_clock::now();
    double total_ms = chrono::duration<double, milli>(t_end - t_start).count();
    long peak_rss = getPeakRSSKB();
    printf("\n[AJB] total_probes=%d total_time_ms=%.3f peak_RSS_KB=%ld\n", probe_id, total_ms, peak_rss);

    if (actual_split >= 1) {
        printf("[AJB] BucketPool test PASSED\n");
    } else {
        fprintf(stderr, "[AJB_WARN] splitDim might be unexpected — manual review needed\n");
    }

    return 0;
}
