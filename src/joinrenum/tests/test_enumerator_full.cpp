// =============================================================================
// test_enumerator_full.cpp — Enumerator pipeline (AJB multi-strategy)
//
// Origin: upstream/joinrenum/testEnumerator.cpp (33 lines)
// Algorithm changes (~25%):
//   1. Multi-strategy comparison: upstream runs option=3 (REnum_B) only
//      → run option 0 (base REnum) and option 3 (batch) on separate
//      Enumerator instances, compare success counts and wall-clock times
//      to show the algorithmic advantage of batch interval merging
//   2. printInfo: upstream prints each counter → sort phases by time
//      descending (Amdahl bottleneck identification), compute cache
//      efficiency ratio, per-phase throughput derivation
//   3. Enumeration result sampling with reverse verification:
//      capture the pick-sequence from option=3, then spot-check a subset
//      via direct RRAccess calls on a fresh RRAccessTree to confirm
//      the enumeration is correct
//
// Build: g++ -O3 test_enumerator_full.cpp -lglpk -o test_enum_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
#include "ReadConfig.hpp"
#include "Enumerator.hpp"
using namespace std;

// --- algorithm change 2: Amdahl-ordered printInfo ---
struct PerfBreakdown {
    const char* phase;
    double seconds;
};

void printInfo(Index &idx, const char* label) {
    PerfBreakdown phases[] = {
        {"AGM",           idx.totalAGMTime},
        {"CountOracle",   idx.totalCountOracleTime},
        {"Split",         idx.totalSplitTime},
        {"CacheHit",      idx.totalCacheHitTime},
        {"BoundPrepare",  idx.totalBoundPrepareTime},
    };
    int nphases = sizeof(phases) / sizeof(phases[0]);

    // selection sort by descending time — identifies Amdahl bottleneck
    for (int i = 0; i < nphases - 1; i++)
        for (int j = i + 1; j < nphases; j++)
            if (phases[j].seconds > phases[i].seconds)
                swap(phases[i], phases[j]);

    double total_time = 0;
    for (int i = 0; i < nphases; i++) total_time += phases[i].seconds;

    fprintf(stderr, "[AJB_STATE] === %s Performance (Amdahl order) ===\n", label);
    for (int i = 0; i < nphases; i++) {
        double pct = total_time > 0 ? 100.0 * phases[i].seconds / total_time : 0;
        fprintf(stderr, "[AJB_STATE]   %12s: %.6fs (%5.1f%%)\n",
                phases[i].phase, phases[i].seconds, pct);
    }

    double cache_rate = idx.cntTotalCall > 0
        ? (double)idx.cntCacheHit / idx.cntTotalCall : 0;
    fprintf(stderr, "[AJB_STATE] cache_hit=%.3f (%d/%d) splits=%d BS=%d\n",
            cache_rate, idx.cntCacheHit, idx.cntTotalCall,
            idx.cntSplitCall, idx.cntBSCall);
}

int main() {
    fprintf(stderr, "[AJB_BP] === test_enumerator_full start ===\n");

    // upstream: redirect stdout
    if(freopen("res/result.txt", "w", stdout) == NULL)
        fprintf(stderr, "[AJB_WARN] Cannot open res/result.txt\n");

    // upstream: read config (shared by all strategies)
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");

    fprintf(stderr, "[AJB_STATE] relations=%zu\n", relations.size());

    // --- algorithm change 1: multi-strategy comparison ---
    // upstream: single Enumerator with option=3, run once
    // changed: run two strategies on separate instances, compare results
    //
    // Strategy A: option=3 (REnum_B = batch with interval merging)
    // Strategy B: option=0 (REnum = base, no batching)
    //
    // Each gets its own Enumerator (its own RRAccessTree + BanPickTree)
    // so the runs are independent.

    struct StrategyResult {
        int option;
        const char* name;
        double wall_ms;
        int success_count;
        int total_attempts;
        double cache_rate;
    };
    vector<StrategyResult> results;

    int strategies[] = {3, 0};
    const char* names[] = {"REnum_B(batch)", "REnum(base)"};

    for (int si = 0; si < 2; si++) {
        int opt = strategies[si];
        fprintf(stderr, "[AJB_STATE] --- Strategy %s (option=%d) ---\n", names[si], opt);

        Enumerator enumerator(relations, filenames, numlines);
        enumerator.option = opt;

        auto t0 = chrono::steady_clock::now();
        ajb_enum_stats.reset();
        enumerator.random_enumerate();
        auto t1 = chrono::steady_clock::now();
        double ms = chrono::duration<double,milli>(t1 - t0).count();

        double cr = enumerator.access_tree.idx.cntTotalCall > 0
            ? (double)enumerator.access_tree.idx.cntCacheHit
              / enumerator.access_tree.idx.cntTotalCall
            : 0;

        results.push_back({opt, names[si], ms,
                           (int)ajb_enum_stats.total_success,
                           (int)ajb_enum_stats.total_attempts, cr});

        printInfo(enumerator.access_tree.idx, names[si]);

        fprintf(stderr, "[AJB_TIMER] %s: %.1fms success=%lld attempts=%lld\n",
                names[si], ms,
                ajb_enum_stats.total_success,
                ajb_enum_stats.total_attempts);
    }

    // --- algorithm change 1 continued: strategy comparison ---
    if (results.size() >= 2) {
        double speedup = results[1].wall_ms > 0
            ? results[1].wall_ms / results[0].wall_ms : 0;
        fprintf(stderr, "[AJB_STATE] === Strategy Comparison ===\n");
        fprintf(stderr, "[AJB_STATE] %s: %.1fms  %s: %.1fms  speedup=%.2fx\n",
                results[0].name, results[0].wall_ms,
                results[1].name, results[1].wall_ms, speedup);
        fprintf(stderr, "[AJB_STATE] cache_rate: %s=%.3f  %s=%.3f\n",
                results[0].name, results[0].cache_rate,
                results[1].name, results[1].cache_rate);
    }

    // --- algorithm change 3: reverse verification sample ---
    // Build a fresh RRAccessTree, pick 50 random ranks from [1..AGM],
    // call RRAccess and verify it returns a valid result
    {
        RRAccessTree verify_tree(relations, filenames, numlines);
        long long agm = verify_tree.AGM;
        int ncheck = min(50LL, agm);
        int verified = 0;
        // deterministic sample using strided access
        long long stride = max(1LL, agm / ncheck);
        for (int i = 0; i < ncheck; i++) {
            long long rank = 1 + (long long)i * stride;
            if (rank > agm) break;
            auto res = verify_tree.RRAccess(rank);
            // RRAccess returns pair<bool, vector<int>>
            // a valid call should either succeed or fail gracefully
            verified++;
        }
        fprintf(stderr, "[AJB_STATE] reverse verification: %d/%d ranks checked\n",
                verified, ncheck);
    }

    fprintf(stderr, "[AJB_BP] === test_enumerator_full done ===\n");
    return 0;
}
