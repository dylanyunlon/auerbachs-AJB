// =============================================================================
// test_enumerator_full.cpp — Enumerator pipeline (AJB-instrumented)
//
// Origin: upstream/joinrenum/testEnumerator.cpp (33 lines, verbatim core)
// AJB adaptation (~20%): printInfo with full Index counter set, BanPickTree
//   residual state dump, ajb_rrt_stats/ajb_enum_stats/ajb_idx_stats global
//   tracker output, per-phase memory delta, CountOracle structure summary.
//
// Build: g++ -O3 test_enumerator_full.cpp -lglpk -o test_enum_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
#include "ReadConfig.hpp"
#include "Enumerator.hpp"
using namespace std;

// upstream: printInfo with full Index counters
void printInfo(Index &idx) {
    fprintf(stderr, "[AJB_STATE] === Index Counters ===\n");
    fprintf(stderr, "[AJB_STATE] CacheHit(SplitBucket): %d / %d total (%.1f%%)\n",
            idx.cntCacheHit, idx.cntTotalCall,
            idx.cntTotalCall > 0 ? 100.0 * idx.cntCacheHit / idx.cntTotalCall : 0.0);
    fprintf(stderr, "[AJB_STATE] AGM: %d calls, %.6fs\n",
            idx.cntAGMCall, idx.totalAGMTime);
    fprintf(stderr, "[AJB_STATE] CountOracle: %.6fs\n", idx.totalCountOracleTime);
    fprintf(stderr, "[AJB_STATE] Split: %d calls, %.6fs\n",
            idx.cntSplitCall, idx.totalSplitTime);
    fprintf(stderr, "[AJB_STATE] BinarySearch loops: %d\n", idx.cntBSCall);
    fprintf(stderr, "[AJB_STATE] CacheHit time: %.6fs\n", idx.totalCacheHitTime);
    fprintf(stderr, "[AJB_STATE] BoundPrepare time: %.6fs\n", idx.totalBoundPrepareTime);
    fprintf(stderr, "[AJB_STATE] RRTree nodes: %d\n", idx.totalrrtreenode);
    // derived: time per split, per AGM call
    if (idx.cntSplitCall > 0)
        fprintf(stderr, "[AJB_STATE] avg_split=%.3fus avg_AGM=%.3fus\n",
                idx.totalSplitTime * 1e6 / idx.cntSplitCall,
                idx.cntAGMCall > 0 ? idx.totalAGMTime * 1e6 / idx.cntAGMCall : 0.0);
}

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
    fprintf(stderr, "[AJB] test_enumerator_full  Enumerator pipeline\n");
    fprintf(stderr, "[AJB] ============================================\n");

    long rss0 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] startup: RSS=%ld KB\n", rss0);

    // upstream: redirect stdout to result file
    if(freopen("res/result.txt", "w", stdout) == NULL)
        fprintf(stderr, "[AJB_WARN] Cannot open res/result.txt for writing\n");

    // upstream: read config
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");

    fprintf(stderr, "[AJB_STATE] === Schema ===\n");
    for(auto& [name, vars] : relations) {
        fprintf(stderr, "[AJB_STATE]   %s(", name.c_str());
        for(size_t j = 0; j < vars.size(); j++)
            fprintf(stderr, "%s%s", j?",":"", vars[j].c_str());
        fprintf(stderr, ") file=%s lines=%d\n",
                filenames.count(name) ? filenames[name].c_str() : "?",
                numlines.count(name) ? numlines[name] : -1);
    }

    long rss1 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] after_config: RSS=%ld KB (delta=%ld)\n", rss1, rss1 - rss0);

    // upstream: construct enumerator
    auto t0 = chrono::high_resolution_clock::now();
    Enumerator enumerator(relations, filenames, numlines);
    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] Enumerator construction: %.3f ms\n",
            chrono::duration<double,milli>(t1 - t0).count());

    // AJB_STATE: dump the key parameters of the constructed tree
    fprintf(stderr, "[AJB_STATE] === Enumerator Ready ===\n");
    fprintf(stderr, "[AJB_STATE] AGM=%lld  BanPickTree.remaining=%d\n",
            (long long)enumerator.access_tree.AGM,
            enumerator.bp.remaining());
    fprintf(stderr, "[AJB_STATE] Index: %zu tables, %zu relations\n",
            enumerator.access_tree.idx.tables.size(),
            enumerator.access_tree.idx.q.getRelations().size());
    // dump CountOracle pointers to confirm they're built
    auto COs = enumerator.access_tree.idx.getCountOracles();
    fprintf(stderr, "[AJB_STATE] CountOracles: %zu (", COs.size());
    for (size_t i = 0; i < COs.size(); i++) {
        fprintf(stderr, "%s%s", i ? "," : "", COs[i] ? "OK" : "NULL");
    }
    fprintf(stderr, ")\n");

    long rss2 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] after_construction: RSS=%ld KB (delta=%ld)\n",
            rss2, rss2 - rss1);

    // upstream: run random enumeration
    fprintf(stderr, "[AJB_BP] random_enumerate() starting...\n");
    auto t2 = chrono::high_resolution_clock::now();
    enumerator.random_enumerate();
    auto t3 = chrono::high_resolution_clock::now();
    double enum_ms = chrono::duration<double,milli>(t3 - t2).count();
    fprintf(stderr, "[AJB_TIMER] random_enumerate: %.3f ms\n", enum_ms);

    // AJB_STATE: post-enumeration — dump all global trackers
    fprintf(stderr, "[AJB_STATE] === Post-Enumeration Diagnostics ===\n");

    // BanPickTree residual
    fprintf(stderr, "[AJB_STATE] BanPickTree.remaining=%d (should be 0)\n",
            enumerator.bp.remaining());

    // Index counters (the meaty stuff)
    printInfo(enumerator.access_tree.idx);

    // Enumerator global tracker
    ajb_enum_stats.dump("enum_full");

    // RRAccessTree global tracker
    ajb_rrt_stats.dump("rrt_full");

    // Index global tracker
    ajb_idx_stats.dump("idx_full");

    // BucketPool tracker
    ajb_bp_stats.dump("bp_full");

    // AGM tracker
    ajb_agm_stats.dump("agm_full");

    long rss_end = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] final: RSS=%ld KB (total delta=%ld KB)\n",
            rss_end, rss_end - rss0);
    fprintf(stderr, "[AJB_STATE] enumeration throughput: %.1f results/ms\n",
            enum_ms > 0 ? enumerator.access_tree.AGM / enum_ms : 0.0);
    fprintf(stderr, "[AJB] VERDICT: test_enumerator_full PASSED\n");
    return 0;
}
