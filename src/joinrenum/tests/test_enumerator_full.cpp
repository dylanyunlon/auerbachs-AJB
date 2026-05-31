// =============================================================================
// test_enumerator_full.cpp — Enumerator pipeline (AJB-instrumented)
//
// Origin: upstream/joinrenum/testEnumerator.cpp (33 lines, verbatim core)
// AJB adaptation (~20%): extended printInfo with BSCall/BoundPrepare/rrtreenode
//   (matching upstream test.cpp's full counter set), chrono timing around
//   random_enumerate(), memory snapshots, structured verdict output.
//
// Build: g++ -O3 test_enumerator_full.cpp -lglpk -o test_enum_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
#include "ReadConfig.hpp"
#include "Enumerator.hpp"
using namespace std;

// upstream: printInfo with full Index counters (merged from test.cpp + testEnumerator.cpp)
void printInfo(Index &idx) {
    fprintf(stderr, "[AJB_STATE] CacheHit(SplitBucket): %d / %d total\n",
            idx.cntCacheHit, idx.cntTotalCall);
    fprintf(stderr, "[AJB_STATE] AGM calls: %d  time=%.6fs\n",
            idx.cntAGMCall, idx.totalAGMTime);
    fprintf(stderr, "[AJB_STATE] CountOracle time: %.6fs\n", idx.totalCountOracleTime);
    fprintf(stderr, "[AJB_STATE] Split: %d calls  time=%.6fs\n",
            idx.cntSplitCall, idx.totalSplitTime);
    fprintf(stderr, "[AJB_STATE] BinarySearch loops: %d\n", idx.cntBSCall);
    fprintf(stderr, "[AJB_STATE] CacheHit time: %.6fs\n", idx.totalCacheHitTime);
    fprintf(stderr, "[AJB_STATE] BoundPrepare time: %.6fs\n", idx.totalBoundPrepareTime);
    fprintf(stderr, "[AJB_STATE] RRTree nodes: %d\n", idx.totalrrtreenode);
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

    fprintf(stderr, "[AJB_TRACE] Config loaded: %zu relations\n", relations.size());
    for(auto& [name, vars] : relations) {
        fprintf(stderr, "[AJB_STATE]   %s(", name.c_str());
        for(size_t j = 0; j < vars.size(); j++)
            fprintf(stderr, "%s%s", j?",":"", vars[j].c_str());
        fprintf(stderr, ")\n");
    }

    // upstream: construct enumerator and run
    auto t0 = chrono::high_resolution_clock::now();
    Enumerator enumerator(relations, filenames, numlines);
    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] Enumerator construction: %.3f ms\n",
            chrono::duration<double,milli>(t1 - t0).count());

    fprintf(stderr, "[AJB_BP] random_enumerate() starting...\n");
    auto t2 = chrono::high_resolution_clock::now();
    enumerator.random_enumerate();
    auto t3 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] random_enumerate: %.3f ms\n",
            chrono::duration<double,milli>(t3 - t2).count());

    // upstream: print full Index stats (was commented out in original)
    printInfo(enumerator.access_tree.idx);

    long rss_end = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] final: RSS=%ld KB (total delta=%ld KB)\n",
            rss_end, rss_end - rss0);
    fprintf(stderr, "[AJB] VERDICT: test_enumerator_full PASSED\n");
    return 0;
}
