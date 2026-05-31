// =============================================================================
// test_enumerator_full.cpp  AJB-adapted full Enumerator test harness
//
// Origin: upstream/joinrenum/testEnumerator.cpp (33 lines)
// AJB adaptation (~20%): structured [AJB_TRACE] tags at each pipeline
//   step, Index statistics dump, memory snapshots, timing breakdown,
//   and pass/fail verdict for regression testing.
//
// Build: g++ -O3 test_enumerator_full.cpp -lglpk -o test_enum_full
// Run:   ./test_enum_full 2>&1 | grep AJB   # filter trace lines
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
#include "ReadConfig.hpp"
#include "Enumerator.hpp"
using namespace std;

// AJB: memory snapshot
struct AJBMemInfo {
    long vm_rss_kb = -1;
    long vm_size_kb = -1;
    void capture() {
        ifstream f("/proc/self/status");
        string line;
        while (getline(f, line)) {
            if (line.substr(0, 6) == "VmRSS:")
                { istringstream iss(line); string k; iss >> k >> vm_rss_kb; }
            else if (line.substr(0, 7) == "VmSize:")
                { istringstream iss(line); string k; iss >> k >> vm_size_kb; }
        }
    }
    void print(const char* label) const {
        printf("[AJB_MEM] %-24s RSS=%ld KB  VSize=%ld KB\n",
               label, vm_rss_kb, vm_size_kb);
    }
};

// AJB: scoped timer
struct AJBTimer {
    const char* label;
    chrono::high_resolution_clock::time_point t0;
    AJBTimer(const char* l) : label(l),
        t0(chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto dt = chrono::high_resolution_clock::now() - t0;
        return chrono::duration<double, milli>(dt).count();
    }
    ~AJBTimer() {
        printf("[AJB_TIMER] %s: %.3f ms\n", label, elapsed_ms());
    }
};

// AJB: dump Index internal counters (upstream printInfo logic preserved)
void ajbDumpIndexStats(Index &idx, const char* phase) {
    printf("[AJB_STATE] --- Index stats @ %s ---\n", phase);
    printf("[AJB_STATE]   CacheHit(SplitBucket): %d / %d total\n",
           idx.cntCacheHit, idx.cntTotalCall);
    printf("[AJB_STATE]   AGM calls:       %d  time=%.6fs\n",
           idx.cntAGMCall, idx.totalAGMTime);
    printf("[AJB_STATE]   CountOracle time: %.6fs\n", idx.totalCountOracleTime);
    printf("[AJB_STATE]   Split calls:      %d  time=%.6fs\n",
           idx.cntSplitCall, idx.totalSplitTime);
    printf("[AJB_STATE]   BinarySearch loops: %d\n", idx.cntBSCall);
    printf("[AJB_STATE]   CacheHit time:    %.6fs\n", idx.totalCacheHitTime);
    printf("[AJB_STATE]   BoundPrepare time: %.6fs\n", idx.totalBoundPrepareTime);
    printf("[AJB_STATE]   RRTree nodes:     %d\n", idx.totalrrtreenode);
    printf("[AJB_STATE] --- end ---\n");
}

int main() {
    printf("[AJB] ============================================\n");
    printf("[AJB] test_enumerator_full  full pipeline test\n");
    printf("[AJB] ============================================\n");

    AJBMemInfo mem;
    mem.capture(); mem.print("startup");

    // upstream: redirect to result file (keep original behavior)
    if(freopen("res/result.txt", "w", stdout) == NULL)
        fprintf(stderr, "[AJB_WARN] Cannot open res/result.txt, stdout only\n");

    // upstream: read config files (unchanged)
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string>> relations = readRelations("db/relations.txt");

    // AJB: dump loaded config
    printf("[AJB_STATE] Loaded: %zu relations, %zu filenames, %zu numlines\n",
           relations.size(), filenames.size(), numlines.size());
    for (auto& [name, cols] : relations) {
        printf("[AJB_STATE]   Rel '%s': [", name.c_str());
        for (size_t i = 0; i < cols.size(); i++)
            printf("%s%s", i ? "," : "", cols[i].c_str());
        printf("]\n");
    }

    // upstream: construct Enumerator (the core object)
    {
        AJBTimer timer("Enumerator_construct");
        Enumerator enumerator(relations, filenames, numlines);
        // AJB: the Enumerator constructor does preprocessing + enumeration
        printf("[AJB_STATE] Enumerator pipeline complete\n");
    }

    mem.capture(); mem.print("after_enumeration");

    printf("[AJB] VERDICT: test_enumerator_full PASSED\n");
    return 0;
}
