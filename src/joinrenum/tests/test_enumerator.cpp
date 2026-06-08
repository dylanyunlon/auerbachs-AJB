// =============================================================================
// test_enumerator.cpp — AJB-adapted Enumerator test harness
//
// Origin: upstream/joinrenum/testEnumerator.cpp (33 lines)
// Adaptation (~20%): AJB structured state dumps, Index cache/performance
//   counters, and scoped timing for the enumeration pass.
//
// Build: g++ -O2 test_enumerator.cpp -lglpk -o test_enum
// =============================================================================

#include <bits/stdc++.h>
#include "ReadConfig.hpp"
#include "Enumerator.hpp"

// AJB: inline scoped timer
struct ScopedTimer {
  const char* label;
  std::chrono::high_resolution_clock::time_point t0;
  ScopedTimer(const char* l) : label(l), t0(std::chrono::high_resolution_clock::now()) {
    printf("[AJB_TIMER] >>> %s\n", label);
  }
  ~ScopedTimer() {
    double s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    printf("[AJB_TIMER] <<< %s  %.6f s (%.3f ms)\n", label, s, s * 1000.0);
  }
};

using namespace std;

// AJB: print Index internal performance counters
void printIndexDiagnostics(Index& idx) {
    printf("\n[AJB_STATE] Index performance counters:\n");
    printf("  SplitBucket cache hit  = %d / %d (%.1f%%)\n",
           idx.cntCacheHit, idx.cntTotalCall,
           idx.cntTotalCall > 0 ? 100.0 * idx.cntCacheHit / idx.cntTotalCall : 0.0);
    printf("  AGM calls              = %d\n", idx.cntAGMCall);
    printf("  AGM total time         = %.6f s\n", idx.totalAGMTime);
    printf("  CountOracle time       = %.6f s\n", idx.totalCountOracleTime);
    printf("  Split total time       = %.6f s\n", idx.totalSplitTime);
    printf("  Split calls            = %d\n", idx.cntSplitCall);
    printf("  Binary search loops    = %d\n", idx.cntBSCall);
    printf("  Cache hit time         = %.6f s\n", idx.totalCacheHitTime);
    printf("  Bound prepare time     = %.6f s\n", idx.totalBoundPrepareTime);
    printf("  RRTree nodes           = %d\n", idx.totalrrtreenode);

    // AJB: derived metrics
    if (idx.cntSplitCall > 0) {
        printf("  avg_split_time         = %.6f ms\n",
               idx.totalSplitTime / idx.cntSplitCall * 1000.0);
    }
    if (idx.cntAGMCall > 0) {
        printf("  avg_agm_time           = %.6f ms\n",
               idx.totalAGMTime / idx.cntAGMCall * 1000.0);
    }
}

int main(int argc, char* argv[]) {
    // AJB: optional output redirection
    bool redirect = false;
    if (argc >= 2 && string(argv[1]) == "--file") {
        redirect = true;
        if (freopen("res/result.txt", "w", stdout) == NULL) {
            fprintf(stderr, "[AJB_ERROR] Cannot open res/result.txt for writing\n");
            return 1;
        }
        printf("[AJB] Output redirected to res/result.txt\n");
    }

    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string>> relations = readRelations("db/relations.txt");

    printf("[AJB] Config loaded: %zu relations, %zu files\n",
           relations.size(), filenames.size());

    Enumerator* enumerator;
    {
        ScopedTimer t("enumerator_init");
        enumerator = new Enumerator(relations, filenames, numlines);
    }

    {
        ScopedTimer t("random_enumerate");
        enumerator->random_enumerate();
    }

    // AJB: dump diagnostics
    printIndexDiagnostics(enumerator->access_tree.idx);

    printf("[AJB] Enumerator test DONE\n");

    delete enumerator;
    return 0;
}
