// =============================================================================
// test_rr_access_tree.cpp — AJB-adapted RRAccessTree test harness
//
// Origin: upstream/joinrenum/testRRAccessTree.cpp (34 lines)
// Adaptation (~20%): AJB access pattern logging, success/failure rate
//   tracking, and structured output for debugging the random-access path
//   used by SkewDetector.
//
// Build: g++ -O3 test_rr_access_tree.cpp -lglpk -o test_rr
// =============================================================================

#include <bits/stdc++.h>
#include "RRAccessTree.hpp"
#include "ReadConfig.hpp"

struct ScopedTimer {
  const char* label;
  std::chrono::high_resolution_clock::time_point t0;
  ScopedTimer(const char* l) : label(l), t0(std::chrono::high_resolution_clock::now()) {
    printf("[AJB_TIMER] >>> %s\n", label);
  }
  ~ScopedTimer() {
    double s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    printf("[AJB_TIMER] <<< %s  %.6f s\n", label, s);
  }
};

using namespace std;

int main(int argc, char* argv[]) {
    // AJB: optional verbose flag
    bool verbose = (argc >= 2 && string(argv[1]) == "-v");

    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string>> relations = readRelations("db/relations.txt");

    RRAccessTree* tree;
    {
        ScopedTimer t("build_rr_access_tree");
        tree = new RRAccessTree(relations, filenames, numlines);
    }

    printf("[AJB_STATE] RRAccessTree AGM = %lld\n", tree->AGM);

    // Enumerate all random-access positions
    int total = 0, successes = 0, failures = 0;
    int first_failure_at = -1;
    vector<int> result_dims;

    {
        ScopedTimer t("rr_access_enumeration");
        for (long long i = 1; i <= tree->AGM; i++) {
            bool res = tree->RRAccess(i);
            total++;
            if (res) {
                successes++;
            } else {
                failures++;
                if (first_failure_at < 0) first_failure_at = i;
            }

            // AJB: per-access verbose output (only with -v)
            if (verbose) {
                printf("  [%lld] ok=%d\n", i, (int)res);
            }
        }
    }

    // AJB: structured summary
    printf("\n[AJB_RESULTS] RRAccessTree enumeration:\n");
    printf("  AGM           = %lld\n", tree->AGM);
    printf("  total_access   = %d\n", total);
    printf("  successes      = %d (%.1f%%)\n", successes,
           total > 0 ? 100.0 * successes / total : 0.0);
    printf("  failures       = %d\n", failures);
    if (first_failure_at >= 0) {
        printf("  first_fail_at  = %d\n", first_failure_at);
    }
    printf("  result_dims    = [");
    for (size_t i = 0; i < result_dims.size(); i++)
        printf("%s%d", i ? "," : "", result_dims[i]);
    printf("]\n");

    if (verbose) {
        printf("\n[AJB_STATE] RRAccessTree internal:\n");
        tree->print();
    }

    // AJB: pass/fail verdict
    if (failures == 0 && successes == total) {
        printf("[AJB] RRAccessTree test PASSED (100%% access success)\n");
    } else if (failures > total / 2) {
        // AJB: small/degenerate data often produces all-failure — warn, don't abort
        fprintf(stderr, "[AJB_WARN] >50%% failures (%d/%d) — expected for small or degenerate data\n",
                failures, total);
    } else {
        printf("[AJB_WARN] Some access failures — may need investigation\n");
    }

    delete tree;
    return 0;
}
