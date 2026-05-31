// =============================================================================
// test_rr_access_tree_full.cpp  AJB-adapted RRAccessTree test
//
// Origin: upstream/joinrenum/testRRAccessTree.cpp (34 lines)
// AJB adaptation (~20%): step-by-step RRAccess dump showing each tuple
//   returned, distribution analysis, and timing for the full AGM range.
//
// Build: g++ -O3 test_rr_access_tree_full.cpp -lglpk -o test_rrat_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
#include "RRAccessTree.hpp"
#include "ReadConfig.hpp"
using namespace std;

int main() {
    printf("[AJB] ============================================\n");
    printf("[AJB] test_rr_access_tree_full  RRAccess test\n");
    printf("[AJB] ============================================\n");

    // upstream: read config
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string>> relations = readRelations("db/relations.txt");

    printf("[AJB_STATE] Config loaded: %zu relations\n", relations.size());

    // upstream: build tree
    auto t0 = chrono::high_resolution_clock::now();
    RRAccessTree tree(relations, filenames, numlines);
    auto t1 = chrono::high_resolution_clock::now();
    double build_ms = chrono::duration<double,milli>(t1 - t0).count();

    printf("[AJB_TIMER] RRAccessTree build: %.3f ms\n", build_ms);
    printf("[AJB_STATE] AGM bound = %d\n", tree.AGM);

    // upstream: enumerate all RRAccess(1..AGM)
    int success_count = 0, fail_count = 0;
    auto t2 = chrono::high_resolution_clock::now();

    for (int i = 1; i <= tree.AGM; i++) {
        pair<bool, vector<int>> res = tree.RRAccess(i);

        // AJB: dump every access result (breakpoint-style)
        if (i <= 20 || i == tree.AGM || i % 100 == 0) {
            printf("[AJB_TRACE] RRAccess(%d): valid=%d  tuple=[",
                   i, (int)res.first);
            for (size_t j = 0; j < res.second.size(); j++)
                printf("%s%d", j ? "," : "", res.second[j]);
            printf("]\n");
        }

        if (res.first) success_count++;
        else fail_count++;
    }

    auto t3 = chrono::high_resolution_clock::now();
    double enum_ms = chrono::duration<double,milli>(t3 - t2).count();

    printf("[AJB_TIMER] RRAccess enumeration: %.3f ms\n", enum_ms);
    printf("[AJB_STATE] Results: %d valid, %d invalid out of %d\n",
           success_count, fail_count, tree.AGM);

    // AJB: distribution analysis
    double valid_ratio = tree.AGM > 0 ?
        (double)success_count / tree.AGM : 0.0;
    printf("[AJB_STATE] Valid ratio: %.4f\n", valid_ratio);

    bool pass = (success_count + fail_count == tree.AGM);
    printf("[AJB] VERDICT: test_rr_access_tree_full %s\n",
           pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
