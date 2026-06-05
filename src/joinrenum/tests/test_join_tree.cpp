// =============================================================================
// test_join_tree.cpp — AJB-adapted JoinTree test harness
//
// Origin: upstream/joinrenum/testJoinTree.cpp (64 lines)
// Adaptation (~20%): AJB state dumps, structured bound printing, treeUpp
//   result verification, and scoped timing.
//
// Build: g++ -O3 test_join_tree.cpp -lglpk -o test_jt
// =============================================================================

#include <bits/stdc++.h>
#include "Index.hpp"
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

int main() {
    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});

    unordered_map<string, vector<string>> relations = readRelations("db/relations.txt");
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");

    Index idx(q);
    {
        ScopedTimer t("preprocessing");
        idx.preProcessing(relations, filenames, numlines);
    }

    // AJB: dump query structure
    printf("\n[AJB_STATE] Query structure:\n");
    q.print();

    vector<CountOracle<int>*> CO = idx.getCountOracles();
    printf("[AJB_STATE] CountOracle count = %zu\n", CO.size());

    JoinTree tree = idx.jt;
    printf("\n[AJB_STATE] JoinTree:\n");
    tree.print();
    tree.printChildren();

    Bucket B = idx.getFullBucket();
    printf("\n[AJB_STATE] FullBucket:\n");
    B.print();

    // Compute bounds per relation
    vector<vector<int>> relation = q.getRelations();
    vector<pair<vector<int>, vector<int>>> bound;
    for (size_t i = 0; i < relation.size(); i++) {
        vector<int> lower_bound, upper_bound;
        for (size_t j = 0; j < relation[i].size(); j++) {
            lower_bound.push_back(B.getLowerBound()[relation[i][j]]);
            upper_bound.push_back(B.getUpperBound()[relation[i][j]]);
        }
        bound.push_back({lower_bound, upper_bound});
    }

    // AJB: structured bound output
    printf("\n[AJB_STATE] Per-relation bounds:\n");
    for (size_t i = 0; i < bound.size(); i++) {
        printf("  R%zu lower=[", i);
        for (size_t j = 0; j < bound[i].first.size(); j++)
            printf("%s%d", j ? "," : "", bound[i].first[j]);
        printf("] upper=[");
        for (size_t j = 0; j < bound[i].second.size(); j++)
            printf("%s%d", j ? "," : "", bound[i].second[j]);
        printf("]\n");
    }

    // Test treeUpp with both bound-based and bucket-based calls
    long long upp_bound = tree.treeUpp(B.splitDim, bound);
    long long upp_iter  = tree.treeUpp(B);

    printf("\n[AJB_RESULTS] treeUpp comparison:\n");
    printf("  bound-based = %d\n", upp_bound);
    printf("  bucket-based = %d\n", upp_iter);

    // AJB: consistency check
    if (upp_bound != upp_iter) {
        fprintf(stderr, "[AJB_WARN] treeUpp results differ — check bound/iter consistency\n");
    } else {
        printf("[AJB] JoinTree test PASSED (treeUpp consistent)\n");
    }

    return 0;
}
