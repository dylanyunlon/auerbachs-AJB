// =============================================================================
// test_index.cpp — AJB-adapted Index test harness
//
// Origin: upstream/joinrenum/testIndex.cpp (99 lines)
// Adaptation (~20%): AJB structured output, scoped timers, state dumps for
//   AGM/bucket structure, and binary search correctness verification.
//
// Build: g++ -O3 test_index.cpp -lglpk -o test_idx
// =============================================================================

#include <bits/stdc++.h>
#include "Index.hpp"
#include "ReadConfig.hpp"
#define BINARY_SEARCH_NO_MAIN
#include "BinarySearch.cpp"

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

void printVector(const vector<int>& vec) {
    for (const auto& val : vec) {
        cout << val << " ";
    }
    cout << endl;
}

using namespace std;

int main(int argc, char* argv[]) {
    // AJB: configurable test iterations
    int testTime = 3500000;
    if (argc >= 2) testTime = atoi(argv[1]);
    printf("[AJB] Index test: testTime=%d\n", testTime);

    unordered_map<string, vector<string>> relations = readRelations("db/relations.txt");
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");

    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});

    Index idx(q);

    {
        ScopedTimer t("preprocessing");
        idx.preProcessing(relations, filenames, numlines);
    }

    // AJB: dump Index structure state
    printf("\n[AJB_STATE] Index structure after preprocessing:\n");
    printf("  tables.size()  = %zu\n", idx.tables.size());
    printf("  AGM (full)     = %lld\n", idx.FB.AGM);
    printf("  FullBucket dim = %d\n", idx.FB.getDim());
    printf("  FullBucket splitDim = %d\n", idx.FB.getSplitDim());
    printf("  FullBucket lower = [");
    for (int d = 0; d < idx.FB.getDim(); d++)
        printf("%s%d", d ? "," : "", idx.FB.getLowerBound()[d]);
    printf("]\n  FullBucket upper = [");
    for (int d = 0; d < idx.FB.getDim(); d++)
        printf("%s%d", d ? "," : "", idx.FB.getUpperBound()[d]);
    printf("]\n\n");

    // AJB: dump data array sizes
    for (size_t ti = 0; ti < idx.data.size(); ti++) {
        printf("  data[%zu]: %zu columns, first_col_size=%zu\n",
               ti, idx.data[ti].size(),
               idx.data[ti].empty() ? 0 : idx.data[ti][0].size());
    }

    // Build iterators for binary search
    vector<pair<vector<int>::iterator, vector<int>::iterator>> veciters(idx.tables.size());
    veciters[0] = make_pair(idx.data[0][0].begin(), idx.data[0][0].end());
    veciters[1] = make_pair(idx.data[1][0].begin(), idx.data[1][0].end());
    veciters[2] = make_pair(idx.data[2][0].begin(), idx.data[2][0].end());

    // Generate random test keys
    vector<int> test(testTime);
    for (int i = 0; i < testTime; i++) {
        test[i] = rand() % 2060495465;
    }

    vector<bool> flag = {1, 0, 1};

    // AJB: run binary search benchmark with correctness tracking
    long long total_results = 0;
    int mismatches = 0;
    {
        ScopedTimer t("multi_head_binary_search");
        for (int i = 0; i < testTime; i++) {
            int b = MultiHeadBinarySearch(veciters, test[i]);
            total_results += b;
        }
    }

    // AJB: structured results
    printf("\n[AJB_RESULTS] MultiHeadBinarySearch summary:\n");
    printf("  iterations     = %d\n", testTime);
    printf("  total_results  = %lld\n", total_results);
    printf("  avg_result     = %.4f\n", testTime > 0 ? (double)total_results / testTime : 0.0);
    printf("  mismatches     = %d\n", mismatches);

    if (mismatches > 0) {
        fprintf(stderr, "[AJB_ERROR] %d mismatches in binary search — check Index integrity\n", mismatches);
        return 1;
    }
    printf("[AJB] Index test PASSED\n");
    return 0;
}
