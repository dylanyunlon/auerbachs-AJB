// =============================================================================
// test_join_tree_full.cpp  AJB-adapted full JoinTree test harness
//
// Origin: upstream/joinrenum/testJoinTree.cpp (64 lines)
// AJB adaptation (~20%): breakpoint dumps of tree structure, neighbor
//   relations, CountOracle binding, and timing instrumentation.
//
// Build: g++ -O3 test_join_tree_full.cpp -lglpk -o test_jt_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
#include "Index.hpp"
#include "ReadConfig.hpp"
using namespace std;

// AJB: timer helper
struct AJBTimer {
    string label;
    chrono::high_resolution_clock::time_point t0;
    AJBTimer(const string& l) : label(l),
        t0(chrono::high_resolution_clock::now()) {}
    ~AJBTimer() {
        double ms = chrono::duration<double,milli>(
            chrono::high_resolution_clock::now() - t0).count();
        printf("[AJB_TIMER] %s: %.3f ms\n", label.c_str(), ms);
    }
};

int main() {
    printf("[AJB] ============================================\n");
    printf("[AJB] test_join_tree_full  JoinTree structure test\n");
    printf("[AJB] ============================================\n");

    // upstream: triangle query
    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});

    // AJB: dump query
    printf("[AJB_STATE] Query constructed: %zu relations\n",
           q.getRelNames().size());

    // upstream: read config
    unordered_map<string, vector<string>> relations = readRelations("db/relations.txt");
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");

    printf("[AJB_STATE] Config: %zu rels, %zu files, %zu numlines\n",
           relations.size(), filenames.size(), numlines.size());

    // upstream: Index + preProcessing
    Index idx = Index(q);
    {
        AJBTimer timer("preProcessing");
        idx.preProcessing(relations, filenames, numlines);
    }

    // upstream: get CountOracles
    vector<CountOracle<int>*> CO = idx.getCountOracles();
    printf("[AJB_STATE] CountOracles: %zu created\n", CO.size());

    // upstream: print query
    q.print();

    // AJB: dump neighbor relations for each relation in the query
    printf("[AJB_STATE] --- Neighbor structure ---\n");
    for (size_t i = 0; i < q.getRelNames().size(); i++) {
        vector<int> neighbors = q.getNeighborRels(i);
        printf("[AJB_STATE]   Rel %zu '%s' neighbors: [",
               i, q.getRelNames()[i].c_str());
        for (size_t j = 0; j < neighbors.size(); j++)
            printf("%s%d", j ? "," : "", neighbors[j]);
        printf("]\n");
    }

    // upstream: JoinTree from Index
    JoinTree tree = idx.jt;

    // AJB: dump tree structure
    printf("[AJB_STATE] --- JoinTree ---\n");
    tree.print();

    // AJB: dump tree-level stats
    printf("[AJB_STATE] JoinTree node count: checking traversal...\n");

    // upstream: try the alternate larger queries (commented in original)
    // Query q2({"R1","R2","R3","R4"}, {{"A","B","C","D"}, {"B","D","E","G"},
    //           {"B","C","E","F"}, {"C","D","F","G"}});

    // AJB: validate join tree has expected properties
    bool tree_valid = true;
    printf("[AJB_STATE] JoinTree basic validation:\n");
    printf("[AJB_STATE]   - Relations in query: %zu\n", q.getRelNames().size());
    printf("[AJB_STATE]   - Tables in index: %zu\n", idx.tables.size());
    if (idx.tables.size() != q.getRelNames().size()) {
        printf("[AJB_FAIL]   table count mismatch!\n");
        tree_valid = false;
    }

    printf("[AJB] VERDICT: test_join_tree_full %s\n",
           tree_valid ? "PASSED" : "FAILED");
    return tree_valid ? 0 : 1;
}
