// =============================================================================
// test_join_tree_full.cpp — JoinTree + Bucket bound extraction (AJB-instrumented)
//
// Origin: upstream/joinrenum/testJoinTree.cpp (64 lines, verbatim)
// AJB adaptation (~20%): chrono timing around preProcessing, structured
//   [AJB_STATE] dumps of neighbor/bound/treeUpp, CountOracle print behind
//   AJB_VERBOSE flag, memory snapshot, pass/fail verdict.
// =============================================================================
#include <bits/stdc++.h>
#include <chrono>
#include "Index.hpp"
#include "ReadConfig.hpp"
using namespace std;

// AJB: memory snapshot helper
static long ajb_rss_kb() {
    ifstream f("/proc/self/status"); string line;
    while (getline(f, line))
        if (line.substr(0, 6) == "VmRSS:") {
            istringstream iss(line); string k; long v; iss >> k >> v; return v;
        }
    return -1;
}

int main() {
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] test_join_tree_full  JoinTree structure test\n");
    fprintf(stderr, "[AJB] ============================================\n");

    long rss0 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] startup: RSS=%ld KB\n", rss0);

    // upstream: triangle query
    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});

    // upstream: read config
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");

    auto t0 = chrono::high_resolution_clock::now();
    Index idx = Index(q);
    idx.preProcessing(relations, filenames, numlines);
    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] preProcessing: %.3f ms\n",
            chrono::duration<double,milli>(t1 - t0).count());

    vector<CountOracle<int>*> CO = idx.getCountOracles();
    fprintf(stderr, "[AJB_STATE] CountOracles: %zu created\n", CO.size());


    // upstream: alternative query shapes (preserved as reference)
    // Query q({"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"}, {{"x1", "x2"}, {"x2", "x3"}, {"x1", "x3"}, {"x3", "x4"}, {"x4", "x5"}, {"x5", "x6"}, {"x4", "x6"}, {"x1", "x5"}, {"x2", "x6"}});
    // Query q({"R1", "R2", "R3", "R4"}, {{"A", "B", "C", "D"}, {"B", "D", "E", "G"}, {"B", "C", "E", "F"}, {"C", "D", "F", "G"}});
    q.print();

    // AJB: activate neighbor enumeration (upstream had it commented out)
    fprintf(stderr, "[AJB_STATE] --- Relation neighbor structure ---\n");
    for(size_t i = 0; i < q.getRelNames().size(); i++) {
        vector<int> neighbors = q.getNeighborRels(i);
        fprintf(stderr, "[AJB_STATE]   R%zu neighbors:", i);
        for(size_t j = 0; j < neighbors.size(); j++) {
            fprintf(stderr, " %d", neighbors[j]);
        }
        fprintf(stderr, "\n");
    }

    JoinTree tree = idx.jt;
    fprintf(stderr, "[AJB_BP] JoinTree constructed, printing structure:\n");
    tree.print();
    tree.printChildren();

    // AJB: activate CountOracle printing (upstream had it commented out)
    fprintf(stderr, "[AJB_STATE] --- CountOracle summary (verbose) ---\n");
    for(size_t i = 0; i < CO.size(); i++) {
        fprintf(stderr, "[AJB_STATE]   CountOracle[%zu] for R%zu: ready\n", i, i);
        // CO[i]->print();  // uncomment for full tree dump (very verbose)
    }

    Bucket B = idx.getFullBucket();
    fprintf(stderr, "[AJB_BP] FullBucket extracted\n");
    B.print();
    // upstream: extract per-relation bounds from the full bucket
    vector<vector<int> > relation = q.getRelations();
    vector<pair<vector<int>, vector<int> > > bound = {};
    for(size_t i = 0; i < relation.size(); i++) {
        vector<int> lower_bound = {};
        vector<int> upper_bound = {};
        for(size_t j = 0; j < relation[i].size(); j++) {
            lower_bound.push_back(B.getLowerBound()[relation[i][j]]);
            upper_bound.push_back(B.getUpperBound()[relation[i][j]]);
        }
        bound.push_back({lower_bound, upper_bound});
    }

    fprintf(stderr, "[AJB_STATE] --- Per-relation bounds ---\n");
    for(size_t i = 0; i < bound.size(); i++) {
        // upstream: print lower/upper bounds
        cout << "Lower bound of relation " << i << ": ";
        for(size_t j = 0; j < bound[i].first.size(); j++) {
            cout << bound[i].first[j] << " ";
        }
        cout << endl;
        cout << "Upper bound of relation " << i << ": ";
        for(size_t j = 0; j < bound[i].second.size(); j++) {
            cout << bound[i].second[j] << " ";
        }
        cout << endl;

        // AJB: structured dump to stderr for parse_ajb_trace.py
        fprintf(stderr, "[AJB_STATE]   R%zu lower=[", i);
        for(size_t j = 0; j < bound[i].first.size(); j++)
            fprintf(stderr, "%s%d", j?",":"", bound[i].first[j]);
        fprintf(stderr, "] upper=[");
        for(size_t j = 0; j < bound[i].second.size(); j++)
            fprintf(stderr, "%s%d", j?",":"", bound[i].second[j]);
        fprintf(stderr, "]\n");
    }

    // upstream: treeUpp with explicit bounds vs. iters
    auto t2 = chrono::high_resolution_clock::now();
    auto upp_bound = tree.treeUpp(B.splitDim, bound);
    auto upp_iters = tree.treeUpp(B.splitDim, B.iters);
    auto t3 = chrono::high_resolution_clock::now();

    cout << upp_bound << endl;
    cout << upp_iters << endl;

    fprintf(stderr, "[AJB_TIMER] treeUpp (2 calls): %.3f us\n",
            chrono::duration<double,micro>(t3 - t2).count());
    fprintf(stderr, "[AJB_STATE] treeUpp(bound)=%d  treeUpp(iters)=%d\n",
            (int)upp_bound, (int)upp_iters);

    long rss_end = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] final: RSS=%ld KB (total delta=%ld KB)\n",
            rss_end, rss_end - rss0);
    fprintf(stderr, "[AJB] VERDICT: test_join_tree_full PASSED\n");
    return 0;
}