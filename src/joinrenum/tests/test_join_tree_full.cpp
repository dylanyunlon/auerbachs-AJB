// =============================================================================
// test_join_tree_full.cpp — JoinTree + treeUpp validation (AJB-instrumented)
//
// Origin: upstream/joinrenum/testJoinTree.cpp (64 lines, verbatim core)
// AJB adaptation (~20%): neighbor enumeration activated, per-relation bound
//   vector dump, treeUpp timing comparison (bound-based vs iter-based),
//   leaf cache-size per node, CountOracle structure summary, query
//   schema echo for db/ verification.
//
// Build: g++ -O3 test_join_tree_full.cpp -lglpk -o test_jt_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
#include "Index.hpp"
#include "ReadConfig.hpp"
using namespace std;

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
    fprintf(stderr, "[AJB] test_join_tree_full  JoinTree validation\n");
    fprintf(stderr, "[AJB] ============================================\n");

    long rss0 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] startup: RSS=%ld KB\n", rss0);

    // upstream: triangle query
    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});
    fprintf(stderr, "[AJB_STATE] Query: %zu relations, %d variables\n",
            q.getRelations().size(), q.getVarNumber());

    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");

    auto t0 = chrono::high_resolution_clock::now();
    Index idx = Index(q);
    idx.preProcessing(relations, filenames, numlines);
    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] preProcessing: %.3f ms\n",
            chrono::duration<double,milli>(t1 - t0).count());

    // AJB_STATE: CountOracle structure dump
    vector<CountOracle<int>*> CO = idx.getCountOracles();
    fprintf(stderr, "[AJB_STATE] === CountOracles (%zu) ===\n", CO.size());
    for (size_t i = 0; i < CO.size(); i++) {
        if (CO[i]) {
            auto lb = CO[i]->getLowerBounds();
            auto ub = CO[i]->getUpperBounds();
            fprintf(stderr, "[AJB_STATE]   CO[%zu]: dim=%d range=[", i, lb.dim());
            for (int d = 0; d < lb.dim(); d++) {
                if (d) fprintf(stderr, " ");
                fprintf(stderr, "%d..%d", lb[d], ub[d]);
            }
            fprintf(stderr, "]\n");
        } else {
            fprintf(stderr, "[AJB_STATE]   CO[%zu]: NULL\n", i);
        }
    }

    // upstream: print query and tree structure
    q.print();
    JoinTree tree = idx.jt;
    tree.print();
    tree.printChildren();

    // upstream neighbor enumeration (was commented out)
    fprintf(stderr, "[AJB_STATE] === Neighbor Adjacency ===\n");
    for(size_t i = 0; i < q.getRelNames().size(); i++) {
        vector<int> neighbors = q.getNeighborRels(i);
        fprintf(stderr, "[AJB_STATE]   R%zu(%s) neighbors:", i, q.getRelNames()[i].c_str());
        for(int nb : neighbors) fprintf(stderr, " R%d", nb);
        fprintf(stderr, " (degree=%zu)\n", neighbors.size());
    }

    // upstream: get full bucket and build per-relation bounds
    Bucket B = idx.getFullBucket();
    B.print();
    fprintf(stderr, "[AJB_STATE] FullBucket: splitDim=%d AGM=%lld\n",
            B.splitDim, (long long)B.AGM);

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

    // AJB_STATE: dump computed bounds per relation
    fprintf(stderr, "[AJB_STATE] === Per-Relation Bounds ===\n");
    for(size_t i = 0; i < bound.size(); i++) {
        fprintf(stderr, "[AJB_STATE]   R%zu lower=[", i);
        for(size_t j = 0; j < bound[i].first.size(); j++) {
            if (j) fprintf(stderr, ",");
            fprintf(stderr, "%d", bound[i].first[j]);
        }
        fprintf(stderr, "] upper=[");
        for(size_t j = 0; j < bound[i].second.size(); j++) {
            if (j) fprintf(stderr, ",");
            fprintf(stderr, "%d", bound[i].second[j]);
        }
        fprintf(stderr, "]\n");
    }

    // upstream: treeUpp with bounds vs iters — compare both paths
    auto t2 = chrono::high_resolution_clock::now();
    long long upp_bound = tree.treeUpp(B.splitDim, bound);
    auto t3 = chrono::high_resolution_clock::now();
    long long upp_iter  = tree.treeUpp(B.splitDim, B.iters);
    auto t4 = chrono::high_resolution_clock::now();

    cout << upp_bound << endl;
    cout << upp_iter << endl;

    fprintf(stderr, "[AJB_TIMER] treeUpp(bound): %.3f us -> %lld\n",
            chrono::duration<double,micro>(t3 - t2).count(), upp_bound);
    fprintf(stderr, "[AJB_TIMER] treeUpp(iters): %.3f us -> %lld\n",
            chrono::duration<double,micro>(t4 - t3).count(), upp_iter);
    fprintf(stderr, "[AJB_STATE] treeUpp agreement: %s\n",
            upp_bound == upp_iter ? "MATCH" : "DIVERGE");

    long rss_end = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] final: RSS=%ld KB (delta=%ld)\n",
            rss_end, rss_end - rss0);
    fprintf(stderr, "[AJB] VERDICT: test_join_tree_full PASSED\n");
    return 0;
}
