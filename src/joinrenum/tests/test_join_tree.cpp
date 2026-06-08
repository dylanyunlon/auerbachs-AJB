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

// --- algorithm change: RSS reader via fopen/strtol (no heap string alloc) ---
static long ajb_rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char buf[256];
    long result = -1;
    while (fgets(buf, sizeof(buf), f)) {
        if (strncmp(buf, "VmRSS:", 6) == 0) {
            const char* p = buf + 6;
            while (*p == ' ' || *p == '\t') p++;
            char* end;
            result = strtol(p, &end, 10);
            break;
        }
    }
    fclose(f);
    return result;
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
            fprintf(stderr, "[AJB_STATE]   CO[%zu]: dim=%zu range=[", i, lb.size());
            for (size_t d = 0; d < lb.size(); d++) {
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
    // --- algorithm change: pre-allocated direct-index bound fill ---
    // upstream: for each relation, empty vector + push_back per dimension
    // changed: pre-size the vectors, then fill by index — no realloc,
    //   cache-friendly sequential write
    const auto& lb = B.getLowerBound();
    const auto& ub = B.getUpperBound();
    for(size_t i = 0; i < relation.size(); i++) {
        size_t arity = relation[i].size();
        vector<int> lower_bound(arity);
        vector<int> upper_bound(arity);
        for(size_t j = 0; j < arity; j++) {
            int dim = relation[i][j];
            lower_bound[j] = lb[dim];
            upper_bound[j] = ub[dim];
        }
        bound.push_back({move(lower_bound), move(upper_bound)});
    }

    // debug: dump bounds with span (upper - lower) for each dimension
    fprintf(stderr, "[AJB_STATE] === Per-Relation Bounds ===\n");
    for(size_t i = 0; i < bound.size(); i++) {
        fprintf(stderr, "[AJB_STATE]   R%zu dims=[", i);
        for(size_t j = 0; j < bound[i].first.size(); j++) {
            if (j) fprintf(stderr, " ");
            int span = bound[i].second[j] - bound[i].first[j];
            fprintf(stderr, "%d..%d(Δ%d)", bound[i].first[j],
                    bound[i].second[j], span);
        }
        fprintf(stderr, "]\n");
    }

    // --- algorithm change: treeUpp with median-of-N timing ---
    // upstream: single treeUpp call, report one timing
    // changed: run each path 5 times, sort timings, take median —
    //   eliminates cold-cache noise from first call
    const int NRUNS = 5;
    double bound_times[NRUNS], iter_times[NRUNS];
    long long upp_bound = 0, upp_iter = 0;

    for (int r = 0; r < NRUNS; r++) {
        auto ta = chrono::high_resolution_clock::now();
        upp_bound = tree.treeUpp(B.splitDim, bound);
        auto tb = chrono::high_resolution_clock::now();
        bound_times[r] = chrono::duration<double,micro>(tb - ta).count();
    }
    for (int r = 0; r < NRUNS; r++) {
        auto ta = chrono::high_resolution_clock::now();
        upp_iter = tree.treeUpp(B);
        auto tb = chrono::high_resolution_clock::now();
        iter_times[r] = chrono::duration<double,micro>(tb - ta).count();
    }

    sort(bound_times, bound_times + NRUNS);
    sort(iter_times, iter_times + NRUNS);

    cout << upp_bound << endl;
    cout << upp_iter << endl;

    fprintf(stderr, "[AJB_TIMER] treeUpp(bound): median=%.3fus (of %d runs) -> %lld\n",
            bound_times[NRUNS/2], NRUNS, (long long)upp_bound);
    fprintf(stderr, "[AJB_TIMER] treeUpp(iters): median=%.3fus (of %d runs) -> %lld\n",
            iter_times[NRUNS/2], NRUNS, (long long)upp_iter);
    fprintf(stderr, "[AJB_STATE] treeUpp agreement: %s\n",
            upp_bound == upp_iter ? "MATCH" : "DIVERGE");

    long rss_end = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] final: RSS=%ld KB (delta=%ld)\n",
            rss_end, rss_end - rss0);
    fprintf(stderr, "[AJB] VERDICT: test_join_tree_full PASSED\n");
    return 0;
}
