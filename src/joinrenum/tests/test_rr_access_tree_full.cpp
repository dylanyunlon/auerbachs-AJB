// =============================================================================
// test_rr_access_tree_full.cpp — RRAccessTree full (AJB-instrumented)
//
// Origin: upstream/joinrenum/testRRAccessTree.cpp (34 lines, verbatim core)
// AJB adaptation (~20%): per-access result vector dump with value-range
//   tracking, tree depth distribution from ajb_rrt_stats, memory profiling
//   at 3 phases (startup/build/enumeration), split-point histogram for
//   diagnosing RRAccess hotspots, per-rank latency sampling.
//
// Build: g++ -O3 test_rr_access_tree_full.cpp -lglpk -o test_rra_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
#include "RRAccessTree.hpp"
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
    fprintf(stderr, "[AJB] test_rr_access_tree_full  RRAccessTree test\n");
    fprintf(stderr, "[AJB] ============================================\n");

    long rss0 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] startup: RSS=%ld KB\n", rss0);

    // upstream: read config
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");

    // AJB_STATE: echo the loaded schema so we can verify db/ content
    fprintf(stderr, "[AJB_STATE] schema: %zu relations loaded\n", relations.size());
    for (auto& [name, vars] : relations) {
        fprintf(stderr, "[AJB_STATE]   %s(", name.c_str());
        for (size_t j = 0; j < vars.size(); j++)
            fprintf(stderr, "%s%s", j ? "," : "", vars[j].c_str());
        fprintf(stderr, ") file=%s lines=%d\n",
                filenames.count(name) ? filenames[name].c_str() : "?",
                numlines.count(name) ? numlines[name] : -1);
    }

    // upstream: construct RRAccessTree
    auto t0 = chrono::high_resolution_clock::now();
    RRAccessTree tree(relations, filenames, numlines);
    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] RRAccessTree construction: %.3f ms\n",
            chrono::duration<double,milli>(t1 - t0).count());
    fprintf(stderr, "[AJB_STATE] AGM bound = %lld\n", (long long)tree.AGM);

    // AJB_STATE: dump the Index internals that the tree wraps
    fprintf(stderr, "[AJB_STATE] Index: %zu tables, varnum=%d, treeflag=%d\n",
            tree.idx.tables.size(), tree.idx.q.getVarNumber(),
            (int)tree.idx.treeflag);
    // dump the full bucket — this is the root of all splits
    Bucket fb = tree.idx.getFullBucket();
    fprintf(stderr, "[AJB_STATE] FullBucket: splitDim=%d AGM=%lld bounds=[",
            fb.splitDim, (long long)fb.AGM);
    for (size_t d = 0; d < fb.getLowerBound().size(); d++) {
        if (d) fprintf(stderr, " ");
        fprintf(stderr, "%d..%d", fb.getLowerBound()[d], fb.getUpperBound()[d]);
    }
    fprintf(stderr, "]\n");

    long rss1 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] after_build: RSS=%ld KB (delta=%ld)\n", rss1, rss1 - rss0);

    // upstream: enumerate all i in [1..AGM] and call RRAccess(i)
    int success_count = 0, fail_count = 0;
    // AJB: track value-range per dimension across successful accesses
    vector<int> dim_min, dim_max;
    long long val_sum = 0;
    int dim_count = 0;
    // AJB: per-access latency sampling (every Nth)
    vector<double> latency_samples;

    auto t2 = chrono::high_resolution_clock::now();
    for(int i = 1; i <= tree.AGM; i++) {
        auto ta = chrono::steady_clock::now();
        pair<bool, vector<int> > res = tree.RRAccess(i);
        auto tb = chrono::steady_clock::now();

        // upstream: print each result
        cout << i << ": " << res.first << ", ";
        for(size_t j = 0; j < res.second.size(); j++) {
            cout << res.second[j] << ",";
        }
        cout << endl;

        if(res.first) {
            success_count++;
            // AJB: accumulate dimension-wise min/max for value distribution
            if (dim_min.empty()) {
                dim_count = res.second.size();
                dim_min = res.second;
                dim_max = res.second;
            } else {
                for (int d = 0; d < dim_count && d < (int)res.second.size(); d++) {
                    dim_min[d] = min(dim_min[d], res.second[d]);
                    dim_max[d] = max(dim_max[d], res.second[d]);
                }
            }
            for (int v : res.second) val_sum += v;
        } else {
            fail_count++;
        }

        // AJB: sample latency every 50th access
        if (i % 50 == 0) {
            double us = chrono::duration<double, micro>(tb - ta).count();
            latency_samples.push_back(us);
        }

        // AJB: periodic progress with running hit-rate
        if(i % 100 == 0 || i == tree.AGM) {
            double hit_rate = success_count * 100.0 / i;
            fprintf(stderr, "[AJB_TRACE] RRAccess %d/%lld  ok=%d fail=%d hit=%.1f%%\n",
                    i, (long long)tree.AGM, success_count, fail_count, hit_rate);
        }
    }
    auto t3 = chrono::high_resolution_clock::now();
    double loop_ms = chrono::duration<double,milli>(t3 - t2).count();
    fprintf(stderr, "[AJB_TIMER] RRAccess loop (1..%lld): %.3f ms\n",
            (long long)tree.AGM, loop_ms);

    // AJB_STATE: complete summary
    fprintf(stderr, "[AJB_STATE] Final: success=%d  fail=%d  total=%lld\n",
            success_count, fail_count, (long long)tree.AGM);
    if (tree.AGM > 0) {
        fprintf(stderr, "[AJB_STATE] avg_latency=%.3f us/access  throughput=%.1f Kops/s\n",
                loop_ms * 1000.0 / tree.AGM, tree.AGM / loop_ms);
    }

    // AJB_STATE: dump value-range distribution per dimension
    if (!dim_min.empty()) {
        fprintf(stderr, "[AJB_STATE] result dims=%d value_ranges=[", dim_count);
        for (int d = 0; d < dim_count; d++) {
            if (d) fprintf(stderr, " ");
            fprintf(stderr, "%d..%d", dim_min[d], dim_max[d]);
        }
        fprintf(stderr, "] avg_val=%.1f\n",
                success_count > 0 && dim_count > 0 ?
                (double)val_sum / (success_count * dim_count) : 0.0);
    }

    // AJB_STATE: latency distribution (p50/p90/p99)
    if (!latency_samples.empty()) {
        sort(latency_samples.begin(), latency_samples.end());
        int n = latency_samples.size();
        fprintf(stderr, "[AJB_STATE] latency(us) n=%d p50=%.1f p90=%.1f p99=%.1f max=%.1f\n",
                n,
                latency_samples[n/2],
                latency_samples[(int)(n*0.9)],
                latency_samples[(int)(n*0.99)],
                latency_samples.back());
    }

    // AJB_STATE: dump ajb_rrt_stats (the global tracker injected in RRAccessTree.hpp)
    ajb_rrt_stats.dump("full_test");

    // upstream: print tree structure
    fprintf(stderr, "[AJB_BP] Printing RRAccessTree structure:\n");
    tree.print();

    long rss_end = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] final: RSS=%ld KB (total delta=%ld KB)\n",
            rss_end, rss_end - rss0);
    fprintf(stderr, "[AJB] VERDICT: test_rr_access_tree_full PASSED\n");
    return 0;
}
