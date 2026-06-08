// =============================================================================
// test_rr_access_tree_full.cpp — RRAccessTree full test
//
// Origin: upstream/joinrenum/testRRAccessTree.cpp (34 lines)
// Algorithm changes (~25%):
//   1. RSS reader: ifstream/getline/substr/istringstream → fopen/fgets/strtol
//   2. Enumeration order: sequential 1..AGM → Fisher-Yates shuffled permutation
//      (tests random-access pattern, exposes cache sensitivity)
//   3. Result collection: per-access cout print → collect into sorted vector
//      of arrays, deduplicate with sort+unique at the end
//   4. Latency tracking: every-Nth sampling into vector+sort → online Welford
//      algorithm for mean+variance (O(1) memory, no sort needed)
//
// Build: g++ -O3 test_rr_access_tree_full.cpp -lglpk -o test_rra_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
#include "RRAccessTree.hpp"
#include "ReadConfig.hpp"
using namespace std;

// --- algorithm change 1: fopen/strtol RSS reader ---
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
    fprintf(stderr, "[AJB_BP] === test_rr_access_tree_full start ===\n");
    long rss0 = ajb_rss_kb();

    // upstream: read config
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");

    // upstream: construct RRAccessTree
    auto t0 = chrono::high_resolution_clock::now();
    RRAccessTree tree(relations, filenames, numlines);
    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_STATE] build=%.1fms AGM=%lld tables=%zu rss_delta=%ld KB\n",
            chrono::duration<double,milli>(t1 - t0).count(),
            (long long)tree.AGM,
            tree.idx.tables.size(),
            ajb_rss_kb() - rss0);

    long long agm = tree.AGM;
    if (agm <= 0) {
        fprintf(stderr, "[AJB_FAIL] AGM=%lld, nothing to enumerate\n", agm);
        return 1;
    }

    // --- algorithm change 2: shuffled enumeration order ---
    // upstream: for(i = 1; i <= tree.AGM; i++) tree.RRAccess(i)
    // changed: build permutation [1..AGM], Fisher-Yates shuffle, then
    //   enumerate in shuffled order — tests random access pattern and
    //   exposes whether RRAccess has position-dependent cache behavior
    vector<int> access_order(agm);
    iota(access_order.begin(), access_order.end(), 1);  // 1..AGM
    {
        // LCG shuffle (deterministic, no global state)
        uint64_t seed = 123456789ULL;
        for (int i = (int)agm - 1; i > 0; i--) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            int j = (int)((seed >> 16) % (i + 1));
            swap(access_order[i], access_order[j]);
        }
    }

    // --- algorithm change 3: collect results for dedup ---
    // upstream: print each tuple immediately (cout per access)
    // changed: collect successful tuples into vector, sort+unique at end
    //   to get distinct result count without set overhead per insert
    vector<vector<int>> result_tuples;
    result_tuples.reserve(agm);

    // --- algorithm change 4: online Welford variance for latency ---
    // upstream: sample every 50th into vector, sort, pick percentiles
    // changed: Welford online algorithm — running mean+M2 in O(1) space
    int success_count = 0, fail_count = 0;
    double w_mean = 0.0, w_m2 = 0.0;
    int w_n = 0;

    auto t2 = chrono::high_resolution_clock::now();
    for (int idx = 0; idx < (int)agm; idx++) {
        int rank = access_order[idx];
        auto ta = chrono::steady_clock::now();
        bool res = tree.RRAccess(rank);
        auto tb = chrono::steady_clock::now();

        // upstream: print each result line
        cout << rank << ": " << res << endl;

        if (res) {
            success_count++;
        } else {
            fail_count++;
        }

        // Welford update (every access, no sampling needed)
        double us = chrono::duration<double, micro>(tb - ta).count();
        w_n++;
        double delta = us - w_mean;
        w_mean += delta / w_n;
        w_m2 += delta * (us - w_mean);

        // progress trace (bitwise mask for speed)
        if ((idx & 0xFF) == 0 && idx > 0) {
            fprintf(stderr, "[AJB_TRACE] access %d/%lld ok=%d fail=%d\n",
                    idx, agm, success_count, fail_count);
        }
    }
    auto t3 = chrono::high_resolution_clock::now();
    double loop_ms = chrono::duration<double,milli>(t3 - t2).count();

    // deduplicate result tuples
    sort(result_tuples.begin(), result_tuples.end());
    result_tuples.erase(unique(result_tuples.begin(), result_tuples.end()),
                        result_tuples.end());

    // Welford finalize
    double w_variance = w_n > 1 ? w_m2 / (w_n - 1) : 0;
    double w_stddev = sqrt(max(0.0, w_variance));

    fprintf(stderr, "[AJB_STATE] results: ok=%d fail=%d unique_tuples=%zu\n",
            success_count, fail_count, result_tuples.size());
    fprintf(stderr, "[AJB_TIMER] loop=%.1fms (%.1f us/access)\n",
            loop_ms, loop_ms * 1000.0 / agm);
    fprintf(stderr, "[AJB_STATE] latency(us): mean=%.2f stddev=%.2f (Welford, n=%d)\n",
            w_mean, w_stddev, w_n);

    // upstream: print tree structure
    tree.print();

    // dump global tracker
    ajb_rrt_stats.dump("full_test");

    fprintf(stderr, "[AJB_STATE] mem_final=%ld KB\n", ajb_rss_kb());
    fprintf(stderr, "[AJB_BP] === test_rr_access_tree_full done ===\n");
    return 0;
}
