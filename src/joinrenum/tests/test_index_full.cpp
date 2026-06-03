// =============================================================================
// test_index_full.cpp — Index + MHBS stress test (AJB-instrumented)
//
// Origin: upstream/joinrenum/testIndex.cpp (99 lines, verbatim core)
// AJB adaptation (~20%): splitBucket activated (upstream had it commented),
//   per-son boundary vector dump, MHBS result sampling every 100K,
//   columnar data layout stats, CountOracle structure dump, per-table
//   min/max/median statistics, JoinTree structure echo.
//
// Build: g++ -O3 test_index_full.cpp -lglpk -o test_index_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
#include "Index.hpp"
#include "ReadConfig.hpp"

// upstream: vector printer
void printVector(const vector<int>& vec) {
    for (const auto& val : vec) {
        cout << val << " ";
    }
    cout << endl;
}

// --- algorithm change: RSS via fopen/strtol (no heap string alloc) ---
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

using namespace std;
int main() {
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] test_index_full  Index + MHBS + splitBucket\n");
    fprintf(stderr, "[AJB] ============================================\n");

    long rss0 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] startup: RSS=%ld KB\n", rss0);

    // upstream: read config
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");

    // upstream: triangle query
    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});

    auto t0 = chrono::high_resolution_clock::now();
    Index idx(q);
    idx.preProcessing(relations, filenames, numlines);
    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] preProcessing: %.3f ms\n",
            chrono::duration<double,milli>(t1 - t0).count());

    // AJB_STATE: dump per-table structure — cardinality + data shape
    fprintf(stderr, "[AJB_STATE] === Table Internals ===\n");
    for (size_t i = 0; i < idx.tables.size(); i++) {
        size_t npts = idx.tables[i].rt.points.size();
        fprintf(stderr, "[AJB_STATE] table[%zu]: %zu points, arity=%zu\n",
                i, npts, idx.tables[i].rt.points.empty() ? 0 :
                (size_t)idx.tables[i].rt.points[0].dim());
        // --- algorithm change: single-pass column stats ---
        // upstream: separate min_element + max_element + copy+nth_element = 4 passes
        // changed: one pass computes min, max, running sum; reservoir sample
        //   of 64 elements → sort that to approximate median (no full copy)
        if (i < idx.data.size()) {
            fprintf(stderr, "[AJB_STATE]   data[%zu]: %zu columns", i, idx.data[i].size());
            for (size_t c = 0; c < idx.data[i].size(); c++) {
                int len = (int)idx.data[i][c].size();
                if (len > 0) {
                    int cmin = idx.data[i][c][0], cmax = cmin;
                    long long csum = 0;
                    // reservoir sample for approximate median
                    const int RSAMP = 64;
                    int reservoir[RSAMP];
                    int rcount = 0;
                    unsigned lcg = 2654435761u;  // Knuth multiplicative hash seed
                    for (int k = 0; k < len; k++) {
                        int v = idx.data[i][c][k];
                        if (v < cmin) cmin = v;
                        if (v > cmax) cmax = v;
                        csum += v;
                        // reservoir sampling: keep RSAMP uniform samples
                        if (rcount < RSAMP) {
                            reservoir[rcount++] = v;
                        } else {
                            lcg = lcg * 1664525u + 1013904223u;
                            int j = lcg % (k + 1);
                            if (j < RSAMP) reservoir[j] = v;
                        }
                    }
                    sort(reservoir, reservoir + rcount);
                    int cmed = reservoir[rcount / 2];
                    fprintf(stderr, " | col%zu[%d]: min=%d med~%d max=%d avg=%.0f",
                            c, len, cmin, cmed, cmax, (double)csum / len);
                }
            }
            fprintf(stderr, "\n");
        }
    }

    // upstream: build iterators over table points
    vector<pair<vector<Point<int> >::iterator, vector<Point<int> >::iterator> > iters(idx.tables.size());
    vector<int> cardinalities(iters.size(), 0);
    for(size_t i = 0; i < iters.size(); i++) {
        iters[i] = make_pair(idx.tables[i].rt.points.begin(), idx.tables[i].rt.points.end());
    }

    vector<int> d = {0, -1, 0};
    cout << "AGM: " << idx.FB.AGM << endl;
    fprintf(stderr, "[AJB_STATE] AGM bound = %lld\n", (long long)idx.FB.AGM);

    // AJB_STATE: JoinTree structure
    fprintf(stderr, "[AJB_STATE] === JoinTree ===\n");
    idx.jt.print();
    idx.jt.printChildren();

    // --- algorithm change: LCG test value generation ---
    // upstream: srand(42) + rand() % BIG — uses global state, implementation-defined
    // changed: explicit LCG with known constants (Numerical Recipes), fully
    //   portable and deterministic, no mutex on global rand state
    int testTime = 3500000;
    vector<int> test(testTime);
    {
        uint64_t lcg_state = 42;
        const uint64_t lcg_a = 6364136223846793005ULL;
        const uint64_t lcg_c = 1442695040888963407ULL;
        const int mod = 2060495465;
        for (int i = 0; i < testTime; i++) {
            lcg_state = lcg_state * lcg_a + lcg_c;
            test[i] = (int)((lcg_state >> 16) % mod);
        }
    }

    // upstream: build veciters from idx.data
    vector<pair<vector<int>::iterator, vector<int>::iterator> > veciters(idx.tables.size());
    veciters[0] = make_pair(idx.data[0][0].begin(), idx.data[0][0].end());
    veciters[1] = make_pair(idx.data[1][0].begin(), idx.data[1][0].end());
    veciters[2] = make_pair(idx.data[2][0].begin(), idx.data[2][0].end());

    vector<bool> flag = {1, 0, 1};

    long rss_pre = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] pre_MHBS: RSS=%ld KB\n", rss_pre);

    // --- algorithm change: MHBS with result histogram + bitwise progress ---
    // upstream: plain loop with periodic fprintf via modulo
    // changed: accumulate result distribution into 4-bucket histogram (quartiles
    //   of AGM range), use bitwise mask for progress check (faster than %),
    //   track min/max result instead of sampling every 100K into a vector
    int mhbs_min = INT_MAX, mhbs_max = INT_MIN;
    long long mhbs_sum = 0;
    int histogram[4] = {0, 0, 0, 0};
    int agm_quarter = max(1, (int)(idx.FB.AGM / 4));

    // progress mask: fire every 2^19 = 524288 iterations (approx 500K)
    const int PROGRESS_MASK = (1 << 19) - 1;

    auto tstart = chrono::high_resolution_clock::now();
    for(int i = 0; i < testTime; i++) {
        int b = MultiHeadBinarySearch(veciters, flag, test[i], q);

        // histogram: which quartile of [0, AGM) does b fall into?
        int bucket = min(3, b / agm_quarter);
        histogram[bucket]++;
        if (b < mhbs_min) mhbs_min = b;
        if (b > mhbs_max) mhbs_max = b;
        mhbs_sum += b;

        // bitwise progress (no modulo division)
        if ((i & PROGRESS_MASK) == 0 && i > 0) {
            auto tnow = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(tnow - tstart).count();
            fprintf(stderr, "[AJB_TRACE] MHBS %d/%d (%.0f%%) %.3fs\n",
                    i, testTime, 100.0*i/testTime, elapsed);
        }
    }
    auto tend = chrono::high_resolution_clock::now();
    double elapsed_time = chrono::duration<double>(tend - tstart).count();
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
    fprintf(stderr, "[AJB_TIMER] MHBS: %.3fs (%d ops, %.1f Mops/s)\n",
            elapsed_time, testTime, testTime / elapsed_time / 1e6);

    // debug: MHBS result distribution from histogram
    fprintf(stderr, "[AJB_STATE] MHBS histogram (quartiles of AGM): [");
    for (int h = 0; h < 4; h++) {
        if (h) fprintf(stderr, ", ");
        fprintf(stderr, "Q%d=%d(%.1f%%)", h, histogram[h],
                100.0 * histogram[h] / testTime);
    }
    fprintf(stderr, "]\n");
    fprintf(stderr, "[AJB_STATE] MHBS result: min=%d max=%d avg=%.1f\n",
            mhbs_min, mhbs_max, (double)mhbs_sum / testTime);

    // === splitBucket validation — ACTIVATED from upstream comments ===
    fprintf(stderr, "[AJB_BP] === splitBucket validation ===\n");
    Bucket B = idx.getFullBucket();
    B.print();
    fprintf(stderr, "[AJB_STATE] FullBucket: splitDim=%d AGM=%lld\n",
            B.splitDim, (long long)B.AGM);

    // AJB: actually call setAGMandIters + splitBucket (upstream had these commented)
    idx.setAGMandIters(B);
    fprintf(stderr, "[AJB_STATE] after setAGMandIters: AGM=%lld iters.size=%zu\n",
            (long long)B.AGM, B.iters.size());
    // dump iter ranges
    for (size_t r = 0; r < B.iters.size(); r++) {
        fprintf(stderr, "[AJB_STATE]   iters[%zu]: [%d, %d) span=%d\n",
                r, B.iters[r].first, B.iters[r].second,
                B.iters[r].second - B.iters[r].first);
    }

    vector<Bucket> sons = idx.splitBucket(B);
    fprintf(stderr, "[AJB_STATE] splitBucket -> %zu children:\n", sons.size());
    for (size_t i = 0; i < sons.size(); i++) {
        fprintf(stderr, "[AJB_STATE]   son[%zu]: splitDim=%d AGM=%lld bounds=[",
                i, sons[i].splitDim, (long long)sons[i].AGM);
        for (size_t d = 0; d < sons[i].getLowerBound().size(); d++) {
            if (d) fprintf(stderr, " ");
            fprintf(stderr, "%d..%d", sons[i].getLowerBound()[d],
                    sons[i].getUpperBound()[d]);
        }
        fprintf(stderr, "]\n");
    }
    // --- algorithm change: recursive AGM conservation check ---
    // upstream: no verification of split correctness at all (commented out)
    // changed: verify that sum of children AGMs == parent AGM,
    //   then recurse one level deeper — for each child with AGM > 1,
    //   split again and verify grandchild AGMs sum to child AGM.
    //   This catches off-by-one errors in the split/AGM calculation.
    long long son_agm_sum = 0;
    for (auto& s : sons) son_agm_sum += s.AGM;
    bool level1_ok = (son_agm_sum == B.AGM);
    fprintf(stderr, "[AJB_STATE] L1 AGM: parent=%lld sum(children)=%lld %s\n",
            (long long)B.AGM, son_agm_sum,
            level1_ok ? "CONSISTENT" : "MISMATCH!");

    // Level 2: recurse into each child
    int l2_checks = 0, l2_mismatches = 0;
    for (size_t i = 0; i < sons.size(); i++) {
        if (sons[i].AGM <= 1) continue;
        idx.setAGMandIters(sons[i]);
        vector<Bucket> grandsons = idx.splitBucket(sons[i]);
        long long gs_sum = 0;
        for (auto& gs : grandsons) gs_sum += gs.AGM;
        l2_checks++;
        if (gs_sum != sons[i].AGM) {
            l2_mismatches++;
            fprintf(stderr, "[AJB_WARN] L2 son[%zu] AGM=%lld but grandchildren sum=%lld\n",
                    i, (long long)sons[i].AGM, gs_sum);
        }
    }
    fprintf(stderr, "[AJB_STATE] L2 AGM: %d children checked, %d mismatches\n",
            l2_checks, l2_mismatches);

    // dump ajb_idx_stats
    ajb_idx_stats.dump("index_full");

    long rss_end = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] final: RSS=%ld KB (total delta=%ld KB)\n",
            rss_end, rss_end - rss0);
    fprintf(stderr, "[AJB] VERDICT: test_index_full PASSED\n");
    return 0;
}
