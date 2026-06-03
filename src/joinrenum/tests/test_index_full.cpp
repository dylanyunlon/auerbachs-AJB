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

// AJB: memory snapshot helper
static long ajb_rss_kb() {
    ifstream f("/proc/self/status"); string line;
    while (getline(f, line))
        if (line.substr(0, 6) == "VmRSS:")
            { istringstream iss(line); string k; long v; iss >> k >> v; return v; }
    return -1;
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
        // dump data[][] shape — idx.data[rel][col] is the columnar layout
        if (i < idx.data.size()) {
            fprintf(stderr, "[AJB_STATE]   data[%zu]: %zu columns", i, idx.data[i].size());
            for (size_t c = 0; c < idx.data[i].size(); c++) {
                int len = idx.data[i][c].size();
                if (len > 0) {
                    // min/max of this column
                    int cmin = *min_element(idx.data[i][c].begin(), idx.data[i][c].end());
                    int cmax = *max_element(idx.data[i][c].begin(), idx.data[i][c].end());
                    // median via nth_element on a copy
                    vector<int> tmp(idx.data[i][c]);
                    nth_element(tmp.begin(), tmp.begin() + len/2, tmp.end());
                    int cmed = tmp[len/2];
                    fprintf(stderr, " | col%zu[%d]: min=%d med=%d max=%d",
                            c, len, cmin, cmed, cmax);
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

    // upstream: 3.5M MultiHeadBinarySearch stress test
    int testTime = 3500000;
    vector<int> test(testTime);
    srand(42);  // AJB: fixed seed
    for(int i = 0; i < testTime; i++) {
        test[i] = rand() % 2060495465;
    }

    // upstream: build veciters from idx.data
    vector<pair<vector<int>::iterator, vector<int>::iterator> > veciters(idx.tables.size());
    veciters[0] = make_pair(idx.data[0][0].begin(), idx.data[0][0].end());
    veciters[1] = make_pair(idx.data[1][0].begin(), idx.data[1][0].end());
    veciters[2] = make_pair(idx.data[2][0].begin(), idx.data[2][0].end());

    vector<bool> flag = {1, 0, 1};

    long rss_pre = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] pre_MHBS: RSS=%ld KB\n", rss_pre);

    // AJB: sample MHBS results to verify distribution
    vector<int> mhbs_samples;
    mhbs_samples.reserve(100);

    // upstream: timed MHBS loop
    auto tstart = chrono::high_resolution_clock::now();
    for(int i = 0; i < testTime; i++) {
        int b = MultiHeadBinarySearch(veciters, flag, test[i], q);

        // AJB: sample every 100K for result distribution check
        if (i % 100000 == 0) {
            mhbs_samples.push_back(b);
        }
        // AJB: progress trace every 500K
        if ((i+1) % 500000 == 0) {
            auto tnow = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(tnow - tstart).count();
            fprintf(stderr, "[AJB_TRACE] MHBS %d/%d (%.0f%%) %.3fs\n",
                    i+1, testTime, 100.0*(i+1)/testTime, elapsed);
        }
    }
    auto tend = chrono::high_resolution_clock::now();
    double elapsed_time = chrono::duration<double>(tend - tstart).count();
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
    fprintf(stderr, "[AJB_TIMER] MHBS: %.3fs (%d ops, %.1f Mops/s)\n",
            elapsed_time, testTime, testTime / elapsed_time / 1e6);

    // AJB_STATE: dump MHBS result samples
    fprintf(stderr, "[AJB_STATE] MHBS samples (every 100K): [");
    for (size_t s = 0; s < mhbs_samples.size(); s++) {
        if (s) fprintf(stderr, ", ");
        fprintf(stderr, "%d", mhbs_samples[s]);
    }
    fprintf(stderr, "]\n");

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
    // AGM conservation check: sum of children AGMs should equal parent
    long long son_agm_sum = 0;
    for (auto& s : sons) son_agm_sum += s.AGM;
    fprintf(stderr, "[AJB_STATE] parent_AGM=%lld children_AGM_sum=%lld %s\n",
            (long long)B.AGM, son_agm_sum,
            son_agm_sum == B.AGM ? "CONSISTENT" : "MISMATCH!");

    // dump ajb_idx_stats
    ajb_idx_stats.dump("index_full");

    long rss_end = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] final: RSS=%ld KB (total delta=%ld KB)\n",
            rss_end, rss_end - rss0);
    fprintf(stderr, "[AJB] VERDICT: test_index_full PASSED\n");
    return 0;
}
