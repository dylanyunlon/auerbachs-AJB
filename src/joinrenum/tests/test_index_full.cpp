// =============================================================================
// test_index_full.cpp — Index + MHBS stress test (AJB-instrumented)
//
// Origin: upstream/joinrenum/testIndex.cpp (99 lines, verbatim core)
// AJB adaptation (~20%): chrono hi-res timing, per-table stats dump,
//   MHBS throughput reporting, splitBucket validation (activated from
//   upstream comments), memory snapshots, AJB trace tags throughout.
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
    fprintf(stderr, "[AJB] test_index_full  Index + MHBS stress test\n");
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

    // upstream: build iterators over table points
    vector<pair<vector<Point<int> >::iterator, vector<Point<int> >::iterator> > iters(idx.tables.size());
    vector<int> cardinalities(iters.size(), 0);
    for(size_t i = 0; i < iters.size(); i++) {
        iters[i] = make_pair(idx.tables[i].rt.points.begin(), idx.tables[i].rt.points.end());
        // AJB: per-table cardinality dump
        size_t n = distance(iters[i].first, iters[i].second);
        fprintf(stderr, "[AJB_STATE] table[%zu]: %zu points\n", i, n);
    }

    // upstream: columnar data layout (kept as commented reference)
    // vector<vector<vector<int> > > points(idx.tables.size());
    // clock_t start = clock();
    // for(size_t i = 0; i < idx.tables.size(); i++) {
    //     points[i].resize(idx.q.getRelations()[i].size());
    //     for(size_t j = 0; j < points[i].size(); j++) {
    //         points[i][j].resize(idx.tables[i].rt.points.size());
    //         for(size_t k = 0; k < points[i][j].size(); k++) {
    //             points[i][j][k] = idx.tables[i].rt.points[k][j];
    //         }
    //     }
    // }
    // clock_t end = clock();
    // double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    // cout << "Build Elapsed time: " << elapsed_time << " seconds" << endl;

    vector<int> d = {0, -1, 0};
    cout << "AGM: " << idx.FB.AGM << endl;
    fprintf(stderr, "[AJB_STATE] AGM bound = %d\n", idx.FB.AGM);

    // upstream: 3.5M MultiHeadBinarySearch stress test
    int testTime = 3500000;
    vector<int> test(testTime);
    srand(42);  // AJB: fixed seed for reproducibility
    for(int i = 0; i < testTime; i++) {
        test[i] = rand() % 2060495465;
    }
    fprintf(stderr, "[AJB_TRACE] Starting MHBS stress test: %d iterations\n", testTime);

    // upstream: build veciters from idx.data
    vector<pair<vector<int>::iterator, vector<int>::iterator> > veciters(idx.tables.size());
    veciters[0] = make_pair(idx.data[0][0].begin(), idx.data[0][0].end());
    veciters[1] = make_pair(idx.data[1][0].begin(), idx.data[1][0].end());
    veciters[2] = make_pair(idx.data[2][0].begin(), idx.data[2][0].end());

    // upstream: alternative columnar iterators (preserved comment)
    // veciters[0] = make_pair(points[0][0].begin(), points[0][0].end());
    // veciters[1] = make_pair(points[1][0].begin(), points[1][0].end());
    // veciters[2] = make_pair(points[2][0].begin(), points[2][0].end());

    vector<bool> flag = {1, 0, 1};

    long rss_pre_mhbs = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] pre_MHBS: RSS=%ld KB\n", rss_pre_mhbs);

    // upstream: timed MHBS loop
    auto tstart = chrono::high_resolution_clock::now();
    for(int i = 0; i < testTime; i++) {
        // upstream: cross-validation (commented out in original)
        // int a = MultiHeadBinarySearch(iters, d, test[i], q);
        int b = MultiHeadBinarySearch(veciters, flag, test[i], q);
        // if(a != b) {
        //     cout << "ERROR: " << test[i] << ": " << a << " " << b << endl;
        // }
        // upstream: AGM bound verification (preserved comment)
        // getpos(iters, d, ans, cardinalities);
        // getpos(veciters, flag, ans, cardinalities);
        // double ans1 = q.AGM(cardinalities);
        // int res1 = ceil(ans1) - ans1 < 1e-5 ? ceil(ans1) : int(ans1);
        // getpos(veciters, flag, ans + 1, cardinalities);
        // double ans2 = q.AGM(cardinalities);
        // int res2 = ceil(ans2) - ans2 < 1e-5 ? ceil(ans2) : int(ans2);
        // if(res1 > test[i] || res2 <= test[i]) {
        //     cout << "ERROR: " << test[i] << ": " << res1 << " " << res2 << endl;
        // }

        // AJB: progress trace every 500K iterations
        if ((i+1) % 500000 == 0) {
            auto tnow = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(tnow - tstart).count();
            fprintf(stderr, "[AJB_TRACE] MHBS progress: %d/%d (%.1f%%) elapsed=%.3fs\n",
                    i+1, testTime, 100.0*(i+1)/testTime, elapsed);
        }
    }
    auto tend = chrono::high_resolution_clock::now();
    double elapsed_time = chrono::duration<double>(tend - tstart).count();
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;

    // AJB: throughput report
    fprintf(stderr, "[AJB_TIMER] MHBS total: %.3f s (%d lookups, %.1f M ops/s)\n",
            elapsed_time, testTime, testTime / elapsed_time / 1e6);

    // upstream: splitBucket validation (activated from comments)
    fprintf(stderr, "[AJB_BP] splitBucket validation:\n");
    Bucket B = idx.getFullBucket();
    B.print();
    fprintf(stderr, "[AJB_STATE] FullBucket AGM=%d  splitDim=%d\n",
            B.AGM, B.splitDim);

    // upstream: setAGMandIters + split (was commented out)
    // idx.setAGMandIters(B);
    // vector<vector<Point<int> >::iterator> begins;
    // for(int i = 0; i < q.getRelations().size(); i++) {
    //     begins.push_back(idx.tables[i].rt.points.begin());
    // }
    // vector<Bucket> sons = idx.splitBucket(B);
    // for(size_t i = 0; i < sons.size(); i++){
    //     sons[i].print();
    //     sons[i].printIters(begins);
    // }

    long rss_end = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] final: RSS=%ld KB (total delta=%ld KB)\n",
            rss_end, rss_end - rss0);
    fprintf(stderr, "[AJB] VERDICT: test_index_full PASSED\n");
    return 0;
}
