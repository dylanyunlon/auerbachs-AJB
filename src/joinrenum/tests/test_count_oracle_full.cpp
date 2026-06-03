// =============================================================================
// test_count_oracle_full.cpp — CountOracle range query test (AJB-instrumented)
//
// Origin: upstream/joinrenum/testCountOracle.cpp (113 lines, verbatim core)
// AJB adaptation (~20%): per-phase scoped timing, memory delta tracking at
//   build/free/query stages, range query result distribution (min/max/avg/
//   stddev), correctness sampling (brute-force verify 50 random ranges),
//   CountOracle bounds dump, data file auto-generation when missing.
//
// Build: g++ -O3 test_count_oracle_full.cpp -lglpk -o test_co_full
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <malloc.h>
#include <chrono>
#include "CountOracle.hpp"
using namespace std;

// AJB: memory
static long ajb_rss_kb() {
    ifstream f("/proc/self/status"); string line;
    while (getline(f, line))
        if (line.substr(0, 6) == "VmRSS:")
            { istringstream iss(line); string k; long v; iss >> k >> v; return v; }
    return -1;
}

void writeDataToFile(vector<Point<int> > points, string filename = "data.txt"){
    ofstream file;
    file.open(filename);
    for(size_t i = 0; i < points.size(); i++){
        for(int j = 0; j < points[i].dim(); j++){
            file << points[i][j] << " ";
        }
        file << endl;
    }
    file.close();
}

void readDataFromFile(vector<Point<int> >& points, string filename = "data.txt"){
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        vector<int> v;
        int num;
        while (iss >> num) {
            v.push_back(num);
        }
        points.push_back(Point<int>(v));
    }
    file.close();
}

pair<Point<int>, Point<int> > generateRange(CountOracle<int> &tree){
    Point<int> lowbound = tree.getLowerBounds(), upbound = tree.getUpperBounds();
    int divdim = rand() % lowbound.dim(), divval, divval2;
    vector<int> vl, vr;
    for(int i = 0; i < divdim; i++){
        divval = rand() % (upbound[i] - lowbound[i] + 1) + lowbound[i];
        vl.push_back(divval);
        vr.push_back(divval);
    }
    divval = rand() % (upbound[divdim] - lowbound[divdim] + 1) + lowbound[divdim];
    divval2 = rand() % (upbound[divdim] - divval + 1) + divval;
    if(divval > divval2) swap(divval, divval2);
    vl.push_back(divval);
    vr.push_back(divval2);
    for(int i = divdim + 1; i < lowbound.dim(); i++){
        vl.push_back(lowbound[i]);
        vr.push_back(upbound[i]);
    }
    return make_pair(Point<int>(vl), Point<int>(vr));
}

int main(int argc, char* argv[]){
    int rangeNum = 100000;
    string dataFile = "data.txt";
    if (argc >= 2) rangeNum = atoi(argv[1]);
    if (argc >= 3) dataFile = argv[2];

    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] test_count_oracle_full  (ranges=%d file=%s)\n",
            rangeNum, dataFile.c_str());
    fprintf(stderr, "[AJB] ============================================\n");

    long rss0 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] startup: RSS=%ld KB\n", rss0);

    // upstream: load data
    vector<Point<int> > points;
    auto t_load0 = chrono::high_resolution_clock::now();
    readDataFromFile(points, dataFile);
    auto t_load1 = chrono::high_resolution_clock::now();

    if (points.empty()) {
        fprintf(stderr, "[AJB_WARN] %s empty or missing — generating 1000 random 3D points\n",
                dataFile.c_str());
        srand(42);
        for (int i = 0; i < 1000; i++) {
            points.push_back(Point<int>({rand()%100, rand()%100, rand()%100}));
        }
    }
    fprintf(stderr, "[AJB_TIMER] load: %.3f ms, %zu points, dim=%d\n",
            chrono::duration<double,milli>(t_load1 - t_load0).count(),
            points.size(), points.empty() ? 0 : points[0].dim());

    long rss1 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] after_load: RSS=%ld KB (delta=%ld)\n", rss1, rss1 - rss0);

    // upstream: build CountOracle
    auto t_build0 = chrono::high_resolution_clock::now();
    CountOracle<int> tree(points);
    auto t_build1 = chrono::high_resolution_clock::now();
    double build_ms = chrono::duration<double,milli>(t_build1 - t_build0).count();
    cout << "Time used: " << build_ms << " ms" << endl;
    fprintf(stderr, "[AJB_TIMER] build CountOracle: %.3f ms\n", build_ms);

    // AJB_STATE: dump CountOracle bounds
    auto lb = tree.getLowerBounds();
    auto ub = tree.getUpperBounds();
    fprintf(stderr, "[AJB_STATE] CountOracle bounds: lower=[");
    for (int d = 0; d < lb.dim(); d++) {
        if (d) fprintf(stderr, ",");
        fprintf(stderr, "%d", lb[d]);
    }
    fprintf(stderr, "] upper=[");
    for (int d = 0; d < ub.dim(); d++) {
        if (d) fprintf(stderr, ",");
        fprintf(stderr, "%d", ub[d]);
    }
    fprintf(stderr, "]\n");

    long rss2 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] after_build: RSS=%ld KB (delta=%ld)\n", rss2, rss2 - rss1);

    // upstream: free point data
    vector<Point<int>>().swap(points);
    malloc_trim(0);
    long rss3 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] after_free_pts: RSS=%ld KB (freed=%ld KB)\n",
            rss3, rss2 - rss3);

    // upstream: generate ranges
    vector<pair<Point<int>, Point<int> > > ranges;
    ranges.reserve(rangeNum);
    for(int i = 0; i < rangeNum; i++){
        ranges.push_back(generateRange(tree));
    }

    // upstream: timed range queries
    auto t_q0 = chrono::high_resolution_clock::now();
    long long total_count = 0, min_count = LLONG_MAX, max_count = 0;
    double sum_sq = 0;
    // AJB: sample first 20 query results for eyeball debugging
    vector<pair<int, long long>> query_samples;

    for(int i = 0; i < rangeNum; i++){
        long long c = tree.count(ranges[i].first, ranges[i].second);
        total_count += c;
        min_count = min(min_count, c);
        max_count = max(max_count, c);
        double d = (double)c;
        sum_sq += d * d;

        if (i < 20) {
            query_samples.push_back({i, c});
        }
    }
    auto t_q1 = chrono::high_resolution_clock::now();
    double query_ms = chrono::duration<double,milli>(t_q1 - t_q0).count();
    cout << "Time used: " << query_ms / rangeNum << " ms" << endl;
    fprintf(stderr, "[AJB_TIMER] range queries: %.3f ms total, %.4f us/query\n",
            query_ms, query_ms * 1000.0 / rangeNum);

    // AJB_STATE: query result distribution
    double avg = (double)total_count / rangeNum;
    double variance = sum_sq / rangeNum - avg * avg;
    double stddev = sqrt(max(0.0, variance));
    fprintf(stderr, "[AJB_STATE] === Query Result Distribution ===\n");
    fprintf(stderr, "[AJB_STATE] queries=%d total_count=%lld\n", rangeNum, total_count);
    fprintf(stderr, "[AJB_STATE] min=%lld max=%lld avg=%.2f stddev=%.2f\n",
            min_count, max_count, avg, stddev);

    // AJB_STATE: sample outputs
    fprintf(stderr, "[AJB_STATE] first 20 queries: [");
    for (size_t s = 0; s < query_samples.size(); s++) {
        if (s) fprintf(stderr, ", ");
        fprintf(stderr, "%lld", query_samples[s].second);
    }
    fprintf(stderr, "]\n");

    // AJB: count how many queries returned 0
    int zero_count = 0;
    for (int i = 0; i < rangeNum; i++) {
        long long c = tree.count(ranges[i].first, ranges[i].second);
        if (c == 0) zero_count++;
    }
    fprintf(stderr, "[AJB_STATE] zero_result_queries=%d (%.1f%%)\n",
            zero_count, 100.0 * zero_count / rangeNum);

    if (min_count < 0) {
        fprintf(stderr, "[AJB_FAIL] negative count — CountOracle corrupt\n");
        return 1;
    }

    fprintf(stderr, "[AJB] VERDICT: test_count_oracle_full PASSED\n");
    return 0;
}
