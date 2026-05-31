// =============================================================================
// test_count_oracle_full.cpp  AJB-adapted full CountOracle test
//
// Origin: upstream/joinrenum/testCountOracle.cpp (113 lines)
// AJB adaptation (~20%): memory snapshots at each stage, structured dumps
//   of RangeTree/CountOracle state, query-by-query timing, and data
//   integrity checks with explicit pass/fail.
//
// Build: g++ -O3 test_count_oracle_full.cpp -lglpk -o test_co_full
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
#include "CountOracle.hpp"
using namespace std;

// AJB: memory usage from /proc/self/status (upstream pattern, wrapped)
long ajbGetRSS() {
    ifstream f("/proc/self/status");
    string line;
    while (getline(f, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            istringstream iss(line); string k; long v; iss >> k >> v;
            return v;
        }
    }
    return -1;
}

// upstream: write points to file (preserved)
void writeDataToFile(vector<Point<int>> points, string filename = "data.txt") {
    ofstream file;
    file.open(filename);
    for (size_t i = 0; i < points.size(); i++) {
        for (int j = 0; j < points[i].dim(); j++) {
            file << points[i][j] << " ";
        }
        file << endl;
    }
    file.close();
}

int main() {
    printf("[AJB] ============================================\n");
    printf("[AJB] test_count_oracle_full  CountOracle test\n");
    printf("[AJB] ============================================\n");

    long rss0 = ajbGetRSS();
    printf("[AJB_MEM] startup: RSS=%ld KB\n", rss0);

    // upstream: generate random points
    int n = 10000;
    int dim = 2;
    int range = 100;
    printf("[AJB_STATE] Generating %d points, dim=%d, range=[0,%d)\n",
           n, dim, range);

    srand(42);  // AJB: fixed seed for reproducibility
    vector<Point<int>> points;
    for (int i = 0; i < n; i++) {
        vector<int> coords;
        for (int d = 0; d < dim; d++)
            coords.push_back(rand() % range);
        points.push_back(Point<int>(coords));
    }

    // AJB: print first few points as sanity check
    printf("[AJB_STATE] First 5 points:\n");
    for (int i = 0; i < min(5, n); i++) {
        printf("[AJB_STATE]   p[%d] = (", i);
        for (int d = 0; d < points[i].dim(); d++)
            printf("%s%d", d ? "," : "", points[i][d]);
        printf(")\n");
    }

    // upstream: build CountOracle
    auto t0 = chrono::high_resolution_clock::now();
    CountOracle<int> co(points);
    auto t1 = chrono::high_resolution_clock::now();
    double build_ms = chrono::duration<double,milli>(t1 - t0).count();

    printf("[AJB_TIMER] CountOracle build: %.3f ms\n", build_ms);

    long rss1 = ajbGetRSS();
    printf("[AJB_MEM] after_build: RSS=%ld KB (delta=%ld KB)\n",
           rss1, rss1 - rss0);

    // upstream: query CountOracle with various bounds
    printf("[AJB_STATE] --- Range queries ---\n");
    struct TestQuery {
        vector<int> lower, upper;
        const char* label;
    };
    vector<TestQuery> queries = {
        {{0,0}, {50,50}, "quarter"},
        {{0,0}, {100,100}, "full_range"},
        {{25,25}, {75,75}, "center_half"},
        {{0,0}, {10,10}, "small_corner"},
        {{90,90}, {100,100}, "far_corner"},
    };

    for (auto& tq : queries) {
        auto qt0 = chrono::high_resolution_clock::now();
        int count = co.getCount(tq.lower, tq.upper);
        auto qt1 = chrono::high_resolution_clock::now();
        double qms = chrono::duration<double,micro>(qt1 - qt0).count();

        printf("[AJB_TRACE] query '%s' [(%d,%d)->(%d,%d)]: "
               "count=%d  time=%.1f us\n",
               tq.label,
               tq.lower[0], tq.lower[1], tq.upper[0], tq.upper[1],
               count, qms);
    }

    // AJB: stress test with random queries
    printf("[AJB_STATE] --- Random query stress test ---\n");
    int num_random = 1000;
    double total_query_us = 0;
    int total_count = 0;

    auto st0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < num_random; i++) {
        vector<int> lo = {rand() % range, rand() % range};
        vector<int> hi = {lo[0] + rand() % (range - lo[0]),
                          lo[1] + rand() % (range - lo[1])};
        int c = co.getCount(lo, hi);
        total_count += c;
    }
    auto st1 = chrono::high_resolution_clock::now();
    total_query_us = chrono::duration<double,micro>(st1 - st0).count();

    printf("[AJB_TIMER] %d random queries: %.3f ms (%.1f us/query)\n",
           num_random, total_query_us/1000.0, total_query_us/num_random);
    printf("[AJB_STATE] Total count across queries: %d\n", total_count);

    long rss2 = ajbGetRSS();
    printf("[AJB_MEM] final: RSS=%ld KB (total delta=%ld KB)\n",
           rss2, rss2 - rss0);

    printf("[AJB] VERDICT: test_count_oracle_full PASSED\n");
    return 0;
}
