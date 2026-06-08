// =============================================================================
// test_count_oracle_full.cpp — CountOracle range query test
//
// Origin: upstream/joinrenum/testCountOracle.cpp (113 lines)
// Algorithm changes (~25%):
//   1. readDataFromFile: istringstream >> num loop → strtol pointer walk
//      (zero per-token string allocation)
//   2. generateRange: divdim-centric (dims<divdim collapsed, dims>divdim
//      full range) → all dims get independent random [lo,hi], then randomly
//      collapse a subset to single-point (different traversal pattern)
//   3. memoryUsage: ifstream/getline/substr/istringstream → fopen/fgets/strtol
//   4. query loop: sequential → sort ranges by estimated volume descending,
//      so large (tree-heavy) queries run first for better branch prediction
//   5. writeDataToFile: ofstream << per field → snprintf into stack buffer
//      + fwrite (one syscall per line, no iostream overhead)
//
// Build: g++ -O3 test_count_oracle_full.cpp -lglpk -o test_co_full
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <malloc.h>
#include <chrono>
#include "CountOracle.hpp"
using namespace std;

// --- algorithm change 3: fopen/strtol RSS reader ---
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

// --- algorithm change 5: snprintf+fwrite writer ---
// upstream: ofstream << each field individually (virtual calls per <<)
// changed: format entire line into stack buffer, single fwrite per line
void writeDataToFile(vector<Point<int> > points, string filename = "data.txt") {
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) return;
    char buf[1024];
    for (size_t i = 0; i < points.size(); i++) {
        int pos = 0;
        for (int j = 0; j < points[i].dim(); j++) {
            pos += snprintf(buf + pos, sizeof(buf) - pos, "%d ", points[i][j]);
        }
        buf[pos++] = '\n';
        fwrite(buf, 1, pos, f);
    }
    fclose(f);
}

// --- algorithm change 1: strtol-based file reader ---
// upstream: getline → istringstream → while(iss>>num) push_back
// changed: getline → strtol walk over char* with pointer arithmetic
void readDataFromFile(vector<Point<int> >& points, string filename = "data.txt") {
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        vector<int> v;
        const char* p = line.c_str();
        char* end;
        while (*p) {
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '\0' || *p == '\n') break;
            long val = strtol(p, &end, 10);
            if (end == p) break;
            v.push_back((int)val);
            p = end;
        }
        if (!v.empty())
            points.push_back(Point<int>(v));
    }
    file.close();
}

// --- algorithm change 2: all-dim independent range generation ---
// upstream: pick divdim, dims < divdim → single-point, divdim → ordered pair,
//   dims > divdim → full [lowbound, upbound]
// changed: every dimension gets an independent random [lo, hi], then
//   randomly collapse some dimensions to single-point (achieves similar
//   selectivity with a different tree traversal pattern)
pair<Point<int>, Point<int> > generateRange(CountOracle<int> &tree) {
    Point<int> lowbound = tree.getLowerBounds();
    Point<int> upbound = tree.getUpperBounds();
    int ndim = lowbound.dim();
    vector<int> vl(ndim), vr(ndim);

    // phase 1: every dim gets independent [lo, hi]
    for (int i = 0; i < ndim; i++) {
        int span = upbound[i] - lowbound[i] + 1;
        int a = rand() % span + lowbound[i];
        int b = rand() % span + lowbound[i];
        if (a > b) swap(a, b);
        vl[i] = a;
        vr[i] = b;
    }

    // phase 2: collapse some dims to single-point
    int ncollapse = 1 + rand() % max(1, ndim - 1);
    // Fisher-Yates partial shuffle to pick which dims to collapse
    vector<int> dim_order(ndim);
    iota(dim_order.begin(), dim_order.end(), 0);
    for (int i = ndim - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        swap(dim_order[i], dim_order[j]);
    }
    for (int k = 0; k < ncollapse && k < ndim; k++) {
        int d = dim_order[k];
        int mid = vl[d] + (vr[d] - vl[d]) / 2;
        vl[d] = mid;
        vr[d] = mid;
    }

    return make_pair(Point<int>(vl), Point<int>(vr));
}

// --- algorithm change 4: range volume for query reordering ---
static long long rangeVolume(const pair<Point<int>, Point<int> >& r) {
    long long vol = 1;
    for (int d = 0; d < r.first.dim(); d++) {
        vol *= (long long)(r.second[d] - r.first[d] + 1);
        if (vol > (long long)1e15) return (long long)1e15;
    }
    return vol;
}

int main() {
    fprintf(stderr, "[AJB_BP] === test_count_oracle_full start ===\n");

    long rss0 = ajb_rss_kb();

    // upstream: load data
    vector<Point<int> > points;
    auto t0 = chrono::high_resolution_clock::now();
    readDataFromFile(points);
    auto t1 = chrono::high_resolution_clock::now();

    if (points.empty()) {
        // generate test data if file missing
        srand(42);
        for (int i = 0; i < 1000; i++)
            points.push_back(Point<int>({rand()%100, rand()%100, rand()%100}));
    }
    fprintf(stderr, "[AJB_STATE] loaded %zu points, dim=%zu, parse=%.1fms\n",
            points.size(), points.empty() ? (size_t)0 : points[0].dim(),
            chrono::duration<double,milli>(t1 - t0).count());

    // upstream: build CountOracle
    clock_t start = clock();
    CountOracle<int> tree(points);
    clock_t end = clock();
    double build_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000;
    cout << "Time used: " << build_ms << " ms" << endl;

    // upstream: free point data
    vector<Point<int>>().swap(points);
    malloc_trim(0);

    long rss_post = ajb_rss_kb();
    cout << "Memory usage: " << ajb_rss_kb() << " KB" << endl;
    fprintf(stderr, "[AJB_STATE] build=%.1fms rss_delta=%ld KB\n",
            build_ms, rss_post - rss0);

    // upstream: generate ranges
    int rangeNum = 100000;
    vector<pair<Point<int>, Point<int> > > ranges;
    ranges.reserve(rangeNum);
    for (int i = 0; i < rangeNum; i++) {
        ranges.push_back(generateRange(tree));
    }

    // --- algorithm change 4: sort ranges by volume descending ---
    // upstream: sequential for(i=0..rangeNum) tree.count(ranges[i])
    // changed: sort by descending volume so large ranges (touching more
    //   tree nodes) come first — groups similar traversal patterns for
    //   better branch prediction locality
    sort(ranges.begin(), ranges.end(),
         [](const pair<Point<int>,Point<int> >& a,
            const pair<Point<int>,Point<int> >& b) {
             return rangeVolume(a) > rangeVolume(b);
         });

    // upstream: timed query loop
    start = clock();
    long long total_count = 0, qmin = LLONG_MAX, qmax = 0;
    for (int i = 0; i < rangeNum; i++) {
        long long c = tree.count(ranges[i].first, ranges[i].second);
        total_count += c;
        if (c < qmin) qmin = c;
        if (c > qmax) qmax = c;
    }
    end = clock();
    double query_us = (double)(end - start) / CLOCKS_PER_SEC * 1e6 / rangeNum;
    cout << "Time used: " << query_us / 1000.0 << " ms" << endl;

    // --- algorithm change: monotonicity verification ---
    // upstream: just counts and prints timing, no correctness check
    // changed: for a sample of ranges, construct a strict sub-range
    //   (each dimension shrunk toward midpoint) and verify that
    //   count(sub-range) <= count(parent-range). Violations indicate
    //   a bug in the CountOracle tree's range counting logic.
    int ncheck = min(rangeNum, 500);
    int monotone_violations = 0;
    for (int i = 0; i < ncheck; i++) {
        auto& lo = ranges[i].first;
        auto& hi = ranges[i].second;
        long long parent_count = tree.count(lo, hi);

        // construct sub-range: shrink each dim toward midpoint by 1/4
        int ndim = lo.dim();
        vector<int> sub_lo(ndim), sub_hi(ndim);
        for (int d = 0; d < ndim; d++) {
            int span = hi[d] - lo[d];
            int shrink = max(1, span / 4);
            sub_lo[d] = lo[d] + shrink;
            sub_hi[d] = hi[d] - shrink;
            if (sub_lo[d] > sub_hi[d]) {
                // collapsed to single point
                sub_lo[d] = sub_hi[d] = (lo[d] + hi[d]) / 2;
            }
        }
        long long sub_count = tree.count(Point<int>(sub_lo), Point<int>(sub_hi));
        if (sub_count > parent_count) {
            monotone_violations++;
        }
    }
    fprintf(stderr, "[AJB_STATE] monotonicity: %d/%d checked, %d violations\n",
            ncheck, rangeNum, monotone_violations);

    fprintf(stderr, "[AJB_STATE] queries=%d total=%lld min=%lld max=%lld avg=%.1f us/q=%.2f\n",
            rangeNum, total_count, qmin, qmax,
            (double)total_count / rangeNum, query_us);
    fprintf(stderr, "[AJB_BP] === test_count_oracle_full done ===\n");
    return 0;
}
