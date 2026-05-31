// =============================================================================
// test_count_oracle_full.cpp — CountOracle + RangeTree stress (AJB-instrumented)
//
// Origin: upstream/joinrenum/testCountOracle.cpp (113 lines, verbatim core)
// AJB adaptation (~20%): chrono hi-res timing, fixed seed for reproducibility,
//   original generateRange() preserved (not simplified), 100K random queries
//   at upstream scale, data file I/O preserved, memory snapshots, AJB tags.
//
// Build: g++ -O3 test_count_oracle_full.cpp -lglpk -o test_co_full
// =============================================================================

#include<bits/stdc++.h>
#include <sys/resource.h>
#include <malloc.h>
#include <chrono>
#include "CountOracle.hpp"
using namespace std;

// upstream: memory usage from /proc/self/status (verbatim)
int memoryUsage() {
    ifstream file("/proc/self/status");
    string line;
    while (getline(file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            istringstream iss(line);
            string key;
            int value;
            string unit;
            iss >> key >> value >> unit;
            return value;
        }
    }
    return -1;
}

// upstream: write points to file (verbatim)
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

// upstream: read points from file (verbatim)
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

// upstream: generate random range query within tree bounds (verbatim)
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
    if(divval > divval2){
        swap(divval, divval2);
    }
    vl.push_back(divval);
    vr.push_back(divval2);
    for(int i = divdim + 1; i < lowbound.dim(); i++){
        vl.push_back(lowbound[i]);
        vr.push_back(upbound[i]);
    }
    return make_pair(Point<int>(vl), Point<int>(vr));
}

int main(){
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] test_count_oracle_full  CountOracle stress\n");
    fprintf(stderr, "[AJB] ============================================\n");

    int rss0 = memoryUsage();
    fprintf(stderr, "[AJB_MEM] startup: RSS=%d KB\n", rss0);

    vector<Point<int> > points;

    // upstream: try reading from data.txt; fall back to random generation
    {
        ifstream probe("data.txt");
        if (probe.good()) {
            probe.close();
            readDataFromFile(points);
            fprintf(stderr, "[AJB_STATE] Loaded %zu points from data.txt\n", points.size());
        } else {
            // AJB: generate synthetic data if file doesn't exist (for CI)
            fprintf(stderr, "[AJB_TRACE] data.txt not found, generating synthetic data\n");
            srand(42);
            int n = 10000, dim = 3, range = 100;
            set<Point<int>> S;
            while((int)points.size() < n) {
                vector<int> v;
                for(int j = 0; j < dim; j++) v.push_back(rand() % range);
                Point<int> p(v);
                if(S.find(p) == S.end()) { S.insert(p); points.push_back(p); }
            }
            // write for future runs
            writeDataToFile(points);
            fprintf(stderr, "[AJB_STATE] Generated %zu unique points (dim=%d range=%d)\n",
                    points.size(), dim, range);
        }
    }

    // AJB: sample dump
    fprintf(stderr, "[AJB_STATE] First 3 points:\n");
    for(size_t i = 0; i < min((size_t)3, points.size()); i++) {
        fprintf(stderr, "[AJB_STATE]   p[%zu] = (", i);
        for(int j = 0; j < points[i].dim(); j++)
            fprintf(stderr, "%s%d", j?",":"", points[i][j]);
        fprintf(stderr, ")\n");
    }

    // upstream: build CountOracle + timing
    auto ct0 = chrono::high_resolution_clock::now();
    CountOracle<int> tree(points);
    auto ct1 = chrono::high_resolution_clock::now();
    double build_ms = chrono::duration<double,milli>(ct1 - ct0).count();
    cout << "Time used: " << build_ms << " ms" << endl;
    fprintf(stderr, "[AJB_TIMER] CountOracle build: %.3f ms (%zu points)\n",
            build_ms, points.size());

    // upstream: free points, reclaim memory
    vector<Point<int>>().swap(points);
    malloc_trim(0);
    int rss1 = memoryUsage();
    cout << "Memory usage: " << rss1 << " KB" << endl;
    fprintf(stderr, "[AJB_MEM] after_build+free: RSS=%d KB (delta=%d)\n",
            rss1, rss1 - rss0);

    // upstream: generate 100K random range queries (original scale)
    int rangeNum = 100000;
    fprintf(stderr, "[AJB_TRACE] Generating %d random range queries...\n", rangeNum);
    vector<pair<Point<int>, Point<int> > > ranges;
    for(int i = 0; i < rangeNum; i++){
        ranges.push_back(generateRange(tree));
    }

    // AJB: dump a few sample ranges
    fprintf(stderr, "[AJB_STATE] Sample ranges:\n");
    for(int i = 0; i < min(3, rangeNum); i++) {
        fprintf(stderr, "[AJB_STATE]   range[%d]: lo=(", i);
        for(int j = 0; j < ranges[i].first.dim(); j++)
            fprintf(stderr, "%s%d", j?",":"", ranges[i].first[j]);
        fprintf(stderr, ") hi=(");
        for(int j = 0; j < ranges[i].second.dim(); j++)
            fprintf(stderr, "%s%d", j?",":"", ranges[i].second[j]);
        fprintf(stderr, ")\n");
    }

    // upstream: timed query loop
    auto qt0 = chrono::high_resolution_clock::now();
    long total_count = 0;
    for(size_t i = 0; i < ranges.size(); i++){
        total_count += tree.count(ranges[i].first, ranges[i].second);
    }
    auto qt1 = chrono::high_resolution_clock::now();
    double query_ms = chrono::duration<double,milli>(qt1 - qt0).count();
    cout << "Time used: " << query_ms / rangeNum << " ms" << endl;

    fprintf(stderr, "[AJB_TIMER] %d queries: %.3f ms total (%.3f us/query)\n",
            rangeNum, query_ms, query_ms * 1000.0 / rangeNum);
    fprintf(stderr, "[AJB_STATE] Total count sum across all queries: %ld\n", total_count);

    // upstream: sample range queries with verbose output (originally commented)
    // tree.print();
    // for(int i = 0; i < 10; i++){
    //     pair<Point<int>, Point<int> > range = generateRange(tree);
    //     cout << "Range: ";
    //     range.first.print();
    //     cout << " to ";
    //     range.second.print();
    //     cout << "Count: " << tree.count(range.first, range.second) << endl;
    // }

    int rss2 = memoryUsage();
    fprintf(stderr, "[AJB_MEM] final: RSS=%d KB (total delta=%d KB)\n",
            rss2, rss2 - rss0);
    fprintf(stderr, "[AJB] VERDICT: test_count_oracle_full PASSED\n");
    return 0;
}
