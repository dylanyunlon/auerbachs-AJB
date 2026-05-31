// =============================================================================
// gen_co_data_full.cpp — CountOracle data generator (AJB-instrumented)
//
// Origin: upstream/joinrenum/genCOData.cpp (59 lines, verbatim core)
// AJB adaptation (~20%): CLI args for n/dim/range, distribution analysis,
//   chrono timing, uniqueness stats, AJB trace tags.
//
// Build: g++ -O3 gen_co_data_full.cpp -lglpk -o gen_co_data_full
// Usage: ./gen_co_data_full [n] [dim] [range]  (defaults: 1000000 10 1000)
// =============================================================================

#include<bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
#include "LexRangeTree.hpp"
using namespace std;

// upstream: memory usage (verbatim)
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

int main(int argc, char** argv){
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] gen_co_data_full  data generator\n");
    fprintf(stderr, "[AJB] ============================================\n");

    // AJB: CLI args (upstream used hardcoded values)
    int n = (argc > 1) ? atoi(argv[1]) : 1000000;
    int dim = (argc > 2) ? atoi(argv[2]) : 10;
    int range = (argc > 3) ? atoi(argv[3]) : 1000;
    fprintf(stderr, "[AJB_STATE] params: n=%d dim=%d range=%d\n", n, dim, range);

    // upstream: generate unique random points
    auto t0 = chrono::high_resolution_clock::now();
    vector<Point<int> > points;
    set<Point<int> > S;
    int attempts = 0;
    while((int)points.size() < n){
        vector<int> v;
        for(int j = 0; j < dim; j++){
            v.push_back(rand() % range);
        }
        attempts++;
        if(S.find(Point<int>(v)) == S.end()){
            S.insert(Point<int>(v));
            points.push_back(Point<int>(v));
        }
        // AJB: progress trace
        if(points.size() % 100000 == 0) {
            fprintf(stderr, "[AJB_TRACE] generated %zu/%d unique (attempts=%d, dup_rate=%.1f%%)\n",
                    points.size(), n, attempts, 100.0*(attempts - points.size())/attempts);
        }
    }
    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] generation: %.3f ms (%d unique from %d attempts)\n",
            chrono::duration<double,milli>(t1 - t0).count(), n, attempts);

    // upstream: write to file
    // sort(points.begin(), points.end());  // upstream: commented out
    auto t2 = chrono::high_resolution_clock::now();
    writeDataToFile(points);
    auto t3 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] write: %.3f ms\n",
            chrono::duration<double,milli>(t3 - t2).count());

    // AJB: distribution analysis
    fprintf(stderr, "[AJB_STATE] --- Distribution analysis (dim 0) ---\n");
    map<int, int> freq;
    for(auto& p : points) freq[p[0]]++;
    int max_freq = 0, min_freq = INT_MAX;
    for(auto& [k,v] : freq) { max_freq = max(max_freq, v); min_freq = min(min_freq, v); }
    fprintf(stderr, "[AJB_STATE]   distinct_vals=%zu  min_freq=%d  max_freq=%d\n",
            freq.size(), min_freq, max_freq);

    fprintf(stderr, "[AJB_MEM] final: RSS=%d KB\n", memoryUsage());
    fprintf(stderr, "[AJB] gen_co_data_full COMPLETE\n");
    return 0;
}
