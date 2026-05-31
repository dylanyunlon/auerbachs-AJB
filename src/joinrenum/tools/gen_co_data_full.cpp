// =============================================================================
// gen_co_data_full.cpp  AJB-adapted CountOracle data generator
//
// Origin: upstream/joinrenum/genCOData.cpp (59 lines)
// AJB adaptation (~20%): parameterized CLI, structured memory tracking,
//   progress output, data distribution analysis, and output validation.
//
// Build: g++ -O3 gen_co_data_full.cpp -lglpk -o gen_co_full
// Usage: ./gen_co_full [num_points] [dim] [range]
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
#include "CountOracle.hpp"
using namespace std;

// upstream: memory usage (preserved, wrapped)
long ajbMemoryKB() {
    ifstream file("/proc/self/status");
    string line;
    while (getline(file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            istringstream iss(line); string k; long v; iss >> k >> v;
            return v;
        }
    }
    return -1;
}

// upstream: write data to file (preserved)
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

int main(int argc, char* argv[]) {
    printf("[AJB] ============================================\n");
    printf("[AJB] gen_co_data_full  CountOracle data gen\n");
    printf("[AJB] ============================================\n");

    // AJB: parameterized (upstream hardcoded)
    int n = (argc > 1) ? atoi(argv[1]) : 10000;
    int dim = (argc > 2) ? atoi(argv[2]) : 2;
    int range = (argc > 3) ? atoi(argv[3]) : 100;

    printf("[AJB_STATE] Params: n=%d dim=%d range=%d\n", n, dim, range);

    long mem0 = ajbMemoryKB();
    printf("[AJB_MEM] startup: %ld KB\n", mem0);

    // upstream: generate random points
    srand(42);
    auto t0 = chrono::high_resolution_clock::now();

    vector<Point<int>> points;
    points.reserve(n);
    for (int i = 0; i < n; i++) {
        vector<int> coords;
        for (int d = 0; d < dim; d++)
            coords.push_back(rand() % range);
        points.push_back(Point<int>(coords));

        // AJB: progress every 25%
        if (n >= 100 && (i + 1) % (n / 4) == 0)
            printf("[AJB_TRACE] generated %d / %d points\n", i + 1, n);
    }

    auto t1 = chrono::high_resolution_clock::now();
    double gen_ms = chrono::duration<double,milli>(t1 - t0).count();
    printf("[AJB_TIMER] point generation: %.3f ms\n", gen_ms);

    // AJB: distribution analysis
    vector<double> coord_means(dim, 0.0);
    vector<int> coord_min(dim, INT_MAX), coord_max(dim, INT_MIN);
    for (auto& p : points) {
        for (int d = 0; d < dim; d++) {
            coord_means[d] += p[d];
            coord_min[d] = min(coord_min[d], p[d]);
            coord_max[d] = max(coord_max[d], p[d]);
        }
    }
    printf("[AJB_STATE] Distribution per dimension:\n");
    for (int d = 0; d < dim; d++) {
        coord_means[d] /= n;
        printf("[AJB_STATE]   dim[%d]: mean=%.1f min=%d max=%d\n",
               d, coord_means[d], coord_min[d], coord_max[d]);
    }

    // upstream: build CountOracle
    auto t2 = chrono::high_resolution_clock::now();
    CountOracle<int> co(points);
    auto t3 = chrono::high_resolution_clock::now();
    double build_ms = chrono::duration<double,milli>(t3 - t2).count();
    printf("[AJB_TIMER] CountOracle build: %.3f ms\n", build_ms);

    long mem1 = ajbMemoryKB();
    printf("[AJB_MEM] after CO build: %ld KB (delta=%ld KB)\n",
           mem1, mem1 - mem0);

    // upstream: write data
    string outfile = "data.txt";
    writeDataToFile(points, outfile);
    printf("[AJB_STATE] Written %zu points to %s\n", points.size(), outfile.c_str());

    // AJB: validation query
    int full_count = co.getCount(
        vector<int>(dim, 0), vector<int>(dim, range));
    printf("[AJB_STATE] Validation: full-range count=%d (expect %d) -> %s\n",
           full_count, n, full_count == n ? "OK" : "MISMATCH");

    long mem2 = ajbMemoryKB();
    printf("[AJB_MEM] final: %ld KB\n", mem2);
    printf("[AJB] VERDICT: gen_co_data_full %s\n",
           full_count == n ? "PASSED" : "CHECK");
    return 0;
}
