// =============================================================================
// gen_co_data.cpp — AJB-adapted CountOracle data generator
//
// Origin: upstream/joinrenum/genCOData.cpp (59 lines)
// Adaptation (~20%): AJB configurable dimensions/cardinality via CLI,
//   progress reporting, data distribution summary, and output format
//   compatible with the AJB skew detector's sampling path.
//
// Build: g++ -O3 gen_co_data.cpp -o gen_co
// Usage: ./gen_co [n_points] [n_dims] [max_val] [output_file]
// =============================================================================

#include <bits/stdc++.h>
using namespace std;

// AJB: Note — upstream uses LexRangeTree.hpp here, but that header
// is not present in src/joinrenum. We only need the Point type from
// RangeTree.hpp for file I/O, and the actual generation is pure random.
// So we write raw text (same format) without the RangeTree dependency.

int main(int argc, char* argv[]) {
    // AJB: fully configurable from CLI
    int n_points = 1000000;
    int n_dims   = 10;
    int max_val  = 1000;
    string output = "data.txt";

    if (argc >= 2) n_points = atoi(argv[1]);
    if (argc >= 3) n_dims   = atoi(argv[2]);
    if (argc >= 4) max_val  = atoi(argv[3]);
    if (argc >= 5) output   = argv[4];

    printf("[AJB] gen_co_data: n=%d dims=%d max_val=%d -> %s\n",
           n_points, n_dims, max_val, output.c_str());

    // Generate unique random points
    auto t0 = chrono::high_resolution_clock::now();

    set<vector<int>> seen;
    vector<vector<int>> points;
    points.reserve(n_points);

    int collisions = 0;
    while ((int)points.size() < n_points) {
        vector<int> v(n_dims);
        for (int j = 0; j < n_dims; j++) {
            v[j] = rand() % max_val;
        }
        if (seen.find(v) == seen.end()) {
            seen.insert(v);
            points.push_back(v);
        } else {
            collisions++;
        }

        // AJB: progress report every 10%
        if (points.size() % (n_points / 10 + 1) == 0) {
            printf("[AJB] progress: %zu/%d points (%.0f%%), %d collisions\n",
                   points.size(), n_points,
                   100.0 * points.size() / n_points, collisions);
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double gen_time = chrono::duration<double>(t1 - t0).count();

    // Write to file
    ofstream file(output);
    for (int i = 0; i < n_points; i++) {
        for (int j = 0; j < n_dims; j++) {
            file << points[i][j];
            if (j + 1 < n_dims) file << " ";
        }
        file << "\n";
    }
    file.close();

    auto t2 = chrono::high_resolution_clock::now();
    double write_time = chrono::duration<double>(t2 - t1).count();

    // AJB: generation summary
    printf("\n[AJB_RESULTS] Data generation summary:\n");
    printf("  points      = %d\n", n_points);
    printf("  dimensions  = %d\n", n_dims);
    printf("  value_range = [0, %d)\n", max_val);
    printf("  collisions  = %d (%.2f%%)\n",
           collisions, 100.0 * collisions / (n_points + collisions));
    printf("  gen_time    = %.3f s\n", gen_time);
    printf("  write_time  = %.3f s\n", write_time);
    printf("  output      = %s\n", output.c_str());

    // AJB: quick distribution check on first dimension
    map<int, int> dim0_hist;
    for (auto& p : points) dim0_hist[p[0] / (max_val / 10 + 1)]++;
    printf("  dim0 histogram (10 bins): [");
    for (auto& [bin, cnt] : dim0_hist)
        printf("%s%d:%d", bin == dim0_hist.begin()->first ? "" : ", ", bin, cnt);
    printf("]\n");

    printf("[AJB] gen_co_data DONE\n");
    return 0;
}
