// =============================================================================
// gen_co_data_full.cpp — CountOracle data generator (AJB-instrumented)
//
// Origin: upstream/joinrenum/genCOData.cpp (59 lines, verbatim core)
// AJB adaptation (~20%): CLI params, per-100K progress, uniqueness ratio
//   tracking during dedup, per-dimension value distribution (min/max/mean),
//   output file size reporting, collision rate monitoring.
//
// Build: g++ -O3 gen_co_data_full.cpp -o gen_co_full
// Usage: ./gen_co_full [n_points] [n_dims] [max_val] [output_file]
// =============================================================================

#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    int n_points = (argc >= 2) ? atoi(argv[1]) : 1000000;
    int n_dims   = (argc >= 3) ? atoi(argv[2]) : 10;
    int max_val  = (argc >= 4) ? atoi(argv[3]) : 1000;
    string output = (argc >= 5) ? argv[4] : "data.txt";

    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] gen_co_data_full n=%d dims=%d max=%d -> %s\n",
            n_points, n_dims, max_val, output.c_str());
    fprintf(stderr, "[AJB] ============================================\n");

    auto t0 = chrono::high_resolution_clock::now();
    set<vector<int>> seen;
    vector<vector<int>> points;
    points.reserve(n_points);
    int collisions = 0;

    while ((int)points.size() < n_points) {
        vector<int> v(n_dims);
        for (int j = 0; j < n_dims; j++) v[j] = rand() % max_val;
        if (seen.find(v) == seen.end()) {
            seen.insert(v);
            points.push_back(v);
        } else {
            collisions++;
        }
        // AJB: progress + collision rate every 100K
        if (points.size() % 100000 == 0) {
            int total_tries = points.size() + collisions;
            fprintf(stderr, "[AJB_TRACE] %zu/%d points  collisions=%d (%.1f%%)\n",
                    points.size(), n_points, collisions,
                    100.0 * collisions / total_tries);
        }
    }
    auto t1 = chrono::high_resolution_clock::now();
    double gen_s = chrono::duration<double>(t1 - t0).count();

    // AJB_STATE: per-dimension distribution
    fprintf(stderr, "[AJB_STATE] === Per-Dimension Statistics ===\n");
    for (int d = 0; d < n_dims; d++) {
        long long sum = 0;
        int dmin = INT_MAX, dmax = INT_MIN;
        for (auto& p : points) {
            sum += p[d];
            dmin = min(dmin, p[d]);
            dmax = max(dmax, p[d]);
        }
        double mean = (double)sum / n_points;
        fprintf(stderr, "[AJB_STATE]   dim%d: min=%d max=%d mean=%.1f range=%d\n",
                d, dmin, dmax, mean, dmax - dmin);
    }

    // upstream: write to file
    auto tw0 = chrono::high_resolution_clock::now();
    ofstream file(output);
    for (auto& p : points) {
        for (int j = 0; j < n_dims; j++) {
            file << p[j];
            if (j + 1 < n_dims) file << " ";
        }
        file << "\n";
    }
    file.close();
    auto tw1 = chrono::high_resolution_clock::now();

    // AJB: output file size
    ifstream check(output, ios::ate);
    long fsize = check.tellg();
    check.close();

    fprintf(stderr, "[AJB_TIMER] generate: %.3fs  write: %.3fs\n",
            gen_s, chrono::duration<double>(tw1 - tw0).count());
    fprintf(stderr, "[AJB_STATE] output: %s  size=%ld bytes (%.1f MB)\n",
            output.c_str(), fsize, fsize / 1048576.0);
    fprintf(stderr, "[AJB_STATE] collisions=%d unique_ratio=%.4f\n",
            collisions, (double)n_points / (n_points + collisions));
    fprintf(stderr, "[AJB] gen_co_data_full DONE\n");
    return 0;
}
