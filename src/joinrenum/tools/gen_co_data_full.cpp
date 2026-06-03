// =============================================================================
// gen_co_data_full.cpp — CountOracle data generator
//
// Origin: upstream/joinrenum/genCOData.cpp (59 lines)
// Algorithm changes (~25%):
//   1. Dedup: set<vector<int>> (red-black tree, O(log n) per lookup with
//      full vector comparison) → unordered_set with FNV-1a byte hash
//      (O(1) amortized, different collision behavior)
//   2. Data generation: uniform i.i.d. per dimension → correlated
//      dimensions: dim[0] is uniform, dim[i] = (dim[i-1] + uniform)
//      mod max_val. This produces non-uniform spatial density which
//      exercises CountOracle's range tree differently than random scatter
//   3. Output ordering: upstream writes in generation order (arbitrary)
//      → lexicographic sort before write. This changes the CountOracle
//      build path because the tree insertion order affects balance
//   4. Post-generation verification: partition the bounding box into a
//      coarse grid, count points per cell, compute coefficient of
//      variation to measure spatial uniformity of the generated dataset
//
// Build: g++ -O3 gen_co_data_full.cpp -o gen_co_full
// =============================================================================

#include <bits/stdc++.h>
using namespace std;

// --- algorithm change 1: FNV-1a hash for dedup ---
struct VecHashFNV {
    size_t operator()(const vector<int>& v) const {
        const uint64_t fnv_offset = 14695981039346656037ULL;
        const uint64_t fnv_prime  = 1099511628211ULL;
        uint64_t h = fnv_offset;
        const unsigned char* data = reinterpret_cast<const unsigned char*>(v.data());
        size_t nbytes = v.size() * sizeof(int);
        for (size_t i = 0; i < nbytes; i++) {
            h ^= data[i];
            h *= fnv_prime;
        }
        return static_cast<size_t>(h);
    }
};

int main(int argc, char* argv[]) {
    int n_points = (argc >= 2) ? atoi(argv[1]) : 1000000;
    int n_dims   = (argc >= 3) ? atoi(argv[2]) : 10;
    int max_val  = (argc >= 4) ? atoi(argv[3]) : 1000;
    const char* output = (argc >= 5) ? argv[4] : "data.txt";

    fprintf(stderr, "[AJB_BP] gen_co_data n=%d dims=%d max=%d -> %s\n",
            n_points, n_dims, max_val, output);

    unordered_set<vector<int>, VecHashFNV> seen;
    vector<vector<int>> points;
    points.reserve(n_points);
    int collisions = 0;

    // --- algorithm change 2: correlated dimension generation ---
    // upstream: for(j=0;j<10;j++) v[j] = rand() % 1000
    //   → each dimension is i.i.d. uniform, giving perfectly uncorrelated data
    // changed: dim[0] is uniform, dim[i] = (dim[i-1] + uniform_offset) % max_val
    //   → produces chain-correlated data with non-uniform spatial density,
    //   which exercises different splits in the CountOracle range tree
    uint64_t lcg = 42;
    auto lcg_next = [&](int mod) -> int {
        lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
        return (int)((lcg >> 16) % mod);
    };

    auto t0 = chrono::high_resolution_clock::now();
    while ((int)points.size() < n_points) {
        vector<int> v(n_dims);
        v[0] = lcg_next(max_val);
        for (int j = 1; j < n_dims; j++) {
            // correlated: carry from previous dim + random offset
            int offset = lcg_next(max_val / 2 + 1);
            v[j] = (v[j-1] + offset) % max_val;
        }
        if (seen.insert(v).second) {
            points.push_back(move(v));
        } else {
            collisions++;
        }
    }
    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_STATE] generated %d points (collisions=%d) in %.2fs\n",
            n_points, collisions, chrono::duration<double>(t1 - t0).count());

    // --- algorithm change 3: lexicographic sort before write ---
    // upstream: writes points in generation (insertion) order
    // changed: sort lexicographically so that the CountOracle tree
    //   receives sorted input — a different insertion pattern that
    //   affects internal tree balance and split decisions
    sort(points.begin(), points.end());

    // --- algorithm change 4: spatial uniformity verification ---
    // Partition the bounding box into a coarse grid (4 bins per dim,
    // but only check first 3 dims to keep the grid manageable),
    // count points per cell, compute coefficient of variation
    if (n_dims >= 2) {
        const int GRID = 4;
        int check_dims = min(n_dims, 3);
        int ncells = 1;
        for (int d = 0; d < check_dims; d++) ncells *= GRID;

        vector<int> cell_counts(ncells, 0);
        for (auto& p : points) {
            int cell = 0;
            int stride = 1;
            for (int d = 0; d < check_dims; d++) {
                int bin = min(p[d] * GRID / max_val, GRID - 1);
                cell += bin * stride;
                stride *= GRID;
            }
            cell_counts[cell]++;
        }
        // compute mean and stddev of cell counts
        double sum = 0, sum_sq = 0;
        int nonempty = 0;
        for (int c : cell_counts) {
            sum += c;
            sum_sq += (double)c * c;
            if (c > 0) nonempty++;
        }
        double mean = sum / ncells;
        double var = sum_sq / ncells - mean * mean;
        double cv = mean > 0 ? sqrt(max(0.0, var)) / mean : 0;
        fprintf(stderr, "[AJB_STATE] grid(%d^%d=%d cells): nonempty=%d mean=%.0f cv=%.3f\n",
                GRID, check_dims, ncells, nonempty, mean, cv);
        // cv near 0 = uniform, cv >> 0 = clustered (expected for correlated data)
    }

    // write sorted data to file
    auto tw0 = chrono::high_resolution_clock::now();
    FILE* f = fopen(output, "w");
    char buf[2048];
    for (auto& p : points) {
        int pos = 0;
        for (int j = 0; j < n_dims; j++) {
            if (j) buf[pos++] = ' ';
            pos += snprintf(buf + pos, sizeof(buf) - pos, "%d", p[j]);
        }
        buf[pos++] = '\n';
        fwrite(buf, 1, pos, f);
    }
    fclose(f);
    auto tw1 = chrono::high_resolution_clock::now();

    fprintf(stderr, "[AJB_TIMER] write=%.2fs\n",
            chrono::duration<double>(tw1 - tw0).count());
    fprintf(stderr, "[AJB_BP] gen_co_data_full done\n");
    return 0;
}
