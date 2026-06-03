// =============================================================================
// test_join_baseline_upstream.cpp — Triangle join (algorithm-level rewrite)
//
// Origin: upstream/joinrenum/testjoin.cpp (98 lines)
// Algorithm changes (~25%):
//   1. PairHash: xor-shift → FNV-1a byte-mixing (better distribution)
//   2. Adjacency index: map<int,vector<int>> → flat sorted array + offset
//      table + binary search probe (cache-friendly, no pointer chasing)
//   3. flush_cache: per-byte volatile loop → memset + compiler barrier
//   4. File parse: stringstream >> getline('|') → strtol pointer walk
//   5. Result collection: set<vector<int>> → vector + sort + unique
//   6. R-lookup: set<pair> → sorted vector<pair> + binary search
//
// Debug: every phase prints structure sizes, memory, distribution stats.
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
using namespace std;

// --- algorithm change 1: FNV-1a pair hash ---
// upstream: hash(first) ^ (hash(second) << 1)
// changed: FNV-1a feeds each byte of both ints into the hash state,
//   giving better avalanche properties and fewer collisions on
//   clustered integer pairs
struct PairHash {
    size_t operator()(const pair<int,int>& p) const noexcept {
        const uint64_t fnv_offset = 14695981039346656037ULL;
        const uint64_t fnv_prime  = 1099511628211ULL;
        uint64_t h = fnv_offset;
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&p);
        for (size_t i = 0; i < sizeof(p); i++) {
            h ^= bytes[i];
            h *= fnv_prime;
        }
        return static_cast<size_t>(h);
    }
};

// --- algorithm change 3: flush_cache with memset + barrier ---
// upstream: two loops (write + volatile read), vector allocation on heap
// changed: raw new[] + memset (single pass, no per-element store),
//   asm volatile fence to prevent reordering
void flush_cache() {
    const size_t size = 100 * 1024 * 1024;
    volatile char* p = new char[size];
    memset((char*)p, 0xAA, size);
#ifdef __GNUC__
    asm volatile("" ::: "memory");
#endif
    delete[] p;
}

// --- algorithm change 4: strtol-based line parser ---
// upstream: getline(ss, x_str, '|') → stoi
// changed: strtol pointer walk, no string allocation per field
static bool parse_edge(const char* line, int& x, int& y) {
    char* end;
    x = (int)strtol(line, &end, 10);
    if (*end != '|') return false;
    y = (int)strtol(end + 1, &end, 10);
    if (end == line) return false;
    return true;
}

// --- debug: print RSS in MB ---
static double rss_mb() {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
#ifdef __APPLE__
    return r.ru_maxrss / (1024.0 * 1024.0);
#else
    return r.ru_maxrss / 1024.0;
#endif
}

int main() {
    fprintf(stderr, "[AJB_BP] === test_join_baseline_upstream start ===\n");
    fprintf(stderr, "[AJB_BP] phase=cache_flush\n");
    flush_cache();

    string filename = "db/Ra.tbl";
    ifstream infile(filename);
    if (!infile.is_open()) {
        fprintf(stderr, "[AJB_FAIL] cannot open %s\n", filename.c_str());
        return 1;
    }

    // --- phase 1: parse file with strtol ---
    fprintf(stderr, "[AJB_BP] phase=file_parse file=%s\n", filename.c_str());
    auto t_parse_start = chrono::high_resolution_clock::now();

    vector<pair<int,int>> data;
    data.reserve(1 << 20);
    string line;
    int parse_errors = 0;
    while (getline(infile, line)) {
        if (line.empty()) continue;
        int x, y;
        if (parse_edge(line.c_str(), x, y)) {
            data.emplace_back(x, y);
        } else {
            parse_errors++;
        }
    }
    infile.close();

    auto t_parse_end = chrono::high_resolution_clock::now();
    double parse_ms = chrono::duration<double,milli>(t_parse_end - t_parse_start).count();

    // debug: edge distribution stats
    int xmin = INT_MAX, xmax = INT_MIN, ymin = INT_MAX, ymax = INT_MIN;
    for (auto& [x,y] : data) {
        xmin = min(xmin, x); xmax = max(xmax, x);
        ymin = min(ymin, y); ymax = max(ymax, y);
    }
    fprintf(stderr, "[AJB_STATE] edges=%zu parse_errors=%d parse_time=%.1fms\n",
            data.size(), parse_errors, parse_ms);
    fprintf(stderr, "[AJB_STATE] x_range=[%d,%d] y_range=[%d,%d]\n",
            xmin, xmax, ymin, ymax);
    fprintf(stderr, "[AJB_STATE] mem_after_parse=%.1fMB\n", rss_mb());

    // --- algorithm change 6: R-lookup as sorted vector + binary search ---
    // upstream: set<pair<int,int>> R(data.begin(), data.end())
    // changed: sort data copy, unique, then use lower_bound for lookup —
    //   contiguous memory, better cache behavior on large R
    fprintf(stderr, "[AJB_BP] phase=build_R_index\n");
    auto t_ridx = chrono::high_resolution_clock::now();

    vector<pair<int,int>> R_sorted(data.begin(), data.end());
    sort(R_sorted.begin(), R_sorted.end());
    R_sorted.erase(unique(R_sorted.begin(), R_sorted.end()), R_sorted.end());

    auto t_ridx_end = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_STATE] R_sorted: %zu unique edges (from %zu raw) built in %.1fms\n",
            R_sorted.size(), data.size(),
            chrono::duration<double,milli>(t_ridx_end - t_ridx).count());

    // --- algorithm change 2: flat adjacency array ---
    // upstream: map<int, vector<int>> index — tree node per key, vector per adj list
    // changed: sort edges by first element, build offset table into flat array
    //   so adj[y] = flat_adj[offset[y] .. offset[y+1]) — single allocation,
    //   no map traversal, binary search for key lookup
    fprintf(stderr, "[AJB_BP] phase=build_flat_adj\n");
    auto t_adj = chrono::high_resolution_clock::now();

    // sort by the join key (y in (x,y) → y→z lookup means we index by y)
    // collect all (key=y, val=z) pairs from data, sort by key
    vector<pair<int,int>> adj_pairs;
    adj_pairs.reserve(data.size());
    for (auto& [x, z] : data) {
        adj_pairs.emplace_back(x, z);   // edge (x,z) means y=x maps to z
    }
    sort(adj_pairs.begin(), adj_pairs.end());

    // build offset table: keys[i] = distinct key, offsets[i] = start in vals[]
    vector<int> adj_keys;
    vector<int> adj_offsets;
    vector<int> adj_vals;
    adj_keys.reserve(adj_pairs.size());
    adj_vals.reserve(adj_pairs.size());

    for (size_t i = 0; i < adj_pairs.size(); ) {
        int key = adj_pairs[i].first;
        adj_keys.push_back(key);
        adj_offsets.push_back((int)adj_vals.size());
        while (i < adj_pairs.size() && adj_pairs[i].first == key) {
            adj_vals.push_back(adj_pairs[i].second);
            i++;
        }
    }
    adj_offsets.push_back((int)adj_vals.size());  // sentinel

    // free the pairs — we only need keys/offsets/vals now
    { vector<pair<int,int>>().swap(adj_pairs); }

    auto t_adj_end = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_STATE] flat_adj: %zu keys, %zu vals, built in %.1fms\n",
            adj_keys.size(), adj_vals.size(),
            chrono::duration<double,milli>(t_adj_end - t_adj).count());

    // debug: degree distribution of adjacency
    if (!adj_keys.empty()) {
        int dmin = INT_MAX, dmax = 0;
        long long dsum = 0;
        for (size_t i = 0; i < adj_keys.size(); i++) {
            int deg = adj_offsets[i+1] - adj_offsets[i];
            dmin = min(dmin, deg);
            dmax = max(dmax, deg);
            dsum += deg;
        }
        fprintf(stderr, "[AJB_STATE] degree: min=%d avg=%.1f max=%d\n",
                dmin, (double)dsum / adj_keys.size(), dmax);
    }
    fprintf(stderr, "[AJB_STATE] mem_after_index=%.1fMB\n", rss_mb());

    // --- phase: triangle enumeration ---
    // upstream: for (x,y) in data, lookup index[y] → for z in adj, check R.find({x,z})
    // changed: binary search in adj_keys for y, binary search in R_sorted for (x,z)
    fprintf(stderr, "[AJB_BP] phase=triangle_join\n");
    auto t_join = chrono::high_resolution_clock::now();

    // --- algorithm change 5: result collection as vector + sort + unique ---
    // upstream: set<vector<int>> res — tree insertion per result tuple
    // changed: push_back into vector, sort+unique at the end —
    //   avoids per-insert tree rebalancing, amortized O(1) vs O(log n)
    vector<array<int,3>> results;
    results.reserve(1 << 16);

    long long total_probes = 0;
    long long r_hits = 0;

    for (auto& [x, y] : data) {
        // binary search for y in adj_keys
        auto kit = lower_bound(adj_keys.begin(), adj_keys.end(), y);
        if (kit == adj_keys.end() || *kit != y) continue;
        int ki = (int)(kit - adj_keys.begin());
        int vstart = adj_offsets[ki];
        int vend   = adj_offsets[ki + 1];

        for (int vi = vstart; vi < vend; vi++) {
            int z = adj_vals[vi];
            total_probes++;
            // binary search in R_sorted for (x, z)
            if (binary_search(R_sorted.begin(), R_sorted.end(), make_pair(x, z))) {
                results.push_back({x, y, z});
                r_hits++;
            }
        }
    }

    // deduplicate results
    sort(results.begin(), results.end());
    results.erase(unique(results.begin(), results.end()), results.end());

    auto t_join_end = chrono::high_resolution_clock::now();
    double join_s = chrono::duration<double>(t_join_end - t_join).count();

    // --- output (upstream format) ---
    cout << rss_mb() << endl;
    cout << "Time taken: " << join_s << " seconds" << endl;
    cout << rss_mb() << endl;
    cout << "Count = " << (long long)results.size() << endl;
    cout << "Total = " << total_probes << endl;

    // debug: join statistics
    fprintf(stderr, "[AJB_STATE] triangles=%zu probes=%lld hits=%lld selectivity=%.6f\n",
            results.size(), total_probes, r_hits,
            total_probes > 0 ? (double)r_hits / total_probes : 0.0);
    fprintf(stderr, "[AJB_TIMER] join_time=%.3fs throughput=%.1f edges/s\n",
            join_s, data.size() / join_s);

    // debug: sample first 5 result tuples
    fprintf(stderr, "[AJB_STATE] sample_results (first 5):");
    for (size_t i = 0; i < min((size_t)5, results.size()); i++) {
        fprintf(stderr, " (%d,%d,%d)", results[i][0], results[i][1], results[i][2]);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "[AJB_STATE] mem_final=%.1fMB\n", rss_mb());
    fprintf(stderr, "[AJB_BP] === test_join_baseline_upstream done ===\n");
    return 0;
}
