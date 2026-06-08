// =============================================================================
// test_join_baseline_full.cpp — Triangle join baseline
//
// Origin: upstream/joinrenum/testjoin.cpp (98 lines)
// Algorithm changes (~25%):
//   1. PairHash: xor-shift → FNV-1a byte-mixing
//   2. File parse: stoi/line.substr/line.find → strtol pointer walk
//   3. flush_cache: volatile byte loop → memset + compiler barrier
//   4. Adjacency: unordered_map<int,vector<int>> → flat sorted array
//      with offset table + binary search (cache-friendly)
//   5. R-check: unordered_set → sorted vector + binary_search
//   6. RSS: ifstream/istringstream → fopen/strtol
//
// Build: g++ -O3 test_join_baseline_full.cpp -o test_jbl_full
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
using namespace std;

// --- algorithm change 1: FNV-1a pair hash ---
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

// --- algorithm change 3: memset + barrier flush ---
void flush_cache() {
    const size_t size = 100 * 1024 * 1024;
    volatile char* p = new char[size];
    memset((char*)p, 0xAA, size);
#ifdef __GNUC__
    asm volatile("" ::: "memory");
#endif
    delete[] p;
}

// --- algorithm change 6: fopen/strtol RSS reader ---
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

// --- algorithm change 2: strtol-based edge parser ---
static bool parse_edge(const char* line, char sep, int& a, int& b) {
    char* end;
    a = (int)strtol(line, &end, 10);
    if (*end != sep) return false;
    b = (int)strtol(end + 1, &end, 10);
    return true;
}

int main() {
    fprintf(stderr, "[AJB_BP] === test_join_baseline_full start ===\n");
    long rss0 = ajb_rss_kb();

    // load edges
    auto t_load = chrono::high_resolution_clock::now();
    vector<pair<int,int>> edges;
    edges.reserve(1 << 20);

    // try db/Ra.tbl first, then db/R1.tbl, then .csv variants
    const char* paths[] = {"db/Ra.tbl", "db/R1.tbl", "db/Ra.csv", "db/R1.csv", nullptr};
    for (int pi = 0; paths[pi]; pi++) {
        ifstream f(paths[pi]);
        if (!f.is_open()) continue;
        string line;
        while (getline(f, line)) {
            if (line.empty()) continue;
            int a, b;
            if (parse_edge(line.c_str(), '|', a, b)) {
                edges.emplace_back(a, b);
            }
        }
        if (!edges.empty()) {
            fprintf(stderr, "[AJB_STATE] loaded %zu edges from %s\n",
                    edges.size(), paths[pi]);
            break;
        }
    }
    auto t_load_end = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] load=%.1fms\n",
            chrono::duration<double,milli>(t_load_end - t_load).count());

    flush_cache();

    // --- algorithm change 5: R-check as sorted vector ---
    // upstream: unordered_set<pair> R
    // changed: sorted vector + binary_search — contiguous memory
    auto t_idx = chrono::high_resolution_clock::now();
    vector<pair<int,int>> R_sorted(edges.begin(), edges.end());
    sort(R_sorted.begin(), R_sorted.end());
    R_sorted.erase(unique(R_sorted.begin(), R_sorted.end()), R_sorted.end());

    // --- algorithm change 4: flat adjacency array ---
    // upstream: unordered_map<int, vector<int>> adj
    // changed: sort (key,val) pairs, build offset table for O(1) key→range lookup
    vector<pair<int,int>> adj_pairs;
    adj_pairs.reserve(edges.size());
    for (auto& [a, b] : edges) {
        adj_pairs.emplace_back(a, b);
    }
    sort(adj_pairs.begin(), adj_pairs.end());

    vector<int> adj_keys, adj_offsets, adj_vals;
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
    adj_offsets.push_back((int)adj_vals.size());
    { vector<pair<int,int>>().swap(adj_pairs); }

    auto t_idx_end = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_STATE] R_sorted=%zu flat_adj: %zu keys, %zu vals, built in %.1fms\n",
            R_sorted.size(), adj_keys.size(), adj_vals.size(),
            chrono::duration<double,milli>(t_idx_end - t_idx).count());

    // degree stats from flat adj
    if (!adj_keys.empty()) {
        int dmin = INT_MAX, dmax = 0;
        for (size_t i = 0; i < adj_keys.size(); i++) {
            int deg = adj_offsets[i+1] - adj_offsets[i];
            dmin = min(dmin, deg);
            dmax = max(dmax, deg);
        }
        fprintf(stderr, "[AJB_STATE] degree: min=%d max=%d avg=%.1f\n",
                dmin, dmax, (double)adj_vals.size() / adj_keys.size());
    }

    // triangle enumeration with flat adjacency + binary search R-check
    auto t_join = chrono::high_resolution_clock::now();
    long long triangles = 0, probes = 0;

    for (size_t ki = 0; ki < adj_keys.size(); ki++) {
        int a = adj_keys[ki];
        int a_start = adj_offsets[ki], a_end = adj_offsets[ki+1];
        for (int vi = a_start; vi < a_end; vi++) {
            int b = adj_vals[vi];
            // find b in adj_keys
            auto bit = lower_bound(adj_keys.begin(), adj_keys.end(), b);
            if (bit == adj_keys.end() || *bit != b) continue;
            int bi = (int)(bit - adj_keys.begin());
            int b_start = adj_offsets[bi], b_end = adj_offsets[bi+1];
            for (int wi = b_start; wi < b_end; wi++) {
                int c = adj_vals[wi];
                probes++;
                if (binary_search(R_sorted.begin(), R_sorted.end(), make_pair(a, c))) {
                    triangles++;
                }
            }
        }
    }
    auto t_join_end = chrono::high_resolution_clock::now();
    double join_ms = chrono::duration<double,milli>(t_join_end - t_join).count();

    cout << "Triangles: " << triangles << endl;
    fprintf(stderr, "[AJB_STATE] triangles=%lld probes=%lld selectivity=%.6f\n",
            triangles, probes,
            probes > 0 ? (double)triangles / probes : 0.0);
    fprintf(stderr, "[AJB_TIMER] join=%.1fms (%.1f Mprobes/s)\n",
            join_ms, probes / join_ms / 1000.0);
    fprintf(stderr, "[AJB_STATE] mem_final=%ld KB\n", ajb_rss_kb());
    fprintf(stderr, "[AJB_BP] === test_join_baseline_full done ===\n");
    return 0;
}
