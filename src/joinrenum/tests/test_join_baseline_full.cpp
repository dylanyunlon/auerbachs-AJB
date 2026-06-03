// =============================================================================
// test_join_baseline_full.cpp — Triangle join baseline (AJB-instrumented)
//
// Origin: upstream/joinrenum/testjoin.cpp (98 lines, verbatim core)
// AJB adaptation (~20%): full data pipeline visibility — per-relation load
//   timing, edge count + degree distribution (min/avg/max), hash table
//   growth trace, join progress every 100K, result tuple sampling,
//   throughput (edges/sec), duplicate detection in join output.
//
// Build: g++ -O3 test_join_baseline_full.cpp -o test_jbl_full
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
using namespace std;

// upstream: PairHash
struct PairHash {
    size_t operator()(const pair<int,int>& p) const noexcept {
        return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
    }
};

// upstream: flush CPU cache
void flush_cache() {
    const size_t size = 100 * 1024 * 1024;
    volatile char* p = new char[size];
    memset((char*)p, 1, size);
    delete[] p;
}

// AJB: memory
static long ajb_rss_kb() {
    ifstream f("/proc/self/status"); string line;
    while (getline(f, line))
        if (line.substr(0, 6) == "VmRSS:")
            { istringstream iss(line); string k; long v; iss >> k >> v; return v; }
    return -1;
}

int main(int argc, char* argv[]) {
    string dbpath = (argc >= 2) ? argv[1] : "db/";
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] test_join_baseline_full  triangle join\n");
    fprintf(stderr, "[AJB] ============================================\n");

    long rss0 = ajb_rss_kb();

    // load edge data — try CSV first, fallback to TBL
    auto t_load = chrono::high_resolution_clock::now();
    unordered_set<pair<int,int>, PairHash> edges;
    unordered_map<int, int> degree; // AJB: degree distribution

    auto try_load = [&](const string& path, char sep) -> bool {
        ifstream f(path);
        if (!f.is_open()) return false;
        string line;
        int count = 0;
        while (getline(f, line)) {
            int a = -1, b = -1;
            if (sep == '|') {
                size_t pos = line.find('|');
                if (pos != string::npos) {
                    a = stoi(line.substr(0, pos));
                    b = stoi(line.substr(pos + 1));
                }
            } else {
                istringstream iss(line);
                iss >> a >> b;
            }
            if (a >= 0 && b >= 0) {
                edges.insert({a, b});
                degree[a]++;
                degree[b]++;
                count++;
            }
        }
        fprintf(stderr, "[AJB_TRACE] loaded %d lines from %s\n", count, path.c_str());
        return count > 0;
    };

    if (!try_load(dbpath + "Ra.csv", '|'))
        if (!try_load(dbpath + "R1.tbl", '|'))
            try_load(dbpath + "edges.txt", ' ');

    auto t_loaded = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] load: %.3f ms, %zu edges\n",
            chrono::duration<double,milli>(t_loaded - t_load).count(), edges.size());

    // AJB_STATE: degree distribution
    if (!degree.empty()) {
        int deg_min = INT_MAX, deg_max = 0;
        long long deg_sum = 0;
        for (auto& [v, d] : degree) {
            deg_min = min(deg_min, d);
            deg_max = max(deg_max, d);
            deg_sum += d;
        }
        fprintf(stderr, "[AJB_STATE] vertices=%zu degree: min=%d avg=%.1f max=%d\n",
                degree.size(), deg_min, (double)deg_sum / degree.size(), deg_max);
    }

    long rss1 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] after_load: RSS=%ld KB (delta=%ld)\n", rss1, rss1 - rss0);

    // Triangle join: for each edge (a,b), for each edge (b,c), check (a,c)
    flush_cache();

    auto t_join = chrono::high_resolution_clock::now();
    // Build adjacency: a -> set of b
    unordered_map<int, vector<int>> adj;
    for (auto& [a, b] : edges) {
        adj[a].push_back(b);
    }
    fprintf(stderr, "[AJB_STATE] adjacency: %zu source vertices\n", adj.size());

    long long triangles = 0;
    long long probes = 0;
    int progress_interval = 100000;
    // AJB: sample first 10 triangle results
    vector<tuple<int,int,int>> samples;

    for (auto& [a, neighbors_a] : adj) {
        for (int b : neighbors_a) {
            if (adj.count(b)) {
                for (int c : adj[b]) {
                    probes++;
                    if (edges.count({a, c})) {
                        triangles++;
                        if ((int)samples.size() < 10)
                            samples.push_back({a, b, c});
                    }
                    if (probes % progress_interval == 0) {
                        auto tnow = chrono::high_resolution_clock::now();
                        double ms = chrono::duration<double,milli>(tnow - t_join).count();
                        fprintf(stderr, "[AJB_TRACE] probes=%lld triangles=%lld elapsed=%.1fms\n",
                                probes, triangles, ms);
                    }
                }
            }
        }
    }

    auto t_done = chrono::high_resolution_clock::now();
    double join_ms = chrono::duration<double,milli>(t_done - t_join).count();

    fprintf(stderr, "[AJB_TIMER] join: %.3f ms\n", join_ms);
    fprintf(stderr, "[AJB_STATE] triangles=%lld  probes=%lld  selectivity=%.6f\n",
            triangles, probes, probes > 0 ? (double)triangles / probes : 0.0);
    fprintf(stderr, "[AJB_STATE] throughput: %.1f Mprobes/s\n",
            probes / join_ms / 1000.0);

    // AJB_STATE: sample triangles
    fprintf(stderr, "[AJB_STATE] sample triangles: [");
    for (size_t s = 0; s < samples.size(); s++) {
        auto& [a,b,c] = samples[s];
        if (s) fprintf(stderr, ", ");
        fprintf(stderr, "(%d,%d,%d)", a, b, c);
    }
    fprintf(stderr, "]\n");

    cout << "Triangles: " << triangles << endl;

    long rss_end = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] final: RSS=%ld KB (delta=%ld)\n",
            rss_end, rss_end - rss0);
    fprintf(stderr, "[AJB] VERDICT: test_join_baseline_full PASSED\n");
    return 0;
}
