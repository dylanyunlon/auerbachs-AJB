// =============================================================================
// test_join_baseline.cpp — Triangle join baseline benchmark (AJB-instrumented)
//
// Origin: upstream/joinrenum/testjoin.cpp (98 lines, verbatim core)
// AJB adaptation (~20%): cache flush timing, CSV+TBL dual format support,
//   per-phase timing (load, index, join), throughput metrics, edge distribution
//   analysis, memory snapshots, progress reporting during join.
//
// This is the "proper" baseline test that measures CPU join throughput
// for comparison against REnum-BMITU and GPU-accelerated AJB join.
//
// Build: g++ -O3 test_join_baseline.cpp -o test_join_bl
// Usage: ./test_join_bl [dbpath]  (default: db/)
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
using namespace std;

// upstream: PairHash (verbatim)
struct PairHash {
    size_t operator()(const pair<int,int>& p) const noexcept {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

// upstream: flush CPU cache — 100MB to evict L3 (verbatim)
void flush_cache() {
    const size_t size = 100 * 1024 * 1024;
    vector<char> buffer(size);
    for (size_t i = 0; i < size; i++) buffer[i] = i % 256;
    volatile char sink = 0;
    for (size_t i = 0; i < size; i++) sink ^= buffer[i];
}

// AJB: memory helper
static void ajbMem(const char* label) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    fprintf(stderr, "[AJB_MEM] %-24s maxRSS=%ld MB\n", label, ru.ru_maxrss/1024);
}

int main(int argc, char** argv) {
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] test_join_baseline  triangle join benchmark\n");
    fprintf(stderr, "[AJB] ============================================\n");

    string dbpath = (argc > 1) ? argv[1] : "db/";
    if(dbpath.back() != '/') dbpath += '/';

    ajbMem("startup");

    // upstream: flush cache before benchmark
    auto t_flush0 = chrono::high_resolution_clock::now();
    flush_cache();
    auto t_flush1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] cache_flush: %.1f ms\n",
            chrono::duration<double,milli>(t_flush1 - t_flush0).count());

    // upstream: load edge data — try .tbl then .csv
    string filename = dbpath + "Ra.tbl";
    char delim = '|';
    ifstream infile(filename);
    if (!infile.is_open()) {
        filename = dbpath + "Ra.csv";
        delim = ',';
        infile.open(filename);
    }
    if (!infile.is_open()) {
        fprintf(stderr, "[AJB_FAIL] Cannot open %sRa.tbl or %sRa.csv\n",
                dbpath.c_str(), dbpath.c_str());
        return 1;
    }
    fprintf(stderr, "[AJB_TRACE] Loading from %s (delim='%c')\n", filename.c_str(), delim);

    auto t_load0 = chrono::high_resolution_clock::now();
    vector<pair<int,int>> data;
    string line;
    int parse_errors = 0;
    while (getline(infile, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string x_str, y_str;
        if (getline(ss, x_str, delim) && getline(ss, y_str)) {
            try {
                data.emplace_back(stoi(x_str), stoi(y_str));
            } catch (const exception& e) {
                parse_errors++;
                if(parse_errors <= 3)
                    fprintf(stderr, "[AJB_WARN] parse error: \"%s\" (%s)\n", line.c_str(), e.what());
            }
        }
    }
    infile.close();
    auto t_load1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] data_load: %.3f ms (%zu edges, %d errors)\n",
            chrono::duration<double,milli>(t_load1 - t_load0).count(),
            data.size(), parse_errors);

    // AJB: edge distribution analysis
    fprintf(stderr, "[AJB_STATE] --- Edge distribution ---\n");
    set<int> distinct_src, distinct_dst;
    for(auto& [x,y] : data) { distinct_src.insert(x); distinct_dst.insert(y); }
    fprintf(stderr, "[AJB_STATE]   edges=%zu  distinct_src=%zu  distinct_dst=%zu\n",
            data.size(), distinct_src.size(), distinct_dst.size());

    // upstream: build index
    auto t_idx0 = chrono::high_resolution_clock::now();
    set<pair<int,int>> R(data.begin(), data.end());
    map<int, vector<int>> index;
    for (auto &[y,z] : data) {
        index[y].push_back(z);
    }
    auto t_idx1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] index_build: %.3f ms (R=%zu unique, index=%zu keys)\n",
            chrono::duration<double,milli>(t_idx1 - t_idx0).count(),
            R.size(), index.size());

    ajbMem("pre_join");

    // upstream: triangle join — for each (x,y), find z via index, check (x,z) in R
    set<vector<int>> res;
    long long count = 0, total = 0;

    fprintf(stderr, "[AJB_BP] Triangle join starting (%zu edges)...\n", data.size());
    auto t_join0 = chrono::high_resolution_clock::now();
    for (auto &[x,y] : data) {
        auto it = index.find(y);
        if (it != index.end()) {
            for (int z : it->second) {
                total++;
                if (R.find({x,z}) != R.end()) {
                    res.insert({x,y,z});
                }
            }
        }
        // AJB: progress trace every 100K edges processed
        count++;
        if(count % 100000 == 0) {
            auto tnow = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(tnow - t_join0).count();
            fprintf(stderr, "[AJB_TRACE] join progress: %lld/%zu edges, %lld probes, "
                    "%zu results, %.3fs\n",
                    count, data.size(), total, res.size(), elapsed);
        }
    }
    auto t_join1 = chrono::high_resolution_clock::now();
    double join_sec = chrono::duration<double>(t_join1 - t_join0).count();

    // upstream: output
    cout << "Time taken: " << join_sec << " seconds" << endl;
    cout << "Count = " << count << endl;
    cout << "Total = " << total << endl;

    // AJB: structured results
    fprintf(stderr, "[AJB_TIMER] triangle_join: %.3f s\n", join_sec);
    fprintf(stderr, "[AJB_STATE] triangles=%zu  total_probes=%lld  edges_scanned=%lld\n",
            res.size(), total, count);
    if(join_sec > 0)
        fprintf(stderr, "[AJB_STATE] throughput: %.1f M probes/s  %.1f K edges/s\n",
                total / join_sec / 1e6, count / join_sec / 1e3);

    // AJB: sample results
    fprintf(stderr, "[AJB_STATE] First 5 triangles:\n");
    int shown = 0;
    for(auto& tri : res) {
        if(shown >= 5) break;
        fprintf(stderr, "[AJB_STATE]   (%d, %d, %d)\n", tri[0], tri[1], tri[2]);
        shown++;
    }

    ajbMem("post_join");
    fprintf(stderr, "[AJB] VERDICT: test_join_baseline PASSED\n");
    return 0;
}
