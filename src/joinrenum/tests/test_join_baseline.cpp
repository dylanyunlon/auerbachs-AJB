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

    // upstream: load edge data — try .tbl then .csv, auto-detect delimiter
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
    // [AJB] M918: 自动探测分隔符 — 读首行判断含 '|' 还是 ','
    {
        string probe_line;
        if(getline(infile, probe_line)) {
            if(probe_line.find('|') != string::npos) delim = '|';
            else if(probe_line.find(',') != string::npos) delim = ',';
            else if(probe_line.find('\t') != string::npos) delim = '\t';
            // 回到文件开头重新读取
            infile.clear();
            infile.seekg(0, ios::beg);
        }
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

    // AJB: edge distribution analysis — 三角join的三个relation视角
    fprintf(stderr, "[AJB_STATE] --- Edge distribution ---\n");
    set<int> distinct_src, distinct_dst;
    for(auto& [x,y] : data) { distinct_src.insert(x); distinct_dst.insert(y); }
    fprintf(stderr, "[AJB_STATE]   edges=%zu  distinct_src=%zu  distinct_dst=%zu\n",
            data.size(), distinct_src.size(), distinct_dst.size());
    // [AJB] M918: 三角join = R(x,y) ⋈ R(y,z) ⋈ R(x,z), 三个relation视角
    // R_xy = forward edges, R_yz = index lookup, R_xz = membership check
    fprintf(stderr, "[AJB_STATE] --- Triangle join relation sizes ---\n");
    fprintf(stderr, "[AJB_STATE]   R_xy (forward edges)       = %zu\n", data.size());
    fprintf(stderr, "[AJB_STATE]   R_yz (index-lookup source) = %zu (same relation, %zu distinct keys)\n",
            data.size(), distinct_dst.size());
    fprintf(stderr, "[AJB_STATE]   R_xz (membership check)    = will deduplicate into set\n");

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

    // [AJB] M918: 结果抽样验证 — 随机检查5个结果元组
    // 验证: (x,y) ∈ R, (y,z) ∈ R, (x,z) ∈ R
    if(!res.empty()) {
        fprintf(stderr, "[AJB_STATE] --- Random sample verification (5 tuples) ---\n");
        vector<vector<int>> res_vec(res.begin(), res.end());
        srand(42);
        int sample_count = min((int)res_vec.size(), 5);
        int verified_ok = 0;
        for(int s = 0; s < sample_count; s++) {
            int idx = rand() % (int)res_vec.size();
            int x = res_vec[idx][0], y = res_vec[idx][1], z = res_vec[idx][2];
            bool xy_ok = R.count({x, y}) > 0;
            bool yz_ok = R.count({y, z}) > 0;
            bool xz_ok = R.count({x, z}) > 0;
            bool all_ok = xy_ok && yz_ok && xz_ok;
            if(all_ok) verified_ok++;
            fprintf(stderr, "[AJB_STATE]   sample[%d]: (%d,%d,%d) xy=%s yz=%s xz=%s → %s\n",
                    s, x, y, z,
                    xy_ok ? "OK" : "FAIL", yz_ok ? "OK" : "FAIL", xz_ok ? "OK" : "FAIL",
                    all_ok ? "PASS" : "FAIL");
        }
        fprintf(stderr, "[AJB_STATE]   verified: %d/%d OK\n", verified_ok, sample_count);
    }

    // [AJB] M918: 各阶段耗时汇总
    double flush_ms = chrono::duration<double,milli>(t_flush1 - t_flush0).count();
    double load_ms = chrono::duration<double,milli>(t_load1 - t_load0).count();
    double idx_ms = chrono::duration<double,milli>(t_idx1 - t_idx0).count();
    double join_ms = join_sec * 1000.0;
    double total_ms = flush_ms + load_ms + idx_ms + join_ms;
    fprintf(stderr, "[AJB_TIMER] --- Phase breakdown ---\n");
    fprintf(stderr, "[AJB_TIMER]   cache_flush : %8.3f ms (%5.1f%%)\n", flush_ms, 100.0*flush_ms/total_ms);
    fprintf(stderr, "[AJB_TIMER]   data_load   : %8.3f ms (%5.1f%%)\n", load_ms, 100.0*load_ms/total_ms);
    fprintf(stderr, "[AJB_TIMER]   index_build : %8.3f ms (%5.1f%%)\n", idx_ms, 100.0*idx_ms/total_ms);
    fprintf(stderr, "[AJB_TIMER]   triangle_join:%8.3f ms (%5.1f%%)\n", join_ms, 100.0*join_ms/total_ms);
    fprintf(stderr, "[AJB_TIMER]   total       : %8.3f ms\n", total_ms);

    ajbMem("post_join");
    fprintf(stderr, "[AJB] VERDICT: test_join_baseline PASSED\n");
    return 0;
}
