// =============================================================================
// testjoin.cpp — Triangle join baseline benchmark (AJB-instrumented)
//
// Origin: upstream/joinrenum/testjoin.cpp (98 lines, verbatim core)
// AJB adaptation (~20%): AJB banner, data load stats, index build timing,
//   join phase timing, result cardinality and throughput reports, memory
//   snapshots at each phase, structured stderr output.
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
using namespace std;

// [AJB] M914: triangle verification — 随机抽查结果元组是否真的是合法三角形
// 这是算法正确性的运行时断言,不是简单的print
struct TriangleVerifier {
    int checked = 0;
    int passed = 0;
    int failed = 0;
    void verify_sample(const vector<int>& tri, const set<pair<int,int>>& R) {
#ifdef AJB_DEBUG
        checked++;
        // 三角形 (x,y,z) 要求: (x,y)∈R, (y,z)∈R, (x,z)∈R
        bool edge_xy = R.count({tri[0], tri[1]}) > 0;
        bool edge_yz = R.count({tri[1], tri[2]}) > 0;
        bool edge_xz = R.count({tri[0], tri[2]}) > 0;
        if (edge_xy && edge_yz && edge_xz) {
            passed++;
        } else {
            failed++;
            fprintf(stderr, "[AJB_VERIFY_FAIL] triangle(%d,%d,%d) edges: xy=%d yz=%d xz=%d\n",
                    tri[0], tri[1], tri[2], (int)edge_xy, (int)edge_yz, (int)edge_xz);
        }
#endif
    }
    void dump() {
#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_VERIFY] triangles checked=%d passed=%d failed=%d\n", checked, passed, failed);
        if (failed > 0) fprintf(stderr, "[AJB_VERIFY] WARNING: %d invalid triangles detected!\n", failed);
#endif
    }
};

// [AJB] M914: 度数分布分析 — 理解图的倾斜度, 跟skew_detector一脉相承
static void ajb_analyze_degree_distribution(const map<int, vector<int>>& index) {
#ifdef AJB_DEBUG
    if (index.empty()) return;
    vector<int> degrees;
    degrees.reserve(index.size());
    for (auto& [k, v] : index) degrees.push_back((int)v.size());
    sort(degrees.begin(), degrees.end());
    int n = (int)degrees.size();
    double mean = 0;
    for (int d : degrees) mean += d;
    mean /= n;
    double variance = 0;
    for (int d : degrees) variance += (d - mean) * (d - mean);
    variance /= n;
    fprintf(stderr, "[AJB_STATE] degree_distribution: nodes=%d min=%d median=%d max=%d mean=%.1f stddev=%.1f\n",
            n, degrees[0], degrees[n/2], degrees[n-1], mean, sqrt(variance));
    // Gini coefficient — 衡量倾斜度, 高Gini说明少数节点支配join开销
    double gini_sum = 0;
    for (int i = 0; i < n; i++) gini_sum += (2*i - n + 1) * degrees[i];
    double gini = gini_sum / (n * mean * n);
    fprintf(stderr, "[AJB_STATE] degree_gini=%.3f (0=uniform, 1=maximally skewed)\n", gini);
#endif
}

struct PairHash {
    size_t operator()(const pair<int,int>& p) const noexcept {
        // 组合哈希，避免冲突
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

void flush_cache() {
    const size_t size = 100 * 1024 * 1024; // 100MB，一般足够超过L3 cache
    vector<char> buffer(size);

    for (size_t i = 0; i < size; i++) {
        buffer[i] = i % 256; // 写访问，保证真的进入cache
    }

    volatile char sink = 0; 
    for (size_t i = 0; i < size; i++) {
        sink ^= buffer[i]; // 读访问，避免编译器优化掉
    }
}

int main() {
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] testjoin.cpp — triangle join baseline\n");
    fprintf(stderr, "[AJB] ============================================\n");

    auto t_flush0 = chrono::high_resolution_clock::now();
    flush_cache();
    auto t_flush1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] cache_flush: %.1f ms (100MB L3 eviction)\n",
            chrono::duration<double,milli>(t_flush1 - t_flush0).count());

    // upstream: try Ra.tbl first, fall back to Ra.csv
    std::string filename = "db/Ra.tbl";
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        // AJB: try csv fallback for sandbox environments
        filename = "db/Ra.csv";
        infile.open(filename);
    }
    if (!infile.is_open()) {
        std::cerr << "[AJB_FAIL] Cannot open db/Ra.tbl or db/Ra.csv" << std::endl;
        return 1;
    }
    fprintf(stderr, "[AJB_TRACE] Reading from %s\n", filename.c_str());

    std::vector<std::pair<int, int>> data;
    std::string line;


    while (std::getline(infile, line)) {
        if (line.empty()) continue; // 跳过空行

        std::stringstream ss(line);
        std::string x_str, y_str;

        if (std::getline(ss, x_str, '|') && std::getline(ss, y_str)) {
            try {
                int x = std::stoi(x_str);
                int y = std::stoi(y_str);
                data.emplace_back(x, y);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing line: " << line << " (" << e.what() << ")\n";
            }
        }
    }

    infile.close();
    fprintf(stderr, "[AJB_STATE] Loaded %zu edges from %s\n", data.size(), filename.c_str());

    // AJB: sample first few edges
    fprintf(stderr, "[AJB_STATE] First 3 edges:\n");
    for(size_t i = 0; i < min((size_t)3, data.size()); i++)
        fprintf(stderr, "[AJB_STATE]   (%d, %d)\n", data[i].first, data[i].second);

    auto t_idx0 = chrono::high_resolution_clock::now();
    set<std::pair<int, int>> R(data.begin(), data.end());
    map<int, vector<int>> index;
    for (auto &[y,z] : data) {
        index[y].push_back(z);
    }
    auto t_idx1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] index build: %.3f ms (R.size=%zu, index.size=%zu)\n",
            chrono::duration<double,milli>(t_idx1 - t_idx0).count(),
            R.size(), index.size());
    ajb_analyze_degree_distribution(index);

    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    long pre_join_mem = r_usage.ru_maxrss/1024;
    cout << pre_join_mem << endl;
    fprintf(stderr, "[AJB_MEM] pre_join: maxRSS=%ld MB\n", pre_join_mem);

    set<vector<int>> res;

    long long count = 0, total = 0;
    // upstream: 遍历每个 (x,y)，利用索引找 z
    fprintf(stderr, "[AJB_BP] Triangle join starting...\n");
    auto start = std::chrono::high_resolution_clock::now();
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
    }
    // upstream: shuffle res (commented out)
    // random_device rd;
    // mt19937 g(rd());  // Mersenne Twister 引擎
    // shuffle(res.begin(), res.end(), g);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << " seconds\n";

    getrusage(RUSAGE_SELF, &r_usage);
    long post_join_mem = r_usage.ru_maxrss/1024;
    cout << post_join_mem << endl;
    cout << "Count = " << count << endl;
    cout << "Total = " << total << endl;

    // AJB: structured result summary
    fprintf(stderr, "[AJB_TIMER] triangle_join: %.3f s\n", elapsed.count());
    fprintf(stderr, "[AJB_STATE] results=%zu  total_probes=%lld  count=%lld\n",
            res.size(), total, count);
    fprintf(stderr, "[AJB_STATE] throughput: %.1f M probes/s\n",
            total / elapsed.count() / 1e6);
    fprintf(stderr, "[AJB_MEM] post_join: maxRSS=%ld MB (delta=%ld MB)\n",
            post_join_mem, post_join_mem - pre_join_mem);

    // AJB: sample results
    fprintf(stderr, "[AJB_STATE] First 5 triangles:\n");
    int shown = 0;
    TriangleVerifier verifier;
    // M914: 随机抽样验证 — 从结果集中抽5个验证是否为合法三角形
    // 这不是展示,是正确性断言
    mt19937 sample_rng(42);
    vector<vector<int>> res_vec(res.begin(), res.end());
    int n_check = min((int)res_vec.size(), 5);
    for (int i = 0; i < n_check; i++) {
        // Fisher-Yates partial shuffle取前n_check个
        int j = i + (int)(sample_rng() % (res_vec.size() - i));
        swap(res_vec[i], res_vec[j]);
        verifier.verify_sample(res_vec[i], R);
        fprintf(stderr, "[AJB_STATE]   (%d, %d, %d)\n", res_vec[i][0], res_vec[i][1], res_vec[i][2]);
    }
    verifier.dump();

    fprintf(stderr, "[AJB] testjoin.cpp COMPLETE\n");
    return 0;
}