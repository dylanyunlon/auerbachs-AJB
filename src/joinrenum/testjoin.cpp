// =============================================================================
// testjoin.cpp — Triangle join baseline (AJB-instrumented in-place)
//
// Origin: upstream/joinrenum/testjoin.cpp
// AJB adaptation: added [AJB_TRACE] at key pipeline stages, timing
//   breakdown, and memory snapshots.  Original algorithm unchanged.
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
using namespace std;

struct PairHash {
    size_t operator()(const pair<int,int>& p) const noexcept {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

void flush_cache() {
    const size_t size = 100 * 1024 * 1024;
    vector<char> buffer(size);
    for (size_t i = 0; i < size; i++) buffer[i] = i % 256;
    volatile char sink = 0;
    for (size_t i = 0; i < size; i++) sink ^= buffer[i];
}

int main() {
    printf("[AJB_TRACE] testjoin: cache flush...\n");
    flush_cache();

    std::string filename = "db/Ra.tbl";
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "[AJB_FAIL] Cannot open " << filename << std::endl;
        return 1;
    }

    auto t_load = chrono::high_resolution_clock::now();
    std::vector<std::pair<int, int>> data;
    std::string line;

    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string x_str, y_str;
        if (std::getline(ss, x_str, '|') && std::getline(ss, y_str)) {
            try {
                int x = std::stoi(x_str);
                int y = std::stoi(y_str);
                data.emplace_back(x, y);
            } catch (const std::exception& e) {
                std::cerr << "Parse error: " << line << " (" << e.what() << ")\n";
            }
        }
    }
    infile.close();
    auto t_loaded = chrono::high_resolution_clock::now();
    printf("[AJB_TIMER] load: %.3f ms, %zu edges\n",
        chrono::duration<double,milli>(t_loaded - t_load).count(), data.size());

    set<std::pair<int, int>> R(data.begin(), data.end());
    map<int, vector<int>> index;
    for (auto &[y,z] : data) {
        index[y].push_back(z);
    }
    printf("[AJB_STATE] R=%zu unique edges, index=%zu keys\n",
           R.size(), index.size());

    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    printf("[AJB_MEM] pre-join: %ld MB\n", r_usage.ru_maxrss/1024);

    set<vector<int>> res;
    long long count = 0, total = 0;

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
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;

    getrusage(RUSAGE_SELF, &r_usage);
    printf("[AJB_MEM] post-join: %ld MB\n", r_usage.ru_maxrss/1024);
    printf("[AJB_STATE] triangles=%zu probes=%lld\n", res.size(), total);

    cout << "Count = " << count << endl;
    cout << "Total = " << total << endl;
    return 0;
}
