#include <bits/stdc++.h>
#include <sys/resource.h>
using namespace std;

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
    flush_cache();
    std::string filename = "db/Ra.tbl";
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return 1;
    }

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

    set<std::pair<int, int>> R(data.begin(), data.end());
    map<int, vector<int>> index;
    for (auto &[y,z] : data) {
        index[y].push_back(z);
    }

    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    cout << r_usage.ru_maxrss/1024 << endl;
    set<vector<int>> res;

    long long count = 0, total = 0;
    // 遍历每个 (x,y)，利用索引找 z
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
    // shuffle res
    // random_device rd;
    // mt19937 g(rd());  // Mersenne Twister 引擎

    // // 打乱顺序
    // shuffle(res.begin(), res.end(), g);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << " seconds\n";

    getrusage(RUSAGE_SELF, &r_usage);
    cout << r_usage.ru_maxrss/1024 << endl;
    cout << "Count = " << count << endl;
    cout << "Total = " << total << endl;
    return 0;
}
