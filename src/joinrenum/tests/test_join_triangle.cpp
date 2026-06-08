#include <bits/stdc++.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
using namespace std;

// FNV-1a hash for pair<int,int>
struct PairHash {
    static constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    static constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    
    size_t operator()(const pair<int,int>& p) const noexcept {
        uint64_t hash = FNV_OFFSET_BASIS;
        
        // Hash first int
        uint32_t val = p.first;
        hash ^= (val >> 0) & 0xFF;
        hash *= FNV_PRIME;
        hash ^= (val >> 8) & 0xFF;
        hash *= FNV_PRIME;
        hash ^= (val >> 16) & 0xFF;
        hash *= FNV_PRIME;
        hash ^= (val >> 24) & 0xFF;
        hash *= FNV_PRIME;
        
        // Hash second int
        val = p.second;
        hash ^= (val >> 0) & 0xFF;
        hash *= FNV_PRIME;
        hash ^= (val >> 8) & 0xFF;
        hash *= FNV_PRIME;
        hash ^= (val >> 16) & 0xFF;
        hash *= FNV_PRIME;
        hash ^= (val >> 24) & 0xFF;
        hash *= FNV_PRIME;
        
        return hash;
    }
};

void flush_cache() {
    const size_t size = 100 * 1024 * 1024; // 100MB
    int fd = open("/tmp/ajb_cache_flush", O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        cerr << "[AJB_STATE] Warning: Cannot create temp file for cache flush\n";
        return;
    }
    
    // Allocate and map
    if (ftruncate(fd, size) < 0) {
        close(fd);
        return;
    }
    
    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        return;
    }
    
    // Write to cache
    memset(addr, 0xAA, size);
    
    // madvise to drop from cache
    madvise(addr, size, MADV_DONTNEED);
    
    munmap(addr, size);
    close(fd);
    unlink("/tmp/ajb_cache_flush");
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
    int line_count = 0;

    while (std::getline(infile, line)) {
        if (line.empty()) continue;

        char* line_copy = strdup(line.c_str());
        char* saveptr = nullptr;
        char* x_str = strtok_r(line_copy, "|", &saveptr);
        char* y_str = strtok_r(nullptr, "\n", &saveptr);
        
        if (x_str && y_str) {
            char* endptr_x = nullptr;
            char* endptr_y = nullptr;
            int x = strtol(x_str, &endptr_x, 10);
            int y = strtol(y_str, &endptr_y, 10);
            
            if (endptr_x != x_str && endptr_y != y_str) {
                data.emplace_back(x, y);
                line_count++;
            }
        }
        free(line_copy);
    }

    infile.close();

    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    long mem_after_load = r_usage.ru_maxrss / 1024;
    
    cerr << "[AJB_STATE] Data loaded: lines=" << line_count 
         << " pairs=" << data.size() 
         << " memory=" << mem_after_load << "MB\n";

    // Build sorted flat index: vector of (key, values_start, values_count)
    // First collect all (y, z) pairs
    vector<pair<int, int>> yz_pairs;
    for (auto &[y, z] : data) {
        yz_pairs.emplace_back(y, z);
    }
    
    // Sort by y (key)
    sort(yz_pairs.begin(), yz_pairs.end());
    
    // Build index structure
    vector<int> index_keys;
    vector<vector<int>> index_values;
    
    int current_key = -1;
    vector<int> current_values;
    
    for (auto &[y, z] : yz_pairs) {
        if (y != current_key) {
            if (current_key != -1) {
                index_keys.push_back(current_key);
                index_values.push_back(current_values);
                current_values.clear();
            }
            current_key = y;
        }
        current_values.push_back(z);
    }
    if (current_key != -1) {
        index_keys.push_back(current_key);
        index_values.push_back(current_values);
    }
    
    // Build R set for fast lookup
    set<pair<int, int>> R(data.begin(), data.end());

    vector<vector<int>> raw_results;
    long long count = 0, total = 0;
    long long probe_count = 0;

    auto start = std::chrono::high_resolution_clock::now();
    
    for (auto &[x, y] : data) {
        // Binary search in sorted keys
        auto it = lower_bound(index_keys.begin(), index_keys.end(), y);
        
        if (it != index_keys.end() && *it == y) {
            int idx = it - index_keys.begin();
            for (int z : index_values[idx]) {
                total++;
                probe_count++;
                
                if (probe_count % 10000 == 0) {
                    cerr << "[AJB_STATE] Join progress: probes=" << probe_count 
                         << " triangles=" << count << "\n";
                }
                
                if (R.find({x, z}) != R.end()) {
                    raw_results.push_back({x, y, z});
                    count++;
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Deduplicate results using sorted + unique
    sort(raw_results.begin(), raw_results.end());
    raw_results.erase(unique(raw_results.begin(), raw_results.end()), 
                      raw_results.end());
    
    long long unique_count = raw_results.size();

    getrusage(RUSAGE_SELF, &r_usage);
    long mem_after_join = r_usage.ru_maxrss / 1024;

    cerr << "[AJB_STATE] Join complete:\n";
    cerr << "  unique_triangles=" << unique_count << "\n";
    cerr << "  total_probes=" << total << "\n";
    cerr << "  time=" << elapsed.count() << "s\n";
    cerr << "  memory=" << mem_after_join << "MB\n";

    cout << "Time taken: " << elapsed.count() << " seconds\n";
    cout << "Memory peak: " << mem_after_join << "MB\n";
    cout << "Unique triangles: " << unique_count << endl;
    cout << "Total probes: " << total << endl;
    
    return 0;
}
