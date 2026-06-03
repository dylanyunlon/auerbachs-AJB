#include<bits/stdc++.h>
using namespace std;

// --- VectorHash: FNV-1a replacing xor-shift chain ---
// upstream:  hash ^= hash<int>()(i) + 0x9e3779b9 + (hash<<6) + (hash>>2)
// changed: FNV-1a — XOR then multiply per element (different bit mixing)
struct VectorHash {
    size_t operator()(const vector<int>& v) const {
        size_t hash = 14695981039346656037ULL;
        for (int i : v) {
            hash ^= (size_t)(unsigned int)i;
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};

int main() {
    unordered_map<vector<int>, int, VectorHash> cache;
    vector<int> vec;
    for(int i = 0; i < 1700000; i++) {
        vector<int> v = {rand()};
        cache[v] = i;
        vec.push_back(v[0]);
    }

    // --- lookup loop: split into forward + reverse passes ---
    // upstream: single for(i=0..1700000) with rand() + find({vec[i]})
    // changed: first half scans forward, second half scans backward
    //   through vec — different memory access pattern, tests cache
    //   behavior under both sequential and reverse-sequential access
    clock_t start = clock();
    int found = 0;
    int N = 1700000;
    int half = N / 2;

    // forward pass: 0 .. half-1
    for(int i = 0; i < half; i++) {
        if(cache.find({vec[i]}) != cache.end()) {
            found++;
        }
    }
    // reverse pass: N-1 .. half
    for(int i = N - 1; i >= half; i--) {
        if(cache.find({vec[i]}) != cache.end()) {
            found++;
        }
    }
    cout << "Found: " << found << endl; 
    clock_t end = clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    cout << "Time taken: " << elapsed << " seconds" << endl;

    // --- consistency check: find vs count agreement ---
    // upstream: no verification
    // changed: probe random keys and verify find() agrees with count()
    int check_mismatches = 0;
    for(int t = 0; t < 100; t++) {
        int idx = rand() % N;
        vector<int> key = {vec[idx]};
        bool has_find  = (cache.find(key) != cache.end());
        bool has_count = (cache.count(key) > 0);
        if(has_find != has_count) check_mismatches++;
    }
    if(check_mismatches > 0)
        cout << "find/count mismatches: " << check_mismatches << endl;

    return 0;
}
