#include<bits/stdc++.h>
using namespace std;
struct VectorHash {
    size_t operator()(const vector<int>& v) const {
        size_t hash = 0;
        for (int i : v) {
            hash ^= std::hash<int>()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
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
    clock_t start = clock();
    int found = 0;
    for(int i = 0; i < 1700000; i++) {
        int a = rand();
        if(cache.find({vec[i]}) != cache.end()) {
            // Found
            found++;
        }
    }
    cout << "Found: " << found << endl; 
    clock_t end = clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    cout << "Time taken: " << elapsed << " seconds" << endl;
}