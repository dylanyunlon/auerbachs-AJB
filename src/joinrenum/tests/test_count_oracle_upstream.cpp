#include<bits/stdc++.h>
#include <sys/resource.h>
#include <malloc.h>
#include "CountOracle.hpp"
using namespace std;
int memoryUsage() {
    ifstream file("/proc/self/status");
    string line;
    while (getline(file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            istringstream iss(line);
            string key;
            int value;
            string unit;
            iss >> key >> value >> unit;
            return value;
        }
    }
    return -1;
}
void writeDataToFile(vector<Point<int> > points, string filename = "data.txt"){
    ofstream file;
    file.open(filename);
    for(int i = 0; i < points.size(); i++){
        for(int j = 0; j < points[i].dim(); j++){
            file << points[i][j] << " ";
        }
        file << endl;
    }
    file.close();
}
// --- readDataFromFile: strtol parse replacing istringstream ---
// upstream: getline -> istringstream -> while(iss>>num) push_back
// changed: getline -> strtol walk over char* with pointer arithmetic
void readDataFromFile(vector<Point<int> >& points, string filename = "data.txt"){
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        vector<int> v;
        const char* p = line.c_str();
        char* end;
        while(*p) {
            while(*p == ' ' || *p == '\t') p++;
            if(*p == '\0' || *p == '\n') break;
            long val = strtol(p, &end, 10);
            if(end == p) break;   // no conversion
            v.push_back((int)val);
            p = end;
        }
        if(!v.empty())
            points.push_back(Point<int>(v));
    }
    file.close();
}

// --- generateRange: all-dim independent dual-point generation ---
// upstream: pick divdim, dims < divdim get single point, divdim gets
//   ordered pair, dims > divdim get full [low,up] range
// changed: every dimension gets an independent random [lo,hi] pair,
//   then randomly collapse half the dimensions to single-point
//   (achieving similar selectivity with a different traversal pattern)
pair<Point<int>, Point<int> > generateRange(CountOracle<int> &tree){
    Point<int> lowbound = tree.getLowerBounds(), upbound = tree.getUpperBounds();
    int ndim = lowbound.dim();
    vector<int> vl(ndim), vr(ndim);

    // phase 1: every dim gets independent [lo, hi]
    for(int i = 0; i < ndim; i++){
        int span = upbound[i] - lowbound[i] + 1;
        int a = rand() % span + lowbound[i];
        int b = rand() % span + lowbound[i];
        if(a > b) swap(a, b);
        vl[i] = a;
        vr[i] = b;
    }

    // phase 2: collapse some dims to single-point (narrow the range)
    // pick how many to collapse: at least 1, at most ndim-1
    int ncollapse = 1 + rand() % max(1, ndim - 1);
    // walk dims in a shuffled order and collapse the first ncollapse
    vector<int> dim_order(ndim);
    iota(dim_order.begin(), dim_order.end(), 0);
    for(int i = ndim - 1; i > 0; i--){
        int j = rand() % (i + 1);
        swap(dim_order[i], dim_order[j]);
    }
    for(int k = 0; k < ncollapse && k < ndim; k++){
        int d = dim_order[k];
        int mid = vl[d] + (vr[d] - vl[d]) / 2;
        vl[d] = mid;
        vr[d] = mid;
    }

    return make_pair(Point<int>(vl), Point<int>(vr));
}

// --- range volume for sorting ---
static long long rangeVolume(const pair<Point<int>, Point<int> >& r) {
    long long vol = 1;
    for(int d = 0; d < r.first.dim(); d++){
        vol *= (long long)(r.second[d] - r.first[d] + 1);
        if(vol > 1e15) return (long long)1e15;  // cap overflow
    }
    return vol;
}

int main(){
    vector<Point<int> > points;
    readDataFromFile(points);
    clock_t start, end;
    start = clock();
    CountOracle<int> tree(points);
    end = clock();
    cout << "Time used: " << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;
    vector<Point<int>>().swap(points);
    malloc_trim(0);
    cout << "Memory usage: " << memoryUsage() << " KB" << endl;
    vector<pair<Point<int>, Point<int> > > ranges;
    int rangeNum = 100000;
    for(int i = 0; i < rangeNum; i++){
        ranges.push_back(generateRange(tree));
    }

    // --- query loop: sort ranges by volume descending, then query ---
    // upstream: sequential for(i=0..rangeNum) tree.count(ranges[i])
    // changed: sort by descending volume so large ranges (likely
    //   touching more tree nodes) are queried first — groups similar
    //   tree-traversal patterns together for better branch prediction
    sort(ranges.begin(), ranges.end(),
         [](const pair<Point<int>,Point<int>>& a,
            const pair<Point<int>,Point<int>>& b){
             return rangeVolume(a) > rangeVolume(b);
         });

    start = clock();
    for(int i = 0; i < (int)ranges.size(); i++){
        tree.count(ranges[i].first, ranges[i].second);
    }
    end = clock();
    cout << "Time used: " << (double)(end - start) / CLOCKS_PER_SEC * 1000 / rangeNum << " ms" << endl;
    return 0;
}
