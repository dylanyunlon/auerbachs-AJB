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
            return value; // 返回当前的内存使用量，单位是 KB
        }
    }
    return -1; // 如果未找到 VmRSS 字段，则返回 -1
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
void readDataFromFile(vector<Point<int> >& points, string filename = "data.txt"){
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        vector<int> v;
        int num;
        while (iss >> num) {
            v.push_back(num);
        }
        points.push_back(Point<int>(v));
    }
    file.close();
}

pair<Point<int>, Point<int> > generateRange(CountOracle<int> &tree){
    Point<int> lowbound = tree.getLowerBounds(), upbound = tree.getUpperBounds();
    int divdim = rand() % lowbound.dim(), divval, divval2;
    vector<int> vl, vr;
    for(int i = 0; i < divdim; i++){
        divval = rand() % (upbound[i] - lowbound[i] + 1) + lowbound[i];
        vl.push_back(divval);
        vr.push_back(divval);
    }
    divval = rand() % (upbound[divdim] - lowbound[divdim] + 1) + lowbound[divdim];
    divval2 = rand() % (upbound[divdim] - divval + 1) + divval;
    if(divval > divval2){
        swap(divval, divval2);
    }
    vl.push_back(divval);
    vr.push_back(divval2);
    for(int i = divdim + 1; i < lowbound.dim(); i++){
        vl.push_back(lowbound[i]);
        vr.push_back(upbound[i]);
    }
    return make_pair(Point<int>(vl), Point<int>(vr));
}

int main(){
    vector<Point<int> > points;
    readDataFromFile(points);
    // record the time used
    // set<Point<int> > S;
    // while(points.size() < 20){
    //     vector<int> v;
    //     for(int j=0;j<3;j++){
    //         v.push_back(rand()%3);
    //     }
    //     if(S.find(Point<int>(v)) == S.end()){
    //         S.insert(Point<int>(v));
    //         points.push_back(Point<int>(v));
    //     }
    // }
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
    start = clock();
    for(int i = 0; i < ranges.size(); i++){
        tree.count(ranges[i].first, ranges[i].second);
    }
    end = clock();
    cout << "Time used: " << (double)(end - start) / CLOCKS_PER_SEC * 1000 / rangeNum << " ms" << endl;
    // tree.print();
    // for(int i = 0; i < 10; i++){
    //     pair<Point<int>, Point<int> > range = generateRange(tree);
    //     cout << "Range: ";
    //     range.first.print();
    //     cout << " to ";
    //     range.second.print();
    //     cout << "Count: " << tree.count(range.first, range.second) << endl;
    // }
    return 0;
}