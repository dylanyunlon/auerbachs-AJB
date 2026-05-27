#include<bits/stdc++.h>
#include <sys/resource.h>
#include "LexRangeTree.hpp"
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
int main(){
    vector<Point<int> > points;
    set<Point<int> > S;
    while(points.size() < 1000000){
        vector<int> v;
        for(int j=0;j<10;j++){
            v.push_back(rand()%1000);
        }
        if(S.find(Point<int>(v)) == S.end()){
            S.insert(Point<int>(v));
            points.push_back(Point<int>(v));
        }
    }
    // sort(points.begin(), points.end());
    writeDataToFile(points);
}