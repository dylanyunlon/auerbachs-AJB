#include<bits/stdc++.h>
#include <sys/resource.h>
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

// --- writeDataToFile: snprintf to char buffer, batch write ---
// upstream: for each point, for each dim, file << points[i][j] << " "; file << endl;
// changed: format each line into a char buffer with snprintf, accumulate
//   into a string buffer, write in large chunks
void writeDataToFile(vector<Point<int> > points, string filename = "data.txt"){
    FILE* fp = fopen(filename.c_str(), "w");
    if(!fp) return;
    // 64KB write buffer
    vector<char> buf;
    buf.reserve(65536);
    for(int i = 0; i < (int)points.size(); i++){
        char tmp[32];
        for(int j = 0; j < points[i].dim(); j++){
            int n = snprintf(tmp, sizeof(tmp), "%d ", points[i][j]);
            buf.insert(buf.end(), tmp, tmp + n);
        }
        buf.push_back('\n');
        // flush when buffer is large
        if(buf.size() > 60000) {
            fwrite(buf.data(), 1, buf.size(), fp);
            buf.clear();
        }
    }
    if(!buf.empty())
        fwrite(buf.data(), 1, buf.size(), fp);
    fclose(fp);
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
    const int TARGET = 1000000;
    const int DIM = 10;
    const int RANGE = 1000;

    // --- dedup strategy: batch generate + sort + unique ---
    // upstream: while(size < 1M) { generate, set.find, set.insert }
    //   O(N log N) per insert into std::set, total O(N log^2 N)
    // changed: generate 1.1M points (with duplicates), sort lexicographically,
    //   unique to remove dups. If <1M remain, generate more and repeat.
    //   sort+unique is O(N log N) total, one pass.
    vector<Point<int> > points;
    points.reserve(TARGET + TARGET/10);

    while((int)points.size() < TARGET) {
        // generate a batch — overshoot by 10% to account for dups
        int batch = TARGET - (int)points.size();
        batch += batch / 10 + 1000;
        for(int i = 0; i < batch; i++){
            vector<int> v;
            v.reserve(DIM);
            for(int j = 0; j < DIM; j++){
                v.push_back(rand() % RANGE);
            }
            points.push_back(Point<int>(v));
        }
        // sort lexicographically
        sort(points.begin(), points.end());
        // unique: remove consecutive duplicates
        auto last = unique(points.begin(), points.end());
        points.erase(last, points.end());
    }
    // trim to exactly TARGET
    if((int)points.size() > TARGET)
        points.resize(TARGET);

    // shuffle so output isn't sorted (upstream output was insertion order)
    for(int i = (int)points.size() - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        swap(points[i], points[j]);
    }

    writeDataToFile(points);
}
