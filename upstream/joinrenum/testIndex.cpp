#include <bits/stdc++.h>
#include "Index.hpp"
#include "ReadConfig.hpp"

void printVector(const vector<int>& vec) {
    for (const auto& val : vec) {
        cout << val << " ";
    }
    cout << endl;
}

using namespace std;
int main() {
    
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");

    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});

    Index idx(q);

    idx.preProcessing(relations, filenames, numlines);

    vector<pair<vector<Point<int> >::iterator, vector<Point<int> >::iterator> > iters(idx.tables.size());
    vector<int> cardinalities(iters.size(), 0);
    for(size_t i = 0; i < iters.size(); i++) {
        iters[i] = make_pair(idx.tables[i].rt.points.begin(), idx.tables[i].rt.points.end());
    }
    // vector<vector<vector<int> > > points(idx.tables.size());
    // clock_t start = clock();
    // for(size_t i = 0; i < idx.tables.size(); i++) {
    //     points[i].resize(idx.q.getRelations()[i].size());
    //     for(size_t j = 0; j < points[i].size(); j++) {
    //         points[i][j].resize(idx.tables[i].rt.points.size());
    //         for(size_t k = 0; k < points[i][j].size(); k++) {
    //             points[i][j][k] = idx.tables[i].rt.points[k][j];
    //         }
    //     }
    // }
    // clock_t end = clock();
    // double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    // cout << "Build Elapsed time: " << elapsed_time << " seconds" << endl;
    vector<int> d = {0, -1, 0};
    cout << "AGM: " << idx.FB.AGM << endl;
    int testTime = 3500000;
    vector<int> test(testTime);
    for(int i = 0; i < testTime; i++) {
        test[i] = rand() % 2060495465;
    }
    vector<pair<vector<int>::iterator, vector<int>::iterator> > veciters(idx.tables.size());
    veciters[0] = make_pair(idx.data[0][0].begin(), idx.data[0][0].end());
    veciters[1] = make_pair(idx.data[1][0].begin(), idx.data[1][0].end());
    veciters[2] = make_pair(idx.data[2][0].begin(), idx.data[2][0].end());
    
    // veciters[0] = make_pair(points[0][0].begin(), points[0][0].end());
    // veciters[1] = make_pair(points[1][0].begin(), points[1][0].end());
    // veciters[2] = make_pair(points[2][0].begin(), points[2][0].end());
    // record the time
    vector<bool> flag = {1, 0, 1};
    auto start = clock();
    for(int i = 0; i < testTime; i++) {
        // int a = MultiHeadBinarySearch(iters, d, test[i], q);
        int b = MultiHeadBinarySearch(veciters, flag, test[i], q);
        // if(a != b) {
        //     cout << "ERROR: " << test[i] << ": " << a << " " << b << endl;
        // }
        // getpos(iters, d, ans, cardinalities);
        // getpos(veciters, flag, ans, cardinalities);
        // double ans1 = q.AGM(cardinalities);
        // int res1 = ceil(ans1) - ans1 < 1e-5 ? ceil(ans1) : int(ans1);
        // // getpos(iters, d, ans + 1, cardinalities);
        // getpos(veciters, flag, ans + 1, cardinalities);
        // double ans2 = q.AGM(cardinalities);
        // int res2 = ceil(ans2) - ans2 < 1e-5 ? ceil(ans2) : int(ans2);
        // if(res1 > test[i] || res2 <= test[i]) {
        //     cout << "ERROR: " << test[i] << ": " << res1 << " " << res2 << endl;
        // }
    }
    auto end = clock();
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;


    // Bucket B = idx.getFullBucket();
    // B.print();
    // idx.setAGMandIters(B);
    // B.print();
    // vector<vector<Point<int> >::iterator> begins;
    // for(int i = 0; i < q.getRelations().size(); i++) {
    //     begins.push_back(idx.tables[i].rt.points.begin());
    // }
    // vector<Bucket> sons = idx.splitBucket(B);
    // for(size_t i = 0; i < sons.size(); i++){
    //     sons[i].print();
    //     sons[i].printIters(begins);
    // }

    return 0;
}