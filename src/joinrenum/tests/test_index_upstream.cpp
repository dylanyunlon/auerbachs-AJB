#include <bits/stdc++.h>
#include "Index.hpp"
#include "ReadConfig.hpp"
#define BINARY_SEARCH_NO_MAIN
#include "BinarySearch.cpp"

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

    // --- columnar transpose: activated from upstream comments ---
    // upstream had this entire block commented out; now runs to build
    // a transposed column-major layout for cross-validation
    vector<vector<vector<int> > > points(idx.tables.size());
    clock_t col_start = clock();
    for(size_t i = 0; i < idx.tables.size(); i++) {
        points[i].resize(idx.q.getRelations()[i].size());
        for(size_t j = 0; j < points[i].size(); j++) {
            points[i][j].resize(idx.tables[i].rt.points.size());
            for(size_t k = 0; k < points[i][j].size(); k++) {
                points[i][j][k] = idx.tables[i].rt.points[k][j];
            }
        }
    }
    clock_t col_end = clock();
    double col_elapsed = double(col_end - col_start) / CLOCKS_PER_SEC;
    cout << "Build Elapsed time: " << col_elapsed << " seconds" << endl;

    vector<int> d = {0, -1, 0};
    cout << "AGM: " << idx.FB.AGM << endl;

    // --- test value generation: LCG replacing rand() ---
    // upstream: test[i] = rand() % 2060495465
    // changed: linear congruential generator, no libc call per value
    //   x_{n+1} = (a * x_n + c) mod m
    int testTime = 3500000;
    vector<int> test(testTime);
    {
        uint64_t x = 12345;  // seed
        const uint64_t a = 6364136223846793005ULL;
        const uint64_t c = 1442695040888963407ULL;
        for(int i = 0; i < testTime; i++) {
            x = a * x + c;
            test[i] = (int)((x >> 33) % 2060495465);
        }
    }

    vector<pair<vector<int>::iterator, vector<int>::iterator> > veciters(idx.tables.size());
    veciters[0] = make_pair(idx.data[0][0].begin(), idx.data[0][0].end());
    veciters[1] = make_pair(idx.data[1][0].begin(), idx.data[1][0].end());
    veciters[2] = make_pair(idx.data[2][0].begin(), idx.data[2][0].end());

    // columnar iterators for the second half
    vector<pair<vector<int>::iterator, vector<int>::iterator> > coliters(idx.tables.size());
    coliters[0] = make_pair(points[0][0].begin(), points[0][0].end());
    coliters[1] = make_pair(points[1][0].begin(), points[1][0].end());
    coliters[2] = make_pair(points[2][0].begin(), points[2][0].end());

    vector<bool> flag = {1, 0, 1};

    // --- MHBS loop: split in half, first half uses veciters, ---
    // --- second half switches to columnar iters for cross-check ---
    // upstream: single loop, all veciters
    // changed: two-phase loop with data source switch at midpoint
    int halfTime = testTime / 2;
    auto start = clock();
    int mismatch = 0;
    for(int i = 0; i < halfTime; i++) {
        int b = MultiHeadBinarySearch(veciters, test[i]);
        (void)b;
    }
    for(int i = halfTime; i < testTime; i++) {
        int b_vec = MultiHeadBinarySearch(veciters, test[i]);
        int b_col = MultiHeadBinarySearch(coliters, test[i]);
        if(b_vec != b_col) mismatch++;
    }
    auto end = clock();
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
    if(mismatch > 0)
        cout << "Cross-validation mismatches: " << mismatch << endl;

    // --- splitBucket: activated from upstream comments ---
    // upstream had this block commented out; now exercises the full
    // bucket -> split -> walk children pipeline
    Bucket B = idx.getFullBucket();
    B.print();
    idx.setAGMandIters(B);
    B.print();
    vector<vector<Point<int> >::iterator> begins;
    for(int i = 0; i < (int)q.getRelations().size(); i++) {
        begins.push_back(idx.tables[i].rt.points.begin());
    }
    vector<Bucket> sons = idx.splitBucket(B);
    for(size_t i = 0; i < sons.size(); i++){
        sons[i].print();
        sons[i].printIters(begins);
    }

    return 0;
}
