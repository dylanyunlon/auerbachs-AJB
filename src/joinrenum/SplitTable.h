#ifndef SPLITTABLE_H
#define SPLITTABLE_H
// [AJB] SplitTable: per-relation split tracking for BucketPool
// 记录每个relation在每个splitDim上的值域边界(用于快速Split)
// Created by shai.zeevi on 04/06/2019.
#include <vector>
#include <iostream>
#include <cstdio>
using namespace std;

class SplitTable {
public:
    // splitRanges[dim] = 这个relation在维度dim上的split point列表(已排序)
    vector<vector<int> > splitRanges;
    int numDims;

    SplitTable() : numDims(0) {}
    SplitTable(int d) : numDims(d) {
        splitRanges.resize(d);
    }

    void addSplitPoint(int dim, int val) {
        if(dim >= 0 && dim < numDims)
            splitRanges[dim].push_back(val);
    }

    void sortAll() {
        for(int d = 0; d < numDims; d++){
            sort(splitRanges[d].begin(), splitRanges[d].end());
            splitRanges[d].erase(
                unique(splitRanges[d].begin(), splitRanges[d].end()),
                splitRanges[d].end());
        }
    }

    const vector<int>& getSplitPoints(int dim) const {
        return splitRanges[dim];
    }

    int numSplitPoints(int dim) const {
        return dim < numDims ? splitRanges[dim].size() : 0;
    }

    void print() const {
        cout << "SplitTable (" << numDims << " dims):" << endl;
        for(int d = 0; d < numDims; d++){
            cout << "  dim" << d << ": " << splitRanges[d].size() << " splits";
            if(!splitRanges[d].empty())
                cout << " range=[" << splitRanges[d].front() << "," << splitRanges[d].back() << "]";
            cout << endl;
        }
    }

    // [AJB] structured dump — 每维的split points数量和值域
    void ajb_dump(const char* label = "") const {
        fprintf(stderr, "[AJB_STATE][SplitTable] %s dims=%d\n", label, numDims);
        for(int d = 0; d < numDims; d++){
            fprintf(stderr, "[AJB_STATE][SplitTable]   dim%d: %zu splits",
                    d, splitRanges[d].size());
            if(!splitRanges[d].empty())
                fprintf(stderr, " [%d..%d]", splitRanges[d].front(), splitRanges[d].back());
            fprintf(stderr, "\n");
        }
    }

    // [AJB] 总split点数 — 用于评估预处理开销
    int ajb_total_splits() const {
        int total = 0;
        for(int d = 0; d < numDims; d++) total += splitRanges[d].size();
        return total;
    }
};

#endif // SPLITTABLE_H
