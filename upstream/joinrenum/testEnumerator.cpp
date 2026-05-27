#include <bits/stdc++.h>
#include "ReadConfig.hpp"
#include "Enumerator.hpp"
using namespace std;


void printInfo(Index &idx) {
    
    cout << "Cache Hit of SplitBucket: " << idx.cntCacheHit << " Total Call: " << idx.cntTotalCall << endl;

    cout << "Total AGM Call: " << idx.cntAGMCall << endl;
    cout << "Total AGM Time: " << idx.totalAGMTime << endl;
    cout << "Total Count Oracle Time: " << idx.totalCountOracleTime << endl;
    cout << "Total Split Time: " << idx.totalSplitTime << endl;
    cout << "Total Split Call: " << idx.cntSplitCall << endl;
    cout << "Total Binary Search Loop: " << idx.cntBSCall << endl;
    cout << "Total Cache Hit Time: " << idx.totalCacheHitTime << endl;
    cout << "Total Bound Prepare Time: " << idx.totalBoundPrepareTime << endl;
    cout << "Total RRTree Nodes: " << idx.totalrrtreenode << endl;
    return;
}

int main() {
    if(freopen("res/result.txt", "w", stdout) == NULL)cout << "WRITEERR" << endl;

    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");

    Enumerator enumerator(relations, filenames, numlines);
    enumerator.random_enumerate();
    // printInfo(enumerator.access_tree.idx);
    return 0;
}