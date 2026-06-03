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

    // --- derived ratio: cache hit rate ---
    // upstream: only printed raw counts
    // changed: compute and output the ratio directly
    if(idx.cntTotalCall > 0) {
        double rate = (double)idx.cntCacheHit / idx.cntTotalCall;
        cout << "Cache Hit Rate: " << rate << endl;
    }
    return;
}

int main() {
    // --- output redirect: ofstream + rdbuf replacing freopen ---
    // upstream: freopen("res/result.txt", "w", stdout)
    // changed: ofstream redirect so stderr remains functional
    ofstream result_out("res/result.txt");
    streambuf* saved_cout = nullptr;
    if(result_out.is_open()) {
        saved_cout = cout.rdbuf(result_out.rdbuf());
    } else {
        cout << "WRITEERR" << endl;
    }

    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");

    Enumerator enumerator(relations, filenames, numlines);
    enumerator.random_enumerate();

    // --- printInfo: activated from upstream comments ---
    // upstream: // printInfo(enumerator.access_tree.idx);
    // changed: call is now live
    printInfo(enumerator.access_tree.idx);

    // restore cout
    if(saved_cout) {
        cout.rdbuf(saved_cout);
        result_out.close();
    }
    return 0;
}
