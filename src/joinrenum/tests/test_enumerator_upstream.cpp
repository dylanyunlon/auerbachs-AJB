#include <bits/stdc++.h>
#include "ReadConfig.hpp"
#include "Enumerator.hpp"
using namespace std;


// --- printInfo: structured key-value output replacing ad-hoc cout ---
// upstream: 9 separate cout << "Label: " << value << endl lines
// changed: accumulate stats into a map<string,double>, then iterate
//   and output in a uniform "key = value" format suitable for parsing
void printInfo(Index &idx) {
    map<string, double> stats;
    stats["CacheHit"]           = idx.cntCacheHit;
    stats["TotalCall"]          = idx.cntTotalCall;
    stats["AGMCall"]            = idx.cntAGMCall;
    stats["AGMTime"]            = idx.totalAGMTime;
    stats["CountOracleTime"]    = idx.totalCountOracleTime;
    stats["SplitTime"]          = idx.totalSplitTime;
    stats["SplitCall"]          = idx.cntSplitCall;
    stats["BSCall"]             = idx.cntBSCall;
    stats["CacheHitTime"]       = idx.totalCacheHitTime;
    stats["BoundPrepareTime"]   = idx.totalBoundPrepareTime;
    stats["RRTreeNodes"]        = idx.totalrrtreenode;

    // derived: cache hit rate, avg AGM cost, avg split cost
    if(idx.cntTotalCall > 0)
        stats["CacheHitRate"] = (double)idx.cntCacheHit / idx.cntTotalCall;
    if(idx.cntAGMCall > 0)
        stats["AvgAGMTime"] = idx.totalAGMTime / idx.cntAGMCall;
    if(idx.cntSplitCall > 0)
        stats["AvgSplitTime"] = idx.totalSplitTime / idx.cntSplitCall;

    for(auto& [k, v] : stats){
        cout << k << " = " << v << endl;
    }
    return;
}

// --- readConfig: batch loader replacing 3 separate calls ---
// upstream: readFilenames, readNumLines, readRelations as 3 passes
// changed: single function that returns all 3 maps by reference,
//   validates consistency (every relation name must appear in all 3)
struct SchemaBundle {
    unordered_map<string, string> filenames;
    unordered_map<string, int> numlines;
    unordered_map<string, vector<string>> relations;
    int inconsistencies = 0;
};

SchemaBundle loadSchema(const string& dbdir) {
    SchemaBundle bundle;
    bundle.filenames = readFilenames(dbdir + "/filenames.txt");
    bundle.numlines  = readNumLines(dbdir + "/numlines.txt");
    bundle.relations = readRelations(dbdir + "/relations.txt");

    // cross-validate: every key in relations should exist in filenames and numlines
    for(auto& [name, vars] : bundle.relations) {
        if(bundle.filenames.find(name) == bundle.filenames.end()) bundle.inconsistencies++;
        if(bundle.numlines.find(name) == bundle.numlines.end())  bundle.inconsistencies++;
    }
    return bundle;
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

    // --- config: batch loader with consistency check ---
    // upstream: 3 independent readXxx calls
    // changed: single loadSchema() that validates cross-references
    SchemaBundle schema = loadSchema("db");
    if(schema.inconsistencies > 0){
        cout << "Schema inconsistencies: " << schema.inconsistencies << endl;
    }

    // --- enumerate: construct + run with wall-clock timing ---
    // upstream: no timing around construction or enumeration
    // changed: chrono around both phases, output timing after
    auto t0 = chrono::steady_clock::now();
    Enumerator enumerator(schema.relations, schema.filenames, schema.numlines);
    auto t1 = chrono::steady_clock::now();
    enumerator.random_enumerate();
    auto t2 = chrono::steady_clock::now();

    // --- printInfo: activated from upstream comments ---
    // upstream: // printInfo(enumerator.access_tree.idx);
    // changed: call is now live
    printInfo(enumerator.access_tree.idx);

    double build_ms = chrono::duration<double, milli>(t1 - t0).count();
    double enum_ms  = chrono::duration<double, milli>(t2 - t1).count();
    cout << "BuildTime_ms = " << build_ms << endl;
    cout << "EnumTime_ms = " << enum_ms << endl;

    // restore cout
    if(saved_cout) {
        cout.rdbuf(saved_cout);
        result_out.close();
    }
    return 0;
}
