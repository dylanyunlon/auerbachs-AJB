#include <bits/stdc++.h>
#include "RRAccessTree.hpp"
#include "ReadConfig.hpp"
using namespace std;
int main() {
    
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");

    RRAccessTree tree(relations, filenames, numlines);

    int agm = tree.AGM;

    // --- Phase 1: binary probing to find failure boundary ---
    // upstream: linear for(i=1; i<=AGM; i++) RRAccess(i)
    // changed: exponential jump (1,2,4,8,...) to locate the first i
    //   where RRAccess fails, then linear scan around that boundary
    int first_fail = agm + 1;
    for(int probe = 1; probe <= agm; probe *= 2) {
        pair<bool, vector<int> > res = tree.RRAccess(probe);
        if(!res.first) {
            first_fail = probe;
            break;
        }
    }

    // linear scan from max(1, first_fail/2) to find exact boundary
    int scan_start = max(1, first_fail / 2);
    int exact_boundary = agm + 1;
    for(int i = scan_start; i <= min(first_fail, agm); i++) {
        pair<bool, vector<int> > res = tree.RRAccess(i);
        if(!res.first) {
            exact_boundary = i;
            break;
        }
    }

    // --- Phase 2: full sweep with batch result collection ---
    // upstream: cout inside loop body per iteration
    // changed: collect all results into a vector, output after loop
    struct Result { int rank; bool ok; vector<int> vals; };
    vector<Result> results;
    results.reserve(agm);
    for(int i = 1; i <= agm; i++) {
        pair<bool, vector<int> > res = tree.RRAccess(i);
        results.push_back({i, res.first, res.second});
    }

    // batch output
    for(auto& r : results) {
        cout << r.rank << ": " << r.ok << ", ";
        for(size_t j = 0; j < r.vals.size(); j++) {
            cout << r.vals[j] << ",";
        }
        cout << endl;
    }

    // report boundary findings
    if(exact_boundary <= agm) {
        cout << "Failure boundary at rank " << exact_boundary << endl;
    }

    tree.print();

    // --- printBucketTree: activated from upstream comments ---
    // upstream: // tree.idx.printBucketTree(tree.idx.getFullBucket());
    tree.idx.printBucketTree(tree.idx.getFullBucket());
    
    return 0;
}
