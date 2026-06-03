#include <bits/stdc++.h>
#include "Index.hpp"
#include "ReadConfig.hpp"
using namespace std;
int main() {
    
    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});
    

    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");

    Index idx = Index(q);
    idx.preProcessing(relations, filenames, numlines);

    vector<CountOracle<int>*> CO = idx.getCountOracles();

    q.print();

    // --- neighbor traversal: activated from upstream comments ---
    for(int i = 0; i < (int)q.getRelNames().size(); i++) {
        vector<int> neighbors = q.getNeighborRels(i);
        for(int j = 0; j < (int)neighbors.size(); j++) {
            cout << "Relation " << i << " has neighbor: " << neighbors[j] << endl;
        }
    }

    JoinTree tree = idx.jt;
    tree.print();
    tree.printChildren();

    // --- CountOracle printing: activated from upstream comments ---
    for(int i = 0; i < (int)CO.size(); i++) {
        cout << "Count Oracle of R" << i << "---------------" << endl;
        CO[i]->print();
    }

    Bucket B = idx.getFullBucket();
    B.print();
    vector<vector<int> > relation = q.getRelations();

    // --- bound extraction: reverse iteration + direct index fill ---
    // upstream: forward loop i=0..nrel, inner loop j with push_back
    // changed: reverse iterate relations (last to first) and fill by
    //   pre-sized vectors using operator[] instead of push_back
    int nrel = (int)relation.size();
    vector<pair<vector<int>, vector<int> > > bound(nrel);
    for(int i = nrel - 1; i >= 0; i--) {
        int ncols = (int)relation[i].size();
        bound[i].first.resize(ncols);
        bound[i].second.resize(ncols);
        for(int j = ncols - 1; j >= 0; j--) {
            bound[i].first[j]  = B.getLowerBound()[relation[i][j]];
            bound[i].second[j] = B.getUpperBound()[relation[i][j]];
        }
    }
    // print in forward order (same output as upstream)
    for(int i = 0; i < nrel; i++) {
        cout << "Lower bound of relation " << i << ": ";
        for(int j = 0; j < (int)bound[i].first.size(); j++) {
            cout << bound[i].first[j] << " ";
        }
        cout << endl;
        cout << "Upper bound of relation " << i << ": ";
        for(int j = 0; j < (int)bound[i].second.size(); j++) {
            cout << bound[i].second[j] << " ";
        }
        cout << endl;
    }

    // --- treeUpp cross-validation: two representations, check agreement ---
    // upstream: just prints both results
    // changed: compute both, check they agree (same data, different rep)
    double upp_bound = tree.treeUpp(B.splitDim, bound);
    double upp_iter  = tree.treeUpp(B.splitDim, B.iters);
    cout << upp_bound << endl;
    cout << upp_iter << endl;
    if(abs(upp_bound) + abs(upp_iter) > 1e-12) {
        double rdiff = abs(upp_bound - upp_iter) / max(abs(upp_bound), abs(upp_iter));
        if(rdiff > 1e-9) {
            cout << "DIVERGENCE: bound_path=" << upp_bound
                 << " iter_path=" << upp_iter
                 << " rel_diff=" << rdiff << endl;
        }
    }
    return 0;
}
