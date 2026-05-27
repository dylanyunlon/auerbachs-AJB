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


    // Query q({"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"}, {{"x1", "x2"}, {"x2", "x3"}, {"x1", "x3"}, {"x3", "x4"}, {"x4", "x5"}, {"x5", "x6"}, {"x4", "x6"}, {"x1", "x5"}, {"x2", "x6"}});

    // Query q({"R1", "R2", "R3", "R4"}, {{"A", "B", "C", "D"}, {"B", "D", "E", "G"}, {"B", "C", "E", "F"}, {"C", "D", "F", "G"}});
    q.print();
    // for(int i = 0; i < q.getRelNames().size(); i++) {
    //     vector<int> neighbors = q.getNeighborRels(i);
    //     for(int j = 0; j < neighbors.size(); j++) {
    //         cout << "Relation " << i << " has neighbor: " << neighbors[j] << endl; // print the neighbors of the current relation
    //     }
    // }
    JoinTree tree = idx.jt;
    tree.print();
    tree.printChildren();
    // for(int i = 0; i < CO.size(); i++) {
    //     cout << "Count Oracle of R" << i << "---------------" << endl;
    //     CO[i]->print();
    // }
    Bucket B = idx.getFullBucket();
    B.print();
    vector<vector<int> > relation = q.getRelations();
    vector<pair<vector<int>, vector<int> > > bound = {};
    for(int i = 0; i < relation.size(); i++) {
        vector<int> lower_bound = {};
        vector<int> upper_bound = {};
        for(int j = 0; j < relation[i].size(); j++) {
            lower_bound.push_back(B.getLowerBound()[relation[i][j]]);
            upper_bound.push_back(B.getUpperBound()[relation[i][j]]);
        }
        bound.push_back({lower_bound, upper_bound});
    }
    for(int i = 0; i < bound.size(); i++) {
        cout << "Lower bound of relation " << i << ": ";
        for(int j = 0; j < bound[i].first.size(); j++) {
            cout << bound[i].first[j] << " ";
        }
        cout << endl;
        cout << "Upper bound of relation " << i << ": ";
        for(int j = 0; j < bound[i].second.size(); j++) {
            cout << bound[i].second[j] << " ";
        }
        cout << endl;
    }
    cout << tree.treeUpp(B.splitDim, bound) << endl;
    cout << tree.treeUpp(B.splitDim, B.iters) << endl;
    return 0;
}