#include <bits/stdc++.h>
#include "RRAccessTree.hpp"
#include "ReadConfig.hpp"
using namespace std;
int main() {
    // Bucket B = Bucket({0, 0}, {10, 10});
    // B.print();
    // RRAccessTreeNode* node = new RRAccessTreeNode(B, 0, vector<pair<Bucket, int> >(), vector<RRAccessTreeNode*>());
    // node->print();
    
    
    // Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});
    
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");

    RRAccessTree tree(relations, filenames, numlines);



    for(int i = 1; i <= tree.AGM; i++) {
        pair<bool, vector<int> > res = tree.RRAccess(i);
        cout << i << ": " << res.first << ", ";
        for(int j = 0; j < res.second.size(); j++) {
            cout << res.second[j] << ",";
        }
        cout << endl;
    }

    tree.print();
    // tree.idx.printBucketTree(tree.idx.getFullBucket());
    
    return 0;
}