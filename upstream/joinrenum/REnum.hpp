#include "Table.h"
#include "Parcel.h"
#include "Index.hpp"
#include "ReadConfig.hpp"

using namespace std;
class RandOrderEnum {
    private:
    Index idx;

    public:
    RandOrderEnum(string filenames_dir, string numlines_dir, string relations_dir, vector<string> relationNames, vector<vector<string> > relationVars) {        
        unordered_map<string, string> filenames = readFilenames(filenames_dir);
        unordered_map<string, int> numlines = readNumLines(numlines_dir);
        unordered_map<string, vector<string> > relations = readRelations(relations_dir);
        Query q(relationNames, relationVars);
        this->idx = Index(q);
        this->idx.preProcessing(relations, filenames, numlines);
    }

    void enumerate() {
        for(int i = 1; i <= idx.AGM(); i++) {
            pair<bool, vector<int> > res = idx.randomAccess(idx.getFullBucket(), i);
            cout << i << ": ";
            cout << res.first << "::";
            for(int j = 0; j < res.second.size(); j++) {
                cout << res.second[j] << ",";
            }
            cout << endl;
        }
        return;
    }

    vector<int> sample() {
        return idx.sample(idx.getFullBucket());
    }
};