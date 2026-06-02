#include "Table.h"
// [AJB] REnum: top-level pipeline — ReadConfig→Query→Index→preProcessing→enumerate/sample
// 这是整个joinrenum系统的入口, 调试从这里开始
#include "Parcel.h"
#include "Index.hpp"
#include "ReadConfig.hpp"
#include <chrono>

using namespace std;
class RandOrderEnum {
    private:
    Index idx;

    public:
    RandOrderEnum(string filenames_dir, string numlines_dir, string relations_dir, vector<string> relationNames, vector<vector<string> > relationVars) {
        auto t0 = std::chrono::steady_clock::now();
        unordered_map<string, string> filenames = readFilenames(filenames_dir);
        unordered_map<string, int> numlines = readNumLines(numlines_dir);
        unordered_map<string, vector<string> > relations = readRelations(relations_dir);
        fprintf(stderr, "[AJB_BP][REnum] config loaded: %zu filenames, %zu numlines, %zu relations\n",
                filenames.size(), numlines.size(), relations.size());
        Query q(relationNames, relationVars);
        this->idx = Index(q);
        this->idx.preProcessing(relations, filenames, numlines);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "[AJB_TIMER][REnum] construction=%.3fms AGM=%lld\n", ms, (long long)idx.AGM());
    }

    void enumerate() {
        fprintf(stderr, "[AJB_BP][REnum] enumerate start: AGM=%lld\n", (long long)idx.AGM());
        auto t0 = std::chrono::steady_clock::now();
        int found = 0;
        for(int i = 1; i <= idx.AGM(); i++) {
            pair<bool, vector<int> > res = idx.randomAccess(idx.getFullBucket(), i);
            cout << i << ": ";
            cout << res.first << "::";
            for(int j = 0; j < res.second.size(); j++) {
                cout << res.second[j] << ",";
            }
            cout << endl;
            if(res.first) found++;
            // [AJB_TRACE] 每1000次输出进度
            if(i % 1000 == 0 || i == idx.AGM())
                fprintf(stderr, "[AJB_TRACE][REnum] enumerate progress: %d/%lld found=%d\n",
                        i, (long long)idx.AGM(), found);
        }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stderr, "[AJB_TIMER][REnum] enumerate done: found=%d/%lld elapsed=%.3fs\n",
                found, (long long)idx.AGM(), sec);
        return;
    }

    vector<int> sample() {
        return idx.sample(idx.getFullBucket());
    }
};
