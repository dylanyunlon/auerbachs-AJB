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
        // --- output strategy: batch buffered write replacing per-line cout ---
        // upstream: for each i, cout << i << ": " << res.first << ...
        // changed: format each line into a char buffer, accumulate, flush
        //   every 32KB. Avoids iostream overhead in tight loops.
        Bucket fb = idx.getFullBucket();
        vector<char> obuf;
        obuf.reserve(32768);
        char tmp[256];
        for(int i = 1; i <= idx.AGM(); i++) {
            pair<bool, vector<int> > res = idx.randomAccess(fb, i);
            int n = snprintf(tmp, sizeof(tmp), "%d: %d::", i, (int)res.first);
            obuf.insert(obuf.end(), tmp, tmp + n);
            for(int j = 0; j < (int)res.second.size(); j++) {
                n = snprintf(tmp, sizeof(tmp), "%d,", res.second[j]);
                obuf.insert(obuf.end(), tmp, tmp + n);
            }
            obuf.push_back('\n');
            if(res.first) found++;
            // flush buffer periodically
            if(obuf.size() > 30000) {
                fwrite(obuf.data(), 1, obuf.size(), stdout);
                obuf.clear();
            }
            if(i % 1000 == 0 || i == idx.AGM())
                fprintf(stderr, "[AJB_TRACE][REnum] enumerate progress: %d/%lld found=%d\n",
                        i, (long long)idx.AGM(), found);
        }
        if(!obuf.empty())
            fwrite(obuf.data(), 1, obuf.size(), stdout);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        // M1015: cardinality estimation accuracy ratio
        // AGM is the upper bound; found is the actual count of valid tuples
        // ratio < 1.0: good (AGM is tight); ratio << 1.0: loose bound
        long long agm = (long long)idx.AGM();
        double accuracy_ratio = agm > 0 ? (double)found / agm : 0.0;
        double tightness = 1.0 - accuracy_ratio;  // 0=perfectly tight, 1=completely loose
        fprintf(stderr, "[AJB_BP][REnum] cardinality_accuracy: agm=%lld actual=%d ratio=%.6f tightness=%.6f\n",
                agm, found, accuracy_ratio, tightness);
        if(accuracy_ratio < 0.01)
            fprintf(stderr, "[AJB_BP][REnum] WARN: very loose AGM bound (actual=%.2f%% of AGM)\n",
                    accuracy_ratio * 100.0);
        else if(accuracy_ratio > 0.8)
            fprintf(stderr, "[AJB_BP][REnum] INFO: tight AGM bound (actual=%.2f%% of AGM)\n",
                    accuracy_ratio * 100.0);
        fprintf(stderr, "[AJB_TIMER][REnum] enumerate done: found=%d/%lld elapsed=%.3fs throughput=%.0f tuples/s\n",
                found, agm, sec, sec > 0 ? found / sec : 0.0);
        return;
    }

    vector<int> sample() {
        return idx.sample(idx.getFullBucket());
    }
};
