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

        // [AJB] M1015: cardinality estimation accuracy tracking
        // AGM() gives the predicted upper bound on join result size.
        // As we enumerate, we track the ratio of actual found tuples to
        // the predicted AGM cardinality, reporting running accuracy.
        long long ajb_predicted_agm = (long long)idx.AGM();
        // Welford accumulator for per-batch accuracy ratios
        struct {
            long long n = 0; double mean = 0.0; double m2 = 0.0;
            void update(double x) {
                n++; double d = x - mean; mean += d / n;
                double d2 = x - mean; m2 += d * d2;
            }
            double variance() const { return n < 2 ? 0.0 : m2 / (n - 1); }
            double stddev() const { return n < 2 ? 0.0 : std::sqrt(m2 / (n - 1)); }
        } ajb_accuracy_acc;

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

            // [AJB_BP] M1015: at each 1000-iteration checkpoint, compute
            // current cardinality ratio (found/expected) and record
            if(i % 1000 == 0 || i == idx.AGM()) {
                // Expected found at position i: (i / AGM) * actual_total ≈ i for full enum
                // Accuracy ratio = found / (i * (totalFound/AGM)) where totalFound is unknown
                // Simpler: selectivity = found / i (fraction of probes that yield tuples)
                double selectivity = (double)found / i;
                ajb_accuracy_acc.update(selectivity);
                fprintf(stderr, "[AJB_BP][REnum] cardinality_accuracy: i=%d found=%d agm=%lld selectivity=%.6f mean_sel=%.6f stddev_sel=%.6f\n",
                        i, found, ajb_predicted_agm, selectivity,
                        ajb_accuracy_acc.mean, ajb_accuracy_acc.stddev());
            }

            if(i % 1000 == 0 || i == idx.AGM())
                fprintf(stderr, "[AJB_TRACE][REnum] enumerate progress: %d/%lld found=%d\n",
                        i, (long long)idx.AGM(), found);
        }
        if(!obuf.empty())
            fwrite(obuf.data(), 1, obuf.size(), stdout);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();

        // [AJB_BP] M1015: final cardinality estimation accuracy
        double final_ratio = ajb_predicted_agm > 0 ? (double)found / ajb_predicted_agm : 0.0;
        fprintf(stderr, "[AJB_BP][REnum] FINAL cardinality_accuracy: found=%d predicted_agm=%lld ratio=%.6f (1.0=perfect, <1=overestimate)\n",
                found, ajb_predicted_agm, final_ratio);
        fprintf(stderr, "[AJB_BP][REnum] selectivity_stats: mean=%.6f stddev=%.6f variance=%.8f\n",
                ajb_accuracy_acc.mean, ajb_accuracy_acc.stddev(), ajb_accuracy_acc.variance());

        fprintf(stderr, "[AJB_TIMER][REnum] enumerate done: found=%d/%lld elapsed=%.3fs\n",
                found, (long long)idx.AGM(), sec);
        return;
    }

    vector<int> sample() {
        return idx.sample(idx.getFullBucket());
    }
};
