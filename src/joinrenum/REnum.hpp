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

    // --- M1128: Lagrangian relaxation lower bound ---
    // Computes a relaxed bound by dualizing the join constraints.
    // For each variable, we assign a Lagrange multiplier (price) and
    // compute the relaxed cost = sum of per-relation costs adjusted
    // by multiplier weights.  This gives a valid lower bound on the
    // optimal join plan cost, useful for branch-and-bound pruning.
    struct LagrangianRelaxation {
        std::vector<double> multipliers;    // one per variable
        double best_bound = 0.0;            // best relaxed bound found
        int iterations = 0;
        double step_size = 1.0;             // subgradient step size

        void initialize(int num_variables) {
            multipliers.assign(num_variables, 1.0);
            best_bound = 0.0;
            iterations = 0;
            step_size = 1.0;
        }

        // Compute relaxed bound given current multipliers and relation costs
        double computeBound(const std::vector<double>& relation_costs,
                           const std::vector<std::vector<int>>& var_to_rel_map) {
            double bound = 0.0;
            for (size_t r = 0; r < relation_costs.size(); ++r) {
                // Each relation's relaxed cost is its base cost weighted by
                // the product of multipliers of its variables
                double weight = 1.0;
                for (int v : var_to_rel_map[r]) {
                    if (v >= 0 && v < (int)multipliers.size())
                        weight *= multipliers[v];
                }
                bound += relation_costs[r] * weight;
            }
            return bound;
        }

        // Subgradient update: move multipliers toward constraint violation
        void subgradientStep(const std::vector<double>& violations) {
            iterations++;
            // Diminishing step size: step / sqrt(iterations)
            double effective_step = step_size / std::sqrt(static_cast<double>(iterations));

            double violation_norm_sq = 0.0;
            for (double v : violations) violation_norm_sq += v * v;
            if (violation_norm_sq < 1e-12) return;

            for (size_t i = 0; i < multipliers.size() && i < violations.size(); ++i) {
                multipliers[i] += effective_step * violations[i] / std::sqrt(violation_norm_sq);
                // Keep multipliers non-negative
                if (multipliers[i] < 0.0) multipliers[i] = 0.0;
            }
        }

        void dump() const {
            fprintf(stderr, "[AJB_BP][Lagrangian] iterations=%d best_bound=%.4f step=%.6f\n",
                    iterations, best_bound, step_size);
        }
    };

    LagrangianRelaxation lagrangian;

    // --- M1129: Iterative convergence detection ---
    // Monitors the change in bound/cost across iterations.
    // Stops early when delta < epsilon for k consecutive iterations.
    struct ConvergenceDetector {
        double epsilon = 1e-6;        // convergence threshold
        int required_stable = 5;      // consecutive stable iterations needed
        int stable_count = 0;         // current streak of stable iterations
        double prev_value = 1e18;     // previous iteration's value
        int total_iterations = 0;
        bool converged = false;

        bool check(double current_value) {
            total_iterations++;
            double delta = std::abs(current_value - prev_value);

            if (delta < epsilon) {
                stable_count++;
                if (stable_count >= required_stable && !converged) {
                    converged = true;
                    fprintf(stderr, "[AJB_BP][Convergence] CONVERGED at iter=%d delta=%.2e "
                            "value=%.6f (stable for %d iters)\n",
                            total_iterations, delta, current_value, stable_count);
                }
            } else {
                stable_count = 0;
            }
            prev_value = current_value;
            return converged;
        }

        void reset() {
            stable_count = 0;
            prev_value = 1e18;
            total_iterations = 0;
            converged = false;
        }

        void dump() const {
            fprintf(stderr, "[AJB_BP][Convergence] iters=%d stable=%d converged=%s prev=%.6f\n",
                    total_iterations, stable_count,
                    converged ? "YES" : "NO", prev_value);
        }
    };

    ConvergenceDetector convergence;
};
