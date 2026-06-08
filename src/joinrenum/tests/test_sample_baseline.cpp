// =============================================================================
// test_sample_baseline.cpp — Random sampling baseline (AJB-instrumented)
//
// Origin: upstream/joinrenum/test.cpp lines 165-178 (commented Sample section)
// AJB adaptation (~25%): sorted flat_vector + std::lower_bound replaces
//   set<vector<int>> for dedup (cache-friendly for large result sets).
//   Collision rate EMA tracking, early termination when collision rate > 0.99,
//   [AJB_STATE] dumps with dedup stats, RSS memory profile, convergence
//   detection via success rate plateau.
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
#include "Table.h"
#include "Parcel.h"
#include "Index.hpp"
#include "ReadConfig.hpp"

using namespace std;

// AJB: sorted-vector dedup — replaces set<vector<int>> for better cache behavior
struct FlatDedup {
    vector<vector<int>> results;
    bool sorted_flag;

    FlatDedup() : sorted_flag(true) {}

    bool insert_unique(const vector<int>& v) {
        // binary search in sorted portion
        auto it = lower_bound(results.begin(), results.end(), v);
        if (it != results.end() && *it == v) {
            return false;  // duplicate
        }
        results.insert(it, v);  // insert in sorted position
        return true;
    }

    size_t size() const { return results.size(); }

    // AJB: memory estimate (bytes)
    size_t memory_bytes() const {
        size_t total = sizeof(results) + results.capacity() * sizeof(vector<int>);
        for (const auto& r : results) {
            total += r.capacity() * sizeof(int);
        }
        return total;
    }
};

struct AjbSampleState {
    int total_attempts;
    int successes;
    int collisions;          // sample was valid but already in result set
    double ema_collision_rate;
    double alpha;
    int window_attempts;
    int window_collisions;

    AjbSampleState() : total_attempts(0), successes(0), collisions(0),
                       ema_collision_rate(0), alpha(0.02),
                       window_attempts(0), window_collisions(0) {}

    void record_collision() {
        collisions++;
        window_collisions++;
        update_ema();
    }

    void record_success() {
        successes++;
        update_ema();
    }

    void record_attempt() {
        total_attempts++;
        window_attempts++;
    }

    bool should_terminate() const {
        // AJB: early termination when collision rate > 0.99
        // (almost all samples are duplicates)
        return total_attempts > 1000 && ema_collision_rate > 0.99;
    }

    void dump(const char* tag, double elapsed_s, size_t dedup_mem) const {
        double throughput = (elapsed_s > 0) ? successes / elapsed_s : 0;
        fprintf(stderr,
            "[AJB_STATE][Sample] %s: attempts=%d successes=%d collisions=%d "
            "collision_rate=%.4f throughput=%.1f/s dedup_mem_kb=%zu\n",
            tag, total_attempts, successes, collisions,
            ema_collision_rate, throughput, dedup_mem / 1024);
    }

private:
    void update_ema() {
        if (window_attempts >= 100) {
            double rate = (double)window_collisions / window_attempts;
            ema_collision_rate = alpha * rate + (1.0 - alpha) * ema_collision_rate;
            window_attempts = 0;
            window_collisions = 0;
        }
    }
};

static long getPeakRSSKB() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

void printInfo(Index &idx) {
    fprintf(stderr, "[AJB_STATE] CacheHit=%d TotalCall=%d AGMCall=%d "
            "AGMTime=%.4f COTime=%.4f SplitTime=%.4f SplitCall=%d\n",
            idx.cntCacheHit, idx.cntTotalCall, idx.cntAGMCall,
            idx.totalAGMTime, idx.totalCountOracleTime,
            idx.totalSplitTime, idx.cntSplitCall);
}

int main() {
    fprintf(stderr, "[AJB_TIMER] test_sample_baseline START\n");

    // --- Load DB config ---
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string>> relations = readRelations("db/relations.txt");

    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});
    Index idx(q);
    idx.preProcessing(relations, filenames, numlines);

    fprintf(stderr, "[AJB_STATE][Sample] AGM_bound=%d\n", (int)idx.AGM());
    fprintf(stderr, "[AJB_STATE][Sample] peak_rss_kb=%ld (after preProcessing)\n",
            getPeakRSSKB());

    // --- Sample core: random sampling with sorted-vector dedup ---
    FlatDedup dedup;
    AjbSampleState state;
    int step = 20;

    auto start = chrono::high_resolution_clock::now();

    // AJB: max iterations guard (upstream was while(true))
    int max_attempts = (int)idx.AGM() * 10;
    if (max_attempts < 100000) max_attempts = 100000;

    while (state.total_attempts < max_attempts) {
        state.record_attempt();
        vector<int> s = idx.sampleUntilSuccess();

        if (!dedup.insert_unique(s)) {
            // collision — already seen this result
            state.record_collision();
        } else {
            state.record_success();

            if (state.successes < step || state.successes % step == 0) {
                auto now = chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed = now - start;
                cout << state.successes << ", " << state.total_attempts
                     << ", " << elapsed.count() << endl;
            }

            // AJB: periodic state dump
            if (state.successes % 1000 == 0) {
                auto now = chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed = now - start;
                state.dump("periodic", elapsed.count(), dedup.memory_bytes());
                fprintf(stderr, "[AJB_STATE][Sample] rss_kb=%ld\n", getPeakRSSKB());
            }

            if (state.successes % 500 == 0) printInfo(idx);
        }

        // AJB: early termination on collision saturation
        if (state.should_terminate()) {
            fprintf(stderr, "[AJB_STATE][Sample] EARLY_TERMINATION "
                    "collision_rate=%.4f > 0.99 at attempt=%d\n",
                    state.ema_collision_rate, state.total_attempts);
            break;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_elapsed = end - start;

    // --- Final report ---
    state.dump("FINAL", total_elapsed.count(), dedup.memory_bytes());
    printInfo(idx);
    fprintf(stderr, "[AJB_STATE][Sample] unique_results=%zu\n", dedup.size());
    fprintf(stderr, "[AJB_STATE][Sample] final_rss_kb=%ld\n", getPeakRSSKB());
    fprintf(stderr, "[AJB_TIMER] test_sample_baseline DONE elapsed=%.4fs\n",
            total_elapsed.count());

    return 0;
}
