// =============================================================================
// test_renum_baseline.cpp — REnum random permutation access (AJB-instrumented)
//
// Origin: upstream/joinrenum/test.cpp lines 137-160 (commented REnum section)
// AJB adaptation (~25%): xoshiro256** PRNG replaces mt19937 (2x throughput for
//   uniform draws), splitmix64 seeding, [AJB_STATE] dumps every 1000 successes
//   with success rate / throughput / RSS, convergence tracking via exponential
//   moving average of success intervals.
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
#include "Table.h"
#include "Parcel.h"
#include "Index.hpp"
#include "ReadConfig.hpp"

using namespace std;

// AJB: xoshiro256** PRNG — replaces mt19937 for lower overhead per draw
struct Xoshiro256ss {
    uint64_t s[4];

    Xoshiro256ss(uint64_t seed) {
        // splitmix64 seeding (better than single-seed mt19937)
        for (int i = 0; i < 4; i++) {
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }

    uint64_t next() {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // uniform in [lo, hi] inclusive
    int uniform(int lo, int hi) {
        if (lo >= hi) return lo;
        uint64_t range = (uint64_t)(hi - lo) + 1;
        return lo + (int)(next() % range);
    }

private:
    static uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

// AJB: state tracking struct
struct AjbREnumState {
    int total_attempts;
    int successes;
    double ema_gap;         // exponential moving average of attempts between successes
    int last_success_at;    // attempt number of last success
    double alpha;           // EMA smoothing factor

    AjbREnumState() : total_attempts(0), successes(0), ema_gap(0),
                      last_success_at(0), alpha(0.05) {}

    void record_success(int attempt_num) {
        int gap = attempt_num - last_success_at;
        if (successes == 0) ema_gap = gap;
        else ema_gap = alpha * gap + (1.0 - alpha) * ema_gap;
        last_success_at = attempt_num;
        successes++;
    }

    void dump(const char* tag, double elapsed_s) const {
        double rate = (total_attempts > 0) ? (double)successes / total_attempts : 0;
        double throughput = (elapsed_s > 0) ? successes / elapsed_s : 0;
        fprintf(stderr,
            "[AJB_STATE][REnum] %s: attempts=%d successes=%d rate=%.4f "
            "throughput=%.1f/s ema_gap=%.1f\n",
            tag, total_attempts, successes, rate, throughput, ema_gap);
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
    fprintf(stderr, "[AJB_TIMER] test_renum_baseline START\n");
    auto t0 = chrono::high_resolution_clock::now();

    // --- Load DB config ---
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string>> relations = readRelations("db/relations.txt");

    // Triangle query: R1(A,B), R2(B,C), R3(A,C)
    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});
    Index idx(q);
    idx.preProcessing(relations, filenames, numlines);

    int N = idx.AGM();
    fprintf(stderr, "[AJB_STATE][REnum] AGM_bound=%d\n", N);
    fprintf(stderr, "[AJB_STATE][REnum] peak_rss_kb=%ld (after preProcessing)\n", getPeakRSSKB());

    // --- REnum core: Fisher-Yates permutation with xoshiro256** ---
    vector<int> A(N + 1, 0);  // +1 for 1-indexed
    Xoshiro256ss rng(42);      // AJB: deterministic seed for reproducibility

    AjbREnumState state;
    int step = 20;

    auto start = chrono::high_resolution_clock::now();

    for (int i = 1; i <= N; i++) {
        state.total_attempts++;
        // AJB: xoshiro256** draw replaces mt19937 + uniform_int_distribution
        int j = rng.uniform(i, N);
        int pos = (A[j] > 0) ? A[j] : j;
        A[j] = (A[i] > 0) ? A[i] : i;

        pair<bool, vector<int>> res = idx.randomAccess(idx.getFullBucket(), pos);
        if (res.first) {
            state.record_success(i);

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
                state.dump("periodic", elapsed.count());
                fprintf(stderr, "[AJB_STATE][REnum] rss_kb=%ld\n", getPeakRSSKB());
            }

            if (state.successes % 500 == 0) printInfo(idx);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_elapsed = end - start;

    // --- Final report ---
    state.dump("FINAL", total_elapsed.count());
    printInfo(idx);
    fprintf(stderr, "[AJB_STATE][REnum] final_rss_kb=%ld\n", getPeakRSSKB());
    fprintf(stderr, "[AJB_TIMER] test_renum_baseline DONE elapsed=%.4fs\n",
            total_elapsed.count());

    return 0;
}
