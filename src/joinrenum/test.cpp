// =============================================================================
// test.cpp — REnum-BMITU full pipeline (AJB-instrumented)
//
// Origin: upstream/joinrenum/test.cpp (191 lines)
// AJB adaptation (~20%): [AJB_TRACE] at each enumeration milestone,
//   structured timer around each phase, per-500 stats dump preserved,
//   and progress percentage reporting.
// =============================================================================

#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include "Table.h"
#include "Parcel.h"
#include "Index.hpp"
#include "ReadConfig.hpp"
#include "BanPickTree.hpp"
using namespace std;

void printInfo(Index &idx) {
    // upstream: print all Index counters
    printf("[AJB_STATE] CacheHit(SplitBucket): %d / %d total\n",
           idx.cntCacheHit, idx.cntTotalCall);
    printf("[AJB_STATE] AGM calls: %d  time=%.6fs\n",
           idx.cntAGMCall, idx.totalAGMTime);
    printf("[AJB_STATE] CountOracle time: %.6fs\n", idx.totalCountOracleTime);
    printf("[AJB_STATE] Split: %d calls  time=%.6fs\n",
           idx.cntSplitCall, idx.totalSplitTime);
    printf("[AJB_STATE] CacheHit time: %.6fs\n", idx.totalCacheHitTime);
}

int main() {
    printf("[AJB] ============================================\n");
    printf("[AJB] test.cpp — REnum-BMITU full pipeline\n");
    printf("[AJB] ============================================\n");

    // upstream: read config
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string>> relations = readRelations("db/relations.txt");

    // upstream: print loaded schema
    for (auto& [name, vars] : relations) {
        cout << name << ": ";
        for (auto& v : vars) cout << v << " ";
        cout << endl;
    }

    // upstream: triangle query (hardcoded)
    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});

    // upstream: Index + preprocessing
    auto t_pre0 = chrono::high_resolution_clock::now();
    Index idx(q);
    idx.preProcessing(relations, filenames, numlines);
    auto t_pre1 = chrono::high_resolution_clock::now();

    printf("[AJB_TIMER] preprocessing: %.3f ms\n",
        chrono::duration<double,milli>(t_pre1 - t_pre0).count());

    cout << "Variables: ";
    for (size_t i = 0; i < q.getVarNames().size(); i++)
        cout << q.getVarNames()[i] << " ";
    cout << endl;

    int cntsuccess = 0, cnt = 0;
    int step = 20;

    printf("[AJB_STATE] AGM bound = %d\n", idx.AGM());
    cout << idx.AGM() << endl;

    // upstream: REnum-BMITU algorithm
    BanPickTree bp(idx.AGM());

    if (freopen("res/result.txt", "w", stdout) == NULL)
        fprintf(stderr, "[AJB_WARN] Cannot open res/result.txt\n");

    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    double last_pct_report = 0;

    printf("[AJB_TRACE] REnum-BMITU loop starting, AGM=%d\n", idx.AGM());

    while (bp.remaining()) {
        cnt++;
        int s = bp.pick();
        pair<bool, vector<int>> res = idx.randomAccess_opt(idx.getFullBucket(), s);

        if (res.first) {
            cntsuccess++;
            if (cntsuccess < step || cntsuccess % step == 0) {
                end = chrono::high_resolution_clock::now();
                elapsed = end - start;
                cout << cntsuccess << ", " << cnt << ", " << bp.remaining()
                     << ", " << bp.getPercentage() << ", "
                     << elapsed.count() << endl;
            }
            if (cntsuccess % 500 == 0) {
                printInfo(idx);
                // AJB: progress trace
                double pct = bp.getPercentage();
                fprintf(stderr, "[AJB_TRACE] progress: %d successes, "
                        "%d total, %.1f%% done, %.3fs\n",
                        cntsuccess, cnt, pct * 100, elapsed.count());
            }
        }

        if (res.first) bp.ban(s, s);
        else bp.ban(res.second[0], res.second[1]);
    }

    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << cntsuccess << ", " << cnt << ", " << bp.remaining()
         << ", " << bp.getPercentage() << ", " << elapsed.count() << endl;

    printInfo(idx);

    fprintf(stderr, "[AJB_TIMER] REnum-BMITU total: %.3fs, %d successes / %d probes\n",
            elapsed.count(), cntsuccess, cnt);
    fprintf(stderr, "[AJB] test.cpp COMPLETE\n");
    return 0;
}
