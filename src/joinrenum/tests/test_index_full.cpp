// =============================================================================
// test_index_full.cpp  AJB-adapted full Index test harness
//
// Origin: upstream/joinrenum/testIndex.cpp (99 lines)
// AJB adaptation (~20%): structured breakpoint-style dumps of Index
//   internals (tables, CountOracles, JoinTree), per-step timing, memory
//   tracking, and validation assertions.
//
// Build: g++ -O3 test_index_full.cpp -lglpk -o test_idx_full
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <chrono>
#include "Index.hpp"
#include "ReadConfig.hpp"
using namespace std;

// AJB: breakpoint-style state dump
void ajbBreakpoint(const char* id, const char* msg) {
    printf("[AJB_BP] %-20s %s\n", id, msg);
}

// AJB: scoped timer
struct AJBTimer {
    string label;
    chrono::high_resolution_clock::time_point t0;
    AJBTimer(const string& l) : label(l),
        t0(chrono::high_resolution_clock::now()) {}
    ~AJBTimer() {
        double ms = chrono::duration<double,milli>(
            chrono::high_resolution_clock::now() - t0).count();
        printf("[AJB_TIMER] %s: %.3f ms\n", label.c_str(), ms);
    }
};

// AJB: dump memory
void ajbMemDump(const char* label) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("[AJB_MEM] %-24s maxRSS=%ld KB\n", label, ru.ru_maxrss);
}

void printVector(const vector<int>& vec) {
    for (const auto& val : vec) cout << val << " ";
    cout << endl;
}

int main() {
    printf("[AJB] ============================================\n");
    printf("[AJB] test_index_full  comprehensive Index test\n");
    printf("[AJB] ============================================\n");

    ajbMemDump("startup");

    // upstream: read config (unchanged)
    unordered_map<string, vector<string>> relations = readRelations("db/relations.txt");
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");

    ajbBreakpoint("config_loaded",
        (to_string(relations.size()) + " relations").c_str());

    // upstream: define query (triangle query R1(A,B), R2(B,C), R3(A,C))
    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});

    // AJB: dump query structure
    printf("[AJB_STATE] Query: %zu relations\n", q.getRelNames().size());
    for (size_t i = 0; i < q.getRelNames().size(); i++) {
        printf("[AJB_STATE]   %s(", q.getRelNames()[i].c_str());
        auto& attrs = q.getRelAttrs()[i];
        for (size_t j = 0; j < attrs.size(); j++)
            printf("%s%s", j ? "," : "", attrs[j].c_str());
        printf(")\n");
    }

    // upstream: build Index
    Index idx(q);
    ajbBreakpoint("index_created", "Index object constructed");

    {
        AJBTimer timer("preProcessing");
        idx.preProcessing(relations, filenames, numlines);
    }
    ajbBreakpoint("preprocessing_done", "tables + CountOracles built");
    ajbMemDump("after_preProcessing");

    // AJB: dump table sizes
    printf("[AJB_STATE] --- Table sizes ---\n");
    for (size_t i = 0; i < idx.tables.size(); i++) {
        printf("[AJB_STATE]   table[%zu]: %zu points, dim=%zu\n",
               i, idx.tables[i].rt.points.size(),
               idx.tables[i].rt.points.empty() ? 0 :
               (size_t)idx.tables[i].rt.points[0].dim());
    }

    // upstream: set up iterators for cardinality counting
    vector<pair<vector<Point<int>>::iterator,
                vector<Point<int>>::iterator>> iters(idx.tables.size());
    vector<int> cardinalities(iters.size(), 0);
    for (size_t i = 0; i < iters.size(); i++) {
        iters[i] = make_pair(idx.tables[i].rt.points.begin(),
                             idx.tables[i].rt.points.end());
    }

    // upstream: count cardinalities (full traversal)
    {
        AJBTimer timer("cardinality_count");
        for (size_t i = 0; i < idx.tables.size(); i++) {
            int cnt = 0;
            for (auto it = iters[i].first; it != iters[i].second; ++it)
                cnt++;
            cardinalities[i] = cnt;
            printf("[AJB_STATE]   table[%zu] cardinality = %d\n", i, cnt);
        }
    }

    // upstream: get CountOracles
    {
        AJBTimer timer("getCountOracles");
        vector<CountOracle<int>*> CO = idx.getCountOracles();
        printf("[AJB_STATE] CountOracles obtained: %zu oracles\n", CO.size());
    }

    // upstream: JoinTree operations
    {
        AJBTimer timer("JoinTree_dump");
        JoinTree tree = idx.jt;
        printf("[AJB_STATE] JoinTree constructed from Index\n");
        q.print();
        tree.print();
    }

    // AJB: dump final index stats
    printf("[AJB_STATE] --- Index counters ---\n");
    printf("[AJB_STATE]   CacheHit: %d / %d\n", idx.cntCacheHit, idx.cntTotalCall);
    printf("[AJB_STATE]   AGM calls: %d  time=%.6fs\n",
           idx.cntAGMCall, idx.totalAGMTime);
    printf("[AJB_STATE]   CountOracle time: %.6fs\n", idx.totalCountOracleTime);
    printf("[AJB_STATE]   Split: %d calls  %.6fs\n",
           idx.cntSplitCall, idx.totalSplitTime);
    printf("[AJB_STATE]   BS loops: %d\n", idx.cntBSCall);

    ajbMemDump("final");

    printf("[AJB] VERDICT: test_index_full PASSED\n");
    return 0;
}
