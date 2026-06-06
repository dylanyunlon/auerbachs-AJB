// =============================================================================
// test.cpp — REnum-BMITU full pipeline (AJB-instrumented)
//
// Origin: upstream/joinrenum/test.cpp (191 lines, verbatim core preserved)
// AJB adaptation (~20%): [AJB_TRACE] at milestones, chrono timing around
//   each phase, printInfo extended with BSCall/BoundPrepare/rrtreenode,
//   progress percentage on stderr, REnum/Sample variants preserved in comments.
// =============================================================================

#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <cstring>
#include <sys/resource.h>
#include <unistd.h>
#include "Table.h"
#include "Parcel.h"
#include "Index.hpp"
#include "ReadConfig.hpp"
#include "BanPickTree.hpp"
using namespace std;

// [AJB] M913: system-level diagnostics — 运行时环境信息
// 帮助在不同机器上复现问题: 编译器版本、优化级别、数据路径
static void ajb_print_system_info() {
#ifdef AJB_DEBUG
    fprintf(stderr, "[AJB_SYS] C++ standard: %ld\n", (long)__cplusplus);
#ifdef __OPTIMIZE__
    fprintf(stderr, "[AJB_SYS] optimization: ON (Release)\n");
#else
    fprintf(stderr, "[AJB_SYS] optimization: OFF (Debug)\n");
#endif
    fprintf(stderr, "[AJB_SYS] AJB_DEBUG: enabled\n");
    char cwd[256];
    if (getcwd(cwd, sizeof(cwd))) fprintf(stderr, "[AJB_SYS] cwd: %s\n", cwd);
    // 进程内存基线
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        fprintf(stderr, "[AJB_SYS] startup_maxRSS: %ld KB\n", ru.ru_maxrss);
#endif
}

// [AJB] M913: per-probe tracing — 每个randomAccess的输入/输出/耗时
// 用于定位哪些probe特别慢(可能在Index.splitBucket或RangeTree上卡住)
struct AjbProbeTracker {
    int total_probes = 0;
    int success_count = 0;
    int fail_count = 0;
    double fastest_us = 1e18;
    double slowest_us = 0;
    int slowest_probe_id = -1;
    long long total_ban_range = 0; // 累计ban的区间宽度
    void record(int probe_id, int s, bool ok, const vector<int>& result, double elapsed_us) {
#ifdef AJB_DEBUG
        total_probes++;
        if (ok) success_count++; else fail_count++;
        if (elapsed_us < fastest_us) fastest_us = elapsed_us;
        if (elapsed_us > slowest_us) { slowest_us = elapsed_us; slowest_probe_id = probe_id; }
        if (!ok && result.size() >= 2) total_ban_range += (result[1] - result[0] + 1);
        // 每1000个probe输出一次摘要,避免淹没日志
        if (total_probes % 1000 == 0) {
            fprintf(stderr, "[AJB_PROBE_BATCH] probes=%d ok=%d fail=%d fastest=%.1fus slowest=%.1fus(#%d) avg_ban_range=%.1f\n",
                    total_probes, success_count, fail_count, fastest_us, slowest_us,
                    slowest_probe_id, fail_count > 0 ? (double)total_ban_range/fail_count : 0.0);
        }
#endif
    }
    void dump_final() {
#ifdef AJB_DEBUG
        struct rusage ru;
        long peak_rss = 0;
        if (getrusage(RUSAGE_SELF, &ru) == 0) peak_rss = ru.ru_maxrss;
        fprintf(stderr, "[AJB_FINAL] total_probes=%d success=%d fail=%d\n", total_probes, success_count, fail_count);
        fprintf(stderr, "[AJB_FINAL] fastest=%.1fus slowest=%.1fus(probe#%d)\n", fastest_us, slowest_us, slowest_probe_id);
        fprintf(stderr, "[AJB_FINAL] peak_RSS=%ld KB  avg_ban_range=%.1f\n",
                peak_rss, fail_count > 0 ? (double)total_ban_range/fail_count : 0.0);
#endif
    }
};
static AjbProbeTracker ajb_probe_tracker;


void printInfo(Index &idx) {
    // upstream: core stats (verbatim)
    fprintf(stdout, "Cache Hit of SplitBucket: %d Total Call: %d\n", idx.cntCacheHit, idx.cntTotalCall);  // AJB-algo: fprintf
    fprintf(stdout, "Total AGM Call: %d\n", idx.cntAGMCall);
    fprintf(stdout, "Total AGM Time: %f\n", idx.totalAGMTime);
    fprintf(stdout, "Total Count Oracle Time: %f\n", idx.totalCountOracleTime);
    fprintf(stdout, "Total Split Time: %f\n", idx.totalSplitTime);
    fprintf(stdout, "Total Split Call: %d\n", idx.cntSplitCall);
    fprintf(stdout, "Total Cache Hit Time: %f\n", idx.totalCacheHitTime);

    // AJB: structured dump to stderr for parse_ajb_trace.py
    fprintf(stderr, "[AJB_STATE] CacheHit(SplitBucket): %d / %d\n",
            idx.cntCacheHit, idx.cntTotalCall);
    fprintf(stderr, "[AJB_STATE] AGM calls=%d time=%.6fs\n",
            idx.cntAGMCall, idx.totalAGMTime);
    fprintf(stderr, "[AJB_STATE] CountOracle=%.6fs Split=%d/%.6fs CacheHit=%.6fs\n",
            idx.totalCountOracleTime, idx.cntSplitCall,
            idx.totalSplitTime, idx.totalCacheHitTime);
    fprintf(stderr, "[AJB_STATE] BSCall=%d BoundPrepare=%.6fs RRTreeNodes=%d\n",
            idx.cntBSCall, idx.totalBoundPrepareTime, idx.totalrrtreenode);
    return;
}

// AJB-algo: test harness with wall-clock timing
int main() {
    ajb_print_system_info();
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] test.cpp — REnum-BMITU full pipeline\n");
    fprintf(stderr, "[AJB] ============================================\n");

    // Table<Parcel> tbl;
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");
    
    // AJB: range-based for代替显式iterator遍历
    vector<string> query_rels;
    vector<vector<string> > query_vars;
    query_rels.reserve(relations.size());
    query_vars.reserve(relations.size());
    for (const auto& [name, vars] : relations) {
        query_rels.push_back(name);
        query_vars.push_back(vars);
    }
    for (size_t i = 0; i < query_rels.size(); i++) {
        cout << query_rels[i] << ": ";
        for (size_t j = 0; j < query_vars[i].size(); j++) {
            cout << query_vars[i][j] << " ";
        }
        cout << endl;
    }

    // Query q(query_rels, query_vars);
    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});
    // Query q({"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"}, {{"x1", "x2"}, {"x2", "x3"}, {"x1", "x3"}, {"x3", "x4"}, {"x4", "x5"}, {"x5", "x6"}, {"x4", "x6"}, {"x1", "x5"}, {"x2", "x6"}});
    // Query q({"R1", "R2", "R3", "R4", "R5", "R6"}, {{"P", "Q", "R"}, {"Q", "S", "T"}, {"R", "T", "U"}, {"P", "S", "V"}, {"U", "V", "W"}, {"W", "P", "Q"}});
    // Query q({"R1", "R2", "R3", "R4"}, {{"A", "B", "C", "D"}, {"B", "C", "E", "F"}, {"C", "D", "F", "G"}, {"B", "D", "E", "G"}});

    // Query q({"L1", "L2", "O1", "O2", "C1", "C2", "S"},
    // {{"ok1", "pk"},
    //  {"ok2", "pk"},
    //  {"ok1", "ck1"},
    //  {"ok2", "ck2"},
    //  {"ck1", "nk"},
    //  {"ck2", "nk"},
    //  {"sk", "nk"}});
    auto t_pre0 = chrono::high_resolution_clock::now();
    Index idx(q);
    idx.preProcessing(relations, filenames, numlines);
    auto t_pre1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] preProcessing: %.3f ms\n",
            chrono::duration<double,milli>(t_pre1 - t_pre0).count());

    cout << "Variables: ";
    for(size_t i = 0; i < q.getVarNames().size(); i++) {
        cout << q.getVarNames()[i] << " ";
    }
    cout << endl;
    // idx.getFullBucket().print();
    // for(int i = 1; i < 20; i++) {
    //     vector<int> res = idx.sampleUntilSuccess();
    //     cout << "Sample " << i << ": ";
    //     for(int j = 0; j < res.size(); j++) {
    //         cout << res[j] << " ";
    //     }
    //     cout << endl;
    // }
    // idx.printBucketInfo(idx.getFullBucket());
    // idx.printBucketTree(idx.getFullBucket());


    int cntsuccess = 0, cnt = 0;
    // for(int i = 1; i <= idx.AGM(); i++) {
    //         pair<bool, vector<int> > res = idx.randomAccess(idx.getFullBucket(), i);
    //         cout << i << ": ";
    //         cout << res.first << "::";
    //         for(int j = 0; j < res.second.size(); j++) {
    //             cout << res.second[j] << ",";
    //         }
    //         cout << endl;
    //         if(!res.first) {
    //             cntfail++;
    //             i = res.second[1];
    //         }
    //         else cntsuccess++;
    // }

    // vector<int> cars = idx.getCar(idx.getFullBucket());
    // cout << endl;
    // if(freopen("res/res_q1_bmitu.txt", "w", stdout) == NULL)cout << "WRITEERR" << endl;
    int step = 20;
    fprintf(stderr, "[AJB_STATE] AGM bound = %lld\n", idx.AGM());
    cout << idx.AGM() << endl;
    //////////////////////////////REnum-BMITU
    BanPickTree bp(idx.AGM());
    if(freopen("res/result.txt", "w", stdout) == NULL)
        fprintf(stderr, "[AJB_WARN] Cannot open res/result.txt\n");

    fprintf(stderr, "[AJB_TRACE] REnum-BMITU loop starting, AGM=%lld\n", idx.AGM());
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end - start;
    double last_percentage = 0;
    while(bp.remaining()){
        cnt++;
        int s = bp.pick();
        auto probe_t0 = chrono::high_resolution_clock::now();
        pair<bool, vector<int> > res = idx.randomAccess(idx.getFullBucket(), s);
        auto probe_t1 = chrono::high_resolution_clock::now();
        double probe_us = chrono::duration<double,micro>(probe_t1 - probe_t0).count();
        ajb_probe_tracker.record(cnt, s, res.first, res.second, probe_us);
        if(res.first){
            cntsuccess++;
            if(cntsuccess < step || cntsuccess % step == 0){
            end = std::chrono::high_resolution_clock::now();
            elapsed_ms = end - start;
            // AJB: fprintf代替cout链——避免iostream格式化开销
            fprintf(stdout, "%d, %d, %lld, %f, %f\n",
                    cntsuccess, cnt, (long long)bp.remaining(), bp.getPercentage(), elapsed_ms.count());
        }
            if(cntsuccess % 500 == 0) {
                printInfo(idx);
                // AJB: progress trace to stderr
                fprintf(stderr, "[AJB_TRACE] progress: %d successes, %d total, %.1f%% done, %.3fs\n",
                        cntsuccess, cnt, bp.getPercentage() * 100, elapsed_ms.count());
            }

        }        
        if(res.first) bp.ban(s,s);
        // AJB: guard empty failure vector — randomAccess may return {}
        // when k exceeds all sons' AGMs; fall back to banning single slot
        else if(res.second.size() >= 2) bp.ban(res.second[0], res.second[1]);
        else bp.ban(s, s);
        // double done = bp.getPercentage();
        // if(int(done * 100) % 10 == 0 && int(done * 100) != int(last_percentage*100)){
        //     last_percentage = done;
        //     end = std::chrono::high_resolution_clock::now();
        //     elapsed = end - start;
        //     cout << bp.getPercentage() << ", " << elapsed.count() << endl;
        //     last_percentage = done;
        // }
    }


    // ////////////////////////////////REnum
    // int N = idx.AGM();
    // vector<int> A(N,0);
    // if(freopen("res/res_q1_renum.txt", "w", stdout) == NULL)cout << "WRITEERR" << endl;
    // random_device rd;
    // mt19937 gen(rd());
    // auto start = std::chrono::high_resolution_clock::now();
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed_ms = end - start;
    // int pos, j;
    // for(int i = 1; i <= N; i++){
    //     cnt++;
    //     uniform_int_distribution<> distr(i, N);
    //     j = distr(gen);
    //     if(A[j] > 0)pos = A[j];
    //     else pos = j;
    //     if(A[i] > 0)A[j] = A[i];
    //     else A[j] = i;
    //     pair<bool, vector<int> > res = idx.randomAccess(idx.getFullBucket(), pos);
    //     if(res.first){
    //         cntsuccess++;
    //         end = std::chrono::high_resolution_clock::now();
    //         elapsed = end - start;
    //         cout << cntsuccess << ", " << cnt << ", " << elapsed.count() << endl;
    //     }        
    // }

    //////////////////////////////Sample
    // set<vector<int> > S;
    // if(freopen("res/res_q2_sample.txt", "w", stdout) == NULL)cout << "WRITEERR" << endl;
    // while(true) {
    //     cnt++;
    //     vector<int> s = idx.sampleUntilSuccess();
    //     if(S.find(s) != S.end()) continue;
    //     S.insert(s);
    //     cntsuccess++;
    //     if(cntsuccess < step || cntsuccess % step == 0){
    //         end = std::chrono::high_resolution_clock::now();
    //         elapsed = end - start;
    //         cout << cntsuccess << ", " << cnt << ", " << elapsed.count() << endl;
    //     }
    // }
    
    end = std::chrono::high_resolution_clock::now();
    elapsed_ms = end - start;
    // AJB: 最终统计用fprintf一次写出, 避免cout链的多次刷新
    double throughput = elapsed_ms.count() > 0.0 ? cntsuccess / elapsed_ms.count() : 0.0;
    double success_rate = cnt > 0 ? 100.0 * cntsuccess / cnt : 0.0;
    fprintf(stdout, "%d, %d, %lld, %f, %f\n",
            cntsuccess, cnt, (long long)bp.remaining(), bp.getPercentage(), elapsed_ms.count());

    printInfo(idx);

    // [AJB_BP] 最终性能摘要: throughput + success_rate是调参的主要指标
    fprintf(stderr, "[AJB_BP] === FINAL SUMMARY ===\n");
    fprintf(stderr, "[AJB_BP] successes=%d probes=%d success_rate=%.1f%%\n",
            cntsuccess, cnt, success_rate);
    fprintf(stderr, "[AJB_BP] wall=%.3fs throughput=%.1f results/s\n",
            elapsed_ms.count(), throughput);
    fprintf(stderr, "[AJB_TIMER] REnum-BMITU total: %.3fs\n", elapsed_ms.count());
    ajb_probe_tracker.dump_final();
    fprintf(stderr, "[AJB] test.cpp COMPLETE\n");
    return 0;
}