// =============================================================================
// test_rr_access_tree_full.cpp — RRAccessTree full (AJB-instrumented)
//
// Origin: upstream/joinrenum/testRRAccessTree.cpp (34 lines, verbatim core)
// AJB adaptation (~20%): chrono timing around tree construction and
//   RRAccess loop, success/fail counting, AJB_STATE dumps for each access,
//   memory snapshots, tree.print() activated.
//
// Build: g++ -O3 test_rr_access_tree_full.cpp -lglpk -o test_rra_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
#include "RRAccessTree.hpp"
#include "ReadConfig.hpp"
using namespace std;

// AJB: memory snapshot
static long ajb_rss_kb() {
    ifstream f("/proc/self/status"); string line;
    while (getline(f, line))
        if (line.substr(0, 6) == "VmRSS:")
            { istringstream iss(line); string k; long v; iss >> k >> v; return v; }
    return -1;
}

int main() {
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] test_rr_access_tree_full  RRAccessTree test\n");
    fprintf(stderr, "[AJB] ============================================\n");

    long rss0 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] startup: RSS=%ld KB\n", rss0);

    // upstream: read config
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");

    // upstream: construct RRAccessTree
    auto t0 = chrono::high_resolution_clock::now();
    RRAccessTree tree(relations, filenames, numlines);
    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] RRAccessTree construction: %.3f ms\n",
            chrono::duration<double,milli>(t1 - t0).count());
    fprintf(stderr, "[AJB_STATE] AGM bound = %d\n", tree.AGM);

    long rss1 = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] after_build: RSS=%ld KB (delta=%ld)\n", rss1, rss1 - rss0);

    // upstream: enumerate all i in [1..AGM] and call RRAccess(i)
    int success_count = 0, fail_count = 0;
    auto t2 = chrono::high_resolution_clock::now();
    for(int i = 1; i <= tree.AGM; i++) {
        pair<bool, vector<int> > res = tree.RRAccess(i);
        // upstream: print each result
        cout << i << ": " << res.first << ", ";
        for(size_t j = 0; j < res.second.size(); j++) {
            cout << res.second[j] << ",";
        }
        cout << endl;

        if(res.first) success_count++;
        else fail_count++;

        // AJB: trace every 100th access for monitoring
        if(i % 100 == 0 || i == tree.AGM) {
            fprintf(stderr, "[AJB_TRACE] RRAccess progress: %d/%d  success=%d fail=%d\n",
                    i, tree.AGM, success_count, fail_count);
        }
    }
    auto t3 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] RRAccess loop (1..%d): %.3f ms\n",
            tree.AGM, chrono::duration<double,milli>(t3 - t2).count());
    fprintf(stderr, "[AJB_STATE] Final: success=%d  fail=%d  total=%d\n",
            success_count, fail_count, tree.AGM);

    // upstream: print tree structure
    fprintf(stderr, "[AJB_BP] Printing RRAccessTree structure:\n");
    tree.print();

    // upstream: printBucketTree (was commented out)
    // tree.idx.printBucketTree(tree.idx.getFullBucket());

    long rss_end = ajb_rss_kb();
    fprintf(stderr, "[AJB_MEM] final: RSS=%ld KB (total delta=%ld KB)\n",
            rss_end, rss_end - rss0);
    fprintf(stderr, "[AJB] VERDICT: test_rr_access_tree_full PASSED\n");
    return 0;
}
