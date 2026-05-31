// =============================================================================
// upper_bound_full.cpp — STL bound operations + perf benchmark (AJB-instrumented)
//
// Origin: upstream/joinrenum/upb.cpp (13 lines, verbatim core)
// AJB adaptation (~20%): extended with stress test, chrono timing,
//   edge-case validation, structured trace output.
//
// Build: g++ -O3 upper_bound_full.cpp -o upper_bound_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
using namespace std;

int main(){
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] upper_bound_full  STL bound ops test\n");
    fprintf(stderr, "[AJB] ============================================\n");

    // upstream: basic set with lower_bound/upper_bound (verbatim logic)
    set<int> s = {};
    s.insert(1);
    s.insert(3);
    s.insert(4);
    s.insert(6);
    s.insert(8);

    fprintf(stderr, "[AJB_STATE] set contents: {");
    for(auto it = s.begin(); it != s.end(); ++it)
        fprintf(stderr, "%s%d", it != s.begin() ? "," : "", *it);
    fprintf(stderr, "}\n");

    // upstream: lower_bound(4) then decrement, upper_bound(4)
    set<int>::iterator it = s.lower_bound(4);
    int lb_prev = *--it;
    int ub = *s.upper_bound(4);
    cout << lb_prev;
    cout << ub;
    cout << endl;
    fprintf(stderr, "[AJB_TRACE] lower_bound(4)-1 = %d  upper_bound(4) = %d\n", lb_prev, ub);

    // AJB: edge case validation
    fprintf(stderr, "[AJB_STATE] --- Edge case tests ---\n");
    struct BoundTest { int key; const char* label; };
    vector<BoundTest> tests = {{1,"min"}, {8,"max"}, {5,"gap"}, {0,"below_min"}, {9,"above_max"}};
    for(auto& bt : tests) {
        auto lb = s.lower_bound(bt.key);
        auto ub_it = s.upper_bound(bt.key);
        fprintf(stderr, "[AJB_TRACE] key=%d(%s): lb=%s  ub=%s\n",
                bt.key, bt.label,
                lb != s.end() ? to_string(*lb).c_str() : "end",
                ub_it != s.end() ? to_string(*ub_it).c_str() : "end");
    }

    // AJB: stress test with large set
    int N = 1000000;
    fprintf(stderr, "[AJB_TRACE] Stress test: %d element set\n", N);
    set<int> big;
    for(int i = 0; i < N; i++) big.insert(i * 2);  // even numbers

    srand(42);
    int found = 0;
    auto t0 = chrono::high_resolution_clock::now();
    for(int i = 0; i < N; i++) {
        int key = rand() % (N * 2);
        auto lb = big.lower_bound(key);
        if(lb != big.end() && *lb == key) found++;
    }
    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] %d lookups: %.3f ms (%.1f M ops/s)\n",
            N, chrono::duration<double,milli>(t1 - t0).count(),
            N / chrono::duration<double>(t1 - t0).count() / 1e6);
    fprintf(stderr, "[AJB_STATE] exact_matches=%d/%d (%.1f%%)\n",
            found, N, 100.0*found/N);

    fprintf(stderr, "[AJB] upper_bound_full COMPLETE\n");
    return 0;
}
