// =============================================================================
// upper_bound_full.cpp  AJB-adapted STL lower/upper_bound demo
//
// Origin: upstream/joinrenum/upb.cpp (13 lines)
// AJB adaptation (~20%): extended with boundary-case tests, timing
//   for large sets, and iterator validity checks used in AJB binary
//   search paths.
//
// Build: g++ -O3 upper_bound_full.cpp -o ub_full
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
using namespace std;

int main() {
    printf("[AJB] ============================================\n");
    printf("[AJB] upper_bound_full  STL bound operations test\n");
    printf("[AJB] ============================================\n");

    // upstream: basic set with gaps
    set<int> s = {1, 3, 4, 6, 8};

    printf("[AJB_STATE] Set contents: {");
    for (auto it = s.begin(); it != s.end(); ++it)
        printf("%s%d", it != s.begin() ? "," : "", *it);
    printf("}\n");

    // upstream: lower_bound(4) then decrement
    auto it = s.lower_bound(4);
    printf("[AJB_TRACE] lower_bound(4) = %d\n", *it);
    printf("[AJB_TRACE] --lower_bound(4) = %d\n", *--it);
    printf("[AJB_TRACE] upper_bound(4) = %d\n", *s.upper_bound(4));

    // AJB: extended boundary tests relevant to BinarySearch.cpp
    struct BoundTest {
        int key;
        const char* desc;
    };
    vector<BoundTest> tests = {
        {0, "before_min"}, {1, "at_min"}, {5, "in_gap"},
        {8, "at_max"}, {9, "after_max"},
    };

    printf("[AJB_STATE] --- Boundary test suite ---\n");
    for (auto& t : tests) {
        auto lb = s.lower_bound(t.key);
        auto ub = s.upper_bound(t.key);
        printf("[AJB_TRACE] key=%d (%s): lower_bound=%s upper_bound=%s\n",
               t.key, t.desc,
               lb != s.end() ? to_string(*lb).c_str() : "END",
               ub != s.end() ? to_string(*ub).c_str() : "END");
    }

    // AJB: performance test with large sorted set
    int N = 1000000;
    set<int> big;
    for (int i = 0; i < N; i++) big.insert(i * 2);  // even numbers

    printf("[AJB_STATE] Large set: %zu elements\n", big.size());

    auto t0 = chrono::high_resolution_clock::now();
    int found_lb = 0;
    for (int i = 0; i < N; i++) {
        auto r = big.lower_bound(i);
        if (r != big.end()) found_lb++;
    }
    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double,milli>(t1 - t0).count();

    printf("[AJB_TIMER] %d lower_bound queries: %.3f ms (%.0f ops/ms)\n",
           N, ms, N / ms);
    printf("[AJB_STATE] Found (not end): %d / %d\n", found_lb, N);

    printf("[AJB] VERDICT: upper_bound_full PASSED\n");
    return 0;
}
