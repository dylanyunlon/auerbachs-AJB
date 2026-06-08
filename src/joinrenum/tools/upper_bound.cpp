// =============================================================================
// upper_bound_full.cpp — STL lower_bound/upper_bound exploration
//
// Origin: upstream/joinrenum/upb.cpp (13 lines)
// Algorithm changes (~30%):
//   1. Fixed 5-element set → N random elements via LCG
//   2. upstream: single lower_bound(4) + upper_bound(4) test
//      changed: sweep test of lower_bound + upper_bound for every value
//      in [min-1, max+1], verifying the STL contract at each point
//   3. Added manual binary search that mimics the joinrenum MHBS pattern
//      (searching for predecessor in sorted data) and cross-checks
//      against STL iterator arithmetic
//   4. Edge case stress: test with duplicates, boundary values, empty ranges
//
// Build: g++ -O3 upper_bound_full.cpp -o upper_bound_full
// =============================================================================

#include <bits/stdc++.h>
using namespace std;

// --- algorithm change 3: manual predecessor search ---
// This is the pattern MHBS uses internally: find largest element <= target
// in a sorted container. Cross-check against STL.
static int manual_predecessor(const vector<int>& sorted, int target) {
    int lo = 0, hi = (int)sorted.size() - 1;
    int result = -1;  // no predecessor
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (sorted[mid] <= target) {
            result = sorted[mid];
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return result;
}

int main() {
    fprintf(stderr, "[AJB_BP] === upper_bound_full start ===\n");

    // --- upstream core: set with 5 elements ---
    set<int> s = {1, 3, 4, 6, 8};
    auto it = s.lower_bound(4);
    cout << *--it;       // 3
    cout << *s.upper_bound(4);  // 6
    cout << endl;

    // --- algorithm change 1: parameterized random set ---
    const int N = 200;
    uint32_t lcg = 31337;
    auto lcg_next = [&](int mod) -> int {
        lcg = lcg * 1103515245u + 12345u;
        return (int)((lcg >> 8) % mod);
    };

    set<int> big_set;
    for (int i = 0; i < N; i++) {
        big_set.insert(lcg_next(1000));
    }
    // also as sorted vector for binary search comparison
    vector<int> sorted_vec(big_set.begin(), big_set.end());
    fprintf(stderr, "[AJB_STATE] set_size=%zu sorted_vec_size=%zu range=[%d,%d]\n",
            big_set.size(), sorted_vec.size(),
            sorted_vec.front(), sorted_vec.back());

    // --- algorithm change 2: sweep test of lower/upper_bound ---
    int lb_mismatches = 0, ub_mismatches = 0;
    int pred_mismatches = 0;
    int sweep_min = sorted_vec.front() - 1;
    int sweep_max = sorted_vec.back() + 1;

    for (int val = sweep_min; val <= sweep_max; val++) {
        // STL set operations
        auto lb_it = big_set.lower_bound(val);
        auto ub_it = big_set.upper_bound(val);

        // STL vector operations (should match)
        auto vlb = lower_bound(sorted_vec.begin(), sorted_vec.end(), val);
        auto vub = upper_bound(sorted_vec.begin(), sorted_vec.end(), val);

        // cross-check: set and vector should agree
        if (lb_it != big_set.end() && vlb != sorted_vec.end()) {
            if (*lb_it != *vlb) lb_mismatches++;
        }
        if (ub_it != big_set.end() && vub != sorted_vec.end()) {
            if (*ub_it != *vub) ub_mismatches++;
        }

        // --- algorithm change 3: manual predecessor cross-check ---
        int manual_pred = manual_predecessor(sorted_vec, val);
        // STL predecessor: lower_bound then --
        int stl_pred = -1;
        if (vlb != sorted_vec.begin()) {
            stl_pred = *prev(vlb);
            if (*vlb == val) stl_pred = val;  // val itself is in set
        } else if (vlb != sorted_vec.end() && *vlb == val) {
            stl_pred = val;
        }
        if (manual_pred != stl_pred && manual_pred != -1) {
            pred_mismatches++;
        }
    }

    fprintf(stderr, "[AJB_STATE] sweep [%d,%d]: lb_mismatch=%d ub_mismatch=%d pred_mismatch=%d\n",
            sweep_min, sweep_max, lb_mismatches, ub_mismatches, pred_mismatches);

    // --- algorithm change 4: edge case stress ---
    // test with duplicates (set deduplicates, but test the insert pattern)
    set<int> dup_set;
    for (int i = 0; i < 100; i++) dup_set.insert(i % 10);
    // verify boundary: lower_bound(0) == begin, upper_bound(9) == end
    bool edge_ok = (*dup_set.lower_bound(0) == 0) &&
                   (dup_set.upper_bound(9) == dup_set.end());
    fprintf(stderr, "[AJB_STATE] edge_case: dup_set_size=%zu boundary=%s\n",
            dup_set.size(), edge_ok ? "PASS" : "FAIL");

    // empty range test
    auto empty_lb = big_set.lower_bound(sorted_vec.back() + 100);
    fprintf(stderr, "[AJB_STATE] empty_range: lower_bound(max+100)==end? %s\n",
            empty_lb == big_set.end() ? "PASS" : "FAIL");

    fprintf(stderr, "[AJB_BP] === upper_bound_full done ===\n");
    return 0;
}
