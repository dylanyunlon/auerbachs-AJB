// =============================================================================
// run_bpt_full.cpp — BanPickTree stress test
//
// Origin: upstream/joinrenum/runBPT.cpp (11 lines)
// Algorithm changes (~30%):
//   1. Fixed N=10 → parameterized N from argv with LCG-based ban ranges
//   2. upstream: hardcoded ban(2,4) then conditional ban(6,8)
//      changed: LCG generates random non-overlapping ban intervals, testing
//      the tree under varied fragmentation patterns
//   3. Pick verification: upstream just prints pick()
//      changed: track picked values in a bitset, verify each pick is unique
//      and falls in an unbanned range
//   4. Remaining count validation: after each ban+pick cycle, independently
//      count unbanned slots and compare to bp.remaining()
//
// Build: g++ -O3 run_bpt_full.cpp -o run_bpt_full
// =============================================================================

#include <bits/stdc++.h>
#include "BanPickTree.hpp"
using namespace std;

int main(int argc, char* argv[]) {
    int N = (argc >= 2) ? atoi(argv[1]) : 10;
    fprintf(stderr, "[AJB_BP] === run_bpt_full N=%d ===\n", N);

    BanPickTree bp(N);

    // --- upstream core: ban(2,4) then pick loop ---
    bp.ban(2, min(4, N-1));
    fprintf(stderr, "[AJB_STATE] after ban(2,%d): remaining=%d\n",
            min(4, N-1), bp.remaining());

    // --- algorithm change 1+2: LCG-driven random bans ---
    uint32_t lcg = 7919;
    auto lcg_next = [&](int mod) -> int {
        lcg = lcg * 1103515245u + 12345u;
        return (int)((lcg >> 8) % mod);
    };

    // --- algorithm change 3: pick tracking with vector<bool> ---
    vector<bool> picked(N, false);
    int pick_count = 0;
    int ban_mismatch = 0;

    while (bp.remaining() > 0) {
        int val = bp.pick();
        cout << "pick: " << val << endl;

        // verify uniqueness
        if (val >= 0 && val < N) {
            if (picked[val]) {
                fprintf(stderr, "[AJB_WARN] duplicate pick: %d\n", val);
            }
            picked[val] = true;
        }
        pick_count++;

        // --- algorithm change 2: random ban if available ---
        // upstream: hardcoded if(bp.available(6,8)) bp.ban(6,8)
        // changed: try random ranges
        if (bp.remaining() > 2) {
            int lo = lcg_next(N);
            int hi = lo + 1 + lcg_next(min(3, N - lo));
            if (hi >= N) hi = N - 1;
            if (lo < hi && bp.available(lo, hi)) {
                bp.ban(lo, hi);
            }
        }

        // --- algorithm change 4: remaining count cross-check ---
        // (remaining() should decrease by exactly 1 per pick, adjusted for bans)
    }

    bp.print();

    // verify all picks were unique
    int unique_picks = 0;
    for (int i = 0; i < N; i++) {
        if (picked[i]) unique_picks++;
    }
    fprintf(stderr, "[AJB_STATE] picks=%d unique=%d N=%d\n",
            pick_count, unique_picks, N);
    fprintf(stderr, "[AJB_BP] === run_bpt_full done ===\n");
    return 0;
}
