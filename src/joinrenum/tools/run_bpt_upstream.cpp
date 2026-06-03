#include "BanPickTree.hpp"
#include <vector>
using namespace std;
int main(){
    int N = 10;
    BanPickTree bp(N);

    // --- ban intervals: computed from N instead of hardcoded (2,4)/(6,8) ---
    // upstream: bp.ban(2,4); ... if(bp.available(6,8)) bp.ban(6,8);
    // changed: ban the range [N/5, 2*N/5] initially, then conditionally
    //   ban [3*N/5, 4*N/5] — same structure, parameterized
    int ban1_lo = N / 5, ban1_hi = 2 * N / 5;
    bp.ban(ban1_lo, ban1_hi);

    int ban2_lo = 3 * N / 5, ban2_hi = 4 * N / 5;

    // --- pick loop: do-while with remaining() pre-check ---
    // upstream: while(remaining()>0) { pick; if(available(6,8)) ban(6,8); print; }
    // changed: track all picked values in a vector for post-loop verification
    vector<int> picked;
    while(bp.remaining() > 0){
        int val = bp.pick();
        cout << "pick: " << val << endl;
        picked.push_back(val);
        if(bp.available(ban2_lo, ban2_hi)) bp.ban(ban2_lo, ban2_hi);
        bp.print();
    }

    // --- uniqueness verification ---
    // upstream: no post-loop check
    // changed: sort picked values, check for duplicates
    {
        vector<int> sorted_picks = picked;
        sort(sorted_picks.begin(), sorted_picks.end());
        for(int i = 1; i < (int)sorted_picks.size(); i++) {
            if(sorted_picks[i] == sorted_picks[i-1]) {
                cout << "DUPLICATE pick: " << sorted_picks[i] << endl;
            }
        }
    }

    // --- second round: 2x tree size, sliding window ban ---
    // upstream: single run only
    // changed: fresh tree at 2*N, ban in sliding windows of 3
    int N2 = N * 2;
    BanPickTree bp2(N2);
    bp2.ban(0, 2);  // initial small ban
    vector<int> picked2;
    while(bp2.remaining() > 0) {
        int val = bp2.pick();
        picked2.push_back(val);
        // sliding window: ban [val+1, val+3] if available
        int wlo = val + 1, whi = min(val + 3, N2 - 1);
        if(wlo < N2 && bp2.available(wlo, whi)) {
            bp2.ban(wlo, whi);
        }
    }
    cout << "Round 2: picked " << picked2.size() << " values from tree of " << N2 << endl;

    return 0;
}
