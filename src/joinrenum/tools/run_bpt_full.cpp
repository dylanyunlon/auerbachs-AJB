// =============================================================================
// run_bpt_full.cpp  AJB-adapted BanPickTree demo
//
// Origin: upstream/joinrenum/runBPT.cpp (11 lines)
// AJB adaptation (~20%): step-by-step trace of pick/ban operations,
//   remaining-count tracking, state visualization.
//
// Build: g++ -O3 run_bpt_full.cpp -o run_bpt_full
// =============================================================================

#include <chrono>
#include <cstdio>
#include "BanPickTree.hpp"

int main() {
    printf("[AJB] ============================================\n");
    printf("[AJB] run_bpt_full  BanPickTree interactive trace\n");
    printf("[AJB] ============================================\n");

    int tree_size = 10;
    printf("[AJB_STATE] BanPickTree size = %d\n", tree_size);

    BanPickTree bp(tree_size);

    // upstream: ban range [2,4]
    bp.ban(2, 4);
    printf("[AJB_TRACE] ban(2,4) -> remaining=%d\n", bp.remaining());

    int step = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    // upstream: pick loop
    while (bp.remaining() > 0) {
        int picked = bp.pick();
        step++;
        printf("[AJB_TRACE] step %d: pick=%d  remaining=%d\n",
               step, picked, bp.remaining());

        // upstream: conditional ban [6,8]
        if (bp.available(6, 8)) {
            bp.ban(6, 8);
            printf("[AJB_TRACE]   -> ban(6,8) applied, remaining=%d\n",
                   bp.remaining());
        }

        // AJB: print tree state (upstream bp.print())
        printf("[AJB_STATE] tree state after step %d:\n", step);
        bp.print();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

    printf("[AJB_TIMER] %d pick steps: %.3f ms\n", step, ms);
    printf("[AJB] VERDICT: run_bpt_full PASSED (exhausted tree in %d steps)\n",
           step);
    return 0;
}
