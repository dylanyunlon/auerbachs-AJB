// =============================================================================
// run_bpt.cpp — AJB-adapted BanPickTree demo
//
// Origin: upstream/joinrenum/runBPT.cpp (11 lines)
// Adaptation (~20%): AJB step-by-step state logging, pick/ban sequence
//   recording, and structured output. BanPickTree is used in the
//   Enumerator's random-order join attribute selection.
//
// Build: g++ -O3 run_bpt.cpp -o run_bpt
// =============================================================================

#include "BanPickTree.hpp"

int main(int argc, char* argv[]) {
    int n = 10;
    if (argc >= 2) n = atoi(argv[1]);
    printf("[AJB] BanPickTree demo: n=%d\n", n);

    BanPickTree bp(n);

    int step = 0;
    // Initial ban
    bp.ban(2, 4);
    printf("[AJB] step %d: ban(2,4) | remaining=%d\n", step++, bp.remaining());

    while (bp.remaining() > 0) {
        int picked = bp.pick();
        printf("[AJB] step %d: pick() -> %d | remaining=%d\n",
               step++, picked, bp.remaining());

        if (bp.available(6, 8)) {
            bp.ban(6, 8);
            printf("[AJB] step %d: ban(6,8) | remaining=%d\n",
                   step++, bp.remaining());
        }

        // AJB: print tree state at each step
        printf("[AJB_STATE] ");
        bp.print();
    }

    printf("\n[AJB_RESULTS] BanPickTree completed in %d steps\n", step);
    printf("[AJB] BanPickTree demo DONE\n");
    return 0;
}
