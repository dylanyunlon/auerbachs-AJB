// =============================================================================
// run_bpt_full.cpp — BanPickTree step-by-step trace (AJB-instrumented)
//
// Origin: upstream/joinrenum/runBPT.cpp (11 lines, verbatim core)
// AJB adaptation (~20%): parameterized N via CLI, step-by-step AJB trace of
//   pick/ban/available/remaining, timing, structured output.
//
// Build: g++ -O3 run_bpt_full.cpp -o run_bpt_full
// Usage: ./run_bpt_full [N]  (default: 10)
// =============================================================================

#include <chrono>
#include "BanPickTree.hpp"

int main(int argc, char** argv){
    int N = (argc > 1) ? atoi(argv[1]) : 10;
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] run_bpt_full  BanPickTree trace (N=%d)\n", N);
    fprintf(stderr, "[AJB] ============================================\n");

    auto t0 = std::chrono::high_resolution_clock::now();
    BanPickTree bp(N);
    int step = 0;

    // upstream: ban(2,4), then loop
    fprintf(stderr, "[AJB_TRACE] ban(2,4)\n");
    bp.ban(2,4);
    step++;

    while(bp.remaining() > 0){
        int picked = bp.pick();
        fprintf(stderr, "[AJB_TRACE] step=%d  pick=%d  remaining=%d\n",
                step, picked, bp.remaining());
        cout << "pick: " << picked << endl;

        // upstream: conditional ban(6,8)
        if(bp.available(6,8)) {
            bp.ban(6,8);
            fprintf(stderr, "[AJB_TRACE] ban(6,8) applied\n");
        }
        bp.print();
        step++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] BPT run: %.3f us (%d steps)\n",
            std::chrono::duration<double,std::micro>(t1 - t0).count(), step);
    fprintf(stderr, "[AJB] run_bpt_full COMPLETE\n");
    return 0;
}
