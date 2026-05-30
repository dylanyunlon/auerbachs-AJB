// =============================================================================
// ajb_renum_test.cpp — CPU-only test for AJB × joinrenum integration
// =============================================================================
// Reads the sample database (db/), runs RandOrderEnum sampling,
// feeds into AJB's skew detection, and outputs cadence recommendations.
//
// Build (no GPU required):
//   g++ -std=c++17 -O2 -I../joinrenum -DAJB_TRACE_DECISIONS \
//       -o ajb_renum_test ajb_renum_test.cpp
//
// Run:
//   ./ajb_renum_test [num_samples] [num_gpus] [base_K]
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>

#include "ajb_renum_adapter.hpp"

int main(int argc, char* argv[]) {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║     AJB × REnum Integration Test         ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    // Defaults
    size_t num_samples = 500;
    int    num_gpus    = 2;
    int    base_K      = 8;

    if (argc > 1) num_samples = std::atoi(argv[1]);
    if (argc > 2) num_gpus    = std::atoi(argv[2]);
    if (argc > 3) base_K      = std::atoi(argv[3]);

    printf("Configuration:\n");
    printf("  num_samples = %zu\n", num_samples);
    printf("  num_gpus    = %d\n", num_gpus);
    printf("  base_K      = %d\n\n", base_K);

    // --- Setup RandOrderEnum from the sample db/ ---
    // The db/ directory contains:
    //   filenames.txt  — maps relation name → CSV filename
    //   numlines.txt   — maps relation name → row count
    //   relations.txt  — maps relation name → variable list
    //   Ra.csv, Rb.csv, Rc.csv — actual data
    //
    // For the sample data: R(a,b) ⋈ S(b,c) ⋈ T(c,a) triangle query

    std::string db_dir = "db/";

    // Check if we're run from src/joinrenum/ or from project root
    {
        FILE* f = fopen("db/filenames.txt", "r");
        if (!f) {
            f = fopen("src/joinrenum/db/filenames.txt", "r");
            if (f) {
                db_dir = "src/joinrenum/db/";
                fclose(f);
            } else {
                printf("[ERROR] Cannot find db/ directory. Run from src/joinrenum/ or project root.\n");
                return 1;
            }
        } else {
            fclose(f);
        }
    }

    printf("Using database directory: %s\n\n", db_dir.c_str());

    // Read relations via upstream ReadConfig (parses "R1(A,B)" format)
    std::string rel_path = db_dir + "relations.txt";
    std::string fn_path  = db_dir + "filenames.txt";
    std::string nl_path  = db_dir + "numlines.txt";

    auto relations = readRelations(rel_path);
    auto filenames = readFilenames(fn_path);
    auto numlines  = readNumLines(nl_path);

    // Build relationNames and relationVars from parsed data
    std::vector<std::string> relationNames;
    std::vector<std::vector<std::string>> relationVars;
    for (auto& [name, vars] : relations) {
        relationNames.push_back(name);
        relationVars.push_back(vars);
    }

    printf("Parsed %zu relations:\n", relationNames.size());
    for (size_t i = 0; i < relationNames.size(); i++) {
        printf("  %s(", relationNames[i].c_str());
        for (size_t j = 0; j < relationVars[i].size(); j++) {
            if (j > 0) printf(", ");
            printf("%s", relationVars[i][j].c_str());
        }
        printf(")\n");
    }
    printf("\n");

    // --- Construct RandOrderEnum ---
    // Fix paths in filenames map: upstream uses "db/Ra.csv" but we need
    // paths relative to where we're running
    for (auto& [name, path] : filenames) {
        // If path starts with "db/" and db_dir is different, fix it
        if (path.substr(0, 3) == "db/" && db_dir != "db/") {
            path = db_dir + path.substr(3);  // e.g. "src/joinrenum/db/Ra.csv"
        } else if (db_dir == "db/") {
            // already correct
        }
    }

    printf("[AJB-REnum] Constructing RandOrderEnum...\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    RandOrderEnum renum(
        db_dir + "filenames.txt",
        db_dir + "numlines.txt",
        db_dir + "relations.txt",
        relationNames,
        relationVars
    );

    auto t1 = std::chrono::high_resolution_clock::now();
    double construct_time = std::chrono::duration<double>(t1 - t0).count();
    printf("[AJB-REnum] Construction took %.4fs\n\n", construct_time);

    // --- Run single probe ---
    printf("=== Single Probe ===\n");
    auto diag = ajb::RunREnumProbeAndRecommend(
        renum, num_samples, num_gpus, base_K, 0.6f, "renum_diag.csv");

    // --- Run batch probe for stability analysis ---
    printf("\n=== Batch Stability Analysis ===\n");
    auto batch = ajb::BatchREnumProbe(
        renum, {50, 100, 500, 1000}, num_gpus, base_K);

    // --- Final summary ---
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║         Test Complete                     ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Recommended cadence for %d GPUs:         \n", num_gpus);
    printf("║    K_x = %d  (build partitions)           \n", diag.K_x);
    printf("║    K_u = %d  (merge-path boundaries)      \n", diag.K_u);
    printf("║    K_v = %d  (materialization buffers)     \n", diag.K_v);
    printf("║  Skew: CV=%.4f  norm=%.4f  high=%s        \n",
           diag.cv, diag.normalized_skew, diag.is_high_skew ? "Y" : "N");
    printf("╚══════════════════════════════════════════╝\n");

    return 0;
}
