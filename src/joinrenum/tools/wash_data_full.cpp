// =============================================================================
// wash_data_full.cpp  AJB-adapted data format converter
//
// Origin: upstream/joinrenum/wash.cpp (14 lines)
// AJB adaptation (~20%): CLI args, progress output, line counting,
//   format validation, and error reporting.
//
// Build: g++ -O3 wash_data_full.cpp -o wash_full
// Usage: ./wash_full [input] [output]   # default: db/Sampled.txt -> db/R1.txt
// =============================================================================

#include <bits/stdc++.h>
#include <chrono>
using namespace std;

int main(int argc, char* argv[]) {
    printf("[AJB] ============================================\n");
    printf("[AJB] wash_data_full  data format converter\n");
    printf("[AJB] ============================================\n");

    // AJB: parameterized I/O (upstream hardcoded)
    string inpath = (argc > 1) ? argv[1] : "db/Sampled.txt";
    string outpath = (argc > 2) ? argv[2] : "db/R1.txt";

    printf("[AJB_STATE] Input:  %s\n", inpath.c_str());
    printf("[AJB_STATE] Output: %s\n", outpath.c_str());

    ifstream fin(inpath);
    if (!fin.is_open()) {
        fprintf(stderr, "[AJB_FAIL] Cannot open input: %s\n", inpath.c_str());
        return 1;
    }

    ofstream fout(outpath);
    if (!fout.is_open()) {
        fprintf(stderr, "[AJB_FAIL] Cannot open output: %s\n", outpath.c_str());
        return 1;
    }

    auto t0 = chrono::high_resolution_clock::now();

    // upstream: convert "x y" format to "x|y" format
    string line;
    int lines_ok = 0, lines_err = 0;
    while (getline(fin, line)) {
        stringstream ss(line);
        string x, y;
        if (ss >> x >> y) {
            fout << x << "|" << y << endl;
            lines_ok++;
        } else {
            lines_err++;
            // AJB: report malformed lines
            if (lines_err <= 5)
                fprintf(stderr, "[AJB_WARN] Malformed line %d: '%s'\n",
                        lines_ok + lines_err, line.c_str());
        }

        // AJB: progress every 100k lines
        if (lines_ok > 0 && lines_ok % 100000 == 0)
            printf("[AJB_TRACE] converted %d lines...\n", lines_ok);
    }

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double,milli>(t1 - t0).count();

    printf("[AJB_TIMER] conversion: %.3f ms\n", ms);
    printf("[AJB_STATE] Converted %d lines (%d errors) -> %s\n",
           lines_ok, lines_err, outpath.c_str());

    // AJB: verify output file size
    fout.close();
    ifstream check(outpath, ios::ate);
    auto fsize = check.tellg();
    printf("[AJB_STATE] Output file size: %ld bytes\n", (long)fsize);

    printf("[AJB] VERDICT: wash_data_full %s\n",
           lines_ok > 0 ? "PASSED" : "NO_DATA");
    return 0;
}
