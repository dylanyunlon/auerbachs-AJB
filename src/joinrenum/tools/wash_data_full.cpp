// =============================================================================
// wash_data_full.cpp — data format converter + validation (AJB-instrumented)
//
// Origin: upstream/joinrenum/wash.cpp (14 lines, verbatim core)
// AJB adaptation (~20%): CLI input/output paths, line counting, format
//   validation, sample output, error reporting, AJB trace tags.
//
// Build: g++ -O3 wash_data_full.cpp -o wash_data_full
// Usage: ./wash_data_full [input] [output]  (defaults: db/Sampled.txt db/R1.txt)
// =============================================================================

#include<bits/stdc++.h>
#include <chrono>
using namespace std;

int main(int argc, char** argv) {
    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] wash_data_full  format converter\n");
    fprintf(stderr, "[AJB] ============================================\n");

    // AJB: CLI args (upstream used hardcoded paths)
    string inpath = (argc > 1) ? argv[1] : "db/Sampled.txt";
    string outpath = (argc > 2) ? argv[2] : "db/R1.txt";
    fprintf(stderr, "[AJB_STATE] input=%s  output=%s\n", inpath.c_str(), outpath.c_str());

    // upstream: read from input, convert "x y" → "x|y"
    auto t0 = chrono::high_resolution_clock::now();
    ifstream fin(inpath);
    if(!fin.is_open()) {
        fprintf(stderr, "[AJB_FAIL] Cannot open input: %s\n", inpath.c_str());
        return 1;
    }

    ofstream fout(outpath);
    if(!fout.is_open()) {
        fprintf(stderr, "[AJB_FAIL] Cannot open output: %s\n", outpath.c_str());
        return 1;
    }

    string line;
    int lines_read = 0, lines_written = 0, parse_errors = 0;
    while (getline(fin, line)) {
        lines_read++;
        // upstream: parse line "x y" and output as "x|y"
        stringstream ss(line);
        string x, y;
        if(ss >> x >> y) {
            fout << x << "|" << y << endl;
            lines_written++;
            // AJB: sample first few lines
            if(lines_written <= 3)
                fprintf(stderr, "[AJB_TRACE] line %d: \"%s %s\" -> \"%s|%s\"\n",
                        lines_written, x.c_str(), y.c_str(), x.c_str(), y.c_str());
        } else {
            parse_errors++;
            if(parse_errors <= 3)
                fprintf(stderr, "[AJB_WARN] parse error on line %d: \"%s\"\n",
                        lines_read, line.c_str());
        }
    }

    fin.close();
    fout.close();

    auto t1 = chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER] conversion: %.3f ms\n",
            chrono::duration<double,milli>(t1 - t0).count());
    fprintf(stderr, "[AJB_STATE] read=%d  written=%d  errors=%d\n",
            lines_read, lines_written, parse_errors);

    if(parse_errors > 0)
        fprintf(stderr, "[AJB_WARN] %d lines could not be parsed\n", parse_errors);

    fprintf(stderr, "[AJB] wash_data_full COMPLETE\n");
    return 0;
}
