// =============================================================================
// wash_data_full.cpp — Data format converter (AJB-instrumented)
//
// Origin: upstream/joinrenum/wash.cpp (14 lines, verbatim core)
// AJB adaptation (~20%): CLI paths, per-line field count validation,
//   value range tracking per column, malformed line detection with
//   row number, output line count verification, empty-field warnings.
//
// Build: g++ -O3 wash_data_full.cpp -o wash_full
// Usage: ./wash_full [input] [output]
// =============================================================================

#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    string input_path  = (argc >= 2) ? argv[1] : "db/Sampled.txt";
    string output_path = (argc >= 3) ? argv[2] : "db/R1.txt";

    fprintf(stderr, "[AJB] ============================================\n");
    fprintf(stderr, "[AJB] wash_data_full  %s -> %s\n",
            input_path.c_str(), output_path.c_str());
    fprintf(stderr, "[AJB] ============================================\n");

    ifstream fin(input_path);
    if (!fin.is_open()) {
        fprintf(stderr, "[AJB_FAIL] Cannot open input: %s\n", input_path.c_str());
        return 1;
    }
    ofstream fout(output_path);
    if (!fout.is_open()) {
        fprintf(stderr, "[AJB_FAIL] Cannot open output: %s\n", output_path.c_str());
        return 1;
    }

    string line;
    int line_num = 0, written = 0, malformed = 0, empty_fields = 0;
    // AJB: track value ranges per column
    long long col0_min = LLONG_MAX, col0_max = LLONG_MIN;
    long long col1_min = LLONG_MAX, col1_max = LLONG_MIN;

    while (getline(fin, line)) {
        line_num++;
        // upstream: parse "x y" and output "x|y"
        stringstream ss(line);
        string x, y;
        ss >> x >> y;

        if (x.empty() || y.empty()) {
            malformed++;
            if (malformed <= 5)
                fprintf(stderr, "[AJB_WARN] line %d: malformed (x='%s' y='%s')\n",
                        line_num, x.c_str(), y.c_str());
            continue;
        }

        // AJB: check for empty-looking fields
        if (x == "0" || y == "0") empty_fields++;

        fout << x << "|" << y << endl;
        written++;

        // AJB: track value ranges (try numeric parse)
        try {
            long long vx = stoll(x), vy = stoll(y);
            col0_min = min(col0_min, vx); col0_max = max(col0_max, vx);
            col1_min = min(col1_min, vy); col1_max = max(col1_max, vy);
        } catch (...) {
            // non-numeric — skip range tracking
        }
    }

    fprintf(stderr, "[AJB_STATE] === Wash Summary ===\n");
    fprintf(stderr, "[AJB_STATE] input_lines=%d  written=%d  malformed=%d\n",
            line_num, written, malformed);
    if (col0_min != LLONG_MAX) {
        fprintf(stderr, "[AJB_STATE] col0 range: [%lld, %lld]\n", col0_min, col0_max);
        fprintf(stderr, "[AJB_STATE] col1 range: [%lld, %lld]\n", col1_min, col1_max);
    }
    if (empty_fields > 0)
        fprintf(stderr, "[AJB_WARN] %d lines contain zero-valued fields\n", empty_fields);
    if (malformed > 5)
        fprintf(stderr, "[AJB_WARN] ... and %d more malformed lines (suppressed)\n",
                malformed - 5);

    fprintf(stderr, "[AJB] wash_data_full DONE\n");
    return 0;
}
