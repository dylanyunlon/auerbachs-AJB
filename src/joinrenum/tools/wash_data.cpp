// =============================================================================
// wash_data.cpp — AJB-adapted data format converter
//
// Origin: upstream/joinrenum/wash.cpp (14 lines)
// Adaptation (~20%): AJB configurable delimiters, line counting,
//   progress reporting, and validation of output format.
//
// Converts space-delimited relation files to pipe-delimited format
// used by ReadConfig / Index preprocessing.
//
// Build: g++ -O3 wash_data.cpp -o wash_data
// Usage: ./wash_data [input] [output] [in_delim] [out_delim]
// =============================================================================

#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    string input_file  = "db/Sampled.txt";
    string output_file = "db/R1.txt";
    char   in_delim    = ' ';
    string out_delim   = "|";

    if (argc >= 2) input_file  = argv[1];
    if (argc >= 3) output_file = argv[2];
    if (argc >= 4) in_delim    = argv[3][0];
    if (argc >= 5) out_delim   = argv[4];

    printf("[AJB] wash_data: %s -> %s (delim '%c' -> '%s')\n",
           input_file.c_str(), output_file.c_str(), in_delim, out_delim.c_str());

    ifstream fin(input_file);
    if (!fin.is_open()) {
        fprintf(stderr, "[AJB_ERROR] Cannot open input: %s\n", input_file.c_str());
        return 1;
    }

    ofstream fout(output_file);
    if (!fout.is_open()) {
        fprintf(stderr, "[AJB_ERROR] Cannot open output: %s\n", output_file.c_str());
        return 1;
    }

    string line;
    int line_count = 0;
    int field_count_min = INT_MAX, field_count_max = 0;

    while (getline(fin, line)) {
        // Parse fields by input delimiter
        vector<string> fields;
        stringstream ss(line);
        string token;
        while (getline(ss, token, in_delim)) {
            if (!token.empty()) fields.push_back(token);
        }

        // Write with output delimiter
        for (size_t i = 0; i < fields.size(); i++) {
            fout << fields[i];
            if (i + 1 < fields.size()) fout << out_delim;
        }
        fout << "\n";

        field_count_min = min(field_count_min, (int)fields.size());
        field_count_max = max(field_count_max, (int)fields.size());
        line_count++;
    }

    // AJB: summary
    printf("\n[AJB_RESULTS] wash_data summary:\n");
    printf("  lines         = %d\n", line_count);
    printf("  fields/line   = [%d, %d]\n", field_count_min, field_count_max);
    if (field_count_min != field_count_max) {
        printf("[AJB_WARN] Ragged field counts — data may have inconsistent columns\n");
    }
    printf("[AJB] wash_data DONE\n");
    return 0;
}
