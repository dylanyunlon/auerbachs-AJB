#include<bits/stdc++.h>
using namespace std;
int main(int argc, char* argv[]) {
    // --- CLI: argument-based paths replacing hardcoded ---
    // upstream: hardcoded "db/Sampled.txt" -> "db/R1.txt"
    // changed: argv-based paths with defaults
    string input_path  = (argc >= 2) ? argv[1] : "db/Sampled.txt";
    string output_path = (argc >= 3) ? argv[2] : "db/R1.txt";

    // --- I/O: mmap-style read into single string, then parse ---
    // upstream: getline → stringstream → ss >> x >> y
    // changed: read entire file into a string buffer, then walk it
    //   with pointer arithmetic (like a C parser), no istringstream
    ifstream fin(input_path, ios::ate);
    if(!fin.is_open()) { cerr << "Cannot open: " << input_path << endl; return 1; }
    size_t fsize = fin.tellg();
    fin.seekg(0);
    string raw(fsize, '\0');
    fin.read(&raw[0], fsize);
    fin.close();

    ofstream fout(output_path);
    if(!fout.is_open()) { cerr << "Cannot open: " << output_path << endl; return 1; }

    // --- parser: pointer-walk over raw buffer ---
    // upstream: per-line getline + stringstream split
    // changed: single pass over char*, no string copies, no stream objects
    //   parse pairs (field1 delim field2 newline), emit field1|field2
    int converted = 0, skipped = 0;
    const char* p = raw.c_str();
    const char* end = p + raw.size();

    while(p < end) {
        // skip leading whitespace/empty lines
        while(p < end && (*p == '\n' || *p == '\r')) p++;
        if(p >= end) break;

        // field 1: run until whitespace or tab
        const char* f1_start = p;
        while(p < end && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
        size_t f1_len = p - f1_start;
        if(f1_len == 0 || p >= end || (*p == '\n' || *p == '\r')) {
            // no second field — skip to next line
            while(p < end && *p != '\n') p++;
            skipped++;
            continue;
        }

        // skip delimiter(s) between fields
        while(p < end && (*p == ' ' || *p == '\t')) p++;

        // field 2: run until newline
        const char* f2_start = p;
        while(p < end && *p != '\n' && *p != '\r') p++;
        // trim trailing whitespace from field 2
        const char* f2_end = p;
        while(f2_end > f2_start && (*(f2_end-1) == ' ' || *(f2_end-1) == '\t' || *(f2_end-1) == '\r'))
            f2_end--;
        size_t f2_len = f2_end - f2_start;

        if(f2_len == 0) {
            skipped++;
            continue;
        }

        // emit: field1|field2
        fout.write(f1_start, f1_len);
        fout.put('|');
        fout.write(f2_start, f2_len);
        fout.put('\n');
        converted++;
    }

    fout.close();

    if(converted == 0)
        cout << "Warning: no lines converted" << endl;
    cout << "Converted " << converted << " lines";
    if(skipped > 0) cout << " (skipped " << skipped << ")";
    cout << endl;
    return 0;
}
