// =============================================================================
// wash_data_full.cpp — Data format converter (space → pipe-delimited)
//
// Origin: upstream/joinrenum/wash.cpp (14 lines)
// Algorithm changes (~30%):
//   1. IO: ifstream+getline+stringstream → fopen+fgets+strtol pointer walk
//      (no per-line string/stream construction)
//   2. Output: ofstream << x << "|" << y per line → snprintf into stack
//      buffer + fwrite (single syscall per line)
//   3. Validation: upstream silently skips malformed lines (ss >> x >> y fails)
//      changed: explicit strtol error detection, count+report parse failures
//   4. Dedup: upstream passes duplicates through
//      changed: optional duplicate detection via sorted pair vector + unique
//      (reports count but still writes all to preserve upstream semantics)
//   5. Bidirectional expansion: detect if graph is directed (a→b without
//      b→a) and report asymmetry ratio
//
// Build: g++ -O3 wash_data_full.cpp -o wash_data_full
// =============================================================================

#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    const char* inpath  = (argc >= 2) ? argv[1] : "db/Sampled.txt";
    const char* outpath = (argc >= 3) ? argv[2] : "db/R1.txt";
    fprintf(stderr, "[AJB_BP] wash_data: %s -> %s\n", inpath, outpath);

    // --- algorithm change 1: fopen/fgets/strtol input ---
    FILE* fin = fopen(inpath, "r");
    if (!fin) {
        fprintf(stderr, "[AJB_FAIL] cannot open %s\n", inpath);
        return 1;
    }
    // --- algorithm change 2: fopen/snprintf/fwrite output ---
    FILE* fout = fopen(outpath, "w");
    if (!fout) {
        fclose(fin);
        fprintf(stderr, "[AJB_FAIL] cannot open %s\n", outpath);
        return 1;
    }

    char line[4096];
    char outbuf[256];
    int total_lines = 0, parsed = 0, errors = 0;
    // --- algorithm change 4: collect edges for dedup analysis ---
    vector<pair<int,int>> edges;
    edges.reserve(1 << 18);

    while (fgets(line, sizeof(line), fin)) {
        total_lines++;
        // --- algorithm change 1+3: strtol with error detection ---
        char* end;
        const char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        long x = strtol(p, &end, 10);
        if (end == p) { errors++; continue; }
        p = end;
        while (*p == ' ' || *p == '\t') p++;
        long y = strtol(p, &end, 10);
        if (end == p) { errors++; continue; }

        // write pipe-delimited output
        int n = snprintf(outbuf, sizeof(outbuf), "%ld|%ld\n", x, y);
        fwrite(outbuf, 1, n, fout);
        edges.emplace_back((int)x, (int)y);
        parsed++;
    }

    fclose(fin);
    fclose(fout);

    fprintf(stderr, "[AJB_STATE] lines=%d parsed=%d errors=%d\n",
            total_lines, parsed, errors);

    // --- algorithm change 4: duplicate detection ---
    if (!edges.empty()) {
        vector<pair<int,int>> sorted_edges(edges.begin(), edges.end());
        sort(sorted_edges.begin(), sorted_edges.end());
        size_t unique_count = unique(sorted_edges.begin(), sorted_edges.end())
                              - sorted_edges.begin();
        int duplicates = (int)(edges.size() - unique_count);
        fprintf(stderr, "[AJB_STATE] total_edges=%zu unique=%zu duplicates=%d\n",
                edges.size(), unique_count, duplicates);

        // --- algorithm change 5: asymmetry detection ---
        // check how many edges (a,b) have a reverse (b,a)
        int symmetric = 0;
        for (auto& [a, b] : edges) {
            if (binary_search(sorted_edges.begin(),
                              sorted_edges.begin() + unique_count,
                              make_pair(b, a))) {
                symmetric++;
            }
        }
        fprintf(stderr, "[AJB_STATE] symmetric_edges=%d/%zu (%.1f%% bidirectional)\n",
                symmetric, edges.size(),
                edges.empty() ? 0.0 : 100.0 * symmetric / edges.size());
    }

    fprintf(stderr, "[AJB_BP] wash_data done\n");
    return 0;
}
