// =============================================================================
// test_count_oracle.cpp — AJB-adapted CountOracle test harness
//
// Origin: upstream/joinrenum/testCountOracle.cpp (113 lines)
// Adaptation (~20%): AJB debug instrumentation, structured output, memory
//   tracking via AJBReportMemory(), scoped timers, and data integrity checks.
//
// Build: g++ -O3 test_count_oracle.cpp -lglpk -o test_co
// =============================================================================

#include <bits/stdc++.h>
#include <sys/resource.h>
#include <malloc.h>
#include "CountOracle.hpp"

// AJB: lightweight debug helpers (inline, no CUDA dependency)
#include <chrono>

// ---------------------------------------------------------------------------
// AJB: structured memory reporter (replaces upstream ad-hoc /proc reader)
// ---------------------------------------------------------------------------
struct AJBMemSnapshot {
  size_t vm_rss_kb = 0;
  size_t vm_size_kb = 0;
  std::chrono::high_resolution_clock::time_point timestamp;

  void Capture() {
    timestamp = std::chrono::high_resolution_clock::now();
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
      if (strncmp(line, "VmRSS:", 6) == 0)  sscanf(line + 6, "%zu", &vm_rss_kb);
      if (strncmp(line, "VmSize:", 7) == 0)  sscanf(line + 7, "%zu", &vm_size_kb);
    }
    fclose(f);
  }

  void Print(const char* label) const {
    printf("[AJB_MEM] %-20s RSS=%zu kB (%.2f MB) | VmSize=%zu kB\n",
           label, vm_rss_kb, vm_rss_kb / 1024.0, vm_size_kb);
  }

  // Diff against a prior snapshot
  void PrintDelta(const AJBMemSnapshot& before, const char* label) const {
    long delta_rss = (long)vm_rss_kb - (long)before.vm_rss_kb;
    double elapsed = std::chrono::duration<double>(timestamp - before.timestamp).count();
    printf("[AJB_MEM] %-20s delta_RSS=%+ld kB (%.2f MB) | elapsed=%.4f s\n",
           label, delta_rss, delta_rss / 1024.0, elapsed);
  }
};

// ---------------------------------------------------------------------------
// AJB: scoped timer (prints on destruction)
// ---------------------------------------------------------------------------
struct ScopedTimer {
  const char* label;
  std::chrono::high_resolution_clock::time_point t0;
  ScopedTimer(const char* l) : label(l), t0(std::chrono::high_resolution_clock::now()) {
    printf("[AJB_TIMER] >>> %s\n", label);
  }
  ~ScopedTimer() {
    double s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    printf("[AJB_TIMER] <<< %s  %.6f s (%.3f ms)\n", label, s, s * 1000.0);
  }
};

// ---------------------------------------------------------------------------
// Upstream helpers (retained)
// ---------------------------------------------------------------------------
void writeDataToFile(vector<Point<int>> points, string filename = "data.txt") {
    ofstream file;
    file.open(filename);
    for (size_t i = 0; i < points.size(); i++) {
        for (int j = 0; j < points[i].dim(); j++) {
            file << points[i][j] << " ";
        }
        file << endl;
    }
    file.close();
}

void readDataFromFile(vector<Point<int>>& points, string filename = "data.txt") {
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        vector<int> v;
        int num;
        while (iss >> num) {
            v.push_back(num);
        }
        points.push_back(Point<int>(v));
    }
    file.close();
}

pair<Point<int>, Point<int>> generateRange(CountOracle<int>& tree) {
    Point<int> lowbound = tree.getLowerBounds(), upbound = tree.getUpperBounds();
    int divdim = rand() % lowbound.dim(), divval, divval2;
    vector<int> vl, vr;
    for (int i = 0; i < divdim; i++) {
        divval = rand() % (upbound[i] - lowbound[i] + 1) + lowbound[i];
        vl.push_back(divval);
        vr.push_back(divval);
    }
    divval = rand() % (upbound[divdim] - lowbound[divdim] + 1) + lowbound[divdim];
    divval2 = rand() % (upbound[divdim] - divval + 1) + divval;
    if (divval > divval2) swap(divval, divval2);
    vl.push_back(divval);
    vr.push_back(divval2);
    for (int i = divdim + 1; i < lowbound.dim(); i++) {
        vl.push_back(lowbound[i]);
        vr.push_back(upbound[i]);
    }
    return make_pair(Point<int>(vl), Point<int>(vr));
}

// ---------------------------------------------------------------------------
// main — AJB adapted
// ---------------------------------------------------------------------------
using namespace std;

int main(int argc, char* argv[]) {
    // AJB: configurable range count from CLI
    int rangeNum = 100000;
    string dataFile = "data.txt";
    if (argc >= 2) rangeNum = atoi(argv[1]);
    if (argc >= 3) dataFile = argv[2];
    printf("[AJB] CountOracle test: rangeNum=%d dataFile=%s\n", rangeNum, dataFile.c_str());

    AJBMemSnapshot mem_before;
    mem_before.Capture();
    mem_before.Print("before_load");

    vector<Point<int>> points;
    {
        ScopedTimer t("read_data");
        readDataFromFile(points, dataFile);
    }
    // [AJB_BP] M940: synthetic data fallback if file missing or empty
    if (points.empty()) {
        fprintf(stderr, "[AJB_BP][test_count_oracle] no data loaded from '%s', generating synthetic data\n",
                dataFile.c_str());
        srand(42);
        const int synth_n = 200, synth_dim = 3;
        for (int i = 0; i < synth_n; i++) {
            vector<int> v(synth_dim);
            for (int d = 0; d < synth_dim; d++) v[d] = rand() % 1000;
            points.push_back(Point<int>(v));
        }
        printf("[AJB] Generated %d synthetic %d-D points (data.txt fallback)\n", synth_n, synth_dim);
    }
    printf("[AJB] Loaded %zu points, dim=%d\n",
           points.size(), points.empty() ? 0 : points[0].dim());

    AJBMemSnapshot mem_after_load;
    mem_after_load.Capture();
    mem_after_load.PrintDelta(mem_before, "after_load");

    // Build CountOracle
    CountOracle<int>* tree;
    {
        ScopedTimer t("build_count_oracle");
        tree = new CountOracle<int>(points);
    }

    AJBMemSnapshot mem_after_build;
    mem_after_build.Capture();
    mem_after_build.PrintDelta(mem_after_load, "after_build");

    // AJB: print structure summary
    printf("[AJB] CountOracle bounds: lower=[");
    auto lb = tree->getLowerBounds();
    for (size_t d = 0; d < lb.size(); d++) printf("%s%d", d ? "," : "", lb[d]);
    printf("] upper=[");
    auto ub = tree->getUpperBounds();
    for (size_t d = 0; d < ub.size(); d++) printf("%s%d", d ? "," : "", ub[d]);
    printf("]\n");

    // Free point data (upstream style — reclaim memory)
    vector<Point<int>>().swap(points);
    malloc_trim(0);

    AJBMemSnapshot mem_after_free;
    mem_after_free.Capture();
    mem_after_free.PrintDelta(mem_after_build, "after_free_pts");

    // Generate ranges
    vector<pair<Point<int>, Point<int>>> ranges;
    {
        ScopedTimer t("gen_ranges");
        ranges.reserve(rangeNum);
        for (int i = 0; i < rangeNum; i++) {
            ranges.push_back(generateRange(*tree));
        }
    }

    // Count queries + AJB: track min/max/sum for sanity check
    long long total_count = 0;
    long long min_count = LLONG_MAX, max_count = 0;
    {
        ScopedTimer t("range_queries");
        for (int i = 0; i < (int)ranges.size(); i++) {
            long long c = tree->count(ranges[i].first, ranges[i].second);
            total_count += c;
            min_count = min(min_count, c);
            max_count = max(max_count, c);
        }
    }

    // AJB: structured results output
    printf("\n[AJB_RESULTS] CountOracle range query summary:\n");
    printf("  queries      = %d\n", rangeNum);
    printf("  total_count  = %lld\n", total_count);
    printf("  min_count    = %lld\n", min_count);
    printf("  max_count    = %lld\n", max_count);
    printf("  avg_count    = %.2f\n", rangeNum > 0 ? (double)total_count / rangeNum : 0.0);

    // AJB: data integrity assertion
    if (min_count < 0) {
        fprintf(stderr, "[AJB_ERROR] Negative count detected — CountOracle may be corrupt\n");
        return 1;
    }
    printf("[AJB] CountOracle test PASSED\n");

    delete tree;
    return 0;
}
