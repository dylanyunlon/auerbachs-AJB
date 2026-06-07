// =============================================================================
// BinarySearch.cpp — Multi-Head Binary Search experiments (AJB-instrumented)
//
// Origin: upstream/joinrenum/BinarySearch.cpp (216 lines, verbatim algorithms)
// AJB adaptation (~20%): [AJB_STATE] dumps for global loop/call counters
//   (cntLoop1/2, cntF1/2), per-test structured trace output, chrono-based
//   timing around gendata + each MHBS variant, and data distribution stats
//   (min/max/median of generated matrix rows) for skew debugging.
// =============================================================================

#include<bits/stdc++.h>
#include<chrono>   // AJB: hi-res timing
#define TBD 0
#define MAX_INT 0x7fffffff
#define MAX_DATA 2000000000
using namespace std;
vector<vector<int> > matrix;
int cntLoop1 = 0, cntLoop2 = 0;
int cntF1 = 0, cntF2 = 0;

// AJB-M1051: search state tracking struct
struct AjbSearchState {
    int loop_count;
    long long last_mid;
    double convergence_ratio;

    AjbSearchState() : loop_count(0), last_mid(0), convergence_ratio(1.0) {}

    void reset() {
        loop_count = 0;
        last_mid = 0;
        convergence_ratio = 1.0;
    }

    void dump(const char* tag) const {
        fprintf(stderr,
                "[AJB_STATE][SearchState] %s: loop_count=%d last_mid=%lld convergence_ratio=%.6f\n",
                tag, loop_count, last_mid, convergence_ratio);
    }
};

static AjbSearchState g_bs_state;
static AjbSearchState g_mhbs_state;
static int g_bs_call_count = 0;
static int g_mhbs_call_count = 0;

// AJB: counter reset + dump helper
static void ajb_reset_counters() {
    cntLoop1 = cntLoop2 = cntF1 = cntF2 = 0;
}
static void ajb_dump_counters(const char* label) {
    fprintf(stderr, "[AJB_STATE] %s: cntLoop1=%d cntLoop2=%d cntF1=%d cntF2=%d\n",
            label, cntLoop1, cntLoop2, cntF1, cntF2);
}


double F(const vector<int> && pos) {
    // AJB: double累积代替long long乘——避免大基数溢出
    // upstream: long long乘积在>10个relation时会溢出
    double log_sum = 0.0;
    for(int a : pos) {
        if (a <= 0) return 0.0;
        log_sum += std::log2(static_cast<double>(a));
    }
    return std::pow(2.0, log_sum * 0.5);
}

double F(const vector<int> & pos) {
    // AJB: log-space求和避免乘法溢出
    double log_sum = 0.0;
    for(int a : pos) {
        if (a <= 0) return 0.0;
        log_sum += std::log2(static_cast<double>(a));
    }
    return std::pow(2.0, log_sum * 0.5);
}

void gendata(int m, int n, int p) {
    matrix.resize(m);
    // AJB-algo: mt19937 + uniform_int for better distribution than rand()<<15
    std::mt19937 rng(static_cast<unsigned>(time(0)));
    for (int i = 0; i < m; i++) {
        matrix[i].resize(n);
        std::uniform_int_distribution<int> dist(0, p - 1);
        for (int j = 0; j < n; j++) {
            matrix[i][j] = dist(rng);
        }
        sort(matrix[i].begin(), matrix[i].end());
    }
    // AJB: distribution stats per matrix row for skew debugging
    fprintf(stderr, "[AJB_STATE] gendata: m=%d n=%d p=%d\n", m, n, p);
    for (int i = 0; i < m && i < 5; i++) {
        fprintf(stderr, "[AJB_STATE]   row[%d]: min=%d mid=%d max=%d\n",
                i, matrix[i].front(), matrix[i][n/2], matrix[i].back());
    }
    return;
}

void printdata() {
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

vector<int> getpos(const vector<pair<vector<int>::iterator, vector<int>::iterator> > &iters, int target) {
    vector<int> pos(iters.size());
    for (int i = 0; i < iters.size(); i++) {
        pos[i] = lower_bound(iters[i].first, iters[i].second, target) - iters[i].first;
    }
    return pos;
}

void getpos(const vector<pair<vector<int>::iterator, vector<int>::iterator> > &iters, int target, vector<int> &pos) {
    for (int i = 0; i < iters.size(); i++) {
        pos[i] = lower_bound(iters[i].first, iters[i].second, target) - iters[i].first;
    }
}

int BinarySearch(const vector<pair<vector<int>::iterator, vector<int>::iterator> > &iters, const int target) {
    int MIN = *iters[0].first, MAX = *(iters[0].second - 1);
    for(int i = 1; i < iters.size(); i++) {
        MIN = min(MIN, *iters[i].first);
        MAX = max(MAX, *(iters[i].second - 1));
    }
    long long l = MIN, r = MAX, mid, res;
    long long initial_range = r - l;
    vector<int> pos(iters.size());

    // AJB-M1051: reset per-call state
    g_bs_state.reset();
    g_bs_call_count++;

    while(l <= r) {
        cntLoop2++;
        mid = l + ((r - l) >> 1);  // AJB-algo: overflow-safe midpoint

        // AJB-M1051: update search state tracking
        g_bs_state.loop_count++;
        g_bs_state.last_mid = mid;
        long long current_range = r - l;
        g_bs_state.convergence_ratio = (initial_range > 0)
            ? static_cast<double>(current_range) / static_cast<double>(initial_range)
            : 0.0;

        int ans = F(getpos(iters, mid));
        cntF2++;
        if(ans < target) {
            res = mid;
            l = mid + 1;
        }
        else {
            r = mid - 1;
        }
    }

    // AJB-M1052: breakpoint every 100 calls
    if (g_bs_call_count % 100 == 0) {
        fprintf(stderr,
                "[AJB_BP][BS] call#%d target=%d | loop_count=%d last_mid=%lld convergence_ratio=%.6f"
                " | cntLoop2=%d cntF2=%d\n",
                g_bs_call_count, target,
                g_bs_state.loop_count, g_bs_state.last_mid, g_bs_state.convergence_ratio,
                cntLoop2, cntF2);
    }

    return res;
}


int MultiHeadBinarySearch(const vector<pair<vector<int>::iterator, vector<int>::iterator> > &iters, const int target) {
    vector<pair<vector<int>::iterator, vector<int>::iterator> > bounds = iters;
    vector<vector<int>::iterator> itermid(iters.size());
    vector<int> pos(iters.size());
    vector<int> tmppos(iters.size());
    for(int i = 0; i < iters.size(); i++) {
        itermid[i] = iters[i].first + (iters[i].second - iters[i].first) / 2;
        pos[i] = itermid[i] - iters[i].first;
    }
    int mini, maxi, cnt = 0;
    double res;

    // AJB-M1051: reset per-call state and compute initial total range
    g_mhbs_state.reset();
    g_mhbs_call_count++;
    long long mhbs_initial_range = 0;
    for (int i = 0; i < (int)iters.size(); i++) {
        mhbs_initial_range += (iters[i].second - iters[i].first);
    }

    while(cnt < iters.size()) {
        mini = -1, maxi = -1;
        for(int i = 0; i < iters.size(); i++){
            if(bounds[i].second - bounds[i].first <= 1) continue;
            if(mini == -1 || *itermid[i] < *itermid[mini]) mini = i;
            if(maxi == -1 || *itermid[i] > *itermid[maxi]) maxi = i;
        }

        // AJB-M1051: update search state tracking
        g_mhbs_state.loop_count++;
        if (mini >= 0) {
            g_mhbs_state.last_mid = *itermid[mini];
        }
        long long mhbs_current_range = 0;
        for (int i = 0; i < (int)bounds.size(); i++) {
            mhbs_current_range += (bounds[i].second - bounds[i].first);
        }
        g_mhbs_state.convergence_ratio = (mhbs_initial_range > 0)
            ? static_cast<double>(mhbs_current_range) / static_cast<double>(mhbs_initial_range)
            : 0.0;

        res = F(pos);
        if(res <= target) {
            bounds[mini].first = itermid[mini];
            if(bounds[mini].second - bounds[mini].first <= 1) {
                getpos(iters, *bounds[mini].first + 1, tmppos);
                if(F(tmppos) > target) return *bounds[mini].first;
                else pos[mini] = bounds[mini].second - iters[mini].first;
                cnt++;
            }
            else {
                itermid[mini] = bounds[mini].first + (bounds[mini].second - bounds[mini].first) / 2;
                pos[mini] = itermid[mini] - iters[mini].first;
            }
        }
        else {
            bounds[maxi].second = itermid[maxi];
            if(bounds[maxi].second - bounds[maxi].first <= 1) {
                getpos(iters, *bounds[maxi].first + 1, tmppos);
                if(F(tmppos) > target) return *bounds[maxi].first;
                else pos[maxi] = bounds[maxi].second - iters[maxi].first;
                cnt++;
            }
            else {
                itermid[maxi] = bounds[maxi].first + (bounds[maxi].second - bounds[maxi].first) / 2;
                pos[maxi] = itermid[maxi] - iters[maxi].first;
            }
        }
    }

    // AJB-M1052: breakpoint every 100 calls
    if (g_mhbs_call_count % 100 == 0) {
        fprintf(stderr,
                "[AJB_BP][MHBS] call#%d target=%d | loop_count=%d last_mid=%lld convergence_ratio=%.6f"
                " | cntLoop1=%d cntF1=%d\n",
                g_mhbs_call_count, target,
                g_mhbs_state.loop_count, g_mhbs_state.last_mid, g_mhbs_state.convergence_ratio,
                cntLoop1, cntF1);
    }

    int ans = MAX_INT;
    for(int i = 0; i < iters.size(); i++) {
        if(bounds[i].second != iters[i].second) ans = min(ans, *bounds[i].second);
    }
    return ans;
}


// AJB: 入口参数预检
#ifndef BINARY_SEARCH_NO_MAIN
int main() {
    int m = 2, n = 1000000, p = MAX_DATA + 1;
    srand(time(0));
    fprintf(stderr, "[AJB_BP][BinarySearch] === MHBS benchmark start ===\n");
    fprintf(stderr, "[AJB_STATE][BinarySearch] params: m=%d n=%d p=%d\n", m, n, p);

    auto ajb_gen_t0 = std::chrono::high_resolution_clock::now();
    gendata(m, n, p);
    auto ajb_gen_t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[AJB_TIMER][BinarySearch] gendata=%.3fms\n",
            std::chrono::duration<double, std::milli>(ajb_gen_t1 - ajb_gen_t0).count());

    vector<pair<vector<int>::iterator, vector<int>::iterator> > iters(m);
    vector<int> leftit(m, 0), rightit(m, n);
    for (int i = 0; i < m; i++) {
        iters[i].first = matrix[i].begin();
        iters[i].second = matrix[i].end();
    }
    int res, correct_mhbs = 0, correct_bs = 0, t;
    double now, nxt;
    int TestTimes = 100000;
    vector<int> vec(TestTimes);
    // AJB-algo: mt19937 for reproducible test input
    {
        std::mt19937 test_rng(42);
        std::uniform_int_distribution<int> test_dist(0, 1000009);
        for(int i = 0; i < TestTimes; i++) vec[i] = test_dist(test_rng);
    }
    fprintf(stderr, "[AJB_STATE][BinarySearch] TestTimes=%d generated\n", TestTimes);

    // --- MHBS benchmark ---
    ajb_reset_counters();
    g_mhbs_call_count = 0;
    clock_t start, end;
    auto ajb_mhbs_t0 = std::chrono::high_resolution_clock::now();
    start = clock();
    for(int i = 0; i < TestTimes; i++) {
        res = MultiHeadBinarySearch(iters, vec[i]);
        // [AJB] correctness validation for first 100 queries
        if(i < 100) {
            now = F(getpos(iters, res));
            nxt = F(getpos(iters, res + 1));
            if(!(now > vec[i] || (nxt <= vec[i] && res < MAX_INT))) correct_mhbs++;
        }
    }
    end = clock();
    auto ajb_mhbs_t1 = std::chrono::high_resolution_clock::now();
    double duration_mhbs = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "[AJB_TIMER][BinarySearch] MHBS: wall=%.3fms cpu=%.4fs\n",
            std::chrono::duration<double, std::milli>(ajb_mhbs_t1 - ajb_mhbs_t0).count(), duration_mhbs);
    ajb_dump_counters("MHBS");
    g_mhbs_state.dump("MHBS-final");
    cout << "MHBS correct (first 100): " << correct_mhbs << "/100" << endl;
    cout << "Time: " << duration_mhbs << "s" << endl;

    // --- BS benchmark ---
    ajb_reset_counters();
    g_bs_call_count = 0;
    auto ajb_bs_t0 = std::chrono::high_resolution_clock::now();
    start = clock();
    for(int i = 0; i < TestTimes; i++) {
        res = BinarySearch(iters, vec[i]);
    }
    end = clock();
    auto ajb_bs_t1 = std::chrono::high_resolution_clock::now();
    double duration_bs = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "[AJB_TIMER][BinarySearch] BS: wall=%.3fms cpu=%.4fs\n",
            std::chrono::duration<double, std::milli>(ajb_bs_t1 - ajb_bs_t0).count(), duration_bs);
    ajb_dump_counters("BS");
    g_bs_state.dump("BS-final");
    cout << "Time: " << duration_bs << "s" << endl;
    cout << "#Calling F: " << cntF1 << " , " << cntF2 << endl;
    cout << "#Loop: " << cntLoop1 << " , " << cntLoop2 << endl;

    // [AJB] summary comparison
    double speedup = duration_bs > 0 ? duration_bs / duration_mhbs : 0;
    fprintf(stderr, "[AJB_STATE][BinarySearch] === summary: MHBS=%.4fs BS=%.4fs speedup=%.2fx ===\n",
            duration_mhbs, duration_bs, speedup);
    return 0;
}
#endif // BINARY_SEARCH_NO_MAIN



        // if(*itermid[mini] < MIN) {
        //     bounds[mini].first = itermid[mini];
        //     if(bounds[mini].second - bounds[mini].first <= 1) {
        //         if(F(getpos(iters, *bounds[mini].first + 1)) > target)
        //             return *bounds[mini].first;
        //         else pos[mini] = bounds[mini].second - iters[mini].first;
        //         cnt++;
        //     }
        //     else {
        //         itermid[mini] = bounds[mini].first + (bounds[mini].second - bounds[mini].first) / 2;
        //         pos[mini] = itermid[mini] - iters[mini].first;
        //     }
        //     continue;
        // }
        // if(*itermid[maxi] > MAX) {
        //     bounds[maxi].second = itermid[maxi];
        //     if(bounds[maxi].second - bounds[maxi].first <= 1) {
        //         if(F(getpos(iters, *bounds[maxi].first + 1)) > target)
        //             return *bounds[maxi].first;
        //         else pos[maxi] = bounds[maxi].second - iters[maxi].first;
        //         cnt++;
        //     }
        //     else {
        //         itermid[maxi] = bounds[maxi].first + (bounds[maxi].second - bounds[maxi].first) / 2;
        //         pos[maxi] = itermid[maxi] - iters[maxi].first;
        //     }
        //     continue;
        // }
        // cntF1++;
