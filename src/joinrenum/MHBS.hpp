#include <vector>
// [AJB] MHBS: Multi-Head Binary Search — 多relation同时二分, 找splitDim的最优split point
// 这是setAGM的核心子程序, 在每次bucket分裂时被调用
#include <cmath>
#include "AGM.hpp"
using namespace std;

// AJB-algo: safe ceiling — avoids floating-point boundary error
static inline long long ajb_safe_ceil(double res) {
    double c = std::ceil(res);
    return (c - res < 1e-9) ? static_cast<long long>(c) : static_cast<long long>(res);
}

// [AJB] MHBS诊断 + M1012: 搜索深度统计 + 分支偏斜检测
static thread_local struct {
    long long calls = 0;
    long long iterations = 0;     // while循环总轮数
    long long early_returns = 0;  // 单元素range直接返回的次数
    int       max_rels = 0;       // 见过的最大rels.size()
    // [AJB_BP] M929: per-iteration range reduction tracking
    long long total_range_before = 0;
    long long total_range_after = 0;
    long long early_exit_all_converged = 0;
    long long mini_updates = 0;
    long long maxi_updates = 0;

    // M1012: 搜索深度 + 分支偏斜统计
    long long left_branch_visits = 0;   // mid < target 走左(mini更新)的次数
    long long right_branch_visits = 0;  // mid >= target 走右(maxi更新)的次数
    int  max_depth_seen = 0;            // 单次调用最大迭代深度
    // M1012: Welford追踪搜索深度分布
    long long depth_n = 0;
    double depth_mean = 0.0;
    double depth_m2 = 0.0;

    void record_depth(int d) {
        depth_n++;
        double delta = d - depth_mean;
        depth_mean += delta / depth_n;
        double delta2 = d - depth_mean;
        depth_m2 += delta * delta2;
        if(d > max_depth_seen) max_depth_seen = d;
    }
    double depth_stddev() const { return depth_n > 1 ? sqrt(depth_m2 / (depth_n - 1)) : 0.0; }
    double branch_skew() const {
        long long total = left_branch_visits + right_branch_visits;
        if(total == 0) return 0.0;
        double p = (double)left_branch_visits / total;
        // Shannon entropy normalized: H=1 is balanced, H→0 is skewed
        if(p <= 0.0 || p >= 1.0) return 1.0; // max skew
        double H = -(p * log2(p) + (1-p) * log2(1-p));
        return 1.0 - H; // 0=balanced, 1=totally skewed
    }

    void dump(const char* tag = "MHBS") {
        fprintf(stderr, "[AJB_STATE][%s] calls=%lld iters=%lld early_ret=%lld max_rels=%d avg_iters=%.1f\n",
                tag, calls, iterations, early_returns, max_rels,
                calls > 0 ? (double)iterations / calls : 0.0);
        fprintf(stderr, "[AJB_STATE][%s] range_reduction: before=%lld after=%lld ratio=%.4f early_exit_converge=%lld\n",
                tag, total_range_before, total_range_after,
                total_range_before > 0 ? (double)total_range_after / total_range_before : 0.0,
                early_exit_all_converged);
        fprintf(stderr, "[AJB_STATE][%s] bound_updates: mini=%lld maxi=%lld\n",
                tag, mini_updates, maxi_updates);
        // M1012: depth + skew diagnostics
        fprintf(stderr, "[AJB_BP][%s] depth_stats: mean=%.1f σ=%.1f max=%d left=%lld right=%lld skew=%.4f\n",
                tag, depth_mean, depth_stddev(), max_depth_seen,
                left_branch_visits, right_branch_visits, branch_skew());
    }
    void reset() {
        calls = iterations = early_returns = 0; max_rels = 0;
        total_range_before = total_range_after = early_exit_all_converged = 0;
        mini_updates = maxi_updates = 0;
        left_branch_visits = right_branch_visits = 0; max_depth_seen = 0;
        depth_n = 0; depth_mean = depth_m2 = 0.0;
    }
} ajb_mhbs_stats;

void getpos(const vector<pair<vector<int>::iterator, vector<int>::iterator> > &iters, const vector<pair<vector<int>::iterator, vector<int>::iterator> > &bounds, const vector<bool> &flag, const int t, vector<int> &pos) {
    for (int i = 0; i < iters.size(); i++) {
        if(!flag[i]) pos[i] = iters[i].second - iters[i].first;
        else pos[i] = lower_bound(bounds[i].first, bounds[i].second, t) - iters[i].first;
    }
}

int MultiHeadBinarySearch(const vector<pair<vector<int>::iterator, vector<int>::iterator> > &iters, const vector<bool> &flag, vector<int> &rels, const long long target, Query &q) {
    ajb_mhbs_stats.calls++;
    if((int)rels.size() > ajb_mhbs_stats.max_rels) ajb_mhbs_stats.max_rels = rels.size();
    vector<pair<vector<int>::iterator, vector<int>::iterator> > bounds = iters;
    vector<vector<int>::iterator> itermid(iters.size());
    vector<int> pos(iters.size());
    vector<int> tmppos(iters.size());
    int mini, maxi, cnt = 0;
    long long upp;
    double res;
    // [AJB_TRACE] MHBS entry: target是本次搜索的AGM bound目标
    if(ajb_mhbs_stats.calls <= 10)
        fprintf(stderr, "[AJB_TRACE][MHBS] #%lld: target=%lld rels=%zu iters_sizes=[",
                ajb_mhbs_stats.calls, target, rels.size());
    for(int i = 0; i < iters.size(); i++) {
        if(ajb_mhbs_stats.calls <= 10){
            if(i) fprintf(stderr, ",");
            fprintf(stderr, "%ld", iters[i].second - iters[i].first);
        }
        if(!flag[i]) {
            pos[i] = iters[i].second - iters[i].first;
            cnt++;
        }
        else if(iters[i].second - iters[i].first <= 1) {
            ajb_mhbs_stats.early_returns++;
            getpos(iters, bounds, flag, *iters[i].first + 1, tmppos);
            res = q.AGM(tmppos);
            upp = ajb_safe_ceil(res);
            if(upp > target) return *iters[i].first;
            else pos[i] = iters[i].second - iters[i].first;
            cnt++;
        }
        else{
            itermid[i] = iters[i].first + (iters[i].second - iters[i].first) / 2;
            pos[i] = itermid[i] - iters[i].first;
        }
    }
    if(ajb_mhbs_stats.calls <= 10) fprintf(stderr, "]\n");
    // [AJB_BP] M929: measure initial search range
    long long ajb_init_range = 0;
    for (int i = 0; i < (int)iters.size(); i++)
        ajb_init_range += (iters[i].second - iters[i].first);
    ajb_mhbs_stats.total_range_before += ajb_init_range;

    int loop_iters = 0;
    while(cnt < (int)iters.size()) {
        loop_iters++;
        ajb_mhbs_stats.iterations++;
        // --- early exit: if all active bounds have converged to single-element ---
        // upstream: loop continues until cnt == iters.size() via individual convergence
        // changed: check all non-converged bounds; if every remaining one
        //   has range <=1, they'll all converge this iteration — let them,
        //   but if total remaining range is 0 (all bounds already at endpoints)
        //   break immediately to avoid redundant AGM calls
        {
            bool all_converged = true;
            for(size_t i : rels) {
                if(bounds[i].second - bounds[i].first > 1) { all_converged = false; break; }
            }
            if(all_converged) {
                ajb_mhbs_stats.early_exit_all_converged++;
                // [AJB_BP] M929: print early exit trigger reason
                if(ajb_mhbs_stats.early_exit_all_converged <= 10)
                    fprintf(stderr, "[AJB_BP][MHBS] early_exit: all_converged after %d iters, remaining_rels=%zu\n",
                            loop_iters, rels.size());
                break;
            }
        }
        mini = -1, maxi = -1;
        for(size_t i : rels){
            if(bounds[i].second - bounds[i].first <= 1) continue;
            if(mini == -1 || *itermid[i] < *itermid[mini]) mini = i;
            if(maxi == -1 || *itermid[i] > *itermid[maxi]) maxi = i;
        }
        res = q.AGM(pos);
        upp = ajb_safe_ceil(res);
        if(upp <= target) {
            ajb_mhbs_stats.mini_updates++;
            ajb_mhbs_stats.left_branch_visits++;  // M1012: left branch (mini moves up)
            bounds[mini].first = itermid[mini];
            if(bounds[mini].second - bounds[mini].first <= 1) {
                if(*bounds[mini].first == *bounds[mini].second) return *bounds[mini].first;
                getpos(iters, bounds, flag, *bounds[mini].first + 1, tmppos);
                res = q.AGM(tmppos);
                upp = ajb_safe_ceil(res);
                if(upp > target) return *bounds[mini].first;
                else pos[mini] = bounds[mini].second - iters[mini].first;
                cnt++;
            }
            else {
                itermid[mini] = bounds[mini].first + (bounds[mini].second - bounds[mini].first) / 2;
                pos[mini] = itermid[mini] - iters[mini].first;
            }
        }
        else {
            ajb_mhbs_stats.maxi_updates++;
            ajb_mhbs_stats.right_branch_visits++;  // M1012: right branch (maxi moves down)
            bounds[maxi].second = itermid[maxi];
            if(bounds[maxi].second - bounds[maxi].first <= 1) {
                if(*bounds[maxi].first == *bounds[maxi].second) return *bounds[maxi].first;
                getpos(iters, bounds, flag, *bounds[maxi].first + 1, tmppos);
                res = q.AGM(tmppos);
                upp = ajb_safe_ceil(res);
                if(upp > target) return *bounds[maxi].first;
                else pos[maxi] = bounds[maxi].second - iters[maxi].first;
                cnt++;
            }
            else {
                itermid[maxi] = bounds[maxi].first + (bounds[maxi].second - bounds[maxi].first) / 2;
                pos[maxi] = itermid[maxi] - iters[maxi].first;
            }
        }
    }
    int ans = 2147483647;
    for(int i = 0; i < iters.size(); i++) {
        if(bounds[i].second != iters[i].second) ans = min(ans, *bounds[i].second);
    }
    // [AJB_BP] M929: measure final search range reduction
    {
        long long ajb_final_range = 0;
        for (int i = 0; i < (int)iters.size(); i++)
            ajb_final_range += (bounds[i].second - bounds[i].first);
        ajb_mhbs_stats.total_range_after += ajb_final_range;
        // Print reduction rate for first 10 calls or slow convergence
        if (loop_iters > 50 || ajb_mhbs_stats.calls <= 10) {
            double reduction = ajb_init_range > 0
                ? 1.0 - (double)ajb_final_range / ajb_init_range : 1.0;
            fprintf(stderr, "[AJB_BP][MHBS] #%lld range_reduction: %lld→%lld (%.2f%% reduced) iters=%d\n",
                    ajb_mhbs_stats.calls, ajb_init_range, ajb_final_range,
                    reduction * 100.0, loop_iters);
        }
    }
    // [AJB_TRACE] MHBS converged: 轮数多说明值域跨度大
    if(loop_iters > 50 || ajb_mhbs_stats.calls <= 10)
        fprintf(stderr, "[AJB_TRACE][MHBS] converged: iters=%d ans=%d\n", loop_iters, ans);
    // M1012: record depth in Welford tracker; emit skew summary every 500 calls
    ajb_mhbs_stats.record_depth(loop_iters);
    if(ajb_mhbs_stats.calls % 500 == 0) {
        fprintf(stderr, "[AJB_BP][MHBS] depth_summary: n=%lld mean=%.1f σ=%.1f max=%d skew=%.4f (0=balanced,1=skewed)\n",
                ajb_mhbs_stats.depth_n, ajb_mhbs_stats.depth_mean,
                ajb_mhbs_stats.depth_stddev(), ajb_mhbs_stats.max_depth_seen,
                ajb_mhbs_stats.branch_skew());
    }
    return ans;
}
