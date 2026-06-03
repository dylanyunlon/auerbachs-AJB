#include <vector>
// [AJB] MHBS: Multi-Head Binary Search — 多relation同时二分, 找splitDim的最优split point
// 这是setAGM的核心子程序, 在每次bucket分裂时被调用
#include <cmath>
#include "AGM.hpp"
using namespace std;

// [AJB] MHBS诊断
static thread_local struct {
    long long calls = 0;
    long long iterations = 0;     // while循环总轮数
    long long early_returns = 0;  // 单元素range直接返回的次数
    int       max_rels = 0;       // 见过的最大rels.size()
    void dump(const char* tag = "MHBS") {
        fprintf(stderr, "[AJB_STATE][%s] calls=%lld iters=%lld early_ret=%lld max_rels=%d avg_iters=%.1f\n",
                tag, calls, iterations, early_returns, max_rels,
                calls > 0 ? (double)iterations / calls : 0.0);
    }
    void reset() { calls = iterations = early_returns = 0; max_rels = 0; }
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
            upp = ceil(res) - res < 1e-5 ? ceil(res) : (long long)(res);
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
            if(all_converged) break;
        }
        mini = -1, maxi = -1;
        for(size_t i : rels){
            if(bounds[i].second - bounds[i].first <= 1) continue;
            if(mini == -1 || *itermid[i] < *itermid[mini]) mini = i;
            if(maxi == -1 || *itermid[i] > *itermid[maxi]) maxi = i;
        }
        res = q.AGM(pos);
        upp = ceil(res) - res < 1e-5 ? ceil(res) : (long long)(res);
        if(upp <= target) {
            bounds[mini].first = itermid[mini];
            if(bounds[mini].second - bounds[mini].first <= 1) {
                if(*bounds[mini].first == *bounds[mini].second) return *bounds[mini].first;
                getpos(iters, bounds, flag, *bounds[mini].first + 1, tmppos);
                res = q.AGM(tmppos);
                upp = ceil(res) - res < 1e-5 ? ceil(res) : (long long)(res);
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
            bounds[maxi].second = itermid[maxi];
            if(bounds[maxi].second - bounds[maxi].first <= 1) {
                if(*bounds[maxi].first == *bounds[maxi].second) return *bounds[maxi].first;
                getpos(iters, bounds, flag, *bounds[maxi].first + 1, tmppos);
                res = q.AGM(tmppos);
                upp = ceil(res) - res < 1e-5 ? ceil(res) : (long long)(res);
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
    // [AJB_TRACE] MHBS converged: 轮数多说明值域跨度大
    if(loop_iters > 50 || ajb_mhbs_stats.calls <= 10)
        fprintf(stderr, "[AJB_TRACE][MHBS] converged: iters=%d ans=%d\n", loop_iters, ans);
    return ans;
}
