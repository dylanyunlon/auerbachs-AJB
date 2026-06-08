#include <stack>
#include <queue>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <functional>

// [AJB] M912: BucketPool alloc/free tracking + priority_queue pick strategy
// 追踪slot分配碎片化程度, ban原因诊断, top-k候选维护
static thread_local struct {
    long long allocs = 0, reuses = 0, frees = 0;
    long long ban_count = 0;
    long long pick_count = 0;
    long long pick_linear_scans = 0;  // 线性扫描pick的次数
    long long pick_pq_hits = 0;       // priority_queue pick命中次数
    int peak_size = 0;
    int peak_active = 0;  // peak(pool_size - empty_slots)
    void dump(const char* tag = "BucketPool") {
#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_STATE][%s] allocs=%lld reuses=%lld frees=%lld peak=%d peak_active=%d frag_rate=%.2f%%\n",
                tag, allocs, reuses, frees, peak_size, peak_active,
                allocs > 0 ? 100.0 * reuses / allocs : 0.0);
        fprintf(stderr, "[AJB_STATE][%s] picks=%lld(linear=%lld pq=%lld) bans=%lld\n",
                tag, pick_count, pick_linear_scans, pick_pq_hits, ban_count);
#endif
    }
    void reset() { allocs = reuses = frees = ban_count = pick_count = 0;
                    pick_linear_scans = pick_pq_hits = 0;
                    peak_size = peak_active = 0; }
} ajb_bp_stats;

class BucketPool {
    private:
    vector<Bucket> pool;
    stack<int> emptypos;

    // [AJB] M912: banned bucket tracking — 记录每个slot的ban状态和原因
    // ban原因编码: 0=not banned, 1=AGM zero, 2=bound conflict, 3=manual
    vector<uint8_t> ban_status;

    // [AJB] M912: pool级别统计 — 在partial_sort pick中顺带维护
    int max_observed_depth = 0;  // 最大bucket splitDim(近似depth)

    void updatePeakActive() {
        int active = static_cast<int>(pool.size()) - static_cast<int>(emptypos.size());
        if(active > ajb_bp_stats.peak_active) ajb_bp_stats.peak_active = active;
    }

    public:
    Bucket & operator [](const int idx) {
        return pool[idx];
    }

    const Bucket & operator [](const int idx) const {
        return pool[idx];
    }
    
    int newBucket(vector<int> lowerBound, vector<int> upperBound, int splitDim = 0) {
        ajb_bp_stats.allocs++;
        int idx;
        if(emptypos.empty()){
            pool.emplace_back(lowerBound, upperBound, splitDim);
            idx = pool.size() - 1;
            ban_status.push_back(0);
            if((int)pool.size() > ajb_bp_stats.peak_size) ajb_bp_stats.peak_size = pool.size();
        } else {
            ajb_bp_stats.reuses++;
            int pos = emptypos.top();
            emptypos.pop();
            pool[pos].reset(lowerBound, upperBound, splitDim);
            ban_status[pos] = 0;
            idx = pos;
        }
        updatePeakActive();
        // [AJB] M912: 追踪最大观察到的splitDim(近似depth)
        if(pool[idx].splitDim > max_observed_depth) max_observed_depth = pool[idx].splitDim;
#ifdef AJB_DEBUG
        if(ajb_bp_stats.allocs <= 20 || ajb_bp_stats.allocs % 5000 == 0) {
            long long vol = pool[idx].ajb_volume();
            fprintf(stderr, "[AJB_DEBUG][BucketPool] addBucket[%lld]: idx=%d splitDim=%d pool_size=%zu empty=%zu vol=%lld\n",
                    ajb_bp_stats.allocs, idx, pool[idx].splitDim, pool.size(), emptypos.size(), vol);
        }
#endif
        return idx;
    }

    int newCopy(const Bucket &B) {
        int pos;
        if(emptypos.empty()) {
            pool.emplace_back(B);
            pos = pool.size() - 1;
            ban_status.push_back(0);
        } else {
            pos = emptypos.top();
            emptypos.pop();
            pool[pos].reset(B.lowerBound, B.upperBound, B.splitDim);
            ban_status[pos] = 0;
        }
        updatePeakActive();
        return pos;
    }

    int poolSize() {
        return pool.size();
    }

    int fragmentSize() {
        return emptypos.size();
    }

    void free(int idx) {
        if(idx < 0 || idx >= (int)pool.size()){
            fprintf(stderr, "[AJB_WARN][BucketPool] free(%d) out of range [0,%zu)\n", idx, pool.size());
            return;
        }
        ajb_bp_stats.frees++;
        emptypos.push(idx);
    }

    // [AJB] M912: ban a bucket with reason code
    // reason: 1=AGM zero, 2=bound conflict on specific dim, 3=manual
    void banBucket(int idx, uint8_t reason = 3) {
        if(idx < 0 || idx >= (int)pool.size()) return;
        ajb_bp_stats.ban_count++;
        ban_status[idx] = reason;
#ifdef AJB_DEBUG
        if(ajb_bp_stats.ban_count <= 10 || ajb_bp_stats.ban_count % 2000 == 0) {
            // 检查ban原因: 哪个维度的bound冲突
            const char* reason_str = "unknown";
            switch(reason) {
                case 1: reason_str = "AGM_zero"; break;
                case 2: reason_str = "bound_conflict"; break;
                case 3: reason_str = "manual"; break;
            }
            fprintf(stderr, "[AJB_DEBUG][BucketPool] ban[%lld]: bid=%d reason=%s AGM=%lld splitDim=%d\n",
                    ajb_bp_stats.ban_count, idx, reason_str, pool[idx].AGM, pool[idx].splitDim);
            // 打印bound冲突详情
            if(reason == 2) {
                for(size_t d = 0; d < pool[idx].lowerBound.size(); d++) {
                    if(pool[idx].lowerBound[d] > pool[idx].upperBound[d]) {
                        fprintf(stderr, "[AJB_DEBUG][BucketPool]   conflict@dim%zu: lb=%d > ub=%d\n",
                                d, pool[idx].lowerBound[d], pool[idx].upperBound[d]);
                    }
                }
            }
        }
#endif
    }

    // [AJB] M912: pickBucket — 从pool中选择AGM最大的active bucket
    // 策略: 收集所有active候选, 用std::partial_sort维护top-k, 返回最大AGM的bid
    // 相比priority_queue: 减少stale entry开销, 利用cache locality
    int pickBucket() {
        ajb_bp_stats.pick_count++;
        
        // 收集active候选
        struct Candidate { long long agm; int bid; };
        std::vector<Candidate> candidates;
        candidates.reserve(pool.size() - emptypos.size());
        for(size_t i = 0; i < pool.size(); i++) {
            if(i < ban_status.size() && ban_status[i] != 0) continue;
            if(pool[i].AGM > 0) {
                candidates.push_back({pool[i].AGM, static_cast<int>(i)});
            }
        }
        
        if(candidates.empty()) {
#ifdef AJB_DEBUG
            fprintf(stderr, "[AJB_DEBUG][BucketPool] pickBucket: no active candidates in pool of %zu\n", pool.size());
#endif
            return -1;
        }
        
        // partial_sort取top-k(k=min(8, candidates.size()))用于候选分析
        size_t k = std::min(candidates.size(), (size_t)8);
        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                          [](const Candidate& a, const Candidate& b) { return a.agm > b.agm; });
        
        int best = candidates[0].bid;
        
#ifdef AJB_DEBUG
        if(ajb_bp_stats.pick_count <= 5 || ajb_bp_stats.pick_count % 1000 == 0) {
            fprintf(stderr, "[AJB_DEBUG][BucketPool] pickBucket(partial_sort): best_bid=%d AGM=%lld top%zu=[",
                    best, candidates[0].agm, k);
            for(size_t i = 0; i < k && i < 4; i++) {
                if(i) fprintf(stderr, ",");
                fprintf(stderr, "b%d:%lld", candidates[i].bid, candidates[i].agm);
            }
            fprintf(stderr, "] total_candidates=%zu\n", candidates.size());
        }
#endif
        return best;
    }

    // [AJB] M1231: unconditional one-line diagnostic — always prints,
    // so experiment runs give real-time feedback without recompiling
    void ajb_summary_line(const char* phase = "done") const {
        int active_count = 0;
        long long max_agm = 0;
        for(size_t i = 0; i < pool.size(); i++) {
            if(i < ban_status.size() && ban_status[i] != 0) continue;
            active_count++;
            if(pool[i].AGM > max_agm) max_agm = pool[i].AGM;
        }
        fprintf(stderr, "[AJB_BP][%s] pool=%zu active=%d maxAGM=%lld allocs=%lld reuse=%.0f%% bans=%lld\n",
                phase, pool.size(), active_count, max_agm,
                ajb_bp_stats.allocs,
                ajb_bp_stats.allocs > 0 ? 100.0 * ajb_bp_stats.reuses / ajb_bp_stats.allocs : 0.0,
                ajb_bp_stats.ban_count);
    }

    // [AJB] M912: 导出当前pool状态给外部调试代码
    // 增加: 总bucket数、平均volume、最大depth
    void ajb_dump_state() const {
#ifdef AJB_DEBUG
        int active_count = 0, banned_count = 0;
        long long total_agm = 0;
        double total_log_vol = 0.0;
        int local_max_depth = 0;
        for(size_t i = 0; i < pool.size(); i++) {
            if(i < ban_status.size() && ban_status[i] != 0) { banned_count++; continue; }
            active_count++;
            total_agm += pool[i].AGM;
            // volume via log-sum (cheap approximation)
            double lv = 0.0;
            for(size_t d = 0; d < pool[i].lowerBound.size(); d++) {
                int rng = pool[i].upperBound[d] - pool[i].lowerBound[d] + 1;
                if(rng > 0) lv += std::log2((double)rng);
            }
            total_log_vol += lv;
            if(pool[i].splitDim > local_max_depth) local_max_depth = pool[i].splitDim;
        }
        double avg_log_vol = active_count > 0 ? total_log_vol / active_count : 0.0;
        fprintf(stderr, "[AJB_STATE][BucketPool] size=%zu empty=%zu active=%d banned=%d total_agm=%lld utilization=%.1f%%\n",
                pool.size(), emptypos.size(), active_count, banned_count, total_agm,
                pool.size() > 0 ? 100.0 * (pool.size() - emptypos.size()) / pool.size() : 0.0);
        fprintf(stderr, "[AJB_STATE][BucketPool] avg_log2_vol=%.2f max_depth=%d pool_max_depth=%d\n",
                avg_log_vol, local_max_depth, max_observed_depth);
#endif
    }
};
