#include <stack>
#include <cstdio>

// [AJB] BucketPool alloc/free tracking — 追踪slot分配碎片化程度
static thread_local struct {
    long long allocs = 0, reuses = 0, frees = 0;
    int peak_size = 0;
    void dump(const char* tag = "BucketPool") {
        fprintf(stderr, "[AJB_STATE][%s] allocs=%lld reuses=%lld frees=%lld peak=%d frag_rate=%.2f%%\n",
                tag, allocs, reuses, frees, peak_size,
                allocs > 0 ? 100.0 * reuses / allocs : 0.0);
    }
    void reset() { allocs = reuses = frees = 0; peak_size = 0; }
} ajb_bp_stats;

class BucketPool {
    private:
    vector<Bucket> pool;
    stack<int> emptypos;

    public:
    Bucket & operator [](const int idx) {
        return pool[idx];
    }

    const Bucket & operator [](const int idx) const {
        return pool[idx];
    }
    
    int newBucket(vector<int> lowerBound, vector<int> upperBound, int splitDim = 0) {
        ajb_bp_stats.allocs++;
        if(emptypos.empty()){
            pool.emplace_back(lowerBound, upperBound, splitDim);
            int idx = pool.size() - 1;
            if((int)pool.size() > ajb_bp_stats.peak_size) ajb_bp_stats.peak_size = pool.size();
            return idx;
        } else {
            ajb_bp_stats.reuses++;
            int pos = emptypos.top();
            emptypos.pop();
            pool[pos].reset(lowerBound, upperBound, splitDim);
            return pos;
        }
    }

    int newCopy(const Bucket &B) {
        if(emptypos.empty()) {
            pool.emplace_back(B);
            return pool.size() - 1;
        } else {
            int pos = emptypos.top();
            emptypos.pop();
            pool[pos].reset(B.lowerBound, B.upperBound, B.splitDim);
            return pos;
        }
    }

    int poolSize() {
        return pool.size();
    }

    int fragmentSize() {
        return emptypos.size();
    }

    void free(int idx) {
        if(idx < 0 || idx >= pool.size()){
            fprintf(stderr, "[AJB_WARN][BucketPool] free(%d) out of range [0,%zu)\n", idx, pool.size());
            return;
        }
        ajb_bp_stats.frees++;
        emptypos.push(idx);
    }

    // [AJB] 导出当前pool状态给外部调试代码
    void ajb_dump_state() const {
        fprintf(stderr, "[AJB_STATE][BucketPool] size=%zu empty_slots=%zu utilization=%.1f%%\n",
                pool.size(), emptypos.size(),
                pool.size() > 0 ? 100.0 * (pool.size() - emptypos.size()) / pool.size() : 0.0);
    }
};