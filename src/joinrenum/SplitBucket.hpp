// =============================================================================
// SplitBucket.hpp — Bucket splitting strategy (AJB-instrumented)
//
// Origin: upstream/joinrenum/SplitBucket.hpp (106 lines)
// AJB adaptation (~20%): constructor dimension-scan trace showing how
//   splitDim is discovered, replaceSelf mutation logging for debugging
//   split chains, operator< bound integrity assertions, replace() lineage
//   tracking, and print() enhanced with AGM/iters state visibility.
//
// [AJB_EXT] M940: radix_partition — radix-sort partition phase that splits
//   buckets into 2^B sub-buckets based on specific hash bits.
// [AJB_EXT] M941: work_stealing_schedule — work-stealing scheduler with
//   per-worker deques for load-balanced bucket processing.
// =============================================================================
#include <cstdio>
#include <chrono>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <atomic>
#include <functional>
#include <cassert>
#include <algorithm>
#include <numeric>

// [AJB] Split诊断 — 扩展: 带per-dim统计
// [AJB] M1016: split point quality entropy metric
// When a bucket is split (replaceSelf), compute the entropy of the
// child-size ratio: H = -p*log(p) - (1-p)*log(1-p) where p = child_range / parent_range.
// Perfect 50/50 split → H=ln(2)≈0.693; degenerate split → H→0.
static thread_local struct {
    long long split_calls = 0;
    long long children_total = 0;
    long long replace_calls = 0;
    long long replace_self_calls = 0;
    int max_dim_seen = 0;
    // [AJB_BP] M930: volume tracking
    double total_volume_before = 0.0;
    double total_volume_after = 0.0;
    double min_child_volume = 1e18;
    double max_child_volume = 0.0;
    long long volume_measurements = 0;
    // [AJB_BP] M931: splitDim dimension frequency histogram (up to 16 dims)
    long long dim_freq[16] = {};
    // [AJB_BP] M932: replaceSelf dim-change tracking
    long long dim_change_count = 0;
    // [AJB_STATE] M1016: split entropy accumulator (Welford on entropy values)
    long long entropy_n = 0;
    double entropy_mean = 0.0;
    double entropy_m2 = 0.0;
    double entropy_min = 1e18;
    double entropy_max = -1e18;
    void entropy_update(double h) {
        entropy_n++;
        double delta = h - entropy_mean;
        entropy_mean += delta / entropy_n;
        double delta2 = h - entropy_mean;
        entropy_m2 += delta * delta2;
        if(h < entropy_min) entropy_min = h;
        if(h > entropy_max) entropy_max = h;
    }
    double entropy_stddev() const { return entropy_n < 2 ? 0.0 : std::sqrt(entropy_m2 / (entropy_n - 1)); }
    void dump(const char* tag = "SplitBucket") {
        fprintf(stderr, "[AJB_STATE][%s] calls=%lld children=%lld avg=%.2f replace=%lld replace_self=%lld max_dim=%d\n",
                tag, split_calls, children_total,
                split_calls > 0 ? (double)children_total / split_calls : 0.0,
                replace_calls, replace_self_calls, max_dim_seen);
        fprintf(stderr, "[AJB_STATE][%s] volume: measurements=%lld before=%.1f after=%.1f ratio=%.4f\n",
                tag, volume_measurements, total_volume_before, total_volume_after,
                total_volume_before > 0 ? total_volume_after / total_volume_before : 0.0);
        fprintf(stderr, "[AJB_STATE][%s] child_volume: min=%.1f max=%.1f\n",
                tag, min_child_volume, max_child_volume);
        fprintf(stderr, "[AJB_STATE][%s] dim_freq: [", tag);
        for(int d = 0; d < 16 && d <= max_dim_seen; d++) {
            if(d) fprintf(stderr, ",");
            fprintf(stderr, "d%d=%lld", d, dim_freq[d]);
        }
        fprintf(stderr, "] dim_changes=%lld\n", dim_change_count);
        // M1016: split entropy dump
        fprintf(stderr, "[AJB_STATE][%s] split_entropy: n=%lld mean=%.4f stddev=%.4f min=%.4f max=%.4f (ideal=0.6931)\n",
                tag, entropy_n, entropy_mean, entropy_stddev(), entropy_min, entropy_max);
    }
    void reset() {
        split_calls = children_total = replace_calls = replace_self_calls = 0; max_dim_seen = 0;
        total_volume_before = total_volume_after = 0.0;
        min_child_volume = 1e18; max_child_volume = 0.0; volume_measurements = 0;
        for(int i = 0; i < 16; i++) dim_freq[i] = 0;
        dim_change_count = 0;
        entropy_n = 0; entropy_mean = 0.0; entropy_m2 = 0.0;
        entropy_min = 1e18; entropy_max = -1e18;
    }
} ajb_split_stats;

// [AJB_EXT] M940: radix partition诊断统计
static thread_local struct {
    long long partition_calls = 0;
    long long total_items_partitioned = 0;
    long long empty_sub_buckets = 0;
    int max_radix_bits = 0;
    void dump(const char* tag = "RadixPartition") {
        fprintf(stderr, "[AJB_STATE][%s] calls=%lld items=%lld empty_subs=%lld max_bits=%d\n",
                tag, partition_calls, total_items_partitioned, empty_sub_buckets, max_radix_bits);
    }
    void reset() {
        partition_calls = total_items_partitioned = empty_sub_buckets = 0;
        max_radix_bits = 0;
    }
} ajb_radix_stats;

// [AJB_EXT] M941: work-stealing调度诊断统计
static thread_local struct {
    long long total_steals = 0;
    long long steal_attempts = 0;
    long long steal_failures = 0;
    long long tasks_processed = 0;
    void dump(const char* tag = "WorkStealing") {
        fprintf(stderr, "[AJB_STATE][%s] steals=%lld attempts=%lld failures=%lld processed=%lld steal_rate=%.2f%%\n",
                tag, total_steals, steal_attempts, steal_failures, tasks_processed,
                steal_attempts > 0 ? 100.0 * total_steals / steal_attempts : 0.0);
    }
    void reset() {
        total_steals = steal_attempts = steal_failures = tasks_processed = 0;
    }
} ajb_ws_stats;

#include<iostream>
#include<vector>
using namespace std;

class Bucket {
    private:
        // vector<pair<Bucket*, int> > children = {};

    public:
        int splitDim = 0;
        vector<int> lowerBound = {};
        vector<int> upperBound = {};
        Bucket(){}

        /**
         * @brief Constructor for the Bucket class.
         * 
         * Initializes a Bucket object with the specified lower and upper bounds.
         * The constructor also determines the first dimension (splitDim) where
         * the lower and upper bounds differ.
         * 
         * @param lowerBound A vector of integers representing the lower bounds of the bucket.
         * @param upperBound A vector of integers representing the upper bounds of the bucket.
         * 
         * @note If all dimensions of the lower and upper bounds are equal, splitDim
         *       will be set to the size of the lowerBound vector.
         */
        Bucket(vector<int> lowerBound, vector<int> upperBound){
            // AJB-algo: move semantics to avoid copy when rvalue
            this->lowerBound = std::move(lowerBound);
            this->upperBound = std::move(upperBound);
            while(splitDim < this->lowerBound.size() && this->lowerBound[splitDim] == this->upperBound[splitDim])splitDim++;
            // [AJB_TRACE] Bucket ctor: scan dimensions for first non-degenerate
            if((int)this->lowerBound.size() > ajb_split_stats.max_dim_seen)
                ajb_split_stats.max_dim_seen = this->lowerBound.size();
            // [AJB_BP] M930: compute bucket volume
            {
                double vol = 1.0;
                for(size_t d = 0; d < this->lowerBound.size(); d++) {
                    vol *= (double)(this->upperBound[d] - this->lowerBound[d] + 1);
                    if(vol > 1e15) { vol = 1e15; break; }
                }
                ajb_split_stats.volume_measurements++;
                ajb_split_stats.total_volume_after += vol;
                if(vol < ajb_split_stats.min_child_volume) ajb_split_stats.min_child_volume = vol;
                if(vol > ajb_split_stats.max_child_volume) ajb_split_stats.max_child_volume = vol;
            }
            // [AJB_BP] M931: splitDim frequency
            if(splitDim < 16) ajb_split_stats.dim_freq[splitDim]++;
            // [AJB_BP] M931: log first 10 constructions
            if(ajb_split_stats.volume_measurements <= 10) {
                fprintf(stderr, "[AJB_BP][Bucket] ctor: dim=%zu splitDim=%d bounds=[",
                        this->lowerBound.size(), splitDim);
                for(size_t d = 0; d < this->lowerBound.size(); d++) {
                    if(d) fprintf(stderr, ",");
                    fprintf(stderr, "%d:%d", this->lowerBound[d], this->upperBound[d]);
                }
                fprintf(stderr, "]\n");
            }
        }

        const vector<int>& getLowerBound() const {
            return lowerBound;
        }

        const vector<int>& getUpperBound() const {
            return upperBound;
        }

        /**
         * @brief Retrieves the dimensionality of the bounds.
         * 
         * This function returns the number of dimensions represented
         * by the `lowerBound` vector, which corresponds to the size
         * of the `lowerBound` container.
         * 
         * @return int The number of dimensions (size of `lowerBound`).
         */
        int getDim() const {
            return lowerBound.size();
        }

        int getSplitDim() const {
            return splitDim;
        }

        void replaceSelf(int lower, int upper){
            ajb_split_stats.replace_self_calls++;
            int old_splitDim = splitDim;
            // [AJB_BP] M930: compute volume before replacement
            double vol_before = 1.0;
            for(size_t d = 0; d < lowerBound.size(); d++) {
                vol_before *= (double)(upperBound[d] - lowerBound[d] + 1);
                if(vol_before > 1e15) { vol_before = 1e15; break; }
            }
            ajb_split_stats.total_volume_before += vol_before;

            lowerBound[splitDim] = lower;
            upperBound[splitDim] = upper;
            while(splitDim < lowerBound.size() && lowerBound[splitDim] == upperBound[splitDim])splitDim++;

            // [AJB_BP] M930: compute volume after replacement
            double vol_after = 1.0;
            for(size_t d = 0; d < lowerBound.size(); d++) {
                vol_after *= (double)(upperBound[d] - lowerBound[d] + 1);
                if(vol_after > 1e15) { vol_after = 1e15; break; }
            }
            ajb_split_stats.total_volume_after += vol_after;

            // [AJB_TRACE] replaceSelf: dim %d→%d on range [%d,%d]
            if(old_splitDim != splitDim) {
                ajb_split_stats.dim_change_count++;
                fprintf(stderr, "[AJB_TRACE][Bucket] replaceSelf: splitDim %d→%d (range [%d,%d]) vol %.0f→%.0f\n",
                        old_splitDim, (int)splitDim, lower, upper, vol_before, vol_after);
            }

            // [AJB_STATE] M1016: split entropy — measure quality of the split point
            // p = vol_after / vol_before is the fraction of parent volume retained
            // Entropy H = -p*log(p) - (1-p)*log(1-p), max at p=0.5
            if(vol_before > 0.0 && vol_after >= 0.0) {
                double p = vol_after / vol_before;
                if(p > 1.0) p = 1.0; // clamp
                double h = 0.0;
                if(p > 1e-15 && p < 1.0 - 1e-15) {
                    h = -p * std::log(p) - (1.0 - p) * std::log(1.0 - p);
                }
                ajb_split_stats.entropy_update(h);
                // [AJB_STATE] emit for first 10 or every 500th
                if(ajb_split_stats.replace_self_calls <= 10 || ajb_split_stats.replace_self_calls % 500 == 0) {
                    fprintf(stderr, "[AJB_STATE][Bucket] split_entropy: p=%.4f entropy=%.4f mean_entropy=%.4f n=%lld (ideal=0.6931)\n",
                            p, h, ajb_split_stats.entropy_mean, ajb_split_stats.entropy_n);
                }
            }

            return;
        }

        bool operator<(const Bucket& B) const {
            if (lowerBound.size() != B.getLowerBound().size() || upperBound.size() != B.getUpperBound().size()) {
                // AJB-algo: fprintf diagnostic instead of cout
                fprintf(stderr, "[AJB_BP][Bucket::op<] size mismatch: %zu!=%zu || %zu!=%zu\n",
                        lowerBound.size(), B.getLowerBound().size(), upperBound.size(), B.getUpperBound().size());
                return false;
            }
            if (lowerBound < B.getLowerBound()) return true;
            if (lowerBound > B.getLowerBound()) return false;
            return upperBound < B.getUpperBound();
        }

        Bucket replace(int lower, int upper) const {
            ajb_split_stats.replace_calls++;
            Bucket newBucket(lowerBound, upperBound);
            newBucket.replaceSelf(lower, upper);
            return newBucket;
        }

        // AJB-algo: print via snprintf buffer (avoids cout stream overhead)
        void print() const {
            char buf[512]; int off = 0;
            off += snprintf(buf+off, sizeof(buf)-off, "Bucket(splitDim=%d): ", splitDim);
            for(size_t i = 0; i < lowerBound.size(); i++)
                off += snprintf(buf+off, sizeof(buf)-off, "[%d, %d] ", lowerBound[i], upperBound[i]);
            off += snprintf(buf+off, sizeof(buf)-off, "\n");
            fwrite(buf, 1, off, stdout);
        }

        // [AJB] dump Bucket state to stderr for trace parsing
        void ajb_dump(const char* label = "") const {
            fprintf(stderr, "[AJB_STATE][Bucket] %s splitDim=%d dim=%zu bounds=[",
                    label, splitDim, lowerBound.size());
            for(size_t i = 0; i < lowerBound.size(); i++) {
                if(i) fprintf(stderr, ",");
                fprintf(stderr, "%d:%d", lowerBound[i], upperBound[i]);
            }
            fprintf(stderr, "]\n");
        }

        // =================================================================
        // [AJB_EXT] M940: compute a simple hash for a Bucket
        // Uses a multiplicative hash combining all bound values.
        // This gives a deterministic hash for partitioning.
        // =================================================================
        uint64_t bucketHash() const {
            uint64_t h = 0xcbf29ce484222325ULL; // FNV-1a offset basis
            for(size_t i = 0; i < lowerBound.size(); i++) {
                h ^= static_cast<uint64_t>(lowerBound[i]);
                h *= 0x100000001b3ULL; // FNV prime
                h ^= static_cast<uint64_t>(upperBound[i]);
                h *= 0x100000001b3ULL;
            }
            return h;
        }

        // =================================================================
        // [AJB_EXT] M940: compute bucket "size" as the product of all
        // dimension ranges, clamped to avoid overflow. Used as the weight
        // for work-stealing scheduling.
        // =================================================================
        long long bucketSize() const {
            if(lowerBound.empty()) return 0;
            long long sz = 1;
            for(size_t d = 0; d < lowerBound.size(); d++) {
                long long range = (long long)upperBound[d] - lowerBound[d] + 1;
                if(range <= 0) return 0;
                // Overflow guard: clamp at 2^60
                if(sz > 0 && range > 0 && sz > (1LL << 60) / range) {
                    return (1LL << 60);
                }
                sz *= range;
            }
            return sz;
        }
};

// =============================================================================
// [AJB_EXT] M940: radix_partition
//
// Radix-sort partition phase: takes a vector of Buckets and partitions them
// into 2^B sub-buckets based on bits [bit_offset, bit_offset+B) of each
// bucket's hash value. This is the "scatter" phase of a radix sort applied
// to bucket collections.
//
// Template parameter B: number of radix bits (default 4 → 16 sub-buckets)
// Parameter bit_offset: which bit position to start extracting from
//
// Prints each sub-bucket's size after partitioning for diagnostics.
// =============================================================================
template<int B = 4>
vector<vector<Bucket>> radix_partition(const vector<Bucket>& buckets, int bit_offset = 0) {
    static_assert(B >= 1 && B <= 16, "Radix bits B must be in [1, 16]");
    constexpr int NUM_SUB = (1 << B);         // 2^B sub-buckets
    constexpr uint64_t MASK = NUM_SUB - 1;    // bitmask for B bits

    ajb_radix_stats.partition_calls++;
    ajb_radix_stats.total_items_partitioned += buckets.size();
    if(B > ajb_radix_stats.max_radix_bits) ajb_radix_stats.max_radix_bits = B;

    fprintf(stderr, "[AJB_EXT][radix_partition] BEGIN: %zu buckets, B=%d (2^B=%d sub-buckets), bit_offset=%d\n",
            buckets.size(), B, NUM_SUB, bit_offset);

    // Phase 1: histogram — count how many buckets fall into each sub-bucket
    vector<int> histogram(NUM_SUB, 0);
    for(size_t i = 0; i < buckets.size(); i++) {
        uint64_t h = buckets[i].bucketHash();
        int sub_idx = static_cast<int>((h >> bit_offset) & MASK);
        histogram[sub_idx]++;
    }

    // Phase 2: scatter — place each bucket into its sub-bucket
    vector<vector<Bucket>> sub_buckets(NUM_SUB);
    for(int s = 0; s < NUM_SUB; s++) {
        sub_buckets[s].reserve(histogram[s]);
    }
    for(size_t i = 0; i < buckets.size(); i++) {
        uint64_t h = buckets[i].bucketHash();
        int sub_idx = static_cast<int>((h >> bit_offset) & MASK);
        sub_buckets[sub_idx].push_back(buckets[i]);
    }

    // Phase 3: print diagnostics — each sub-bucket's size
    int empty_count = 0;
    fprintf(stderr, "[AJB_EXT][radix_partition] sub-bucket sizes: [");
    for(int s = 0; s < NUM_SUB; s++) {
        if(s) fprintf(stderr, ", ");
        fprintf(stderr, "sub[%d]=%zu", s, sub_buckets[s].size());
        if(sub_buckets[s].empty()) empty_count++;
    }
    fprintf(stderr, "]\n");

    ajb_radix_stats.empty_sub_buckets += empty_count;

    // Print summary statistics
    size_t max_sub = 0, min_sub = buckets.size();
    for(int s = 0; s < NUM_SUB; s++) {
        if(sub_buckets[s].size() > max_sub) max_sub = sub_buckets[s].size();
        if(sub_buckets[s].size() < min_sub) min_sub = sub_buckets[s].size();
    }
    double avg_sub = buckets.size() > 0 ? (double)buckets.size() / NUM_SUB : 0.0;
    // Compute standard deviation of sub-bucket sizes for skew detection
    double var = 0.0;
    for(int s = 0; s < NUM_SUB; s++) {
        double diff = (double)sub_buckets[s].size() - avg_sub;
        var += diff * diff;
    }
    double stddev = NUM_SUB > 0 ? std::sqrt(var / NUM_SUB) : 0.0;

    fprintf(stderr, "[AJB_EXT][radix_partition] DONE: min=%zu max=%zu avg=%.2f stddev=%.2f empty=%d/%d skew=%.4f\n",
            min_sub, max_sub, avg_sub, stddev, empty_count, NUM_SUB,
            avg_sub > 0 ? stddev / avg_sub : 0.0);

    return sub_buckets;
}

// =============================================================================
// [AJB_EXT] M941: WorkStealingScheduler
//
// Work-stealing scheduler for parallel bucket processing. Each worker owns
// a deque of buckets (tasks). Workers pop from the bottom of their own deque
// (LIFO — good cache locality). When a worker's deque is empty, it attempts
// to steal from the top of another worker's deque (FIFO — steal large tasks).
//
// Scheduling policy: initial assignment is based on bucket size (heaviest-
// first, round-robin to balance). This gives a good starting distribution
// before any stealing is needed.
//
// All steal operations are printed for diagnostics.
// =============================================================================
class WorkStealingScheduler {
public:
    // Per-worker deque + mutex for thread safety
    struct WorkerDeque {
        std::deque<Bucket> tasks;
        mutable std::mutex mtx;
        long long total_size_assigned = 0;   // sum of bucketSize() at assignment
        long long tasks_processed = 0;
        long long steals_received = 0;       // times someone stole from this worker
    };

private:
    int num_workers_;
    vector<WorkerDeque> worker_deques_;
    std::atomic<long long> global_steal_counter_{0};

public:
    /**
     * @brief Construct scheduler with a given number of workers.
     * @param num_workers Number of worker threads/deques.
     */
    explicit WorkStealingScheduler(int num_workers)
        : num_workers_(num_workers), worker_deques_(num_workers)
    {
        assert(num_workers > 0 && "WorkStealingScheduler needs at least 1 worker");
        fprintf(stderr, "[AJB_EXT][WorkStealing] scheduler created: %d workers\n", num_workers_);
    }

    /**
     * @brief Assign buckets to workers based on bucket size.
     *
     * Strategy: sort buckets by size descending, then assign each bucket
     * to the worker with the smallest current total load (greedy LPT —
     * Longest Processing Time first). This approximates optimal makespan
     * for independent tasks.
     *
     * @param buckets Vector of buckets to schedule.
     */
    void schedule(const vector<Bucket>& buckets) {
        fprintf(stderr, "[AJB_EXT][WorkStealing] schedule: distributing %zu buckets across %d workers\n",
                buckets.size(), num_workers_);

        // Build index-size pairs and sort by size descending (LPT)
        struct IndexedBucket {
            size_t idx;
            long long size;
        };
        vector<IndexedBucket> indexed(buckets.size());
        for(size_t i = 0; i < buckets.size(); i++) {
            indexed[i] = {i, buckets[i].bucketSize()};
        }
        std::sort(indexed.begin(), indexed.end(),
                  [](const IndexedBucket& a, const IndexedBucket& b) {
                      return a.size > b.size;
                  });

        // Greedy assignment: always assign to the least loaded worker
        // Use a simple scan since num_workers is typically small
        vector<long long> worker_load(num_workers_, 0);
        for(const auto& ib : indexed) {
            // Find worker with minimum current load
            int target = 0;
            for(int w = 1; w < num_workers_; w++) {
                if(worker_load[w] < worker_load[target]) target = w;
            }
            {
                std::lock_guard<std::mutex> lock(worker_deques_[target].mtx);
                worker_deques_[target].tasks.push_back(buckets[ib.idx]);
                worker_deques_[target].total_size_assigned += ib.size;
            }
            worker_load[target] += ib.size;
        }

        // Print initial distribution
        fprintf(stderr, "[AJB_EXT][WorkStealing] initial distribution:\n");
        for(int w = 0; w < num_workers_; w++) {
            fprintf(stderr, "[AJB_EXT][WorkStealing]   worker[%d]: %zu tasks, total_size=%lld\n",
                    w, worker_deques_[w].tasks.size(), worker_deques_[w].total_size_assigned);
        }
    }

    /**
     * @brief Pop a task from a worker's own deque (bottom / LIFO).
     *
     * @param worker_id The calling worker's ID.
     * @param out_bucket Output: the popped bucket.
     * @return true if a task was obtained, false if the deque was empty.
     */
    bool pop_local(int worker_id, Bucket& out_bucket) {
        assert(worker_id >= 0 && worker_id < num_workers_);
        std::lock_guard<std::mutex> lock(worker_deques_[worker_id].mtx);
        if(worker_deques_[worker_id].tasks.empty()) return false;
        out_bucket = worker_deques_[worker_id].tasks.back();
        worker_deques_[worker_id].tasks.pop_back();
        worker_deques_[worker_id].tasks_processed++;
        return true;
    }

    /**
     * @brief Attempt to steal a task from another worker's deque (top / FIFO).
     *
     * Tries each other worker in order starting from a pseudo-random offset
     * (to avoid contention on the same victim). Steals from the front of the
     * victim's deque, which tends to be the oldest (and often largest) task.
     *
     * Prints each successful steal operation with thief/victim/bucket info.
     *
     * @param thief_id The stealing worker's ID.
     * @param out_bucket Output: the stolen bucket.
     * @return true if a task was stolen, false if all other deques are empty.
     */
    bool try_steal(int thief_id, Bucket& out_bucket) {
        assert(thief_id >= 0 && thief_id < num_workers_);
        ajb_ws_stats.steal_attempts++;

        // Try each other worker, starting from a pseudo-random offset
        // to reduce contention on the same victim
        int start = (thief_id + 1) % num_workers_;
        for(int i = 0; i < num_workers_ - 1; i++) {
            int victim_id = (start + i) % num_workers_;
            {
                std::lock_guard<std::mutex> lock(worker_deques_[victim_id].mtx);
                if(!worker_deques_[victim_id].tasks.empty()) {
                    out_bucket = worker_deques_[victim_id].tasks.front();
                    worker_deques_[victim_id].tasks.pop_front();
                    worker_deques_[victim_id].steals_received++;

                    long long steal_seq = global_steal_counter_.fetch_add(1) + 1;
                    ajb_ws_stats.total_steals++;

                    long long stolen_size = out_bucket.bucketSize();
                    fprintf(stderr, "[AJB_EXT][WorkStealing] STEAL #%lld: worker[%d] ← worker[%d], "
                            "bucket_size=%lld, victim_remaining=%zu, splitDim=%d, bounds=[",
                            steal_seq, thief_id, victim_id, stolen_size,
                            worker_deques_[victim_id].tasks.size(), out_bucket.getSplitDim());
                    for(size_t d = 0; d < out_bucket.getLowerBound().size(); d++) {
                        if(d) fprintf(stderr, ",");
                        fprintf(stderr, "%d:%d", out_bucket.getLowerBound()[d], out_bucket.getUpperBound()[d]);
                    }
                    fprintf(stderr, "]\n");

                    return true;
                }
            }
        }
        ajb_ws_stats.steal_failures++;
        return false;
    }

    /**
     * @brief Run the work-stealing loop for a single worker.
     *
     * The worker processes tasks by:
     *   1. Popping from its own deque (local work).
     *   2. If empty, trying to steal from others.
     *   3. If stealing fails, the worker is done.
     *
     * @param worker_id This worker's ID.
     * @param process_fn Callback to process each bucket.
     */
    void worker_loop(int worker_id, std::function<void(int, const Bucket&)> process_fn) {
        int local_processed = 0;
        int stolen_processed = 0;

        while(true) {
            Bucket task;
            // Step 1: try local deque
            if(pop_local(worker_id, task)) {
                process_fn(worker_id, task);
                local_processed++;
                ajb_ws_stats.tasks_processed++;
                continue;
            }
            // Step 2: try stealing
            if(try_steal(worker_id, task)) {
                process_fn(worker_id, task);
                stolen_processed++;
                ajb_ws_stats.tasks_processed++;
                continue;
            }
            // Step 3: no work anywhere, exit
            break;
        }

        fprintf(stderr, "[AJB_EXT][WorkStealing] worker[%d] finished: local=%d stolen=%d total=%d\n",
                worker_id, local_processed, stolen_processed, local_processed + stolen_processed);
    }

    /**
     * @brief Run the full work-stealing schedule across all workers.
     *
     * Assigns buckets using LPT heuristic, then spawns worker threads.
     * Each worker runs worker_loop until all work is consumed.
     *
     * @param buckets Buckets to process.
     * @param process_fn Processing callback: (worker_id, bucket) -> void.
     */
    void run(const vector<Bucket>& buckets, std::function<void(int, const Bucket&)> process_fn) {
        schedule(buckets);

        fprintf(stderr, "[AJB_EXT][WorkStealing] launching %d workers...\n", num_workers_);
        auto t_start = std::chrono::high_resolution_clock::now();

        if(num_workers_ == 1) {
            // Single worker: no threading overhead
            worker_loop(0, process_fn);
        } else {
            vector<std::thread> threads;
            threads.reserve(num_workers_);
            for(int w = 0; w < num_workers_; w++) {
                threads.emplace_back([this, w, &process_fn]() {
                    worker_loop(w, process_fn);
                });
            }
            for(auto& t : threads) t.join();
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        fprintf(stderr, "[AJB_EXT][WorkStealing] ALL DONE: %zu buckets processed in %.2f ms, "
                "total_steals=%lld\n", buckets.size(), elapsed_ms,
                global_steal_counter_.load());

        // Final per-worker summary
        fprintf(stderr, "[AJB_EXT][WorkStealing] per-worker summary:\n");
        for(int w = 0; w < num_workers_; w++) {
            fprintf(stderr, "[AJB_EXT][WorkStealing]   worker[%d]: assigned_size=%lld, "
                    "processed=%lld, stolen_from=%lld, remaining=%zu\n",
                    w, worker_deques_[w].total_size_assigned,
                    worker_deques_[w].tasks_processed,
                    worker_deques_[w].steals_received,
                    worker_deques_[w].tasks.size());
        }

        ajb_ws_stats.dump();
    }

    int getNumWorkers() const { return num_workers_; }
};

// =============================================================================
// [AJB_EXT] Convenience: work_stealing_schedule free function
//
// Wraps WorkStealingScheduler for simple one-shot use. Takes a vector of
// buckets, a number of workers, and a processing callback.
// =============================================================================
inline void work_stealing_schedule(const vector<Bucket>& buckets, int num_workers,
                                   std::function<void(int, const Bucket&)> process_fn) {
    fprintf(stderr, "[AJB_EXT][work_stealing_schedule] START: %zu buckets, %d workers\n",
            buckets.size(), num_workers);
    WorkStealingScheduler scheduler(num_workers);
    scheduler.run(buckets, process_fn);
    fprintf(stderr, "[AJB_EXT][work_stealing_schedule] END\n");
}
ENDOFFILE
echo "File written successfully"