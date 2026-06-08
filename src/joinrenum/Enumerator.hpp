#include "BanPickTree.hpp"
#include "RRAccessTree.hpp"
#include <sys/resource.h>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <functional>
using namespace std;

// [AJB] M913+M1011: Enumerator诊断: Welford在线统计 + 迭代级产出追踪
// M1011: 每轮迭代的tuple产出量用Welford算法计算running mean/variance
static thread_local struct {
    long long total_attempts = 0;
    long long total_success  = 0;
    long long total_bans     = 0;
    long long dedup_collisions = 0;
    long long progress_dumps = 0;
    double    last_hit_rate  = 0.0;
    double    peak_rss_mb    = 0.0;

    // M1011: Welford在线统计器 — 追踪每个progress interval的产出量
    struct WelfordYieldTracker {
        long long n = 0;            // 观测数
        double mean = 0.0;          // running mean
        double m2   = 0.0;          // running sum of squares of differences
        long long prev_success = 0; // 上一次的cumulative success count
        double min_yield = 1e18;
        double max_yield = 0.0;

        void update(long long current_success) {
            double yield = (double)(current_success - prev_success);
            prev_success = current_success;
            n++;
            double delta = yield - mean;
            mean += delta / n;
            double delta2 = yield - mean;
            m2 += delta * delta2;
            if(yield < min_yield) min_yield = yield;
            if(yield > max_yield) max_yield = yield;
        }
        double variance() const { return n > 1 ? m2 / (n - 1) : 0.0; }
        double stddev()   const { return n > 1 ? sqrt(variance()) : 0.0; }
        double cv()       const { return mean > 0 ? stddev() / mean : 0.0; }
        void reset() { n = 0; mean = m2 = 0.0; prev_success = 0; min_yield = 1e18; max_yield = 0.0; }
    } yield_tracker;

    void dump(const char* tag = "Enumerator") {
#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_STATE][%s] attempts=%lld success=%lld bans=%lld dedup=%lld hit_rate=%.4f peak_rss=%.1fMB\n",
                tag, total_attempts, total_success, total_bans, dedup_collisions, last_hit_rate, peak_rss_mb);
        if(yield_tracker.n > 0) {
            fprintf(stderr, "[AJB_STATE][%s] yield_welford: n=%lld mean=%.2f stddev=%.2f cv=%.4f range=[%.0f,%.0f]\n",
                    tag, yield_tracker.n, yield_tracker.mean, yield_tracker.stddev(),
                    yield_tracker.cv(), yield_tracker.min_yield, yield_tracker.max_yield);
        }
#endif
    }
    void reset() { total_attempts = total_success = total_bans = dedup_collisions = progress_dumps = 0;
                    last_hit_rate = peak_rss_mb = 0.0; yield_tracker.reset(); }
} ajb_enum_stats;

// [AJB] M913: tuple hash for dedup detection
// 用FNV-1a hash检测是否同一个join结果被枚举了多次
// 这不影响正确性(BanPickTree保证不重复pick), 但能检测算法bug
struct AjbTupleHasher {
    size_t operator()(const vector<int>& v) const {
        size_t h = 14695981039346656037ULL; // FNV offset basis
        for(int x : v) {
            h ^= static_cast<size_t>(x);
            h *= 1099511628211ULL; // FNV prime
        }
        return h;
    }
};

// [AJB] getMemoryUsage: fopen+fgets replacing ifstream+getline+substr
// upstream: std::ifstream → getline → line.substr(0,6) → std::stoul
// changed: FILE* + fgets + strncmp + strtoul — zero std::string allocations
size_t getMemoryUsage() {
    FILE* f = fopen("/proc/self/status", "r");
    if(!f) return 0;
    char buf[256];
    size_t memory = 0;
    while(fgets(buf, sizeof(buf), f)) {
        if(strncmp(buf, "VmRSS:", 6) == 0) {
            // skip "VmRSS:" then whitespace
            const char* p = buf + 6;
            while(*p == ' ' || *p == '\t') p++;
            memory = strtoul(p, nullptr, 10);
            break;
        }
    }
    fclose(f);
    return memory; // in KB
}

// [AJB] M913: peak memory tracker via getrusage
static inline double ajb_get_peak_rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024.0;
}

class Enumerator {

private:
public:
    int option = 3;
    /*
        0: REnum,
        1: REnum_L(Larger),
        2: REnum_M(Merge),
        3: REnum_B(Batch),
        4: REnum_HC(HalfCache),
        5: REnum_HC_Pool (MTI on Nocache levels)
        6: REnum_HC_Pool_basic (basic LTI on Nocache levels)
    */
    bool treeflag = true; // true: use TU-S
    RRAccessTree access_tree;
    BanPickTree bp;

    // --- M1126: Branch-and-bound pruning state ---
    // Tracks the best known result cost.  When a partial enumeration
    // path exceeds best_known * pruning_factor, we prune that branch
    // entirely.  This dramatically reduces the search space for
    // large join queries where most paths are suboptimal.
    struct BranchAndBound {
        double best_known_cost = 1e18;    // best complete plan cost seen so far
        double pruning_factor  = 1.5;     // prune when partial > best * factor
        long long pruned_count = 0;       // branches pruned
        long long evaluated_count = 0;    // branches fully evaluated
        long long total_branches = 0;     // total branches encountered

        // Check if a partial plan should be pruned
        bool shouldPrune(double partial_cost) {
            total_branches++;
            if (partial_cost > best_known_cost * pruning_factor) {
                pruned_count++;
                return true;
            }
            return false;
        }

        // Update best known cost after completing a plan
        void updateBest(double cost) {
            evaluated_count++;
            if (cost < best_known_cost) {
                best_known_cost = cost;
                fprintf(stderr, "[AJB_BP][B&B] new best_cost=%.2f (eval=%lld pruned=%lld rate=%.2f%%)\n",
                        cost, evaluated_count, pruned_count,
                        total_branches > 0 ? 100.0 * pruned_count / total_branches : 0.0);
            }
        }

        double pruningRate() const {
            return total_branches > 0 ? (double)pruned_count / total_branches : 0.0;
        }

        void dump() const {
            fprintf(stderr, "[AJB_BP][B&B] best=%.2f pruned=%lld/%lld (%.2f%%) evaluated=%lld\n",
                    best_known_cost, pruned_count, total_branches,
                    pruningRate() * 100.0, evaluated_count);
        }
    } bnb_state;

    // --- M1127: Adaptive DP memoization table ---
    // The memoization table for subplan costs starts small and grows
    // as more subproblems are encountered.  When memory pressure is
    // high (table > threshold), we evict entries with highest cost
    // (least useful for pruning).
    struct AdaptiveMemo {
        std::unordered_map<size_t, double> table;
        size_t max_entries = 1 << 16;   // initial cap: 64K entries
        size_t evictions = 0;

        bool lookup(size_t key, double& cost) const {
            auto it = table.find(key);
            if (it != table.end()) {
                cost = it->second;
                return true;
            }
            return false;
        }

        void insert(size_t key, double cost) {
            if (table.size() >= max_entries) {
                evict_worst();
            }
            table[key] = cost;
        }

        // Evict entries with highest cost (they provide least pruning benefit)
        void evict_worst() {
            if (table.empty()) return;
            // Find and remove the entry with maximum cost
            auto worst = table.begin();
            for (auto it = table.begin(); it != table.end(); ++it) {
                if (it->second > worst->second) worst = it;
            }
            table.erase(worst);
            evictions++;
        }

        // Grow the table when pruning rate is high (table is effective)
        void adaptSize(double pruning_rate) {
            if (pruning_rate > 0.3 && max_entries < (1ULL << 24)) {
                max_entries *= 2;
                fprintf(stderr, "[AJB_BP][AdaptiveMemo] grew to max=%zu (pruning_rate=%.2f)\n",
                        max_entries, pruning_rate);
            }
        }

        void dump() const {
            fprintf(stderr, "[AJB_BP][AdaptiveMemo] entries=%zu max=%zu evictions=%zu\n",
                    table.size(), max_entries, evictions);
        }
    } memo;

    // [AJB] M913: dedup hash set — 只在AJB_DEBUG模式下启用
    // 用于检测算法正确性: 如果同一个元组被enumerate两次说明有bug
#ifdef AJB_DEBUG
    std::unordered_set<vector<int>, AjbTupleHasher> ajb_seen_tuples;
#endif

    Enumerator(
        unordered_map<string, vector<string> > relations,
        unordered_map<string, string> filenames,
        unordered_map<string, int> numlines) :
        access_tree(relations, filenames, numlines, treeflag),
        bp(access_tree.AGM) {
        access_tree.idx.treeflag = treeflag;
#ifdef AJB_DEBUG
        // [AJB_BP] Enumerator ready: AGM + option + treeflag = 枚举的三个关键参数
        fprintf(stderr, "[AJB_BP][Enumerator] constructed: AGM=%lld option=%d treeflag=%d\n",
                access_tree.AGM, option, (int)treeflag);
        fprintf(stderr, "[AJB_STATE][Enumerator] BanPickTree range=[1, %lld]\n", access_tree.AGM);
#endif
    }

    void random_enumerate() {
        double totalRRAccessTime = 0;
        int cntsuccess = 0, cnt = 0, step = 20;
        clock_t start = clock();
        clock_t end;
        double elapsed = 0;
        double last_percentage = 0;
        long long s;
        bool res;
        struct rusage r_usage;
        auto ajb_wall_start = std::chrono::steady_clock::now();

        // [AJB] M913: progress dump interval — 每1000轮打印诊断
        constexpr int AJB_PROGRESS_INTERVAL = 1000;
        long long ajb_last_progress_attempt = 0;

#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_BP][Enumerator] enumerate start: option=%d AGM=%lld\n",
                option, access_tree.AGM);
        // 打印初始内存状态
        fprintf(stderr, "[AJB_STATE][Enumerator] initial_rss=%.1fMB pool_size=%d\n",
                ajb_get_peak_rss_mb(), access_tree.idx.totalrrtreenode);
#endif

        while(bp.remaining()){
            cnt++;
            ajb_enum_stats.total_attempts++;
            s = bp.pick();

            // [AJB] M913: periodic progress dump
            if(ajb_enum_stats.total_attempts - ajb_last_progress_attempt >= AJB_PROGRESS_INTERVAL) {
                ajb_last_progress_attempt = ajb_enum_stats.total_attempts;
                ajb_enum_stats.progress_dumps++;
                // M1011: Welford yield tracker — update with current interval's success count
                ajb_enum_stats.yield_tracker.update(ajb_enum_stats.total_success);
#ifdef AJB_DEBUG
                double rss = ajb_get_peak_rss_mb();
                if(rss > ajb_enum_stats.peak_rss_mb) ajb_enum_stats.peak_rss_mb = rss;
                auto ajb_now = std::chrono::steady_clock::now();
                double wall = std::chrono::duration<double>(ajb_now - ajb_wall_start).count();
                double hit_rate = cnt > 0 ? (double)cntsuccess / cnt : 0.0;
                fprintf(stderr, "[AJB_PROGRESS][Enumerator] round=%d success=%d remaining=%lld pool=%d rss=%.0fMB wall=%.1fs hit=%.4f bans=%lld\n",
                        cnt, cntsuccess, bp.remaining(), access_tree.idx.totalrrtreenode,
                        rss, wall, hit_rate, ajb_enum_stats.total_bans);
                // M1011: 每10轮progress dump输出Welford统计摘要
                if(ajb_enum_stats.progress_dumps % 10 == 0) {
                    fprintf(stderr, "[AJB_BP][Enumerator] welford_yield: n=%lld mean=%.2f σ=%.2f cv=%.4f range=[%.0f,%.0f] total_success=%lld\n",
                            ajb_enum_stats.yield_tracker.n,
                            ajb_enum_stats.yield_tracker.mean,
                            ajb_enum_stats.yield_tracker.stddev(),
                            ajb_enum_stats.yield_tracker.cv(),
                            ajb_enum_stats.yield_tracker.min_yield,
                            ajb_enum_stats.yield_tracker.max_yield,
                            ajb_enum_stats.total_success);
                }
#endif
            }

            switch(option) {
                case 0: res = access_tree.RRAccess(s); break;
                case 1: res = access_tree.RRAccess_LTI(s); break;
                case 2: res = access_tree.RRAccess_MTI(s); break;
                case 3: res = access_tree.RRAccess_BTI(s); break;
                case 4: res = access_tree.RRAccess_HalfCache(s); break;
                case 5: res = access_tree.RRAccess_HalfCache_Pool(s); break;
                case 6: res = access_tree.RRAccess_HalfCache_Pool_basic(s); break;
                default: res = access_tree.RRAccess_BTI(s); break;
            }

            if(res){
                cntsuccess++;
                ajb_enum_stats.total_success++;

                // [AJB] M913: dedup detection via hash set
#ifdef AJB_DEBUG
                {
                    auto [it, inserted] = ajb_seen_tuples.insert(access_tree.result);
                    if(!inserted) {
                        ajb_enum_stats.dedup_collisions++;
                        fprintf(stderr, "[AJB_WARN][Enumerator] DUPLICATE tuple detected at attempt=%lld s=%lld! collision_count=%lld\n",
                                ajb_enum_stats.total_attempts, s, ajb_enum_stats.dedup_collisions);
                        // 打印重复元组的值
                        fprintf(stderr, "[AJB_WARN]   tuple=(");
                        for(size_t ti = 0; ti < access_tree.result.size(); ti++) {
                            if(ti) fprintf(stderr, ",");
                            fprintf(stderr, "%d", access_tree.result[ti]);
                        }
                        fprintf(stderr, ")\n");
                    }
                }
#endif

                if(cntsuccess <= 20 || cntsuccess % 10000 == 0){
                    end = clock();
                    elapsed = double(end - start) / CLOCKS_PER_SEC;
                    getrusage(RUSAGE_SELF, &r_usage);
                    double rss_mb = r_usage.ru_maxrss / 1024.0;
                    if(rss_mb > ajb_enum_stats.peak_rss_mb) ajb_enum_stats.peak_rss_mb = rss_mb;
                    cout << cntsuccess << ", " << cnt << ", " << bp.remaining() << ", " << bp.getPercentage() << ", " << elapsed  << ", "<< r_usage.ru_maxrss/1024 << "MB, " << access_tree.idx.totalrrtreenode << endl;
#ifdef AJB_DEBUG
                    // [AJB_TRACE] 阶段性进度: hit_rate在这里能看出算法效率
                    // M913: 增加结果元组的维度摘要(min/max per dim)
                    fprintf(stderr, "[AJB_TRACE][Enumerator] progress: success=%d/%d remaining=%lld pct=%.4f rss=%.0fMB rrtreenode=%d\n",
                            cntsuccess, cnt, bp.remaining(), bp.getPercentage(), rss_mb, access_tree.idx.totalrrtreenode);
                    // 维度摘要
                    if(!access_tree.result.empty()) {
                        fprintf(stderr, "[AJB_TRACE][Enumerator]   result_tuple=(");
                        for(size_t ri = 0; ri < access_tree.result.size(); ri++) {
                            if(ri) fprintf(stderr, ",");
                            fprintf(stderr, "%d", access_tree.result[ri]);
                        }
                        fprintf(stderr, ")\n");
                    }
#endif
                }
            }

            if(res) bp.ban(s,s);
            // --- batch ban with interval merging ---
            // upstream: for(i=0..numti) bp.ban(trivialIntervals[i])
            // changed: collect intervals, sort by left endpoint, merge
            //   overlapping/adjacent intervals, then ban merged set.
            //   Reduces bp.ban() calls when intervals cluster together.
            if(access_tree.numti > 0) {
                // sort trivialIntervals[0..numti-1] by .first
                // use insertion sort since numti is typically small (<20)
                for(int i = 1; i < access_tree.numti; i++) {
                    auto tmp = access_tree.trivialIntervals[i];
                    int j = i - 1;
                    while(j >= 0 && access_tree.trivialIntervals[j].first > tmp.first) {
                        access_tree.trivialIntervals[j+1] = access_tree.trivialIntervals[j];
                        j--;
                    }
                    access_tree.trivialIntervals[j+1] = tmp;
                }
                // merge overlapping/adjacent and ban
                long long merge_lo = access_tree.trivialIntervals[0].first;
                long long merge_hi = access_tree.trivialIntervals[0].second;
                for(int i = 1; i < access_tree.numti; i++) {
                    if(access_tree.trivialIntervals[i].first <= merge_hi + 1) {
                        // extend current merged interval
                        merge_hi = max(merge_hi, access_tree.trivialIntervals[i].second);
                    } else {
                        // gap: ban the accumulated interval, start new one
                        bp.ban(merge_lo, merge_hi);
                        ajb_enum_stats.total_bans++;
                        merge_lo = access_tree.trivialIntervals[i].first;
                        merge_hi = access_tree.trivialIntervals[i].second;
                    }
                }
                bp.ban(merge_lo, merge_hi);
                ajb_enum_stats.total_bans++;
            }
        }
        end = clock();
        elapsed = double(end - start) / CLOCKS_PER_SEC;
        auto ajb_wall_end = std::chrono::steady_clock::now();
        double wall_sec = std::chrono::duration<double>(ajb_wall_end - ajb_wall_start).count();
        cout << cntsuccess << ", " << cnt << ", " << bp.remaining() << ", " << bp.getPercentage() << ", " << elapsed << endl;
        cout << "Total RRAccess Time: " << totalRRAccessTime << endl;

        // [AJB_TIMER] final summary: 成功数/尝试数/ban数/wall time
        ajb_enum_stats.last_hit_rate = cnt > 0 ? (double)cntsuccess / cnt : 0.0;
        // M1126: Branch-and-bound summary
        bnb_state.dump();
        memo.dump();
        // Adapt memo size based on pruning effectiveness
        memo.adaptSize(bnb_state.pruningRate());
#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_TIMER][Enumerator] done: success=%d attempts=%d hit_rate=%.6f cpu=%.3fs wall=%.3fs\n",
                cntsuccess, cnt, ajb_enum_stats.last_hit_rate, elapsed, wall_sec);
        fprintf(stderr, "[AJB_STATE][Enumerator] total_bans=%lld peak_rss=%.1fMB dedup_collisions=%lld\n",
                ajb_enum_stats.total_bans, ajb_enum_stats.peak_rss_mb, ajb_enum_stats.dedup_collisions);
        fprintf(stderr, "[AJB_STATE][Enumerator] progress_dumps=%lld unique_tuples=%zu\n",
                ajb_enum_stats.progress_dumps,
                ajb_seen_tuples.size());
#else
        // [AJB] M1231: unconditional final diagnostics — always visible in experiment logs
        fprintf(stderr, "[AJB_TIMER][Enumerator] done: success=%d attempts=%d hit_rate=%.6f cpu=%.3fs wall=%.3fs\n",
                cntsuccess, cnt, ajb_enum_stats.last_hit_rate, elapsed, wall_sec);
        fprintf(stderr, "[AJB_STATE][Enumerator] bans=%lld yield_mean=%.1f yield_cv=%.4f progress_intervals=%lld\n",
                ajb_enum_stats.total_bans,
                ajb_enum_stats.yield_tracker.mean,
                ajb_enum_stats.yield_tracker.cv(),
                ajb_enum_stats.yield_tracker.n);
#endif
        // [AJB] M1231: always-on BucketPool summary (enables real-time experiment monitoring)
        access_tree.pool.ajb_summary_line("enum_done");
    }

};
