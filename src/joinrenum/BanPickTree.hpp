// [AJB] BanPickTree: AVL-based interval tracker for ban/pick operations
// pick(): 随机选一个未ban的位置 (通过G()实现)
// ban(low,high): 把[low,high]标记为已处理
// remaining(): 还剩多少未ban的位置
// 这棵树是Enumerator的调度核心
// [AJB] 底层用AVL树(不是segment tree), 支持O(log n) ban和pick

// [AJB] BanPickTree诊断
// [AJB] M1014: pool utilization tracking — exponential moving average (EMA)
// of banned/total ratio to detect when ban rate is accelerating or decelerating.
// alpha=0.1 gives ~10-observation smoothing window.
#include <cstdio>
static thread_local struct {
    long long pick_calls = 0;
    long long ban_calls  = 0;
    long long ban_overlap = 0;  // ban时与已有区间重叠的次数
    long long total_banned = 0; // ban掉的总元素数
    // [AJB_BP] M933: tree structure tracking
    int       max_tree_height = 0;
    long long leaf_count = 0;    // nodes without children
    long long internal_count = 0; // nodes with at least one child
    // [AJB_BP] M934: ban reason tracking — empty range, clipped, actual
    long long ban_empty = 0;
    long long ban_clipped = 0;
    long long ban_merged = 0;    // merged with adjacent node
    // [AJB_BP] M935: pick distribution (how uniform is G()?)
    long long pick_min = 0;
    long long pick_max = 0;
    double    pick_sum = 0.0;
    // [AJB_STATE] M1014: pool utilization EMA tracking
    double    utilization_ema = 0.0;       // EMA of banned/total ratio
    double    utilization_ema_alpha = 0.1;  // smoothing factor
    long long ema_samples = 0;
    double    util_min = 1.0;
    double    util_max = 0.0;
    void dump(const char* tag = "BanPickTree") {
        fprintf(stderr, "[AJB_STATE][%s] picks=%lld bans=%lld overlaps=%lld total_banned=%lld\n",
                tag, pick_calls, ban_calls, ban_overlap, total_banned);
        fprintf(stderr, "[AJB_STATE][%s] tree: max_height=%d leaf=%lld internal=%lld ratio=%.2f\n",
                tag, max_tree_height, leaf_count, internal_count,
                (leaf_count + internal_count) > 0
                    ? (double)leaf_count / (leaf_count + internal_count) : 0.0);
        fprintf(stderr, "[AJB_STATE][%s] ban_types: empty=%lld clipped=%lld merged=%lld\n",
                tag, ban_empty, ban_clipped, ban_merged);
        fprintf(stderr, "[AJB_STATE][%s] pick_range: min=%lld max=%lld avg=%.1f\n",
                tag, pick_min, pick_max,
                pick_calls > 0 ? pick_sum / pick_calls : 0.0);
        // M1014: pool utilization EMA dump
        fprintf(stderr, "[AJB_STATE][%s] pool_util_ema: current=%.6f samples=%lld min=%.6f max=%.6f\n",
                tag, utilization_ema, ema_samples, util_min, util_max);
    }
    void reset() {
        pick_calls = ban_calls = ban_overlap = total_banned = 0;
        max_tree_height = 0; leaf_count = internal_count = 0;
        ban_empty = ban_clipped = ban_merged = 0;
        pick_min = 0; pick_max = 0; pick_sum = 0.0;
        utilization_ema = 0.0; ema_samples = 0; util_min = 1.0; util_max = 0.0;
    }
} ajb_bpt_stats;

#include <bits/stdc++.h>
using namespace std;

struct node {
    long long low, high, take;
    int left = -1, right = -1, height;
    node() : low(0), high(0), take(0), height(1) {}
    node(long long low, long long high) : low(low), high(high), take(high - low + 1), height(1) {}

    // AJB: 子节点属性访问提取为内联函数——避免散落的三元运算
    static inline int childHeight(const vector<node>& pool, int idx) {
        return idx != -1 ? pool[idx].height : 0;
    }
    static inline long long childTake(const vector<node>& pool, int idx) {
        return idx != -1 ? pool[idx].take : 0;
    }
    void update(vector<node>& pool) {
        take = (high - low + 1) + childTake(pool, left) + childTake(pool, right);
        height = max(childHeight(pool, left), childHeight(pool, right)) + 1;
    }
    int balanceFactor(const vector<node>& pool) const {
        return childHeight(pool, left) - childHeight(pool, right);
    }

    bool operator<(const node& n) const {
        return high < n.low;
    }
};

class BanPickTree {
private:
    vector<node> pool;
    int root = -1;
    long long H = 0;

    // AJB-algo: in-order via explicit stack + snprintf batch output
    void printSubTree(int v) {
        if (v == -1) return;
        std::vector<int> stk;
        stk.reserve(32);
        int cur = v;
        char buf[96];
        while (cur != -1 || !stk.empty()) {
            while (cur != -1) { stk.push_back(cur); cur = pool[cur].left; }
            cur = stk.back(); stk.pop_back();
            int n = snprintf(buf, sizeof(buf), "%lld %lld %lld\n",
                             pool[cur].low, pool[cur].high, pool[cur].take);
            fwrite(buf, 1, n, stdout);
            cur = pool[cur].right;
        }
    }

    void rotateLeft(int& v) {
        int u = pool[v].right;
        pool[v].right = pool[u].left;
        pool[u].left = v;
        pool[v].update(pool);
        pool[u].update(pool);
        v = u;
    }

    void rotateRight(int& v) {
        int u = pool[v].left;
        pool[v].left = pool[u].right;
        pool[u].right = v;
        pool[v].update(pool);
        pool[u].update(pool);
        v = u;
    }

    void insertSubTree(int& v, int nv) {
        if (v == -1) {
            v = nv;
            return;
        }
        if (pool[nv].high == pool[v].low - 1) {
            pool[v].low = pool[nv].low;
            pool[v].update(pool);
            pool.pop_back();
            return;
        }
        if (pool[nv].low == pool[v].high + 1) {
            pool[v].high = pool[nv].high;
            pool[v].update(pool);
            pool.pop_back();
            return;
        }
        if (pool[nv] < pool[v]) {
            insertSubTree(pool[v].left, nv);
            // AJB: balanceFactor代替手写height差
            if (pool[v].balanceFactor(pool) == 2) {
                if (pool[nv] < pool[pool[v].left]) {
                    rotateRight(v);
                } else {
                    rotateLeft(pool[v].left);
                    rotateRight(v);
                }
            }
        } else {
            insertSubTree(pool[v].right, nv);
            // AJB: balanceFactor代替手写height差
            if (pool[v].balanceFactor(pool) == -2) {
                if (pool[pool[v].right] < pool[nv]) {
                    rotateLeft(v);
                } else {
                    rotateRight(pool[v].right);
                    rotateLeft(v);
                }
            }
        }
        pool[v].update(pool);
    }

    long long G() {
        int u = root;
        long long rem = remaining();
        if (rem <= 0) return 0;
        long long y = uniform_int_distribution<long long>(1, rem)(gen);
        long long b = 0;
        while (u != -1) {
            // AJB: 用childTake代替重复的三元运算
            long long ltake = node::childTake(pool, pool[u].left);
            if (y + b + ltake < pool[u].low) {
                u = pool[u].left;
            } else {
                b += ltake + (pool[u].high - pool[u].low + 1);
                u = pool[u].right;
            }
        }
        return y + b;
    }

public:
    mt19937 gen;

    BanPickTree() : gen(random_device{}()) {}

    BanPickTree(long long H) : H(H), gen(random_device{}()) {
        fprintf(stderr, "[AJB_BP][BanPickTree] built: H=%lld\n", H);
    }

    long long getTotal() {
        return H;
    }

    double getPercentage() {
        return 1.0 - 1.0 * remaining() / H;
    }

    void ban(long long low, long long high) {
        ajb_bpt_stats.ban_calls++;
        if (low > high) {
            ajb_bpt_stats.ban_empty++;
            return;
        }
        long long orig_low = low, orig_high = high;
        low = std::max(low, 1LL); high = std::min(high, H);
        if (low > high) {
            ajb_bpt_stats.ban_empty++;
            return;
        }
        // [AJB_BP] M934: detect clipping
        if (low != orig_low || high != orig_high)
            ajb_bpt_stats.ban_clipped++;
        long long before_remaining = remaining();
        if (before_remaining <= 0) return;
        if (pool.size() == pool.capacity())
            pool.reserve(pool.capacity() < 16 ? 16 : pool.capacity() * 2);
        size_t pool_before = pool.size();
        if (root == -1) {
            pool.emplace_back(low, high);
            root = pool.size() - 1;
            // [AJB_BP] M933: first node is leaf
            ajb_bpt_stats.leaf_count++;
            // [AJB_BP] M934: print first 10 bans
            if (ajb_bpt_stats.ban_calls <= 10)
                fprintf(stderr, "[AJB_BP][BanPickTree] ban #%lld: [%lld,%lld] (first node)\n",
                        ajb_bpt_stats.ban_calls, low, high);
            return;
        }
        pool.emplace_back(low, high);
        insertSubTree(root, pool.size() - 1);
        long long actually_banned = before_remaining - remaining();
        ajb_bpt_stats.total_banned += actually_banned;
        if(actually_banned < (high - low + 1)) ajb_bpt_stats.ban_overlap++;

        // [AJB_STATE] M1014: update pool utilization EMA
        // ratio = fraction of total space that has been banned (utilization)
        if(H > 0) {
            double current_util = 1.0 - (double)remaining() / (double)H;
            if(ajb_bpt_stats.ema_samples == 0) {
                ajb_bpt_stats.utilization_ema = current_util;
            } else {
                double a = ajb_bpt_stats.utilization_ema_alpha;
                ajb_bpt_stats.utilization_ema = a * current_util + (1.0 - a) * ajb_bpt_stats.utilization_ema;
            }
            ajb_bpt_stats.ema_samples++;
            if(current_util < ajb_bpt_stats.util_min) ajb_bpt_stats.util_min = current_util;
            if(current_util > ajb_bpt_stats.util_max) ajb_bpt_stats.util_max = current_util;
            // [AJB_STATE] M1014: emit EMA snapshot every 500 bans or first 5
            if(ajb_bpt_stats.ban_calls <= 5 || ajb_bpt_stats.ban_calls % 500 == 0) {
                fprintf(stderr, "[AJB_STATE][BanPickTree] pool_util_ema: ban#=%lld util=%.6f ema=%.6f pool_size=%zu remaining=%lld H=%lld\n",
                        ajb_bpt_stats.ban_calls, current_util, ajb_bpt_stats.utilization_ema,
                        pool.size(), remaining(), H);
            }
        }
        // [AJB_BP] M934: detect merges (pool didn't grow = node was merged)
        if (pool.size() <= pool_before)
            ajb_bpt_stats.ban_merged++;
        // [AJB_BP] M933: track tree height after insertion
        if (root != -1) {
            int h = pool[root].height;
            if (h > ajb_bpt_stats.max_tree_height) {
                ajb_bpt_stats.max_tree_height = h;
                if (h <= 10 || ajb_bpt_stats.ban_calls <= 10)
                    fprintf(stderr, "[AJB_BP][BanPickTree] new max_height=%d at ban #%lld pool_size=%zu\n",
                            h, ajb_bpt_stats.ban_calls, pool.size());
            }
            // [AJB_BP] M933: update leaf/internal counts periodically
            if (ajb_bpt_stats.ban_calls % 1000 == 0 || ajb_bpt_stats.ban_calls <= 10) {
                long long l = 0, n = 0;
                for(size_t i = 0; i < pool.size(); i++) {
                    if(pool[i].left == -1 && pool[i].right == -1) l++;
                    else n++;
                }
                ajb_bpt_stats.leaf_count = l;
                ajb_bpt_stats.internal_count = n;
            }
        }
        // [AJB_BP] M934: print significant bans (first 10 or every 1000th)
        if (ajb_bpt_stats.ban_calls <= 10 || ajb_bpt_stats.ban_calls % 1000 == 0)
            fprintf(stderr, "[AJB_BP][BanPickTree] ban #%lld: [%lld,%lld] banned=%lld remaining=%lld\n",
                    ajb_bpt_stats.ban_calls, low, high, actually_banned, remaining());
    }

    long long pick() {
        ajb_bpt_stats.pick_calls++;
        long long rem = remaining();
        if (rem <= 0) return 0;
        long long result = G();
        // [AJB_BP] M935: track pick distribution
        ajb_bpt_stats.pick_sum += (double)result;
        if (ajb_bpt_stats.pick_calls == 1) {
            ajb_bpt_stats.pick_min = result;
            ajb_bpt_stats.pick_max = result;
        } else {
            if (result < ajb_bpt_stats.pick_min) ajb_bpt_stats.pick_min = result;
            if (result > ajb_bpt_stats.pick_max) ajb_bpt_stats.pick_max = result;
        }
        // [AJB_BP] invariant: picked value must be available
        if (result < 1 || result > H) {
            fprintf(stderr, "[AJB_BP][BanPickTree::pick] OOB result=%lld H=%lld\n", result, H);
        }
        // [AJB_BP] M935: log first 10 picks and every 1000th
        if (ajb_bpt_stats.pick_calls <= 10 || ajb_bpt_stats.pick_calls % 1000 == 0)
            fprintf(stderr, "[AJB_BP][BanPickTree] pick #%lld: result=%lld remaining=%lld range=[%lld,%lld]\n",
                    ajb_bpt_stats.pick_calls, result, rem, ajb_bpt_stats.pick_min, ajb_bpt_stats.pick_max);
        return result;
    }

    long long remaining() {
        return root != -1 ? H - pool[root].take : H;
    }

    // AJB-algo: three-way compare (fewer branches per iteration)
    bool available(long long x) {
        if (root == -1) return (x >= 1 && x <= H);
        int u = root;
        while (u != -1) {
            if (x < pool[u].low) u = pool[u].left;
            else if (x > pool[u].high) u = pool[u].right;
            else return false;
        }
        return (x >= 1 && x <= H);
    }

    bool available(long long low, long long high) {
        // AJB: 区间重叠检测——等价条件但先做不重叠判断(分支预测友好)
        int u = root;
        while (u != -1) {
            if (high < pool[u].low) {
                u = pool[u].left;   // 完全在左边
            } else if (low > pool[u].high) {
                u = pool[u].right;  // 完全在右边
            } else {
                return false;       // 有重叠
            }
        }
        return true;
    }

    void print() {
        printSubTree(root);
    }

    // [AJB] 按percent输出状态快照
    void ajb_dump_state() {
        fprintf(stderr, "[AJB_STATE][BanPickTree] H=%lld remaining=%lld utilization=%.4f pool_size=%zu\n",
                H, remaining(), getPercentage(), pool.size());
        ajb_bpt_stats.dump();
    }
};
