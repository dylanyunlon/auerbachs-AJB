// [AJB] BanPickTree: AVL-based interval tracker for ban/pick operations
// pick(): 随机选一个未ban的位置 (通过G()实现)
// ban(low,high): 把[low,high]标记为已处理
// remaining(): 还剩多少未ban的位置
// 这棵树是Enumerator的调度核心
// [AJB] 底层用AVL树(不是segment tree), 支持O(log n) ban和pick

// [AJB] BanPickTree诊断
#include <cstdio>
static thread_local struct {
    long long pick_calls = 0;
    long long ban_calls  = 0;
    long long ban_overlap = 0;  // ban时与已有区间重叠的次数
    long long total_banned = 0; // ban掉的总元素数
    void dump(const char* tag = "BanPickTree") {
        fprintf(stderr, "[AJB_STATE][%s] picks=%lld bans=%lld overlaps=%lld total_banned=%lld\n",
                tag, pick_calls, ban_calls, ban_overlap, total_banned);
    }
    void reset() { pick_calls = ban_calls = ban_overlap = total_banned = 0; }
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
        if (low > high) return;
        low = std::max(low, 1LL); high = std::min(high, H);
        if (low > high) return;
        long long before_remaining = remaining();
        if (before_remaining <= 0) return;
        if (pool.size() == pool.capacity())
            pool.reserve(pool.capacity() < 16 ? 16 : pool.capacity() * 2);
        if (root == -1) {
            pool.emplace_back(low, high);
            root = pool.size() - 1;
            return;
        }
        pool.emplace_back(low, high);
        insertSubTree(root, pool.size() - 1);
        long long actually_banned = before_remaining - remaining();
        ajb_bpt_stats.total_banned += actually_banned;
        if(actually_banned < (high - low + 1)) ajb_bpt_stats.ban_overlap++;
    }

    long long pick() {
        long long rem = remaining();
        if (rem <= 0) return 0;
        long long result = G();
        // [AJB_BP] invariant: picked value must be available
        if (result < 1 || result > H) {
            fprintf(stderr, "[AJB_BP][BanPickTree::pick] OOB result=%lld H=%lld\n", result, H);
        }
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
