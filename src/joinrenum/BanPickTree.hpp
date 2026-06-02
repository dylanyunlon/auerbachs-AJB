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

    void update(vector<node>& pool) {
        take = high - low + 1;
        if (left != -1) take += pool[left].take;
        if (right != -1) take += pool[right].take;
        height = max(left != -1 ? pool[left].height : 0, right != -1 ? pool[right].height : 0) + 1;
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

    void printSubTree(int v) {
        if (v == -1) return;
        printSubTree(pool[v].left);
        cout << pool[v].low << " " << pool[v].high << " " << pool[v].take << endl;
        printSubTree(pool[v].right);
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
            if (pool[v].left != -1 && pool[pool[v].left].height - (pool[v].right != -1 ? pool[pool[v].right].height : 0) == 2) {
                if (pool[nv] < pool[pool[v].left]) {
                    rotateRight(v);
                } else {
                    rotateLeft(pool[v].left);
                    rotateRight(v);
                }
            }
        } else {
            insertSubTree(pool[v].right, nv);
            if (pool[v].right != -1 && pool[pool[v].right].height - (pool[v].left != -1 ? pool[pool[v].left].height : 0) == 2) {
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
        long long y = uniform_int_distribution<long long>(1, root != -1 ? H - pool[root].take : H)(gen);
        long long b = 0, temp = 0;
        while (u != -1) {
            temp = pool[u].left != -1 ? pool[pool[u].left].take : 0;
            if (y + b + temp < pool[u].low) {
                u = pool[u].left;
            } else {
                b = b + temp + (pool[u].high - pool[u].low + 1);
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
        long long before_remaining = remaining();
        if (low > high) return;
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
        return remaining() ? G() : 0;
    }

    long long remaining() {
        return root != -1 ? H - pool[root].take : H;
    }

    bool available(long long x) {
        int u = root;
        while (u != -1) {
            if (x >= pool[u].low && x <= pool[u].high) return false;
            if (x < pool[u].low)
                u = pool[u].left;
            else
                u = pool[u].right;
        }
        return true;
    }

    bool available(long long low, long long high) {
        int u = root;
        while (u != -1) {
            if (low <= pool[u].high && high >= pool[u].low) return false;
            if (high < pool[u].low)
                u = pool[u].left;
            else
                u = pool[u].right;
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
