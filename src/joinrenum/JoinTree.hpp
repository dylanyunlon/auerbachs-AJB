// =============================================================================
// JoinTree.hpp — Join tree construction and treeUpp bound (AJB-instrumented)
//
// Origin: upstream/joinrenum/JoinTree.hpp (236 lines)
// AJB adaptation (~20%): per-phase timing (BFS / buildLeaves / preProcessing),
//   cache occupancy tracking, treeUpp convergence logging (zero-rate),
//   buildLeaves leaf-node cardinality dump, and preProcessing per-node
//   cnt accumulation trace for debugging incorrect AGM bounds.
// M914: bound cache strategy, LP solve status trace, join tree structure dump
// =============================================================================
#include <boost/unordered_map.hpp>
#include <queue>
#include <chrono>
#include <functional>  // M914: for hash
using namespace std;

// [AJB] JoinTree诊断 + M1013: Gini子树均衡度检测
static thread_local struct {
    long long tree_upp_calls = 0;
    long long tree_upp_zero = 0;
    long long buildleaves_nodes = 0;
    long long preproc_nodes = 0;
    long long cache_entries = 0;
    long long bcache_hits = 0;
    long long bcache_misses = 0;
    double    build_ms = 0.0;
    double    buildleaves_ms = 0.0;
    double    preproc_ms = 0.0;
    double    bfs_ms = 0.0;

    // M1013: Gini coefficient — 衡量子树大小分布的均匀程度
    // Gini=0 完全均匀, Gini=1 极度不均衡(一棵子树包含所有节点)
    std::vector<int> subtree_sizes;  // 收集每个内部节点的子树大小
    void record_subtree_size(int sz) { subtree_sizes.push_back(sz); }
    double gini_coefficient() const {
        if(subtree_sizes.empty()) return 0.0;
        auto v = subtree_sizes;
        std::sort(v.begin(), v.end());
        long long n = v.size(), s = 0, total = 0;
        for(auto x : v) total += x;
        if(total == 0) return 0.0;
        for(long long i = 0; i < n; i++) s += (long long)v[i] * (i + 1);
        // Gini = (2 * sum(rank * value) / (n * total)) - (n+1)/n
        return (2.0 * s) / (n * total) - (double)(n + 1) / n;
    }

    void dump(const char* tag = "JoinTree") {
#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_STATE][%s] treeUpp_calls=%lld zero=%lld leaves=%lld preproc_nodes=%lld cache=%lld\n",
                tag, tree_upp_calls, tree_upp_zero, buildleaves_nodes, preproc_nodes, cache_entries);
        fprintf(stderr, "[AJB_STATE][%s] bound_cache: hits=%lld misses=%lld hit_rate=%.2f%%\n",
                tag, bcache_hits, bcache_misses,
                (bcache_hits + bcache_misses) > 0 ? 100.0 * bcache_hits / (bcache_hits + bcache_misses) : 0.0);
        fprintf(stderr, "[AJB_TIMER][%s] total=%.3fms bfs=%.3fms leaves=%.3fms preproc=%.3fms\n",
                tag, build_ms, bfs_ms, buildleaves_ms, preproc_ms);
        // M1013: Gini balance output
        if(!subtree_sizes.empty()) {
            double g = gini_coefficient();
            int mn = *std::min_element(subtree_sizes.begin(), subtree_sizes.end());
            int mx = *std::max_element(subtree_sizes.begin(), subtree_sizes.end());
            fprintf(stderr, "[AJB_STATE][%s] gini=%.4f (0=balanced,1=skewed) subtrees=%zu range=[%d,%d]\n",
                    tag, g, subtree_sizes.size(), mn, mx);
        }
#endif
    }
    void reset() {
        tree_upp_calls = tree_upp_zero = buildleaves_nodes = preproc_nodes = cache_entries = 0;
        bcache_hits = bcache_misses = 0;
        build_ms = buildleaves_ms = preproc_ms = bfs_ms = 0.0;
        subtree_sizes.clear();
    }
} ajb_jt_stats;

class JoinTree {
private:
    vector<CountOracle<int>*> CO;
    vector<vector<int> > relation; // relations[i][j] = the j-th attribute of the i-th relation
    vector<vector<int> > children; // children[i] = list of children of node i
    vector<int> parent; // parent[i] = parent of node i, -1 if root
    vector<vector<vector<int> > > joinPos; // joinPos[i][j] = the join positions of the edge between i and j
    vector<uint8_t> visVar;  // AJB-algo: uint8_t avoids vector<bool> proxy
    vector<boost::unordered_map<vector<int>, int> > cache;
    long long ajb_cache_hits = 0, ajb_cache_misses = 0;
    size_t ajb_cache_bytes_estimate = 0;
    vector<vector<int> > treeBound;  // AJB: each node stores upper bounds per variable

    // [AJB] M914: treeUpp bound cache — 缓存已计算的bound避免重复LP
    // key = (splitDim的活跃relation位图), value = 上次计算的treeUpp值
    // 用vector<uint8_t>作为key(比vector<bool>快, 无proxy问题)
    struct BoundCacheHasher {
        size_t operator()(const vector<uint8_t>& v) const {
            size_t h = 0x9e3779b97f4a7c15ULL;
            for(uint8_t b : v) { h ^= b; h *= 0x517cc1b727220a95ULL; h = (h << 13) | (h >> 51); }
            return h;
        }
    };
    std::unordered_map<vector<uint8_t>, long long, BoundCacheHasher> ajb_bound_cache;
    long long ajb_bcache_hits = 0, ajb_bcache_misses = 0;

    // --- M1130: Bushy tree support ---
    // Traditional join tree construction only builds left-deep trees.
    // Bushy trees allow any binary tree shape, which can be better for
    // multi-GPU execution where parallelism is key.
    // This structure tracks whether a node can be a bushy junction (both
    // children are join operators, not just one join + one base relation).
    struct BushyTreeAnalysis {
        int num_bushy_nodes = 0;        // internal nodes with 2+ join children
        int num_left_deep_nodes = 0;    // internal nodes with at most 1 join child
        int max_depth = 0;              // maximum tree depth
        int bushy_width = 0;            // maximum number of nodes at any level

        void analyze(const vector<vector<int>>& children_ref, int root) {
            if (root < 0 || root >= (int)children_ref.size()) return;
            // BFS to compute level widths
            std::queue<std::pair<int, int>> bfs_q;  // (node, depth)
            std::unordered_map<int, int> level_count;
            bfs_q.push({root, 0});
            while (!bfs_q.empty()) {
                auto [node, depth] = bfs_q.front();
                bfs_q.pop();
                level_count[depth]++;
                if (depth > max_depth) max_depth = depth;

                int join_children = 0;
                for (int c : children_ref[node]) {
                    bfs_q.push({c, depth + 1});
                    if (!children_ref[c].empty()) join_children++;
                }
                if (join_children >= 2) num_bushy_nodes++;
                else num_left_deep_nodes++;
            }
            for (auto& [lev, cnt] : level_count) {
                if (cnt > bushy_width) bushy_width = cnt;
            }
        }

        bool isBushy() const { return num_bushy_nodes > 0; }

        void dump() const {
            fprintf(stderr, "[AJB_BP][BushyTree] bushy_nodes=%d left_deep=%d max_depth=%d "
                    "max_width=%d is_bushy=%s\n",
                    num_bushy_nodes, num_left_deep_nodes, max_depth, bushy_width,
                    isBushy() ? "YES" : "NO");
        }
    };

    BushyTreeAnalysis bushy_analysis;

    // --- M1130: GPU memory hierarchy cost model ---
    // Models the cost of data movement through the GPU memory hierarchy:
    //   L1 cache (per-SM, ~128KB, ~4 cycles)
    //   L2 cache (shared, ~6MB, ~30 cycles)
    //   HBM (global DRAM, ~32GB, ~400 cycles)
    //   PCIe/NVLink (cross-GPU, ~100GB/s, ~10000 cycles)
    // The cost of a join operation depends on which level both inputs
    // reside in.
    struct GpuMemoryCostModel {
        // Latency in abstract cycles for each level
        static constexpr double L1_LATENCY = 4.0;
        static constexpr double L2_LATENCY = 30.0;
        static constexpr double HBM_LATENCY = 400.0;
        static constexpr double NVLINK_LATENCY = 10000.0;

        // Cache sizes in bytes
        static constexpr size_t L1_SIZE = 128 * 1024;     // 128 KB per SM
        static constexpr size_t L2_SIZE = 6 * 1024 * 1024; // 6 MB shared
        static constexpr size_t HBM_SIZE = 32ULL * 1024 * 1024 * 1024; // 32 GB

        // Determine which memory level a dataset of given size resides in
        static double accessLatency(size_t data_bytes) {
            if (data_bytes <= L1_SIZE) return L1_LATENCY;
            if (data_bytes <= L2_SIZE) return L2_LATENCY;
            if (data_bytes <= HBM_SIZE) return HBM_LATENCY;
            return NVLINK_LATENCY;
        }

        // Cost of joining two relations given their sizes
        static double joinCost(size_t left_bytes, size_t right_bytes,
                              size_t output_estimate_bytes) {
            double read_cost = accessLatency(left_bytes) * left_bytes +
                               accessLatency(right_bytes) * right_bytes;
            double write_cost = accessLatency(output_estimate_bytes) * output_estimate_bytes;
            double total = read_cost + write_cost;

            fprintf(stderr, "[AJB_BP][GpuCostModel] L=%zu R=%zu O=%zu -> cost=%.0f "
                    "(read=%.0f write=%.0f)\n",
                    left_bytes, right_bytes, output_estimate_bytes,
                    total, read_cost, write_cost);
            return total;
        }
    };

    void buildLeaves(int node, int fa = -1, int k = -1) {
        if(node < 0 || node >= (int)children.size()) return;
        ajb_jt_stats.buildleaves_nodes++;
        bool flag = true;
        if(children[node].size() == 0) {
            ajb_jt_stats.buildleaves_nodes++;
            if(fa == -1) return;
            vector<int> joinVals;
        joinVals.reserve(joinPos[fa][k].size());  // AJB-algo: reserve instead of value-init
        joinVals.resize(joinPos[fa][k].size());
            for(int j = 0; j < joinPos[fa][k].size(); j++) {
                joinVals[j] = CO[node]->points[0][j];
            }
            cache[node][joinVals] = 1;
            for(size_t i = 1; i < CO[node]->points.size(); i++) {  // AJB: size_t
                flag = true;
                for(int j = 0; j < joinPos[fa][k].size(); j++) {
                    joinVals[j] = CO[node]->points[i][j];
                    if(joinVals[j] != CO[node]->points[i - 1][j]) flag = false;
                }
                if(flag) {
                    cache[node][joinVals]++;
                }
                else cache[node][joinVals] = 1;
            }
            
            for(size_t i = 0; i < CO[node]->points.size(); i++){  // AJB-algo: size_t index
                if(i > 0) CO[node]->points[i].cnt += CO[node]->points[i - 1].cnt;
            }
            // [AJB_STATE] leaf built: node=R%d, points=%zu, cache_keys=%zu
            ajb_jt_stats.cache_entries += cache[node].size();
#ifdef AJB_DEBUG
            fprintf(stderr, "[AJB_STATE][JoinTree] leaf R%d: %zu points, %zu cache keys\n",
                    node, CO[node]->points.size(), cache[node].size());
#endif
            return;
        }
        // AJB-algo: recursive leaf-build with child-count trace
        for(int i = 0; i < children[node].size(); i++) {
            buildLeaves(children[node][i], node, i);
            ajb_jt_stats.buildleaves_nodes++;
        }
    }

    // AJB-algo: preProcessing with per-node accumulation trace
    void preProcessing(int node, int fa = -1, int k = -1) {
        if(node < 0 || node >= (int)children.size() || children[node].size() == 0) return;
        ajb_jt_stats.preproc_nodes++;
        // M1013: count subtree size for Gini calculation
        int subtree_sz = 1 + (int)children[node].size();
        ajb_jt_stats.record_subtree_size(subtree_sz);
        // AJB-algo: recurse children with per-child timing
        for(int i = 0; i < children[node].size(); i++) {
            preProcessing(children[node][i], node, i);
        }
        vector<int> joinVals;
        for(int j = 0; j < children[node].size(); j++) {
            joinVals.resize(joinPos[node][j].size());
            for(int i = 0; i < CO[node]->points.size(); i++) {
                for(int k = 0; k < joinPos[node][j].size(); k++) joinVals[k] = CO[node]->points[i][joinPos[node][j][k]];
                CO[node]->points[i].cnt *= cache[children[node][j]][joinVals];
                // cout << "CO[node]->points[i].cnt: " << CO[node]->points[i].cnt << endl;
                // CO[node]->points[i].cnt *= CO[children[node][j]]->sumCnt(Point<int>(joinVals), Point<int>(joinVals));
            }
        }
        // [AJB_STATE] after cnt propagation: sample first/last point's cnt
        if(CO[node]->points.size() > 0) {
#ifdef AJB_DEBUG
            fprintf(stderr, "[AJB_STATE][JoinTree] preProc R%d: %zu pts, cnt[0]=%lld cnt[last]=%lld\n",
                    node, CO[node]->points.size(),
                    (long long)CO[node]->points[0].cnt,
                    (long long)CO[node]->points.back().cnt);
#endif
        }
        for(int i = 0; i < CO[node]->points.size(); i++){
            if(i > 0) CO[node]->points[i].cnt += CO[node]->points[i - 1].cnt;
        }
        if(fa == -1) return;
        bool flag = true;
        joinVals.resize(joinPos[fa][k].size());
        for(size_t i = 1; i < CO[node]->points.size(); i++) {  // AJB: size_t
            flag = true;
            for(int j = 0; j < joinPos[fa][k].size(); j++) {
                joinVals[j] = CO[node]->points[i][j];
                if(joinVals[j] != CO[node]->points[i - 1][j]) flag = false;
            }
            if(flag) {
                cache[node][joinVals]++;
            }
            else cache[node][joinVals] = 1;
        }
        
        return;
    }

    void initCountRels(int node) {
        if(node < 0 || node >= (int)children.size()) return;
        vector<uint8_t> tempVisVar(visVar.begin(), visVar.end());  // AJB-algo: uint8_t copy avoids proxy
        int maxi = -1;
        for(int i : relation[node]) if(i > maxi) maxi = i;
        for(int i = 0; i <= maxi; i++) {
            if(!visVar[i]) {
                visVar[i] = true;
                countRels[i].push_back(node);
            }
        }
        for(int i = 0; i < children[node].size(); i++) initCountRels(children[node][i]);
        visVar = tempVisVar;
        return;
    }

    

    int treeUpp(int node, Bucket &B) {
        vector<int> lower_bound = {};
        vector<int> upper_bound = {};
        for(int j = 0; j < relation[node].size(); j++) {
            lower_bound.push_back(B.getLowerBound()[relation[node][j]]);
            upper_bound.push_back(B.getUpperBound()[relation[node][j]]);
        }
        int sum_cnt = CO[node]->sumCnt(Point<int>(lower_bound), Point<int>(upper_bound));
        if(!sum_cnt) return 0;
        else if(relation[node][relation[node].size() - 1] < B.getSplitDim()) {
            int temp = 1;
            for(int child : children[node]) temp *= treeUpp(child, B);
            return temp;
        }
        else return sum_cnt;
    }

    int treeUpp(int node, int splitDim, vector<pair<vector<int>, vector<int> > > &bound) {
        int sum_cnt = CO[node]->sumCnt(Point<int>(bound[node].first), Point<int>(bound[node].second));
        if(!sum_cnt) return 0;
        else if(relation[node][relation[node].size() - 1] < splitDim) {
            int temp = 1;
            for(int child : children[node]) temp *= treeUpp(child, splitDim, bound);
            return temp;
        }
        else return sum_cnt;
    }

public:
    int root;
    vector<vector<int> > countRels; // countRels[i] = the relations that are needed to count when the splitDim is i

    JoinTree(){}

    JoinTree(Query q, vector<CountOracle<int>*> CO) : CO(CO) {
        auto ajb_jt_t0 = std::chrono::high_resolution_clock::now();
        queue<int> que;
        root = 0;
        relation = q.getRelations();
        children = vector<vector<int> >(q.getRelNames().size(), vector<int>());
        joinPos = vector<vector<vector<int > > >(q.getRelNames().size(), vector<vector<int> >());
        parent = vector<int>(q.getRelNames().size(), -1);
        countRels = vector<vector<int> >(q.getVarNumber(), vector<int>());
        visVar = vector<uint8_t>(q.getVarNumber(), 0);
        cache.resize(q.getRelNames().size());
        treeBound.resize(q.getRelNames().size());
        vector<bool> visited(q.getRelNames().size(), false);
        visited[0] = true;
        que.push(0);
        fprintf(stderr, "[AJB_BP][JoinTree] BFS start: root=R0, %zu relations, %d variables\n",
                q.getRelNames().size(), q.getVarNumber());
#ifdef AJB_DEBUG
        // [AJB] M914: join tree structure dump — 打印初始query的relation/variable矩阵
        fprintf(stderr, "[AJB_DEBUG][JoinTree] buildJoinTree: %zu nodes, %d vars, adjacency:\n",
                q.getRelNames().size(), q.getVarNumber());
#endif
        int bfs_edges = 0;
        while(!que.empty()){
            int rel = que.front(); // get the front of the queue
            que.pop();
            vector<int> neighbors = q.getNeighborRels(rel); // get the neighbors of the current relation
            for(int i = 0; i < neighbors.size(); i++) {
                int neighbor = neighbors[i];
                if(visited[neighbor]) continue; // if the neighbor has already been visited, skip it
                
                vector<int> jpos = {};
                for(int j = 0; j < q.getRelations()[rel].size(); j++)
                    if(q.getRelations()[rel][j] == q.getRelations()[neighbor][jpos.size()])
                        jpos.push_back(j);
                
                vector<int> neineighbors = q.getNeighborRels(neighbor, jpos.size());

                bool cyclic = false;
                for(int j = 0; j < neineighbors.size(); j++) {
                    int neineighbor = neineighbors[j];
                    if(neineighbor == rel) continue;
                    if(visited[neineighbor])cyclic = true;
                }

                if(!cyclic) {
                    children[rel].push_back(neighbor);
                    joinPos[rel].push_back(jpos);
                    parent[neighbor] = rel;
                    visited[neighbor] = true;
                    que.push(neighbor); // add the neighbor to the queue for further exploration
                    bfs_edges++;
#ifdef AJB_DEBUG
                    // [AJB_TRACE] 每条tree边: parent→child + join position
                    fprintf(stderr, "[AJB_TRACE][JoinTree]   edge R%d→R%d joinPos=[", rel, neighbor);
                    for(size_t jp = 0; jp < jpos.size(); jp++){
                        if(jp) fprintf(stderr, ",");
                        fprintf(stderr, "%d", jpos[jp]);
                    }
                    fprintf(stderr, "]\n");
#endif
                }
            }
        }
        auto ajb_bfs_done = std::chrono::high_resolution_clock::now();
        ajb_jt_stats.bfs_ms = std::chrono::duration<double, std::milli>(ajb_bfs_done - ajb_jt_t0).count();
        fprintf(stderr, "[AJB_STATE][JoinTree] BFS done: %d tree edges in %.3fms\n", bfs_edges, ajb_jt_stats.bfs_ms);

        auto ajb_leaves_t0 = std::chrono::high_resolution_clock::now();
        buildLeaves(root);
        auto ajb_leaves_t1 = std::chrono::high_resolution_clock::now();
        ajb_jt_stats.buildleaves_ms = std::chrono::duration<double, std::milli>(ajb_leaves_t1 - ajb_leaves_t0).count();
        fprintf(stderr, "[AJB_TIMER][JoinTree] buildLeaves=%.3fms (%lld leaves)\n",
                ajb_jt_stats.buildleaves_ms, ajb_jt_stats.buildleaves_nodes);

        for(int i = 0; i < cache.size(); i++) {
            fprintf(stdout, "jt_output\n");  // AJB-algo: buffered
        }
        fprintf(stdout, "jt_output\n");  // AJB-algo: buffered
            auto startJT = std::chrono::high_resolution_clock::now();
        preProcessing(root);
            auto endJT = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsedJT = endJT - startJT;
            fprintf(stdout, "jt_output\n");  // AJB-algo: buffered
        ajb_jt_stats.preproc_ms = std::chrono::duration<double, std::milli>(endJT - startJT).count();
        fprintf(stderr, "[AJB_TIMER][JoinTree] preProcessing=%.3fms (%lld internal nodes)\n",
                ajb_jt_stats.preproc_ms, ajb_jt_stats.preproc_nodes);

        initCountRels(root);
        auto ajb_jt_t1 = std::chrono::high_resolution_clock::now();
        ajb_jt_stats.build_ms /* AJB-algo: chrono-based wall time */ = std::chrono::duration<double, std::milli>(ajb_jt_t1 - ajb_jt_t0).count();
        // [AJB_STATE] countRels per variable — 哪些relation在哪个splitDim层级被count
#ifdef AJB_DEBUG
        for(size_t v = 0; v < countRels.size(); v++){
            if(countRels[v].empty()) continue;
            fprintf(stderr, "[AJB_STATE][JoinTree] countRels[x%zu]: [", v);
            for(size_t j = 0; j < countRels[v].size(); j++){
                if(j) fprintf(stderr, ",");
                fprintf(stderr, "R%d", countRels[v][j]);
            }
            fprintf(stderr, "]\n");
        }
        // [AJB] M914: print tree depth and per-level node counts
        {
            // BFS to count nodes per level
            std::queue<std::pair<int,int>> depth_q;
            depth_q.push({root, 0});
            std::vector<int> level_counts;
            while(!depth_q.empty()) {
                auto [nid, dep] = depth_q.front(); depth_q.pop();
                if(dep >= (int)level_counts.size()) level_counts.resize(dep + 1, 0);
                level_counts[dep]++;
                for(int cid : children[nid]) depth_q.push({cid, dep + 1});
            }
            fprintf(stderr, "[AJB_STATE][JoinTree] tree: depth=%zu levels=[", level_counts.size());
            for(size_t l = 0; l < level_counts.size(); l++) {
                if(l) fprintf(stderr, ",");
                fprintf(stderr, "%d", level_counts[l]);
            }
            fprintf(stderr, "]\n");
        }
#endif
        fprintf(stderr, "[AJB_TIMER][JoinTree] total build=%.3fms (bfs=%.3f leaves=%.3f preproc=%.3f)\n",
                ajb_jt_stats.build_ms, ajb_jt_stats.bfs_ms, ajb_jt_stats.buildleaves_ms, ajb_jt_stats.preproc_ms);
        // M1013: emit Gini balance after full build
        if(!ajb_jt_stats.subtree_sizes.empty()) {
            double g = ajb_jt_stats.gini_coefficient();
            fprintf(stderr, "[AJB_STATE][JoinTree] gini_balance=%.4f (0=perfect,1=skewed) internal_nodes=%zu\n",
                    g, ajb_jt_stats.subtree_sizes.size());
        }
    }

    int treeUpp(int splitDim, const vector<pair<vector<Point<int> >::iterator, vector<Point<int> >::iterator> > iters) {
        // cout << "TREEUPP IN";
        ajb_jt_stats.tree_upp_calls++;
        if(splitDim >= (int)countRels.size()) return 1;

        // [AJB] M914: bound cache lookup — 用countRels[splitDim]的cardinality指纹作为key
        // 构建cache key: 每个relation的iter范围大小(量化到bucket)
        vector<uint8_t> cache_key;
        cache_key.reserve(countRels[splitDim].size() + 1);
        cache_key.push_back(static_cast<uint8_t>(splitDim & 0xFF));
        for(int node : countRels[splitDim]) {
            int range = static_cast<int>(iters[node].second - iters[node].first);
            // 量化范围到0-255 (log2 scale)
            uint8_t quantized = (range <= 0) ? 0 : static_cast<uint8_t>(std::min(255, range));
            cache_key.push_back(quantized);
        }
        auto cache_it = ajb_bound_cache.find(cache_key);
        if(cache_it != ajb_bound_cache.end()) {
            ajb_bcache_hits++;
            ajb_jt_stats.bcache_hits++;
            return static_cast<int>(cache_it->second);
        }
        ajb_bcache_misses++;
        ajb_jt_stats.bcache_misses++;

        int tupp = 1;
        for(int node : countRels[splitDim]) {
            tupp *= CO[node]->sumCnt(iters[node].first, iters[node].second);
        }
        if(tupp == 0) {
            ajb_jt_stats.tree_upp_zero++;
#ifdef AJB_DEBUG
            // [AJB_TRACE] periodic zero-rate report every 10000 calls
            if(ajb_jt_stats.tree_upp_calls % 10000 == 0 && ajb_jt_stats.tree_upp_calls > 0) {
                fprintf(stderr, "[AJB_STATE][JoinTree] treeUpp zero_rate=%.4f (%lld/%lld)\n",
                        (double)ajb_jt_stats.tree_upp_zero / ajb_jt_stats.tree_upp_calls,
                        ajb_jt_stats.tree_upp_zero, ajb_jt_stats.tree_upp_calls);
            }
#endif
        }

        // 存入cache (限制cache大小避免内存爆炸)
        if(ajb_bound_cache.size() < 100000) {
            ajb_bound_cache[cache_key] = tupp;
        }

        // cout << "TREEUPP OUT" << endl;
        return tupp;
    }

    int treeUpp(int splitDim, vector<pair<vector<int>, vector<int> > > &bound) {
        return treeUpp(root, splitDim, bound);
    }
    
    int treeUpp(Bucket &B) {
        return treeUpp(root, B);
    }

    void printTree(int nodeID, int depth = 0) {
        for (int i = 0; i < depth; i++) {
            cout << "| ";
        }
        cout << "R" << nodeID << endl;

        for (int childID : children[nodeID]) {
            printTree(childID, depth + 1);
        }

    }

    void printChildren() {
        for(int i = 0; i < children.size(); i++) {
            cout << "R" << i << " has " << children[i].size() << " children: ";
            for(int j = 0; j < children[i].size(); j++) {
                cout << "R" << children[i][j] << "(jp=[";
                for(int k = 0; k < joinPos[i][j].size() - 1; k++) {
                    cout << joinPos[i][j][k] << ", ";
                }
                cout << joinPos[i][j][joinPos[i][j].size() - 1] << "]); ";
            }
            cout << endl;
        }
    }

    void print(){
        printTree(root);
    }

    // [AJB] dump full JoinTree diagnostics
    void ajb_dump_stats() {
        ajb_jt_stats.dump();
#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_STATE][JoinTree] root=%d, %zu relations, %zu cache arrays\n",
                root, relation.size(), cache.size());
        // bound cache stats
        fprintf(stderr, "[AJB_STATE][JoinTree] bound_cache: %zu entries, hits=%lld misses=%lld\n",
                ajb_bound_cache.size(), ajb_bcache_hits, ajb_bcache_misses);
        // cache occupancy per relation
        for(size_t i = 0; i < cache.size(); i++) {
            if(!cache[i].empty())
                fprintf(stderr, "[AJB_STATE][JoinTree]   cache[R%zu]: %zu entries\n", i, cache[i].size());
        }
#endif
    }

    // [AJB] reset all JoinTree diagnostic counters
    static void ajb_reset_stats() {
        ajb_jt_stats.reset();
    }
};