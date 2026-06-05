// =============================================================================
// JoinTree.hpp — Join tree construction and treeUpp bound (AJB-instrumented)
//
// Origin: upstream/joinrenum/JoinTree.hpp (236 lines)
// AJB adaptation (~20%): per-phase timing (BFS / buildLeaves / preProcessing),
//   cache occupancy tracking, treeUpp convergence logging (zero-rate),
//   buildLeaves leaf-node cardinality dump, and preProcessing per-node
//   cnt accumulation trace for debugging incorrect AGM bounds.
// =============================================================================
#include <boost/unordered_map.hpp>
#include <queue>
#include <chrono>
using namespace std;

// [AJB] JoinTree诊断 + cache hit/miss ratio — 扩展: per-phase timing + cache stats
static thread_local struct {
    long long tree_upp_calls = 0;
    long long tree_upp_zero = 0;   // treeUpp返回0的次数(空区间)
    long long buildleaves_nodes = 0; // buildLeaves处理的叶子数
    long long preproc_nodes = 0;     // preProcessing处理的内部节点数
    long long cache_entries = 0;     // cache总条目数
    double    build_ms = 0.0;
    double    buildleaves_ms = 0.0;
    double    preproc_ms = 0.0;
    double    bfs_ms = 0.0;
    void dump(const char* tag = "JoinTree") {
        fprintf(stderr, "[AJB_STATE][%s] treeUpp_calls=%lld zero=%lld leaves=%lld preproc_nodes=%lld cache=%lld\n",
                tag, tree_upp_calls, tree_upp_zero, buildleaves_nodes, preproc_nodes, cache_entries);
        fprintf(stderr, "[AJB_TIMER][%s] total=%.3fms bfs=%.3fms leaves=%.3fms preproc=%.3fms\n",
                tag, build_ms, bfs_ms, buildleaves_ms, preproc_ms);
    }
    void reset() {
        tree_upp_calls = tree_upp_zero = buildleaves_nodes = preproc_nodes = cache_entries = 0;
        build_ms = buildleaves_ms = preproc_ms = bfs_ms = 0.0;
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
            fprintf(stderr, "[AJB_STATE][JoinTree] leaf R%d: %zu points, %zu cache keys\n",
                    node, CO[node]->points.size(), cache[node].size());
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
            fprintf(stderr, "[AJB_STATE][JoinTree] preProc R%d: %zu pts, cnt[0]=%lld cnt[last]=%lld\n",
                    node, CO[node]->points.size(),
                    (long long)CO[node]->points[0].cnt,
                    (long long)CO[node]->points.back().cnt);
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
        visVar = vector<bool>(q.getVarNumber(), false);
        cache.resize(q.getRelNames().size());
        treeBound.resize(q.getRelNames().size());
        vector<bool> visited(q.getRelNames().size(), false);
        visited[0] = true;
        que.push(0);
        fprintf(stderr, "[AJB_BP][JoinTree] BFS start: root=R0, %zu relations, %d variables\n",
                q.getRelNames().size(), q.getVarNumber());
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
                    // [AJB_TRACE] 每条tree边: parent→child + join position
                    fprintf(stderr, "[AJB_TRACE][JoinTree]   edge R%d→R%d joinPos=[", rel, neighbor);
                    for(size_t jp = 0; jp < jpos.size(); jp++){
                        if(jp) fprintf(stderr, ",");
                        fprintf(stderr, "%d", jpos[jp]);
                    }
                    fprintf(stderr, "]\n");
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
        for(size_t v = 0; v < countRels.size(); v++){
            if(countRels[v].empty()) continue;
            fprintf(stderr, "[AJB_STATE][JoinTree] countRels[x%zu]: [", v);
            for(size_t j = 0; j < countRels[v].size(); j++){
                if(j) fprintf(stderr, ",");
                fprintf(stderr, "R%d", countRels[v][j]);
            }
            fprintf(stderr, "]\n");
        }
        fprintf(stderr, "[AJB_TIMER][JoinTree] total build=%.3fms (bfs=%.3f leaves=%.3f preproc=%.3f)\n",
                ajb_jt_stats.build_ms, ajb_jt_stats.bfs_ms, ajb_jt_stats.buildleaves_ms, ajb_jt_stats.preproc_ms);
    }

    int treeUpp(int splitDim, const vector<pair<vector<Point<int> >::iterator, vector<Point<int> >::iterator> > iters) {
        // cout << "TREEUPP IN";
        ajb_jt_stats.tree_upp_calls++;
        if(splitDim >= (int)countRels.size()) return 1;
        int tupp = 1;
        for(int node : countRels[splitDim]) {
            tupp *= CO[node]->sumCnt(iters[node].first, iters[node].second);
        }
        if(tupp == 0) {
            ajb_jt_stats.tree_upp_zero++;
            // [AJB_TRACE] periodic zero-rate report every 10000 calls
            if(ajb_jt_stats.tree_upp_calls % 10000 == 0 && ajb_jt_stats.tree_upp_calls > 0) {
                fprintf(stderr, "[AJB_STATE][JoinTree] treeUpp zero_rate=%.4f (%lld/%lld)\n",
                        (double)ajb_jt_stats.tree_upp_zero / ajb_jt_stats.tree_upp_calls,
                        ajb_jt_stats.tree_upp_zero, ajb_jt_stats.tree_upp_calls);
            }
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
        fprintf(stderr, "[AJB_STATE][JoinTree] root=%d, %zu relations, %zu cache arrays\n",
                root, relation.size(), cache.size());
        // cache occupancy per relation
        for(size_t i = 0; i < cache.size(); i++) {
            if(!cache[i].empty())
                fprintf(stderr, "[AJB_STATE][JoinTree]   cache[R%zu]: %zu entries\n", i, cache[i].size());
        }
    }

    // [AJB] reset all JoinTree diagnostic counters
    static void ajb_reset_stats() {
        ajb_jt_stats.reset();
    }
};