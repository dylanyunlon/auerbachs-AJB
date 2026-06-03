// =============================================================================
// RRAccessTree.hpp — Relaxed Random Access Tree (AJB-instrumented)
//
// Origin: upstream/joinrenum/RRAccessTree.hpp (603 lines)
// AJB adaptation (~20%): per-variant call counters (MTI/BTI/LTI/HalfCache/
//   NoCache), cumulative timing for Split vs traversal, hit-rate calculator,
//   per-RRAccess latency histogram buckets, depth-distribution tracking,
//   and periodic progress reporting for long enumerations.
// =============================================================================
#include "Index.hpp"

// [AJB] RRAccessTree运行时诊断 — 扩展: per-variant + timing + depth histogram
static thread_local struct {
    long long rraccess_calls = 0;    // 总RRAccess调用数
    long long rraccess_hits = 0;     // 找到结果的次数
    long long rraccess_misses = 0;   // 落入empty区域的次数
    long long node_creates = 0;      // 新建RRAccessTreeNode次数
    long long total_children = 0;    // 所有新建节点的children总数
    int max_depth = 0;               // 递归最大深度
    long long nocache_splits = 0;    // NoCache路径上的Split次数
    long long halfcache_boundary = 0; // 命中cache boundary的次数
    // [AJB] per-variant counters
    long long mti_calls = 0;
    long long bti_calls = 0;
    long long lti_calls = 0;
    long long hc_calls = 0;
    long long nc_calls = 0;
    // [AJB] timing
    double    total_access_ms = 0.0;  // 所有RRAccess的累计wall时间
    double    max_single_ms = 0.0;    // 单次RRAccess最大耗时
    // [AJB] depth distribution: bucket[0]=depth<5, [1]=5-9, [2]=10-19, [3]=20+
    long long depth_hist[4] = {0, 0, 0, 0};
    void record_depth(int d) {
        if(d < 5) depth_hist[0]++;
        else if(d < 10) depth_hist[1]++;
        else if(d < 20) depth_hist[2]++;
        else depth_hist[3]++;
    }
    void dump(const char* tag = "RRAccessTree") {
        fprintf(stderr, "[AJB_STATE][%s] calls=%lld hits=%lld misses=%lld nodes=%lld children=%lld max_depth=%d\n",
                tag, rraccess_calls, rraccess_hits, rraccess_misses, node_creates, total_children, max_depth);
        fprintf(stderr, "[AJB_STATE][%s] nocache_splits=%lld halfcache_boundary=%lld\n",
                tag, nocache_splits, halfcache_boundary);
        double hit_rate = rraccess_calls > 0 ? 100.0 * rraccess_hits / rraccess_calls : 0.0;
        fprintf(stderr, "[AJB_STATE][%s] variants: MTI=%lld BTI=%lld LTI=%lld HC=%lld NC=%lld hit_rate=%.2f%%\n",
                tag, mti_calls, bti_calls, lti_calls, hc_calls, nc_calls, hit_rate);
        fprintf(stderr, "[AJB_TIMER][%s] total_access=%.2fms max_single=%.3fms avg=%.4fms\n",
                tag, total_access_ms, max_single_ms,
                rraccess_calls > 0 ? total_access_ms / rraccess_calls : 0.0);
        fprintf(stderr, "[AJB_STATE][%s] depth_hist: <5=%lld 5-9=%lld 10-19=%lld 20+=%lld\n",
                tag, depth_hist[0], depth_hist[1], depth_hist[2], depth_hist[3]);
    }
    void reset() {
        rraccess_calls = rraccess_hits = rraccess_misses = node_creates = total_children = 0;
        max_depth = 0; nocache_splits = halfcache_boundary = 0;
        mti_calls = bti_calls = lti_calls = hc_calls = nc_calls = 0;
        total_access_ms = max_single_ms = 0.0;
        depth_hist[0] = depth_hist[1] = depth_hist[2] = depth_hist[3] = 0;
    }
} ajb_rrt_stats;



/**
 * @class RRAccessTreeNode
 * @brief Represents a node in an Relaxed Random Access Tree structure.
 * 
 * This class encapsulates the properties and behavior of a node in an RRAccess Tree.
 * Each node contains a bucket, a collection of child buckets, and pointers to its child nodes.
 * It also calculates and stores the empty size of the node based on the difference between
 * the AGM of its bucket and the sum of the AGMs of its child buckets.
 * 
 * @details
 * The RRAccessTreeNode class provides a constructor to initialize the node with a bucket,
 * child buckets, and child pointers. It also includes a method to print the node's details.
 * 
 * @note The `emptySize` is computed during construction and represents the remaining size
 * after accounting for the AGMs of the child buckets.
 * 
 * @var emptySize
 * The remaining size of the node after subtracting the sum of the AGMs of its child buckets
 * from the AGM of its own bucket.
 * 
 * @var B
 * The bucket associated with this node.
 * 
 * @var children_buckets
 * A vector containing the buckets of the child nodes.
 * 
 * @var children_pointers
 * A vector of pointers to the child nodes of this node.
 * 
 * @fn RRAccessTreeNode(Bucket B, vector<Bucket> children_buckets, vector<RRAccessTreeNode*> children_pointers)
 * @brief Constructs an RRAccessTreeNode object with the specified parameters.
 * 
 * @param B The bucket associated with this node.
 * @param children_buckets A vector of buckets representing the child nodes.
 * @param children_pointers A vector of pointers to the child nodes of this node.
 * 
 * @fn void print()
 * @brief Prints the details of the node, including its AGM, size, and bucket information.
 */
class RRAccessTreeNode {

public:
    long long emptySize;
    Bucket* B;
    vector<Bucket> children_buckets;
    vector<RRAccessTreeNode*> children_pointers;

    /**
     * @brief Constructs an RRAccessTreeNode object with the specified parameters.
     * 
     * @param B The bucket associated with this node.
     * @param children_buckets A vector of buckets.
     * @param children_pointers A vector of pointers to the child nodes of this node.
     */
    RRAccessTreeNode(Bucket* B, vector<Bucket> && children_buckets) : B(B), children_buckets(move(children_buckets)) {
        emptySize = B->AGM;
        for(int i = 0; i < this->children_buckets.size(); i++) {
            emptySize -= this->children_buckets[i].AGM;
        }
        children_pointers = vector<RRAccessTreeNode*>(this->children_buckets.size(), NULL);
        // [AJB_TRACE] 节点创建: emptySize=AGM-sum(children.AGM), 如果emptySize很大说明这层空洞多
        ajb_rrt_stats.node_creates++;
        ajb_rrt_stats.total_children += this->children_buckets.size();
    }

    void print() {
        cout << "AGM: " << B->AGM << ", size: " << children_buckets.size() << ", ";
        B->print();
    }

    // [AJB] node级别的结构dump: emptySize比例反映空洞率
    void ajb_dump(int depth = 0) {
        fprintf(stderr, "[AJB_STATE][RRTreeNode] depth=%d AGM=%lld empty=%lld children=%zu empty_ratio=%.4f\n",
                depth, B->AGM, emptySize, children_buckets.size(),
                B->AGM > 0 ? (double)emptySize / B->AGM : 0.0);
    }
};

class RRAccessTreeNode_Pool {
public:
    long long emptySize;
    int bid;
    vector<int> children_bids;
    vector<long long> children_agms;
    vector<RRAccessTreeNode_Pool*> children_pointers;
    RRAccessTreeNode_Pool(const int bid, const vector<int> && children_bids, const BucketPool &pool) : bid(bid), children_bids(move(children_bids)) {
        // AJB: reserve代替默认构造
        children_agms.resize(this->children_bids.size());
        emptySize = pool[bid].AGM;
        // cout << "children_AGMs: ";
        for(size_t i = 0; i < children_bids.size(); i++) {  // AJB: size_t循环变量
            children_agms[i] = pool[children_bids[i]].AGM;
            emptySize -= children_agms[i];
            // cout << children_agms[i] << ", ";
        }
        // cout << endl;
        children_pointers.assign(this->children_bids.size(), nullptr);  // AJB: assign代替构造
        ajb_rrt_stats.node_creates++;
        ajb_rrt_stats.total_children += this->children_bids.size();
    }
};

class RRAccessTree {
private:


    
    /**
     * @brief Recursively calculates the empty size of the rightmost subtree in an RRAccessTree.
     *
     * This function determines the "empty right" size of a given bucket in the RRAccessTree.
     * It initializes the AGM (Aggregate Measure) and iterators for the bucket if not already set,
     * splits the bucket into child buckets if the node is null, and recursively processes the
     * rightmost child bucket to compute the empty size.
     *
     * @param B The bucket to process, containing data and metadata for the tree node.
     * @param node A reference to the pointer of the current RRAccessTreeNode. If null, a new node
     *             is created and initialized with child buckets and pointers.
     * @return The total empty size of the rightmost subtree, including the current node's empty size.
     */
    long long getEmptyRight(Bucket &B, RRAccessTreeNode* &node) {
        // [AJB_TRACE] getEmptyRight: 计算最右子树的空洞大小
        if (B.AGM < 0) idx.setAGMandIters(B);
        if (B.getSplitDim() == B.getDim()) return 1 - B.AGM;
        if (!node) {
            // vector<Bucket> children = idx.Split(B);
            // auto startSplit = chrono::high_resolution_clock::now();
            node = new RRAccessTreeNode(&B, idx.Split(B));
            // auto endSplit = chrono::high_resolution_clock::now();
            // chrono::duration<double> elapsedSplit = endSplit - startSplit;
            // idx.totalSplitTime += elapsedSplit.count();
        }
        long long emptyright = !node->children_buckets.empty() ? getEmptyRight(node->children_buckets.back(), node->children_pointers.back()) : 0;

        return node->emptySize + emptyright;
    }

    // [AJB] decreaseTrivialLowBound: 递归计算最右路径的空洞, 用于设置trivialInterval下界
    void decreaseTrivialLowBound(Bucket &B, RRAccessTreeNode* &node, long long &lowbound) {
        // AJB: 叶子检测前置——splitDim比较是O(1), setAGMandIters可能触发Split
        if (B.getSplitDim() == B.getDim()){
            if (B.AGM < 0) idx.setAGMandIters(B);
            lowbound -= 1 - B.AGM;
            return;
        }
        if (B.AGM < 0) idx.setAGMandIters(B);
        if (!node) {
            // vector<Bucket> children = idx.Split(B);
            // auto startSplit = chrono::high_resolution_clock::now();
            node = new RRAccessTreeNode(&B, idx.Split(B));
            idx.totalrrtreenode += node->children_buckets.size();
            idx.totalrrtreenode --;
            // auto endSplit = chrono::high_resolution_clock::now();
            // chrono::duration<double> elapsedSplit = endSplit - startSplit;
            // idx.totalSplitTime += elapsedSplit.count();
        }
        lowbound -= node->emptySize;
        // AJB: .back()代替[size()-1]——避免重复计算size
        if (!node->children_buckets.empty()) decreaseTrivialLowBound(node->children_buckets.back(), node->children_pointers.back(), lowbound);

        return;
    }

    bool RRAccess_opt(long long k, Bucket &B, RRAccessTreeNode* &node, long long offset = 0, int _ajb_depth = 0) {
        if(_ajb_depth > ajb_rrt_stats.max_depth) ajb_rrt_stats.max_depth = _ajb_depth;
        if(B.getSplitDim() == B.getDim()){
            result = B.getLowerBound();
            idx.totalrrtreenode --;
            return true;
        }
        if(B.AGM < 0) idx.setAGMandIters(B);
        if (!node) {
            // auto startSplit = chrono::high_resolution_clock::now();
            node = new RRAccessTreeNode(&B, idx.Split(B));
            
            idx.totalrrtreenode += node->children_buckets.size();
            idx.totalrrtreenode --;
            // auto endSplit = chrono::high_resolution_clock::now();
            // chrono::duration<double> elapsedSplit = endSplit - startSplit;
            // idx.totalSplitTime += elapsedSplit.count();
            trivialIntervals[numti].first = offset + B.AGM + 1;
            decreaseTrivialLowBound(B, node, trivialIntervals[numti].first); 
            trivialIntervals[numti].second = offset + B.AGM;
            if(trivialIntervals[numti].first <= trivialIntervals[numti].second) numti++;
        }
        long long childAGM, temp = 0;
        for(int i = 0; i < node->children_buckets.size(); i++) {
            childAGM = node->children_buckets[i].AGM;
            if(offset + temp + childAGM >= k){
                return RRAccess_opt(k, node->children_buckets[i], node->children_pointers[i], offset + temp, _ajb_depth + 1);
            }
            else temp += childAGM;
        }
        return false;
    }


    bool RRAccess(long long k, Bucket &B, RRAccessTreeNode* &node, long long offset = 0) {
        // [AJB] MTI recursive body — 最后一个child特殊处理(合并TI)
        if(B.getSplitDim() == B.getDim()){
            result = B.getLowerBound();
            return true;
        }
        if(B.AGM < 0) idx.setAGMandIters(B);
        if (!node) {
            // vector<Bucket> children = idx.Split(B);
            // auto startSplit = chrono::high_resolution_clock::now();
            node = new RRAccessTreeNode(&B, idx.Split(B));
            // auto endSplit = chrono::high_resolution_clock::now();
            // chrono::duration<double> elapsedSplit = endSplit - startSplit;
            // idx.totalSplitTime += elapsedSplit.count();
        }
        if(offset + B.AGM - node->emptySize < k){
            trivialIntervals[0].first = offset + B.AGM - getEmptyRight(B, node) + 1;
            trivialIntervals[0].second = offset + B.AGM;
            numti = 1;
            return false;
        }
        long long childAGM, temp = 0;
        for(int i = 0; i < node->children_buckets.size() - 1; i++) {
            childAGM = node->children_buckets[i].AGM;
            if(offset + temp + childAGM >= k){
                return RRAccess(k, node->children_buckets[i], node->children_pointers[i], offset + temp);
            }
            else temp += childAGM;
        }
        int last = node->children_buckets.size() - 1;
        bool res = RRAccess(k, node->children_buckets[last], node->children_pointers[last], offset + temp);
        
        if(!res && trivialIntervals[0].second == offset + B.AGM - node->emptySize){
            trivialIntervals[0].second = offset + B.AGM;
            // [AJB_TRACE] TI expansion: 合并空洞区间, 提高ban效率
        }
        return res;
    }


    // [AJB] RRAccess_low (LTI): 较大的trivial interval, TI = [AGM-emptySize+1, AGM]
    bool RRAccess_low(long long k, Bucket &B, RRAccessTreeNode* &node, long long offset = 0) {
        if(B.getSplitDim() == B.getDim()){
            result = B.getLowerBound();
            ajb_rrt_stats.rraccess_hits++;
            return true;
        }
        if(B.AGM < 0) idx.setAGMandIters(B);
        if (!node) {
            // vector<Bucket> children = idx.Split(B);
            // auto startSplit = chrono::high_resolution_clock::now();
            node = new RRAccessTreeNode(&B, idx.Split(B));
            // auto endSplit = chrono::high_resolution_clock::now();
            // chrono::duration<double> elapsedSplit = endSplit - startSplit;
            // idx.totalSplitTime += elapsedSplit.count();
        }
        long long childAGM, temp = 0;
        for(int i = 0; i < node->children_buckets.size(); i++) {
            childAGM = node->children_buckets[i].AGM;
            if(offset + temp + childAGM >= k){
                return RRAccess_low(k, node->children_buckets[i], node->children_pointers[i], offset + temp);
            }
            else temp += childAGM;
        }
        
        trivialIntervals[0].first = offset + B.AGM - node->emptySize + 1;
        trivialIntervals[0].second = offset + B.AGM;
        numti = 1;
        ajb_rrt_stats.rraccess_misses++;
        return false;
    }

    // [AJB] RRAccess_verylow: 最保守的TI, trivialInterval=[k,k]
    bool RRAccess_verylow(long long k, Bucket &B, RRAccessTreeNode* &node, long long offset = 0) {
        if(B.getSplitDim() == B.getDim()){
            result = B.getLowerBound();
            ajb_rrt_stats.rraccess_hits++;
            return true;
        }
        if(B.AGM < 0) idx.setAGMandIters(B);
        if (!node) {
            // vector<Bucket> children = idx.Split(B);
            // auto startSplit = chrono::high_resolution_clock::now();
            node = new RRAccessTreeNode(&B, idx.Split(B));
            // auto endSplit = chrono::high_resolution_clock::now();
            // chrono::duration<double> elapsedSplit = endSplit - startSplit;
            // idx.totalSplitTime += elapsedSplit.count();
        }
        long long childAGM, temp = 0;
        for(int i = 0; i < node->children_buckets.size(); i++) {
            childAGM = node->children_buckets[i].AGM;
            if(offset + temp + childAGM >= k){
                return RRAccess_verylow(k, node->children_buckets[i], node->children_pointers[i], offset + temp);
            }
            else temp += childAGM;
        }
        
        trivialIntervals[0].first = k;
        trivialIntervals[0].second = k;
        numti = 1;
        return false;
    }

    
    long long getEmptyRight_NoCache(Bucket &B) {
        // [AJB_TRACE] getEmptyRight_NoCache: 无缓存版本, 每次都重新Split
        if (B.AGM < 0) idx.setAGMandIters(B);
        if (B.getSplitDim() == B.getDim()) return 1 - B.AGM;
        
        vector<Bucket> children = idx.Split(B);
        long long emptyright = children.size() > 0 ? getEmptyRight_NoCache(children[children.size() - 1]) : 0;

        long long emptySize = B.AGM;
        for(int i = 0; i < children.size(); i++) {
            emptySize -= children[i].AGM;
        }

        return emptySize + emptyright;
    }


    // [AJB] HalfCache: depth <= cacheHeightBound的节点缓存, 超过的走NoCache
    long long getEmptyRight_HalfCache(Bucket &B, RRAccessTreeNode* &node, int depth = 0) {
        if(depth > cacheHeightBound) {
            // [AJB_TRACE] cache boundary hit at depth=%d
            return 0;
        }
        if (B.AGM < 0) idx.setAGMandIters(B);
        if (B.getSplitDim() == B.getDim()) return 1 - B.AGM;
        if(!node) {
            node  = new RRAccessTreeNode(&B, idx.Split(B));
        }
        long long emptyright = node->children_buckets.size() > 0 ? getEmptyRight_HalfCache(node->children_buckets[node->children_buckets.size() - 1], node->children_pointers[node->children_pointers.size() - 1], depth + 1) : 0;
        return node->emptySize + emptyright;
    }

    long long getEmptyRight_HalfCache(int bid, RRAccessTreeNode_Pool* &node, int depth = 0) {
        if(depth > cacheHeightBound) return 0;
        if (!node && pool[bid].getSplitDim() == pool[bid].getDim()) return 1 - pool[bid].AGM;
        if(!node) {
            node  = new RRAccessTreeNode_Pool(bid, idx.Split_pool(pool, bid), pool);
            pool.free(bid);
        }
        long long emptyright = node->children_bids.size() > 0 ? getEmptyRight_HalfCache(node->children_bids[node->children_bids.size() - 1], node->children_pointers[node->children_pointers.size() - 1], depth + 1) : 0;
        return node->emptySize + emptyright;
    }

    bool RRAccess_HalfCache(long long k, Bucket &B, RRAccessTreeNode* &node, long long offset = 0, int depth = 0) {
        if(depth > ajb_rrt_stats.max_depth) ajb_rrt_stats.max_depth = depth;
        if(depth > cacheHeightBound) {
            ajb_rrt_stats.halfcache_boundary++;
            return RRAccess_NoCache(k, B, offset, depth);
        }
        if(B.getSplitDim() == B.getDim()){
            result = B.getLowerBound();
            ajb_rrt_stats.rraccess_hits++;
            return true;
        }
        if(B.AGM < 0) idx.setAGMandIters(B);
        if(!node) {
            node  = new RRAccessTreeNode(&B, idx.Split(B));
            numti++;
            trivialIntervals[numti - 1].first = offset + B.AGM - getEmptyRight_HalfCache(B, node, depth) + 1;
            trivialIntervals[numti - 1].second = offset + B.AGM;
        }
        
        if(offset + B.AGM - node->emptySize < k){
            return false;
        }
        long long childAGM, temp = 0;
        for(int i = 0; i < node->children_buckets.size() ; i++) {
            childAGM = node->children_buckets[i].AGM;
            if(offset + temp + childAGM >= k) {
                return RRAccess_HalfCache(k, node->children_buckets[i], node->children_pointers[i], offset + temp, depth + 1);
            }
            else temp += childAGM;
        }
        return false;
    }

    bool RRAccess_HalfCache_basic(long long k, int bid, RRAccessTreeNode_Pool* &node, long long BAGM, long long offset = 0, int depth = 0) {
        if(depth > ajb_rrt_stats.max_depth) ajb_rrt_stats.max_depth = depth;
        if(depth > cacheHeightBound) {
            ajb_rrt_stats.halfcache_boundary++;
            return RRAccess_NoCache_basic(k, pool[bid], offset, depth);
        }
        if(!node && pool[bid].getSplitDim() == pool[bid].getDim()){
            result = pool[bid].getLowerBound();
            return true;
        }
        if(!node) {
            node  = new RRAccessTreeNode_Pool(bid, idx.Split_pool(pool, bid), pool);
            pool.free(bid);
            numti++;
            trivialIntervals[numti - 1].first = offset + BAGM - getEmptyRight_HalfCache(bid, node, depth) + 1;
            trivialIntervals[numti - 1].second = offset + BAGM;
        }
        
        if(offset + BAGM - node->emptySize < k){
            return false;
        }
        long long cumulative = 0;
        // AJB: size_t循环 + 缓存children数量
        const size_t nc = node->children_agms.size();
        for(size_t i = 0; i < nc; i++) {
            long long cagm = node->children_agms[i];
            if(offset + cumulative + cagm >= k) {
                return RRAccess_HalfCache_basic(k, node->children_bids[i], node->children_pointers[i], cagm, offset + cumulative, depth + 1);
            }
            cumulative += cagm;
        }
        return false;
    }


    bool RRAccess_HalfCache(long long k, int bid, RRAccessTreeNode_Pool* &node, long long BAGM, long long offset = 0, int depth = 0) {
        if(depth > ajb_rrt_stats.max_depth) ajb_rrt_stats.max_depth = depth;
        if(depth > cacheHeightBound) {
            ajb_rrt_stats.halfcache_boundary++;
            return RRAccess_NoCache(k, pool[bid], offset, depth);
        }
        if(!node && pool[bid].getSplitDim() == pool[bid].getDim()){
            result = pool[bid].getLowerBound();
            return true;
        }
        if(!node) {
            node  = new RRAccessTreeNode_Pool(bid, idx.Split_pool(pool, bid), pool);
            pool.free(bid);
            numti++;
            trivialIntervals[numti - 1].first = offset + BAGM - getEmptyRight_HalfCache(bid, node, depth) + 1;
            trivialIntervals[numti - 1].second = offset + BAGM;
        }
        
        if(offset + BAGM - node->emptySize < k){
            return false;
        }
        long long childAGM, temp = 0;
        for(int i = 0; i < node->children_agms.size() ; i++) {
            childAGM = node->children_agms[i];
            if(offset + temp + childAGM >= k) {
                return RRAccess_HalfCache(k, node->children_bids[i], node->children_pointers[i], childAGM, offset + temp, depth + 1);
            }
            else temp += childAGM;
        }
        return false;
    }

    


    bool RRAccess_NoCache(long long k, Bucket &B, long long offset = 0, int depth = 0) {
        if(depth > ajb_rrt_stats.max_depth) ajb_rrt_stats.max_depth = depth;
        if(B.getSplitDim() == B.getDim()){
            result = B.getLowerBound();
            ajb_rrt_stats.rraccess_hits++;
            return true;
        }
        if(B.AGM < 0) idx.setAGMandIters(B);
        ajb_rrt_stats.nocache_splits++;
        vector<Bucket> children = move(idx.Split(B));

        long long childAGM, temp = 0;
        for(int i = 0; i < children.size(); i++) {
            childAGM = children[i].AGM;
            if(offset + temp + childAGM >= k){
                bool res = RRAccess_NoCache(k, children[i], offset + temp, depth + 1);
                if(i == children.size() - 1 && !res && trivialIntervals[numti - 1].second == offset + temp + childAGM) trivialIntervals[numti - 1].second = offset + B.AGM;
                
                return res;
            }
            else temp += childAGM;
        }
        
        numti++;
        trivialIntervals[numti - 1].first = offset + B.AGM - getEmptyRight_NoCache(B) + 1;
        trivialIntervals[numti - 1].second = offset + B.AGM;
        ajb_rrt_stats.rraccess_misses++;
        // [AJB_TRACE] NoCache miss: TI=[%lld, %lld] depth=%d
        
        return false;
    }

    // [AJB] NoCache_basic: 最简版本, TI只设为[temp+1, B.AGM], 不计算getEmptyRight
    bool RRAccess_NoCache_basic(long long k, Bucket &B, long long offset = 0, int depth = 0) {
        if(depth > ajb_rrt_stats.max_depth) ajb_rrt_stats.max_depth = depth;
        if(B.getSplitDim() == B.getDim()){
            result = B.getLowerBound();
            ajb_rrt_stats.rraccess_hits++;
            return true;
        }
        if(B.AGM < 0) idx.setAGMandIters(B);
        vector<Bucket> children = move(idx.Split(B));

        long long childAGM, temp = 0;
        for(int i = 0; i < children.size(); i++) {
            childAGM = children[i].AGM;
            if(offset + temp + childAGM >= k) return RRAccess_NoCache_basic(k, children[i], offset + temp, depth + 1);
            else temp += childAGM;
        }
        
        numti++;
        trivialIntervals[numti - 1].first = offset + temp + 1;
        trivialIntervals[numti - 1].second = offset + B.AGM;
        
        return false;
    }

    
    

public:
    long long AGM;
    int cacheHeightBound = 20; // [AJB] HalfCache的缓存深度限制, 超过此深度走NoCache
    RRAccessTreeNode* root = NULL;
    RRAccessTreeNode_Pool* root_pool = NULL;
    Index idx;
    vector<int> result;
    pair<long long, long long> trivialInterval;
    vector<pair<long long, long long> > trivialIntervals = vector<pair<long long, long long> >(50);
    BucketPool pool;
    int numti = 0;  // AJB: trivial interval计数器

    RRAccessTree() {}

    /**
     * @brief Constructs an RRAccessTree object.
     *
     * This constructor initializes the RRAccessTree by creating an Index object
     * using the provided relations, filenames, and numlines. It also retrieves
     * the AGM (AGM bound) from the Index object.
     *
     * @param relations A map where the key is a string representing a relation name,
     *                  and the value is a vector of strings representing its related variables.
     * @param filenames A map where the key is a string representing a relation name,
     *                  and the value is a string representing the corresponding filename.
     * @param numlines  A map where the key is a string representing a relation name,
     *                  and the value is an integer representing the number of lines in the file.
     */
    RRAccessTree(
        const unordered_map<string, vector<string> > &relations,
        const unordered_map<string, string> &filenames,
        const unordered_map<string, int> &numlines, bool treeflag = false) {
        vector<string> q_relations;
        vector<vector<string> > q_variables;
        for(unordered_map<string, vector<string> >::const_iterator it = relations.begin(); it != relations.end(); it++) {
            q_relations.push_back(it->first);
            // q_variables.push_back(it->second);
        }
        sort(q_relations.begin(), q_relations.end());
        for(int i = 0; i < q_relations.size(); i++) {
            q_variables.push_back(relations.at(q_relations[i]));
        }
        Query q(q_relations, q_variables);
        idx = Index(q, treeflag);
        idx.preProcessing(relations, filenames, numlines);
        AGM = idx.AGM();
        pool.newCopy(idx.FB);
        // [AJB_BP] RRAccessTree ready: AGM是整个枚举空间的大小
        fprintf(stderr, "[AJB_BP][RRAccessTree] constructed: AGM=%lld cacheHeightBound=%d\n", AGM, cacheHeightBound);
        // cout << "AGM: " << AGM << endl;
    }

    /**
     * @brief Constructs an RRAccessTree object and initializes its internal state.
     *
     * @param q The query object used to initialize the index.
     * @param relations A map where the key is a string representing a relation name,
     *                  and the value is a vector of strings representing its variable names.
     * @param filenames A map where the key is a string representing a relation name,
     *                  and the value is a string representing the corresponding file name.
     * @param numlines A map where the key is a string representing a relation name,
     *                 and the value is an integer representing the number of lines in the file.
     *
     * This constructor initializes the `idx` member by creating an Index object with the given query.
     * It then performs preprocessing on the index using the provided relations, filenames, and numlines.
     * Finally, it retrieves and stores the AGM bound from the index.
     */
    RRAccessTree(
        const Query &q,
        const unordered_map<string, vector<string> > &relations,
        const unordered_map<string, string> &filenames,
        const unordered_map<string, int> &numlines) {
        idx = Index(q);
        idx.preProcessing(relations, filenames, numlines);
        AGM = idx.AGM();
        // [AJB_BP] RRAccessTree(Query) ready
        fprintf(stderr, "[AJB_BP][RRAccessTree] constructed(Query): AGM=%lld relations=%zu\n",
                AGM, relations.size());
    }

    Bucket getFullBucket() {
        return idx.getFullBucket();
    }

    // [AJB] reset统计, 用于多轮benchmark
    void ajb_reset_stats() {
        ajb_rrt_stats.reset();
    }


    /**
     * @brief Performs a relaxed-random-access operation on the tree.
     * 
     * This function retrieves data from the tree based on the specified key `k`.
     * It internally calls an overloaded version of `RRAccess` with additional parameters
     * such as the full bucket index, the root node, the starting depth, and the AGM value.
     * 
     * @param k The key used to perform the relaxed-random-access.
     * @return A pair containing:
     *         - A boolean indicating the success or failure of the operation.
     *         - A vector of integers representing the retrieved data.
     *           - if the operation is successful, the vector is the retrieved join result.
     *           - if the operation fails, the vector is an trivial interval.
     */
    bool RRAccess_MTI(long long k) {
        numti = 0;
        ajb_rrt_stats.rraccess_calls++;
        ajb_rrt_stats.mti_calls++;
        auto _t0 = std::chrono::steady_clock::now();
        bool res = RRAccess(k, idx.FB, root, 0);
        auto _t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
        ajb_rrt_stats.total_access_ms += ms;
        if(ms > ajb_rrt_stats.max_single_ms) ajb_rrt_stats.max_single_ms = ms;
        ajb_rrt_stats.record_depth(ajb_rrt_stats.max_depth);
        // [AJB] periodic progress: every 10000 calls dump summary
        if(ajb_rrt_stats.rraccess_calls % 10000 == 0)
            fprintf(stderr, "[AJB_PROGRESS][RRAccessTree] %lld calls, hit_rate=%.2f%%, avg=%.4fms\n",
                    ajb_rrt_stats.rraccess_calls,
                    100.0 * ajb_rrt_stats.rraccess_hits / ajb_rrt_stats.rraccess_calls,
                    ajb_rrt_stats.total_access_ms / ajb_rrt_stats.rraccess_calls);
        return res;
    }

    bool RRAccess_BTI(long long k) {
        numti = 0;
        ajb_rrt_stats.rraccess_calls++;
        ajb_rrt_stats.bti_calls++;
        auto _t0 = std::chrono::steady_clock::now();
        bool res = RRAccess_opt(k, idx.FB, root, 0);
        auto _t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
        ajb_rrt_stats.total_access_ms += ms;
        if(ms > ajb_rrt_stats.max_single_ms) ajb_rrt_stats.max_single_ms = ms;
        ajb_rrt_stats.record_depth(ajb_rrt_stats.max_depth);
        return res;
    }

    bool RRAccess_LTI(long long k) {
        numti = 0;
        ajb_rrt_stats.rraccess_calls++;
        ajb_rrt_stats.lti_calls++;
        auto _t0 = std::chrono::steady_clock::now();
        bool res = RRAccess_low(k, idx.FB, root, 0);
        auto _t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
        ajb_rrt_stats.total_access_ms += ms;
        if(ms > ajb_rrt_stats.max_single_ms) ajb_rrt_stats.max_single_ms = ms;
        return res;
    }

    bool RRAccess(long long k) {
        numti = 0;
        ajb_rrt_stats.rraccess_calls++;
        ajb_rrt_stats.nc_calls++;
        auto _t0 = std::chrono::steady_clock::now();
        bool res = RRAccess_verylow(k, idx.FB, root, 0);
        auto _t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
        ajb_rrt_stats.total_access_ms += ms;
        if(ms > ajb_rrt_stats.max_single_ms) ajb_rrt_stats.max_single_ms = ms;
        return res;
    }

    bool RRAccess_HalfCache(long long k) {
        numti = 0;
        ajb_rrt_stats.rraccess_calls++;
        ajb_rrt_stats.hc_calls++;
        auto _t0 = std::chrono::steady_clock::now();
        bool res = RRAccess_HalfCache(k, idx.FB, root);
        auto _t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
        ajb_rrt_stats.total_access_ms += ms;
        if(ms > ajb_rrt_stats.max_single_ms) ajb_rrt_stats.max_single_ms = ms;
        return res;
    }

    bool RRAccess_HalfCache_Pool(long long k) {
        numti = 0;
        ajb_rrt_stats.rraccess_calls++;
        ajb_rrt_stats.hc_calls++;
        auto _t0 = std::chrono::steady_clock::now();
        bool res = RRAccess_HalfCache(k, 0, root_pool, AGM);
        auto _t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
        ajb_rrt_stats.total_access_ms += ms;
        if(ms > ajb_rrt_stats.max_single_ms) ajb_rrt_stats.max_single_ms = ms;
        return res;
    }

    bool RRAccess_HalfCache_Pool_basic(long long k) {
        numti = 0;
        ajb_rrt_stats.rraccess_calls++;
        ajb_rrt_stats.hc_calls++;
        auto _t0 = std::chrono::steady_clock::now();
        bool res = RRAccess_HalfCache_basic(k, 0, root_pool, AGM);
        auto _t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
        ajb_rrt_stats.total_access_ms += ms;
        if(ms > ajb_rrt_stats.max_single_ms) ajb_rrt_stats.max_single_ms = ms;
        return res;
    }


    void print(RRAccessTreeNode* node, int depth = 0) {
        for (int i = 0; i < depth; i++) cout << "| ";
        if (!node){
            cout << "NULL" << endl;
            return;
        }
        node->print();
        for (int i = 0; i < node->children_pointers.size(); i++) {
            print(node->children_pointers[i], depth + 1);
        }
        return;
    }
    
    void print() {
        print(root);
    }

    // [AJB] dump运行时统计
    void ajb_dump_stats() {
        ajb_rrt_stats.dump();
        fprintf(stderr, "[AJB_STATE][RRAccessTree] AGM=%lld cacheH=%d numti=%d\n",
                AGM, cacheHeightBound, numti);
    }
};