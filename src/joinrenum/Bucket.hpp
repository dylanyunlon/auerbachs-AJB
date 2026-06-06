// [AJB] Bucket: d维区间 [lowerBound, upperBound]
// splitDim = 第一个lower!=upper的维度, 指导分裂方向
// AGM = 这个区间内join结果数的上界(由Index通过RangeTree赋值)
// M911: log-sum-exp volume, split balance ratio, leaf depth histogram
#include <cstdio>
#include <cmath>
#include<iostream>
#include<vector>
#include<cassert>
using namespace std;

// [AJB] M911: leaf depth histogram — 静态累积叶子节点深度分布
// 用于分析bucket tree的形状: 太深说明split效率低
static thread_local struct {
    long long depth_counts[32] = {};  // bucket[i] = depth==i的叶子数
    long long total_leaves = 0;
    long long total_volume_checks = 0;
    long long overflow_detections = 0;  // log-sum-exp触发次数
    void record_leaf_depth(int d) {
        total_leaves++;
        if(d >= 0 && d < 32) depth_counts[d]++;
    }
    void dump(const char* tag = "Bucket") {
#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_STATE][%s] leaf_depth_hist: total=%lld overflow_det=%lld vol_checks=%lld\n",
                tag, total_leaves, overflow_detections, total_volume_checks);
        fprintf(stderr, "[AJB_STATE][%s]   depths:", tag);
        for(int i = 0; i < 32; i++) {
            if(depth_counts[i] > 0) fprintf(stderr, " d%d=%lld", i, depth_counts[i]);
        }
        fprintf(stderr, "\n");
#endif
    }
    void reset() {
        for(int i = 0; i < 32; i++) depth_counts[i] = 0;
        total_leaves = total_volume_checks = overflow_detections = 0;
    }
} ajb_bucket_depth_stats;

class Bucket {
    private:
        // vector<pair<Bucket*, int> > children = {};

    public:
        int splitDim = 0;
        long long AGM = -1;
        vector<int> lowerBound;
        vector<int> upperBound;
        vector<pair<int, int> > iters;
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
        Bucket(const vector<int> &lowerBound, const vector<int> &upperBound, int splitDim = 0) : lowerBound(lowerBound), upperBound(upperBound), splitDim(splitDim) {
            // AJB: 用std::mismatch找第一个lb!=ub的维度
            // upstream: while循环逐个比较
            auto [itL, itU] = std::mismatch(
                this->lowerBound.begin() + splitDim, this->lowerBound.end(),
                this->upperBound.begin() + splitDim);
            this->splitDim = static_cast<int>(itL - this->lowerBound.begin());
        }

        void updateSplitDim() {
            // AJB: mismatch代替while循环
            auto [itL, itU] = std::mismatch(
                lowerBound.begin() + splitDim, lowerBound.end(),
                upperBound.begin() + splitDim);
            splitDim = static_cast<int>(itL - lowerBound.begin());
        }

        const vector<int>& getLowerBound() const {
            return lowerBound;
        }

        const vector<int>& getUpperBound() const {
            return upperBound;
        }

        void reset(const vector<int> &newLowerBound, const vector<int> &newUpperBound, int newSplitDim = 0) {
            // AJB: assign代替size-check+copy——assign自动处理大小差异
            lowerBound.assign(newLowerBound.begin(), newLowerBound.end());
            upperBound.assign(newUpperBound.begin(), newUpperBound.end());
            splitDim = newSplitDim;
            AGM = -1;
            updateSplitDim();
        }

        void reset(const Bucket &B) {
            // AJB: assign自动处理大小差异 + 修正upstream bug
            // (upstream: copy(B.lowerBound.begin, B.upperBound.end, ...) 混用了源迭代器)
            lowerBound.assign(B.lowerBound.begin(), B.lowerBound.end());
            upperBound.assign(B.upperBound.begin(), B.upperBound.end());
            splitDim = B.splitDim;
            AGM = -1;
            updateSplitDim();
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
            lowerBound[splitDim] = lower;
            upperBound[splitDim] = upper;
            updateSplitDim();
        }

        bool operator<(const Bucket& B) const {
            // AJB: 直接字典序比较——不等维度先按维度数排
            // upstream: size不等时打印错误+return false(不安全)
            if (lowerBound.size() != B.lowerBound.size())
                return lowerBound.size() < B.lowerBound.size();
            if (lowerBound != B.lowerBound) return lowerBound < B.lowerBound;
            return upperBound < B.upperBound;
        }

        Bucket replace(int lower, int upper) const {
            // AJB: 直接构造子bucket，避免先拷贝再修改
            Bucket nb;
            nb.lowerBound = lowerBound;
            nb.upperBound = upperBound;
            nb.lowerBound[splitDim] = lower;
            nb.upperBound[splitDim] = upper;
            nb.splitDim = splitDim;
            nb.AGM = -1;
            nb.updateSplitDim();
            return nb;
        }

        void print() const {
            cout << "Bucket( AGM = " << AGM << " ) : ";
            for(int i = 0; i < (int)lowerBound.size(); i++){
                cout << "[" << lowerBound[i] << ", " << upperBound[i] << "] ";
            }
            cout << endl;
        }

        void printIters(vector<vector<Point<int> >::iterator> begins) const {
            for(size_t i = 0; i < iters.size(); i++){
                cout << "R" << i << "(" << iters[i].first << ", " << iters[i].second << "), ";
            }
            cout << endl;
        }

    // [AJB] structured dump — 用于parse_ajb_trace.py解析
    // M911: 增强版 — 额外打印 volume、isLeaf 状态、每维 range 宽度
    void ajb_dump(const char* label = "") const {
#ifdef AJB_DEBUG
        long long vol = const_cast<Bucket*>(this)->ajb_volume();
        bool leaf = splitDim >= (int)lowerBound.size();
        fprintf(stderr, "[AJB_STATE][Bucket] %s d=%zu split=%d AGM=%lld vol=%lld leaf=%s range=[",
                label, lowerBound.size(), splitDim, AGM, vol, leaf ? "Y" : "N");
        for(size_t i = 0; i < lowerBound.size(); i++){
            if(i) fprintf(stderr, " ");
            fprintf(stderr, "%d..%d", lowerBound[i], upperBound[i]);
        }
        fprintf(stderr, "]\n");
        // 每维 range 宽度向量
        fprintf(stderr, "[AJB_STATE][Bucket] %s dim_widths=[", label);
        for(size_t i = 0; i < lowerBound.size(); i++){
            if(i) fprintf(stderr, ",");
            fprintf(stderr, "%d", upperBound[i] - lowerBound[i] + 1);
        }
        fprintf(stderr, "]\n");
#endif
    }

    // [AJB] M911: log-sum-exp volume防溢出
    // 原始: 逐维乘法 vol *= (upper-lower+1), 当维度高或范围大时会溢出
    // 改进: 在log域计算 sum(log(range_i)), 检测是否超过63 bit, 超过时返回saturated值
    // 同时记录每个维度的贡献占比, 用于识别哪个维度主导volume
    long long ajb_volume() const {
        ajb_bucket_depth_stats.total_volume_checks++;
        const size_t ndim = lowerBound.size();
        if(ndim == 0) return 0;
        
        // log-sum方法: 在对数域累加, 检测溢出风险
        double log_vol = 0.0;
        constexpr double LOG2_OVERFLOW = 62.0;  // 2^62 < LLONG_MAX
        
        // 每维 log 贡献 (用于 debug 输出)
        double dim_log[64] = {};  // 最多64维
        
        // 同时计算精确值(如果不溢出)和log域值
        long long exact_vol = 1;
        bool overflowed = false;
        
        for(size_t i = 0; i < ndim; i++){
            long long range = (long long)upperBound[i] - lowerBound[i] + 1;
            if(range <= 0) return 0;
            
            double log_range = std::log2(static_cast<double>(range));
            if(i < 64) dim_log[i] = log_range;
            log_vol += log_range;
            
            // 精确乘法溢出检测
            if(!overflowed) {
                if(exact_vol > 0 && range > 0 && exact_vol > (long long)4611686018427387903LL / range) {
                    overflowed = true;
                    ajb_bucket_depth_stats.overflow_detections++;
#ifdef AJB_DEBUG
                    // 打印溢出发生的维度和累积log值
                    fprintf(stderr, "[AJB_DEBUG][Bucket] volume overflow at dim %zu: log2_vol=%.2f range=%lld\n",
                            i, log_vol, range);
                    // 打印每个维度的贡献比例
                    fprintf(stderr, "[AJB_DEBUG][Bucket]   dim contributions (log2):");
                    for(size_t j = 0; j <= i && j < 64; j++) {
                        fprintf(stderr, " d%zu=%.2f(%.0f%%)", j, dim_log[j],
                                log_vol > 0 ? 100.0 * dim_log[j] / log_vol : 0.0);
                    }
                    fprintf(stderr, "\n");
#endif
                } else {
                    exact_vol *= range;
                }
            }
        }
        
        if(overflowed) {
            // 返回saturated值, 带符号标识溢出
            if(log_vol > LOG2_OVERFLOW) return (long long)4611686018427387903LL;  // ~2^62
            return static_cast<long long>(std::exp2(log_vol));
        }
        return exact_vol;
    }

    // [AJB] M911: split balance ratio — 评估一次split的质量
    // 返回min(left_agm, right_agm) / max(left_agm, right_agm)
    // ratio接近1.0说明split很均衡, 接近0.0说明一边几乎为空
    static double splitBalanceRatio(long long left_agm, long long right_agm) {
        if(left_agm <= 0 && right_agm <= 0) return 0.0;
        long long mx = std::max(left_agm, right_agm);
        long long mn = std::min(left_agm, right_agm);
        return mx > 0 ? static_cast<double>(mn) / mx : 0.0;
    }

    // [AJB] M911: dump split结果 — 分裂后打印两个子bucket的元素分布比
    void ajb_dump_split(const Bucket& left, const Bucket& right, int depth = 0) const {
#ifdef AJB_DEBUG
        double ratio = splitBalanceRatio(left.AGM, right.AGM);
        fprintf(stderr, "[AJB_DEBUG][Bucket] split@dim%d depth=%d: parent_AGM=%lld → L=%lld R=%lld balance=%.4f\n",
                splitDim, depth, AGM, left.AGM, right.AGM, ratio);
        // 打印每个维度的范围缩减
        for(size_t d = 0; d < lowerBound.size(); d++) {
            int parent_range = upperBound[d] - lowerBound[d];
            int left_range = left.upperBound[d] - left.lowerBound[d];
            int right_range = right.upperBound[d] - right.lowerBound[d];
            if(parent_range != left_range || parent_range != right_range) {
                fprintf(stderr, "[AJB_DEBUG][Bucket]   dim%zu: [%d,%d]→L[%d,%d] R[%d,%d]\n",
                        d, lowerBound[d], upperBound[d],
                        left.lowerBound[d], left.upperBound[d],
                        right.lowerBound[d], right.upperBound[d]);
            }
        }
#endif
    }

    // [AJB] M911: splitBucket — 在 splitDim 中点处分裂, 返回左右子bucket
    // 分裂前打印: 当前bucket的lower/upper边界向量、volume、元素数量
    // 分裂后打印: 子bucket的AGM分布比(如果已赋值)
    // 算法: midpoint = (lower + upper) / 2, left=[lower,mid], right=[mid+1,upper]
    pair<Bucket, Bucket> splitBucket(int depth = 0) const {
        assert(splitDim < (int)lowerBound.size() && "splitBucket on leaf bucket");
        int lo = lowerBound[splitDim];
        int hi = upperBound[splitDim];
        int mid = lo + (hi - lo) / 2;

#ifdef AJB_DEBUG
        // 分裂前: 打印边界向量和volume
        long long pre_vol = const_cast<Bucket*>(this)->ajb_volume();
        fprintf(stderr, "[AJB_DEBUG][Bucket] splitBucket PRE depth=%d splitDim=%d vol=%lld lo=[",
                depth, splitDim, pre_vol);
        for(size_t i = 0; i < lowerBound.size(); i++) {
            if(i) fprintf(stderr, ",");
            fprintf(stderr, "%d", lowerBound[i]);
        }
        fprintf(stderr, "] hi=[");
        for(size_t i = 0; i < upperBound.size(); i++) {
            if(i) fprintf(stderr, ",");
            fprintf(stderr, "%d", upperBound[i]);
        }
        fprintf(stderr, "] mid@dim%d=%d\n", splitDim, mid);
#endif

        Bucket left = replace(lo, mid);
        Bucket right = replace(mid + 1, hi);

#ifdef AJB_DEBUG
        // 分裂后: 打印子bucket体积和分布比
        long long lv = const_cast<Bucket&>(left).ajb_volume();
        long long rv = const_cast<Bucket&>(right).ajb_volume();
        double vol_ratio = (lv + rv > 0) ? static_cast<double>(std::min(lv, rv)) / std::max(lv, rv) : 0.0;
        fprintf(stderr, "[AJB_DEBUG][Bucket] splitBucket POST: L_vol=%lld R_vol=%lld vol_balance=%.4f\n",
                lv, rv, vol_ratio);
        // 如果AGM已设置, 也打印AGM分布
        if(left.AGM >= 0 && right.AGM >= 0) {
            ajb_dump_split(left, right, depth);
        }
#endif

        return {left, right};
    }

    // [AJB] isLeaf: splitDim越界说明所有维度都是单点
    // M911: 记录叶子深度到直方图
    bool isLeaf(int depth = -1) const {
        bool leaf = splitDim >= (int)lowerBound.size();
        if(leaf && depth >= 0) {
            ajb_bucket_depth_stats.record_leaf_depth(depth);
        }
        return leaf;
    }

    // [AJB] M911: per-dim range spread — 返回每个维度的范围向量
    // 用于cardinality估算和split维度选择
    vector<long long> dimRanges() const {
        vector<long long> ranges(lowerBound.size());
        for(size_t i = 0; i < lowerBound.size(); i++) {
            ranges[i] = (long long)upperBound[i] - lowerBound[i] + 1;
        }
        return ranges;
    }

    // [AJB] M911: 检查bucket是否为单点(所有维度lower==upper)
    bool isSinglePoint() const {
        for(size_t i = 0; i < lowerBound.size(); i++) {
            if(lowerBound[i] != upperBound[i]) return false;
        }
        return true;
    }
};

