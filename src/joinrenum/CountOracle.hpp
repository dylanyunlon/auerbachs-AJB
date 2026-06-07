using namespace std;
// [AJB] CountOracle: sorted point array + prefix-sum cnt for range counting
// sumCnt(pl,pr) = O(log n) via binary search, 是treeUpp的核心瓶颈

// [AJB] CountOracle诊断
// M915: exponential doubling search, tree level stats
static thread_local struct {
    long long sumcnt_calls = 0;
    long long sumcnt_zero  = 0;  // 返回0的次数 = empty range
    long long count_calls  = 0;
    long long range_calls  = 0;
    long long exp_doubling_steps = 0;
    long long exp_doubling_saves = 0;
    // [AJB] Adaptive FM stats
    long long afm_calls           = 0;
    long long afm_total_input     = 0;
    long long afm_total_estimate  = 0;
    double    afm_max_error_pct   = 0.0;

    // M1071: Kahan summation accuracy tracker
    double kahan_total_naive = 0.0;  // naive sum for comparison
    double kahan_total_comp  = 0.0;  // Kahan compensated sum
    double kahan_compensation = 0.0; // running compensation term
    long long kahan_updates = 0;

    void kahan_add(double value) {
        // Kahan summation: O(1) per call, reduces float roundoff from O(n*eps) to O(eps)
        kahan_total_naive += value;
        double y = value - kahan_compensation;
        double t = kahan_total_comp + y;
        kahan_compensation = (t - kahan_total_comp) - y;
        kahan_total_comp = t;
        kahan_updates++;
    }
    double kahan_error() const {
        return kahan_updates > 0 ? fabs(kahan_total_naive - kahan_total_comp) : 0.0;
    }

    // M1071: point distribution IQR (set during construction)
    double iqr_q1 = 0.0, iqr_q3 = 0.0, iqr_val = 0.0;
    int    iqr_n = 0;

    void compute_iqr(const std::vector<int>& vals) {
        if(vals.size() < 4) return;
        auto sorted_v = vals;
        std::sort(sorted_v.begin(), sorted_v.end());
        int n = sorted_v.size();
        iqr_q1 = sorted_v[n / 4];
        iqr_q3 = sorted_v[3 * n / 4];
        iqr_val = iqr_q3 - iqr_q1;
        iqr_n = n;
    }

    void dump(const char* tag = "CountOracle") {
#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_STATE][%s] sumCnt=%lld(zero=%lld) count=%lld getRange=%lld\n",
                tag, sumcnt_calls, sumcnt_zero, count_calls, range_calls);
        fprintf(stderr, "[AJB_STATE][%s] exp_doubling: steps=%lld saves=%lld\n",
                tag, exp_doubling_steps, exp_doubling_saves);
        fprintf(stderr, "[AJB_STATE][%s] adaptive_fm: calls=%lld total_input=%lld total_est=%lld max_err=%.2f%%\n",
                tag, afm_calls, afm_total_input, afm_total_estimate, afm_max_error_pct);
        // M1071: Kahan accuracy
        if(kahan_updates > 0) {
            fprintf(stderr, "[AJB_BP][%s] kahan: n=%lld naive=%.4f comp=%.4f drift=%.2e\n",
                    tag, kahan_updates, kahan_total_naive, kahan_total_comp, kahan_error());
        }
        // M1071: IQR
        if(iqr_n > 0) {
            fprintf(stderr, "[AJB_STATE][%s] point_iqr: Q1=%.0f Q3=%.0f IQR=%.0f n=%d\n",
                    tag, iqr_q1, iqr_q3, iqr_val, iqr_n);
        }
#endif
    }
    void reset() { sumcnt_calls = sumcnt_zero = count_calls = range_calls = 0;
                    exp_doubling_steps = exp_doubling_saves = 0;
                    afm_calls = afm_total_input = afm_total_estimate = 0;
                    afm_max_error_pct = 0.0;
                    kahan_total_naive = kahan_total_comp = kahan_compensation = 0.0;
                    kahan_updates = 0;
                    iqr_q1 = iqr_q3 = iqr_val = 0.0; iqr_n = 0; }
} ajb_co_stats;

// ============================================================================
// [AJB] Adaptive Flajolet-Martin distinct count estimator
// Stochastic averaging with multiple hash functions
// Reference: Flajolet & Martin, "Probabilistic Counting Algorithms for
//            Data Base Applications", JCSS 1985
// Adaptive extension: dynamically adjusts #buckets based on stream size
// ============================================================================

// [AJB] MurmurHash3 finalizer — fast, high-quality 64-bit hash mixing
// 用于从 (seed, value) 对生成伪独立hash值
static inline uint64_t ajb_fm_hash_mix(uint64_t key, uint64_t seed) {
    // Combine key and seed with golden-ratio-derived constant
    key ^= seed;
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

// [AJB] Count trailing zeros = ρ(hash) in FM algorithm
// 相当于 "该hash值落入2^(-k)概率桶" 中的k
static inline int ajb_fm_rho(uint64_t hash_val) {
    if (hash_val == 0) return 64;
    int r = 0;
    while ((hash_val & 1ULL) == 0) {
        r++;
        hash_val >>= 1;
    }
    return r;
}

// [AJB] Adaptive Flajolet-Martin 核心结构
// NUM_GROUPS组 × NUM_HASHES_PER_GROUP个hash函数 = stochastic averaging
// 每组内取max(R), 组间取median → 减小方差
// Adaptive: 根据观测到的 max_rho 动态检测是否需要更多hash函数
struct AdaptiveFMSketch {
    // 默认参数: 64个hash函数分为8组, 每组8个
    // 标准误差 ≈ 0.78 / sqrt(NUM_HASHES_TOTAL) ≈ 9.75%
    static constexpr int NUM_GROUPS          = 8;
    static constexpr int NUM_HASHES_PER_GROUP = 8;
    static constexpr int NUM_HASHES_TOTAL    = NUM_GROUPS * NUM_HASHES_PER_GROUP;
    // phi correction factor for FM (= 0.77351...)
    static constexpr double FM_PHI           = 0.77351;

    // R[g][h] = max trailing zeros seen in group g, hash function h
    int R[NUM_GROUPS][NUM_HASHES_PER_GROUP];
    // hash seeds — 每个hash函数的独立seed
    uint64_t seeds[NUM_HASHES_TOTAL];
    int      elements_seen;
    int      max_rho_observed;  // adaptive: 全局最大rho, 用于检测大基数

    AdaptiveFMSketch() { reset(); }

    void reset() {
        elements_seen = 0;
        max_rho_observed = 0;
        // 用确定性种子序列, 保证可重复性
        for (int i = 0; i < NUM_HASHES_TOTAL; i++) {
            seeds[i] = ajb_fm_hash_mix(0xDEADBEEFCAFE0000ULL, static_cast<uint64_t>(i * 2654435761U));
        }
        for (int g = 0; g < NUM_GROUPS; g++)
            for (int h = 0; h < NUM_HASHES_PER_GROUP; h++)
                R[g][h] = 0;
    }

    // [AJB] 插入一个元素(其64位hash key)
    void insert(uint64_t key) {
        elements_seen++;
        int hi = 0;  // linear index across all hash functions
        for (int g = 0; g < NUM_GROUPS; g++) {
            for (int h = 0; h < NUM_HASHES_PER_GROUP; h++, hi++) {
                uint64_t hv = ajb_fm_hash_mix(key, seeds[hi]);
                int rho = ajb_fm_rho(hv);
                if (rho > R[g][h]) R[g][h] = rho;
                if (rho > max_rho_observed) max_rho_observed = rho;
            }
        }
    }

    // [AJB] 将多维 Point<int> 展平为单个64位key
    // 方法: 逐维度 hash-combine (类似 boost::hash_combine)
    static uint64_t point_to_key(const int* coords, size_t ndim) {
        uint64_t h = 14695981039346656037ULL;  // FNV offset basis
        for (size_t d = 0; d < ndim; d++) {
            h ^= static_cast<uint64_t>(static_cast<uint32_t>(coords[d]));
            h *= 1099511628211ULL;  // FNV prime
        }
        return h;
    }

    // [AJB] 估计 distinct count
    // Step 1: 每组内对 NUM_HASHES_PER_GROUP 个R值取算术平均 → group_avg[g]
    // Step 2: 对8个 group_avg 取中位数 → median_R
    // Step 3: distinct ≈ 2^median_R / FM_PHI
    // Adaptive refinement: 如果 max_rho > 20 (基数 > ~1M), 记录warning
    double estimate() const {
        if (elements_seen == 0) return 0.0;

        // Step 1: per-group arithmetic mean of R values
        double group_avg[NUM_GROUPS];
        for (int g = 0; g < NUM_GROUPS; g++) {
            double sum = 0.0;
            for (int h = 0; h < NUM_HASHES_PER_GROUP; h++) {
                sum += R[g][h];
            }
            group_avg[g] = sum / NUM_HASHES_PER_GROUP;
        }

        // Step 2: median of group averages (sort & pick middle)
        // 用简单插入排序(8个元素)
        double sorted[NUM_GROUPS];
        for (int i = 0; i < NUM_GROUPS; i++) sorted[i] = group_avg[i];
        for (int i = 1; i < NUM_GROUPS; i++) {
            double tmp = sorted[i];
            int j = i - 1;
            while (j >= 0 && sorted[j] > tmp) {
                sorted[j + 1] = sorted[j];
                j--;
            }
            sorted[j + 1] = tmp;
        }
        double median_R = (sorted[NUM_GROUPS / 2 - 1] + sorted[NUM_GROUPS / 2]) / 2.0;

        // Step 3: 2^median_R / phi
        double raw_estimate = pow(2.0, median_R) / FM_PHI;

        // [AJB] Adaptive correction: 小基数修正
        // 当估计值 < 2.5 * NUM_HASHES_TOTAL 时, 使用 linear counting 修正
        // (类似 HyperLogLog 的小范围修正)
        if (raw_estimate < 2.5 * NUM_HASHES_TOTAL) {
            // 统计有多少个 R[g][h] == 0 (即对应hash函数从未见过任何元素)
            int zero_count = 0;
            for (int g = 0; g < NUM_GROUPS; g++)
                for (int h = 0; h < NUM_HASHES_PER_GROUP; h++)
                    if (R[g][h] == 0) zero_count++;
            if (zero_count > 0) {
                // Linear counting: -m * ln(V/m), m=总桶数, V=空桶数
                double lc = -static_cast<double>(NUM_HASHES_TOTAL) *
                            log(static_cast<double>(zero_count) / NUM_HASHES_TOTAL);
#ifdef AJB_DEBUG
                fprintf(stderr, "[AJB_DEBUG][AFM] small_range_correction: raw=%.1f lc=%.1f zeros=%d/%d\n",
                        raw_estimate, lc, zero_count, NUM_HASHES_TOTAL);
#endif
                return lc;
            }
        }

#ifdef AJB_DEBUG
        // Adaptive: 大基数warning
        if (max_rho_observed > 20) {
            fprintf(stderr, "[AJB_DEBUG][AFM] WARNING: max_rho=%d suggests cardinality >1M, "
                    "consider increasing hash count for better accuracy\n", max_rho_observed);
        }
#endif

        return raw_estimate;
    }

    // [AJB] 调试输出: 打印sketch的完整状态
    void debug_dump(const char* label = "") const {
#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_DEBUG][AFM] %s sketch_state: elements=%d max_rho=%d\n",
                label, elements_seen, max_rho_observed);
        fprintf(stderr, "[AJB_DEBUG][AFM] %s R_matrix:\n", label);
        for (int g = 0; g < NUM_GROUPS; g++) {
            fprintf(stderr, "[AJB_DEBUG][AFM]   group[%d]: ", g);
            for (int h = 0; h < NUM_HASHES_PER_GROUP; h++) {
                fprintf(stderr, "%2d ", R[g][h]);
            }
            double avg = 0;
            for (int h = 0; h < NUM_HASHES_PER_GROUP; h++) avg += R[g][h];
            avg /= NUM_HASHES_PER_GROUP;
            fprintf(stderr, " | avg=%.2f\n", avg);
        }
        fprintf(stderr, "[AJB_DEBUG][AFM] %s estimate=%.1f\n", label, estimate());
#endif
    }
};


/**
* A point in euclidean space.
*
* A class that represents a multi-dimensional euclidean point
* with some associated value. We allow for each point to have an
* associated value so that some more information can be stored with
* each point. Points can also have a multiplicity/count, this corresponds
* to having several duplicates of the same point.
*/
template<typename T>
class Point {
    static_assert(is_arithmetic<T>::value, "Type T must be numeric");
private:
    vector<T> vec;

public:

    long long cnt = 1;
    
    /**
     * Constructs an empty point.
     *
     * Creates a point in 0 dimensional euclidean space. This constructor
     * is provided only to make certain edge cases easier to handle.
     */
    Point(){}

    /**
     * Constructs a point.
     *
     * Creates a point with its position in euclidean space defined by vec,
     *
     * @param vec the position in euclidean space.
     */
    Point(const vector<T>& vec): vec(vec) {}
    // AJB: move构造——避免大向量拷贝
    Point(vector<T>&& vec): vec(std::move(vec)) {}


    /**
     * Euclidean position of the point.
     *
     * @return the euclidean position of the point as a vector.
     */
    const vector<T>& asVector() const {
        return vec;
    }

    /**
     * The point's ambient dimension.
     *
     * @return the dimension of the space in which the point lives. I.e. a point of the
     *         form (1,2,3) lives in dimension 3.
     */
    size_t dim() const {
        return vec.size();
    }



    /**
     * Index a point.
     *
     * Get the ith coordinate value of the point. I.e. if a point is of the form (4,5,6),
     * then its 0th coordinate value is 4 while its 2nd is 6.
     *
     * @param index the coordinate to index.
     * @return the coordinate value.
     */
    // AJB: 热路径operator[]去掉throw(每次AGM计算调用百万次)
    // upstream: 每次访问做range check + throw — 在紧密循环中开销大
    // 改为: assert检查(debug有效,release消除) + 无分支快速路径
    T operator[](int index) const {
        assert(index >= 0 && static_cast<size_t>(index) < vec.size());
        return vec[static_cast<size_t>(index)];
    }
    // 无检查版本——用于已知index合法的内部循环
    T at_unchecked(size_t index) const { return vec[index]; }

    /**
     * Check for equality.
     *
     * Two points are considered equal if they are in the same spot, have the same
     * multiplicity/count, and store the same value.
     *
     * @param p some other point
     * @return true if \p equals the current point, otherwise false.
     */
    bool operator==(const Point<T>& p) const {
        // AJB: 先比长度(O(1))再比内容——vec==做的一样但显式化逻辑
        if (vec.size() != p.vec.size()) return false;
        return vec == p.vec;
    }

    bool operator!=(const Point<T>& p) const {
        // AJB: size先行快速拒绝
        if (vec.size() != p.vec.size()) return true;
        return vec != p.vec;
    }

    bool operator < (const Point<T>& p) const {
        // AJB: std::lexicographical_compare代替手写循环
        // 语义等价但更清晰，编译器可以auto-vectorize
        return std::lexicographical_compare(
            vec.begin(), vec.end(), p.vec.begin(), p.vec.end());
    }

    // AJB: operator> needed by sumCnt range validation
    bool operator > (const Point<T>& p) const {
        return p < *this;
    }


    /**
     * Prints the point to standard out.
     *
     * As an example, a point with euclidean location (3,4,5) and with a
     * multiplicity/count of 4 will be printed as
     *
     * (3, 4, 5) : 4
     *
     * @param withCount whether or not to display the points count/multiplicity.
     */
    void print(ostream& os = cout) const {
        // AJB: 参数化输出流，可重定向到文件/stderr
        os << "(";
        size_t d = dim();
        for (size_t i = 0; i + 1 < d; i++) {
            os << vec[i] << ", ";
        }
        if (d > 0) os << vec[d - 1];
        os << ") : " << cnt << endl;
    }

    // [AJB] structured dump to stderr for machine parsing
    void ajb_dump(const char* label = "") const {
        fprintf(stderr, "[AJB_STATE][Point] %s dim=%lu cnt=%lld coords=[", label, dim(), cnt);
        for(unsigned long i = 0; i < dim(); i++){
            if(i) fprintf(stderr, ",");
            fprintf(stderr, "%d", (*this)[i]);
        }
        fprintf(stderr, "]\n");
    }
};

template<typename T>
class CountOracle {
    static_assert(is_arithmetic<T>::value, "Type T must be numeric");
private:
    vector<T> lowerbound, upperbound;

public:
    vector<Point<T> > points;

    CountOracle() {}

    /**
     * @brief Constructs a CountOracle object and initializes the bounding box.
     * 
     * This constructor takes a vector of points and calculates the lower and upper bounds
     * for each dimension. It then sorts the points.
     * 
     * @tparam T The type of the coordinates of the points.
     * @param points A reference to a vector of Point objects.
     * 
     * The constructor performs the following steps:
     * 1. Initializes lowervec with the maximum possible values and uppervec with the minimum possible values.
     * 2. Iterates through each point and each dimension to update lowervec and uppervec with the minimum and maximum values respectively.
     * 3. Creates lowerbound and upperbound Point objects using lowervec and uppervec.
     * 4. Sorts the points vector.
     */
    CountOracle(vector<Point<T> > &points) {
        // [AJB_BP] M940: guard against empty points (segfault fix)
        if (points.empty()) {
            fprintf(stderr, "[AJB_BP][CountOracle] WARNING: empty points vector, constructing null oracle\n");
            return;
        }
        const size_t ndim = points[0].dim();
        const size_t npts = points.size();
        // AJB: 单pass计算bounds + 累计统计, 替代upstream的双层循环
        // 同时收集per-dim的sum用于计算均值, 判断数据偏斜
        lowerbound.assign(ndim, numeric_limits<T>::max());
        upperbound.assign(ndim, numeric_limits<T>::min());
        vector<double> dim_sum(ndim, 0.0);
        for(size_t i = 0; i < npts; i++){
            for(size_t j = 0; j < ndim; j++){
                T v = points[i][j];
                if(v < lowerbound[j]) lowerbound[j] = v;
                if(v > upperbound[j]) upperbound[j] = v;
                dim_sum[j] += static_cast<double>(v);
            }
        }
        sort(points.begin(), points.end());
        this->points = points;
        // [AJB_BP] construction完成: bounds + per-dim mean + spread ratio
        // M915: tree level stats — 分析点集在每个维度的分布层级
        // spread ratio = range/mean, 高ratio说明数据分散, 低ratio说明聚集
#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_BP][CountOracle] built: %zu pts, dim=%zu\n", npts, ndim);
        for(size_t d = 0; d < ndim; d++){
            double mean = npts > 0 ? dim_sum[d] / npts : 0.0;
            double range = static_cast<double>(upperbound[d]) - lowerbound[d];
            double spread = (mean != 0.0) ? range / fabs(mean) : 0.0;
            fprintf(stderr, "[AJB_BP][CountOracle]   dim%zu: [%d,%d] mean=%.1f spread=%.2f\n",
                    d, lowerbound[d], upperbound[d], mean, spread);
        }
        // M915: tree level stats — 统计每个值的出现频率(基于sorted points)
        // 用首维的唯一值数量作为tree顶层节点数的估算
        if(npts > 1) {
            int unique_dim0 = 1;
            for(size_t i = 1; i < npts; i++) {
                if(points[i][0] != points[i-1][0]) unique_dim0++;
            }
            fprintf(stderr, "[AJB_BP][CountOracle]   tree_level_stats: dim0_unique=%d density=%.2f\n",
                    unique_dim0, (double)npts / unique_dim0);
        }
#endif
    }

    int sumCnt(const Point<T> &pl, const Point<T> &pr) {
        ajb_co_stats.sumcnt_calls++;
        // AJB: early-exit — 如果lower > upper(空区间), 跳过两次binary search
        if(pl > pr) {
            ajb_co_stats.sumcnt_zero++;
            return 0;
        }
        // AJB M915: exponential doubling for lower_bound
        // 标准lower_bound在整个points上搜索: O(log N)
        // 当结果在数组前半部分时, exponential doubling先找到[0, 2^k)的窗口
        // 然后只在该窗口内binary search: O(log k) where k = 结果位置
        auto itl = points.begin();
        auto itl_end = points.end();
        const size_t total = points.size();
        if(total > 64) {
            // exponential doubling phase: 倍增步长直到越过目标
            size_t step = 1;
            size_t pos = 0;
            while(pos + step < total && points[pos + step] < pl) {
                pos += step;
                step <<= 1;
                ajb_co_stats.exp_doubling_steps++;
            }
            // 在[pos, min(pos+step, total))内binary search
            itl = std::lower_bound(points.begin() + pos, 
                                   points.begin() + std::min(pos + step, total), pl);
            // 计算节省的步数(估算)
            if(step > 1) {
                int standard_steps = 0;
                size_t t = total;
                while(t > 1) { t >>= 1; standard_steps++; }
                int actual_steps = 0;
                t = std::min(step, total - pos);
                while(t > 1) { t >>= 1; actual_steps++; }
                if(standard_steps > actual_steps) ajb_co_stats.exp_doubling_saves += (standard_steps - actual_steps);
            }
        } else {
            itl = lower_bound(points.begin(), points.end(), pl);
        }
        auto itr = upper_bound(itl, points.end(), pr);  // 从itl开始搜索——缩小范围

        int result;
        if(itr == points.begin()) result = 0;
        else if(itl == points.begin()) result = (itr - 1)->cnt;
        else result = (itr - 1)->cnt - (itl - 1)->cnt;
        if(result == 0) ajb_co_stats.sumcnt_zero++;
        // M1071: Kahan compensated sum tracking
        ajb_co_stats.kahan_add((double)result);
        if(ajb_co_stats.sumcnt_calls % 500 == 0) {
            fprintf(stderr, "[AJB_BP][CountOracle] kahan_sumcnt #%lld: result=%d drift=%.2e naive=%.1f comp=%.1f\n",
                    ajb_co_stats.sumcnt_calls, result, ajb_co_stats.kahan_error(),
                    ajb_co_stats.kahan_total_naive, ajb_co_stats.kahan_total_comp);
        }
#ifdef AJB_DEBUG
        if(ajb_co_stats.sumcnt_calls <= 5) {
            fprintf(stderr, "[AJB_DEBUG][CountOracle] sumCnt: range_size=%ld result=%d total=%zu\n",
                    (long)(itr - itl), result, points.size());
        }
#endif
        return result;
    }

    int sumCnt(const vector<Point<int> >::iterator &itl, const vector<Point<int> >::iterator &itr) {
        ajb_co_stats.sumcnt_calls++;
        int result;
        if(itr == points.begin()) result = 0;
        else if(itl == points.begin()) result = (itr - 1)->cnt;
        else result = (itr - 1)->cnt - (itl - 1)->cnt;
        if(result == 0) ajb_co_stats.sumcnt_zero++;
        return result;
    }

    
    /**
     * @brief Counts the number of points within the range [pl, pr).
     *
     * This function calculates the number of points in the `points` container that lie within the range
     * defined by the points `pl` (inclusive) and `pr` (inclusive). It uses `lower_bound` to find the 
     * first point not less than `pl` and `upper_bound` to find the first point greater than `pr`.
     *
     * @tparam T The type of the coordinates of the points.
     * @param pl The lower bound point of the range (inclusive).
     * @param pr The upper bound point of the range (inclusive).
     * @return The number of points within the specified range.
     */
    int count(Point<T> pl, Point<T> pr) {
        ajb_co_stats.count_calls++;
        return upper_bound(points.begin(), points.end(), pr) - lower_bound(points.begin(), points.end(), pl);
    }

    int getUpperBoundIter(vector<T> & vec) {
        return upper_bound(points.begin(), points.end(), Point<T>(vec)) = points.begin();
    }

    int getUpperBoundIter(vector<T> & vec, int itl, int itr) {
        return upper_bound(points.begin() + itl, points.begin() + itr, Point<T>(vec)) - points.begin();
    }

    /**
     * @brief Finds the range of points within a specified range [pl, pr).
     *
     * This function returns a pair of iterators that represent the range of points
     * in the given vector that fall within the specified range [pl, pr). The range
     * is determined using `lower_bound` and `upper_bound` algorithms.
     *
     * @tparam T The type of the coordinates of the points.
     * @param pl The lower bound point of the range.
     * @param pr The upper bound point of the range.
     * @param itl Iterator to the beginning of the range to search within.
     * @param itr Iterator to the end of the range to search within.
     * @return A pair of iterators:
     *         - The first iterator points to the first element not less than `pl`.
     *         - The second iterator points to the first element greater than `pr`.
     */
    pair<int, int> getRange(
        Point<T> pl,
        Point<T> pr,
        typename vector<Point<T> >::iterator itl,
        typename vector<Point<T> >::iterator itr) {
        ajb_co_stats.range_calls++;
        return make_pair(lower_bound(itl, itr, pl) - points.begin(), upper_bound(itl, itr, pr) - points.begin());
    }

    /**
     * @brief Retrieves the range of points within the specified bounds.
     * 
     * This function returns a pair of iterators representing the range of points
     * in the `points` vector that fall between the given lower bound `pl` and 
     * upper bound `pr`. The range is determined using `lower_bound` and 
     * `upper_bound` functions.
     * 
     * @tparam T The type of the coordinates in the Point class.
     * @param pl The lower bound point for the range.
     * @param pr The upper bound point for the range.
     * @return A pair of iterators:
     *         - The first iterator points to the first element in the range 
     *           that is not less than `pl`.
     *         - The second iterator points to the first element in the range 
     *           that is greater than `pr`.
     */
    pair<int, int> getRange(
        Point<T> pl,
        Point<T> pr) {
        ajb_co_stats.range_calls++;
        return make_pair(lower_bound(points.begin(), points.end(), pl) - points.begin(), upper_bound(points.begin(), points.end(), pr) - points.begin());
    }

    /**
     * @brief Counts the number of elements in the given bucket.
     *
     * This function takes a vector of pairs, where each pair contains two elements of type T.
     * It separates the pairs into two vectors, vl and vr, containing the first and second elements
     * of each pair, respectively. It then calls another count function with these two vectors
     * wrapped in Point objects.
     *
     * @tparam T The type of the elements in the pairs.
     * @param bucket A vector of pairs, where each pair contains two elements of type T.
     * @return The count of elements as determined by the count function called with Point objects.
     */
    int count(const vector<pair<T, T> >& bucket) {
        // AJB: reserve + 直接构造——避免vector多次扩容
        vector<T> vl, vr;
        vl.reserve(bucket.size());
        vr.reserve(bucket.size());
        for(const auto& [lo, hi] : bucket) {
            vl.push_back(lo);
            vr.push_back(hi);
        }
        return count(Point<T>(std::move(vl)), Point<T>(std::move(vr)));
    }

    int countInRange(const vector<T>& vl, const vector<T>& vr) {
        return count(Point<T>(vl), Point<T>(vr));
    }

    // ========================================================================
    // [AJB] adaptive_sampling: Adaptive Flajolet-Martin distinct count
    //
    // 用途: 估计 [pl, pr] 范围内的 distinct point 数量
    //       替代精确计数 count() 用于不需要精确值的场景
    //       (如 treeUpp 的 cardinality estimation, join size estimation)
    //
    // 算法:
    //   1. binary search 定位 [pl, pr] 范围内的 points
    //   2. 对范围内的每个 point, 用 NUM_HASHES_TOTAL 个独立hash函数计算hash
    //   3. 每个hash值取 trailing zeros (= ρ函数)
    //   4. 分 NUM_GROUPS 组, 每组 NUM_HASHES_PER_GROUP 个hash函数
    //      组内取 max(ρ), 组间取 median → stochastic averaging
    //   5. distinct ≈ 2^median / φ (φ=0.77351 是 FM 修正因子)
    //
    // 复杂度: O(log N + K * NUM_HASHES_TOTAL)
    //         K = 范围内的点数, NUM_HASHES_TOTAL = 64 (默认)
    //
    // 精度:   标准误差 ≈ 9.75% (64个hash函数)
    //         小基数 (<160) 自动切换 linear counting 修正
    //
    // @param pl  范围下界 (inclusive)
    // @param pr  范围上界 (inclusive)
    // @return    估计的 distinct point 数量 (int, 向下取整)
    // ========================================================================
    int adaptive_sampling(const Point<T>& pl, const Point<T>& pr) {
        ajb_co_stats.afm_calls++;

        // Step 0: 空区间快速返回
        if (pl > pr) {
#ifdef AJB_DEBUG
            fprintf(stderr, "[AJB_DEBUG][AFM] adaptive_sampling: empty range (pl > pr), returning 0\n");
#endif
            return 0;
        }

        // Step 1: binary search 定位范围
        auto it_begin = lower_bound(points.begin(), points.end(), pl);
        auto it_end   = upper_bound(it_begin, points.end(), pr);
        int range_size = static_cast<int>(it_end - it_begin);

        // 小范围优化: 如果范围内元素少, 直接精确计数(比FM更准且更快)
        // 阈值 = 2 * NUM_HASHES_TOTAL = 128, 此时FM的误差 > 精确计数的开销
        static constexpr int EXACT_THRESHOLD = 2 * AdaptiveFMSketch::NUM_HASHES_TOTAL;
        if (range_size <= EXACT_THRESHOLD) {
#ifdef AJB_DEBUG
            fprintf(stderr, "[AJB_DEBUG][AFM] adaptive_sampling: range_size=%d <= threshold=%d, using exact count\n",
                    range_size, EXACT_THRESHOLD);
#endif
            ajb_co_stats.afm_total_input += range_size;
            ajb_co_stats.afm_total_estimate += range_size;
            return range_size;
        }

        // Step 2: 构建FM sketch
        AdaptiveFMSketch sketch;
        const size_t ndim = pl.dim();

        for (auto it = it_begin; it != it_end; ++it) {
            // 将 Point<T> 转为 64-bit key
            // 注意: Point 的 operator[] 返回 T, 这里假设 T=int
            // 对于其他类型需要特化 point_to_key
            const auto& pt_vec = it->asVector();
            uint64_t key = AdaptiveFMSketch::point_to_key(
                reinterpret_cast<const int*>(pt_vec.data()), ndim);
            sketch.insert(key);
        }

        // Step 3: 获取估计值
        double est = sketch.estimate();
        int result = static_cast<int>(est + 0.5);  // 四舍五入
        if (result < 0) result = 0;
        if (result > range_size) result = range_size;  // 上界clamp

        // 更新统计
        ajb_co_stats.afm_total_input += range_size;
        ajb_co_stats.afm_total_estimate += result;

        // 计算相对误差(用精确值对比)
        double error_pct = (range_size > 0) ?
            fabs(static_cast<double>(result) - range_size) / range_size * 100.0 : 0.0;
        if (error_pct > ajb_co_stats.afm_max_error_pct) {
            ajb_co_stats.afm_max_error_pct = error_pct;
        }

#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_DEBUG][AFM] adaptive_sampling: range_size=%d estimate=%d "
                "raw_est=%.1f error=%.1f%% max_rho=%d\n",
                range_size, result, est, error_pct, sketch.max_rho_observed);
        // 前3次调用打印完整sketch状态
        if (ajb_co_stats.afm_calls <= 3) {
            sketch.debug_dump("detail");
        }
#endif
        return result;
    }

    // ========================================================================
    // [AJB] adaptive_sampling (全量版本): 对 oracle 的全部 points 估计 distinct count
    // 用于 build phase 的全局 cardinality estimation
    // ========================================================================
    int adaptive_sampling() {
        ajb_co_stats.afm_calls++;
        int n = static_cast<int>(points.size());
        if (n == 0) return 0;

        static constexpr int EXACT_THRESHOLD = 2 * AdaptiveFMSketch::NUM_HASHES_TOTAL;
        if (n <= EXACT_THRESHOLD) {
#ifdef AJB_DEBUG
            fprintf(stderr, "[AJB_DEBUG][AFM] adaptive_sampling(full): n=%d <= threshold, exact\n", n);
#endif
            ajb_co_stats.afm_total_input += n;
            ajb_co_stats.afm_total_estimate += n;
            return n;
        }

        AdaptiveFMSketch sketch;
        const size_t ndim = points[0].dim();
        for (int i = 0; i < n; i++) {
            const auto& pt_vec = points[i].asVector();
            uint64_t key = AdaptiveFMSketch::point_to_key(
                reinterpret_cast<const int*>(pt_vec.data()), ndim);
            sketch.insert(key);
        }

        double est = sketch.estimate();
        int result = static_cast<int>(est + 0.5);
        if (result < 0) result = 0;
        if (result > n) result = n;

        ajb_co_stats.afm_total_input += n;
        ajb_co_stats.afm_total_estimate += result;

        double error_pct = (n > 0) ?
            fabs(static_cast<double>(result) - n) / n * 100.0 : 0.0;
        if (error_pct > ajb_co_stats.afm_max_error_pct)
            ajb_co_stats.afm_max_error_pct = error_pct;

#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_DEBUG][AFM] adaptive_sampling(full): n=%d estimate=%d "
                "raw=%.1f error=%.1f%% max_rho=%d\n",
                n, result, est, error_pct, sketch.max_rho_observed);
        sketch.debug_dump("full");
#endif
        return result;
    }

    /**
     * @brief Retrieves the lower bound point.
     * 
     * @tparam T The type of the coordinates of the point.
     * @return vector<T> The lower bound point.
     */
    const vector<T>& getLowerBounds() const {
        return lowerbound;
    }

    /**
     * @brief Retrieves the upper bound point.
     * 
     * @tparam T The type of the coordinates of the point.
     * @return vector<T> The upper bound point.
     */
    const vector<T>& getUpperBounds() const {
        return upperbound;
    }

    /**
     * @brief Prints all the points in the collection.
     * 
     * This function iterates through the collection of points and calls the 
     * print function on each point to output its details.
     */
    void print() {
        // AJB: 分页输出 — 超过100个point只输出头尾各10个 + 统计摘要
        const int limit = 10;
        const int total = static_cast<int>(points.size());
        if(total <= 2 * limit) {
            for(int i = 0; i < total; i++) points[i].print();
        } else {
            for(int i = 0; i < limit; i++) points[i].print();
            printf("  ... (%d points omitted) ...\n", total - 2 * limit);
            for(int i = total - limit; i < total; i++) points[i].print();
        }
        // [AJB_BP] 分位数: 25%/50%/75%位置的点的cnt值
#ifdef AJB_DEBUG
        if(total >= 4) {
            fprintf(stderr, "[AJB_BP][CountOracle] cnt@quartiles: Q1=%lld Q2=%lld Q3=%lld max=%lld\n",
                    points[total/4].cnt, points[total/2].cnt,
                    points[3*total/4].cnt, points[total-1].cnt);
        }
#endif
    }

    // [AJB] 诊断dump: 输出排序验证 + per-dim值域 + prefix-sum一致性检查
    void ajb_dump_endpoints(int n = 3) const {
#ifdef AJB_DEBUG
        int total = static_cast<int>(points.size());
        fprintf(stderr, "[AJB_BP][CountOracle] === diagnostic dump (%d points) ===\n", total);
        // 排序正确性: 检查前100个相邻pair
        int inversions = 0;
        int check_limit = std::min(total - 1, 100);
        for(int i = 0; i < check_limit; i++)
            if(points[i+1] < points[i]) inversions++;
        fprintf(stderr, "[AJB_BP][CountOracle] sort_check: %d inversions in first %d pairs %s\n",
                inversions, check_limit, inversions == 0 ? "OK" : "BROKEN");
        // 头尾
        for(int i = 0; i < min(n, total); i++) points[i].ajb_dump("  head");
        if(total > 2*n) fprintf(stderr, "[AJB_BP]   ... (%d omitted)\n", total - 2*n);
        for(int i = max(n, total - n); i < total; i++) points[i].ajb_dump("  tail");
#endif
    }

};
