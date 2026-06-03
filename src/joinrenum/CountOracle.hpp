using namespace std;
// [AJB] CountOracle: sorted point array + prefix-sum cnt for range counting
// sumCnt(pl,pr) = O(log n) via binary search, 是treeUpp的核心瓶颈

// [AJB] CountOracle诊断
static thread_local struct {
    long long sumcnt_calls = 0;
    long long sumcnt_zero  = 0;  // 返回0的次数 = empty range
    long long count_calls  = 0;
    long long range_calls  = 0;
    void dump(const char* tag = "CountOracle") {
        fprintf(stderr, "[AJB_STATE][%s] sumCnt=%lld(zero=%lld) count=%lld getRange=%lld\n",
                tag, sumcnt_calls, sumcnt_zero, count_calls, range_calls);
    }
    void reset() { sumcnt_calls = sumcnt_zero = count_calls = range_calls = 0; }
} ajb_co_stats;

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
        lowerbound = vector<T>(points[0].dim(), numeric_limits<T>::max());
        upperbound = vector<T>(points[0].dim(), numeric_limits<T>::min());
        for(int i = 0; i < points.size(); i++){
            for(int j = 0; j < points[i].dim(); j++){
                lowerbound[j] = min(lowerbound[j], points[i][j]);
                upperbound[j] = max(upperbound[j], points[i][j]);
            }
        }
        // lowerbound = Point<T>(lowervec);
        // upperbound = Point<T>(uppervec);
        sort(points.begin(), points.end());
        this->points = points;
        // [AJB_BP] CountOracle constructed: 点数+维度+值域是debug的第一站
        fprintf(stderr, "[AJB_BP][CountOracle] built: %zu points, dim=%lu, bounds=[",
                points.size(), points[0].dim());
        for(unsigned long d = 0; d < points[0].dim(); d++){
            if(d) fprintf(stderr, " ");
            fprintf(stderr, "%d..%d", lowerbound[d], upperbound[d]);
        }
        fprintf(stderr, "]\n");
        // [AJB_STATE] prefix-sum验证: 最后一个point的cnt应该等于total
        if(!points.empty()){
            fprintf(stderr, "[AJB_STATE][CountOracle] last_prefix_cnt=%lld (should be total weighted count)\n",
                    this->points.back().cnt);
        }
    }

    int sumCnt(const Point<T> &pl, const Point<T> &pr) {
        ajb_co_stats.sumcnt_calls++;
        vector<Point<int> >::iterator itl = lower_bound(points.begin(), points.end(), pl);
        vector<Point<int> >::iterator itr = upper_bound(points.begin(), points.end(), pr);
        int result;
        if(itr == points.begin()) result = 0;
        else if(itl == points.begin()) result = (itr - 1)->cnt;
        else result = (itr - 1)->cnt - (itl - 1)->cnt;
        if(result == 0) ajb_co_stats.sumcnt_zero++;
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
        for (int i = 0; i < points.size(); i++) {
            points[i].print();
        }
    }

    // [AJB] dump前N个和后N个point到stderr, 用于验证排序正确性
    void ajb_dump_endpoints(int n = 3) const {
        int total = points.size();
        fprintf(stderr, "[AJB_STATE][CountOracle] %d points, showing first/last %d:\n", total, n);
        for(int i = 0; i < min(n, total); i++) points[i].ajb_dump("  head");
        if(total > 2*n) fprintf(stderr, "[AJB_STATE][CountOracle]   ... (%d omitted)\n", total - 2*n);
        for(int i = max(n, total - n); i < total; i++) points[i].ajb_dump("  tail");
    }

};
