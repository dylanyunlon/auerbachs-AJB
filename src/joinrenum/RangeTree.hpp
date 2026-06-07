/*
 * MIT License
 *
 * Copyright (c) 2016 Luca Weihs
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
* Implements the RangeTree datastructure.
*
* See the documentation of the class RangeTree for more details regarding the purpose
* of RangeTree's.
*
* This file contains two main abstractions meant for use:
* 1. The RangeTree class.
* 2. A Point class which captures the idea of a d-dimensional euclidean point.
*/

#ifndef RANGETREE_H
#define RANGETREE_H

#include <vector>
#include <iostream>
#include <sstream>
#include <numeric>
#include<memory>
#include <type_traits>
#include <deque>
#include <cmath>
#include <algorithm>

// [AJB] RangeTree诊断: 追踪查询热度和构建开销
static thread_local struct {
    long long build_calls = 0;
    long long query_calls = 0;
    long long total_points_inserted = 0;
    long long total_query_results = 0;
    int max_dim = 0;
    // [AJB_BP] M921: countInRange recursive depth tracking
    long long count_range_calls = 0;
    int       count_range_max_depth = 0;
    long long count_range_leaf_hits = 0;
    long long count_range_prune_left = 0;   // pruned right subtree
    long long count_range_prune_right = 0;  // pruned left subtree
    // [AJB_BP] M922: splitOnMid balance tracking
    long long split_calls = 0;
    long long split_bitvec_ones = 0;
    long long split_bitvec_total = 0;
    double    split_worst_balance = 1.0;    // min(nLeft,nRight)/max(nLeft,nRight)
    // [AJB_BP] M923: binarySearch convergence
    long long bsearch_geq_calls = 0;
    long long bsearch_leq_calls = 0;
    long long bsearch_total_steps = 0;
    int       bsearch_max_steps = 0;
    // [AJB_BP] M924: getAllPoints DFS depth
    long long getall_calls = 0;
    int       getall_max_stack = 0;
    long long getall_total_collected = 0;
    // [AJB_BP] M925: cycle-following permutation stats
    long long cycle_rearrange_calls = 0;
    long long cycle_total_swaps = 0;
    int       cycle_max_len = 0;
    long long cycle_movenext_calls = 0;
    // [AJB_BP] M921-enhanced: pointsInRange tracking
    long long pts_in_range_calls = 0;
    long long pts_in_range_total_returned = 0;
    int       pts_in_range_max_returned = 0;
    // [AJB_BP] M921-enhanced: fractional cascade stats
    long long frac_cascade_left_calls = 0;
    long long frac_cascade_right_calls = 0;
    long long frac_cascade_total_nodes = 0;
    // [AJB_BP] M921-enhanced: canonical node decomposition stats
    long long canonical_left_calls = 0;
    long long canonical_right_calls = 0;
    long long canonical_total_nodes = 0;
    int       canonical_max_nodes = 0;
    // [AJB_BP] M921-enhanced: cycle length distribution histogram (buckets: 1, 2-3, 4-7, 8-15, 16+)
    long long cycle_hist[5] = {0,0,0,0,0};
    void dump(const char* tag = "RangeTree") {
        fprintf(stderr, "[AJB_STATE][%s] builds=%lld queries=%lld points_inserted=%lld query_results=%lld max_dim=%d\n",
                tag, build_calls, query_calls, total_points_inserted, total_query_results, max_dim);
        fprintf(stderr, "[AJB_STATE][%s] countInRange: calls=%lld max_depth=%d leaf_hits=%lld prune_L=%lld prune_R=%lld\n",
                tag, count_range_calls, count_range_max_depth, count_range_leaf_hits,
                count_range_prune_left, count_range_prune_right);
        fprintf(stderr, "[AJB_STATE][%s] pointsInRange: calls=%lld total_returned=%lld max_returned=%d\n",
                tag, pts_in_range_calls, pts_in_range_total_returned, pts_in_range_max_returned);
        fprintf(stderr, "[AJB_STATE][%s] splitOnMid: calls=%lld worst_balance=%.4f bitvec_density=%.4f\n",
                tag, split_calls, split_worst_balance,
                split_bitvec_total > 0 ? (double)split_bitvec_ones / split_bitvec_total : 0.0);
        fprintf(stderr, "[AJB_STATE][%s] binarySearch: geq=%lld leq=%lld total_steps=%lld max_steps=%d\n",
                tag, bsearch_geq_calls, bsearch_leq_calls, bsearch_total_steps, bsearch_max_steps);
        fprintf(stderr, "[AJB_STATE][%s] getAllPoints: calls=%lld max_stack=%d total_collected=%lld\n",
                tag, getall_calls, getall_max_stack, getall_total_collected);
        fprintf(stderr, "[AJB_STATE][%s] fracCascade: left=%lld right=%lld total_nodes=%lld\n",
                tag, frac_cascade_left_calls, frac_cascade_right_calls, frac_cascade_total_nodes);
        fprintf(stderr, "[AJB_STATE][%s] canonical: left=%lld right=%lld total=%lld max=%d\n",
                tag, canonical_left_calls, canonical_right_calls, canonical_total_nodes, canonical_max_nodes);
        fprintf(stderr, "[AJB_STATE][%s] cycle-follow: rearrange=%lld movenext=%lld total_swaps=%lld max_cycle=%d\n",
                tag, cycle_rearrange_calls, cycle_movenext_calls, cycle_total_swaps, cycle_max_len);
        fprintf(stderr, "[AJB_STATE][%s] cycle_hist: [1]=%lld [2-3]=%lld [4-7]=%lld [8-15]=%lld [16+]=%lld\n",
                tag, cycle_hist[0], cycle_hist[1], cycle_hist[2], cycle_hist[3], cycle_hist[4]);
    }
    void reset() {
        build_calls = query_calls = total_points_inserted = total_query_results = 0; max_dim = 0;
        count_range_calls = count_range_leaf_hits = count_range_prune_left = count_range_prune_right = 0;
        count_range_max_depth = 0;
        pts_in_range_calls = pts_in_range_total_returned = 0; pts_in_range_max_returned = 0;
        split_calls = split_bitvec_ones = split_bitvec_total = 0; split_worst_balance = 1.0;
        bsearch_geq_calls = bsearch_leq_calls = bsearch_total_steps = 0; bsearch_max_steps = 0;
        getall_calls = 0; getall_max_stack = 0; getall_total_collected = 0;
        frac_cascade_left_calls = frac_cascade_right_calls = frac_cascade_total_nodes = 0;
        canonical_left_calls = canonical_right_calls = canonical_total_nodes = 0; canonical_max_nodes = 0;
        cycle_rearrange_calls = cycle_total_swaps = cycle_movenext_calls = 0; cycle_max_len = 0;
        for (int i = 0; i < 5; i++) cycle_hist[i] = 0;
    }
} ajb_rt_stats;

namespace RangeTree {

    /**
    * A point in euclidean space.
    *
    * A class that represents a multi-dimensional euclidean point
    * with some associated value. We allow for each point to have an
    * associated value so that some more information can be stored with
    * each point. Points can also have a multiplicity/count, this corresponds
    * to having several duplicates of the same point.
    */
    template<typename T, class S>
    class Point {
        static_assert(std::is_arithmetic<T>::value, "Type T must be numeric");
    private:
        std::vector<T> vec;
        S val;
        int multiplicity;

    public:
        /**
        * Constructs an empty point.
        *
        * Creates a point in 0 dimensional euclidean space. This constructor
        * is provided only to make certain edge cases easier to handle.
        */
        Point() : multiplicity(0) {}

        /**
        * Constructs a point.
        *
        * Creates a point with its position in euclidean space defined by vec,
        * value defined by val, and a multiplicity/count of 1.
        *
        * @param vec the position in euclidean space.
        * @param val the value associated with the point.
        */
        Point(const std::vector<T>& vec, const S& val): val(val), vec(vec), multiplicity(1) {}

        /**
        * Constructs a point.
        *
        * Copies a point.
        *
        * @param vec the position in euclidean space.
        * @param val the value associated with the point.
        */
        Point(const Point<T,S>& p): val(p.val), vec(p.vec), multiplicity(p.count()) {}


        /**
        * Euclidean position of the point.
        *
        * @return the euclidean position of the point as a std::vector.
        */
        const std::vector<T>& asVector() const {
            return vec;
        }

        /**
        * The point's ambient dimension.
        *
        * @return the dimension of the space in which the point lives. I.e. a point of the
        *         form (1,2,3) lives in dimension 3.
        */
        unsigned long dim() const {
            return vec.size();
        }

        /**
        * The point's count/multiplicity.
        *
        * @return returns the count/multiplicity.
        */
        int count() const {
            return multiplicity;
        }

        /**
        * Increase the point's count/multiplicity.
        *
        * @param n amount to increase by.
        */
        void increaseCountBy(int n) {
            if (n < 0) {
                throw std::logic_error("Can't increase by a negative amount");
            }
            multiplicity += n;
        }

        /**
        * The point's value.
        *
        * @return the value stored in the point.
        */
        S value() const {
            return val;
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
        T operator[](int index) const {
            if(index < 0 || index >= dim()) {
                throw std::out_of_range("[] access index for point is out of range.");
            }
            return vec[index];
        }

        /**
        * Check for equality.
        *
        * Two points are considered equal if they are in the same spot, have the same
        * multiplicity/count, and store the same value.
        *
        * @param p some other point
        * @return true if \p equals the current point, otherwise false.
        */
        bool operator==(const Point<T,S>& p) const {
            return vec == p.vec && multiplicity == p.multiplicity && val == p.val;
        }

        /**
        * Check for inequality.
        *
        * The opposite of ==.
        *
        * @param p some other point.
        * @return false if \p equals the current point, otherwise true.
        */
        bool operator!=(const Point<T,S>& p) const {
            return !((*this) == p);
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
        void print(bool withCount=true) const {
            // Buffer-based output: build the string in a local buffer, then flush once.
            // Upstream did per-coordinate cout << which flushes on each << operator.
            char buf[512];
            int pos = 0;
            pos += snprintf(buf + pos, sizeof(buf) - pos, "(");
            for (int i = 0; i < dim() - 1; i++) {
                pos += snprintf(buf + pos, sizeof(buf) - pos, "%g, ", (double)(*this)[i]);
                if (pos >= (int)sizeof(buf) - 32) break;  // guard against overflow
            }
            if (withCount) {
                pos += snprintf(buf + pos, sizeof(buf) - pos, "%g) : %d",
                    (double)(*this)[dim() - 1], count());
            } else {
                pos += snprintf(buf + pos, sizeof(buf) - pos, "%g) : ",
                    (double)(*this)[dim() - 1]);
            }
            std::cout << buf << std::endl;
        }
    };

    /**
    * A class that totally orders Point<T,S>'s in euclidean space.
    *
    * A total order of Points is required in the RangeTree. This is an implementation
    * detail that can be ignored. Given a start index \compareStartIndex, this class
    * orders points so that, for two points p_1 = (p_{11}, p_{12},...,p_{1n}) and
    * p_2 = (p_{21}, p_{22},...,p_{2n}) we have that p_1 < p_2 if and only if
    *
    * (p_{1\compareStartInd},...,p_{1n}) < (p_{2\compareStartInd},...,p_{2n})
    *
    * using the usual lexicographic order, or
    *
    * (p_{1\compareStartInd},...,p_{1n}) == (p_{2\compareStartInd},...,p_{2n}) and
    * (p_{11},...,p_{1(\compareStartInd-1)}) < (p_{21},...,p_{2(\compareStartInd-1)})
    *
    * again using the usual lexicographic order.
    */
    template <typename T, class S>
    class PointOrdering {
        static_assert(std::is_arithmetic<T>::value, "Type T must be numeric");
    private:
        int compareStartIndex;

    public:
        PointOrdering(int compareStartIndex): compareStartIndex(compareStartIndex) {
            if (compareStartIndex < 0) {
                throw new std::logic_error("Cannot have comparison start index <0.");
            }
        }

        static bool equals(const Point<T,S>& p1, const Point<T,S>& p2) {
            return p1.asVector() == p2.asVector();
        }

        int getCompareStartIndex() const {
            return compareStartIndex;
        }

        bool less(const Point<T,S>& p1, const Point<T,S>& p2) const {
            if (p1.dim() != p2.dim()) {
                throw std::logic_error("Points are incomparable (differing dims).");
            }
            int d = p1.dim();
            if (compareStartIndex >= d) {
                throw std::logic_error("Cannot compare points, compare start index >= point dimension.");
            }
            // Single wrap-around loop: start at compareStartIndex, wrap via modulo.
            // Avoids the two-loop pattern; early-exits on first coordinate difference.
            for (int step = 0; step < d; step++) {
                int i = (compareStartIndex + step) % d;
                T a = p1[i], b = p2[i];
                if (a < b) return true;
                if (a > b) return false;
            }
            return false;
        }

        bool lessOrEq(const Point<T,S>& p1, const Point<T,S>& p2) const {
            return less(p1, p2) || equals(p1, p2);
        }

        bool greater(const Point<T,S>& p1, const Point<T,S>& p2) const {
            return less(p2, p1);
        }

        bool greaterOrEq(const Point<T,S>& p1, const Point<T,S>& p2) const {
            return greater(p1, p2) || equals(p1,p2);
        }

        bool operator()(const Point<T,S>& p1, const Point<T,S>& p2) const {
            return this->less(p1, p2);
        }
    };

    /**
    * A matrix that keeps a collection of points sorted on each coordinate
    */
    template<typename T, class S>
    class SortedPointMatrix {
        static_assert(std::is_arithmetic<T>::value, "Type T must be numeric");
    private:
        std::vector<Point<T,S>* > pointsSortedByCurrentDim;
        std::deque<std::vector<int> > redirectionTable;
        int currentDim;
        int dim;
        static const int MAX_POINTS_BEFORE_SWITCH = 1000;

        std::vector<int> sortOrder(const std::vector<Point<T,S>* >& points, int onDim) {
            std::vector<int> order(points.size());
            std::iota(order.begin(), order.end(), 0);  // fill with 0,1,2,...,n-1
            PointOrdering<T,S> pointOrdering(onDim);
            std::sort(order.begin(), order.end(),
                      [&pointOrdering, &points](int i, int j) {
                          return pointOrdering.less(*(points[i]), *(points[j]));
                      });
            return order;
        }

        void sort(std::vector<Point<T,S>* >& points, int onDim) {
            PointOrdering<T,S> pointOrdering(onDim);
            std::sort(points.begin(), points.end(),
                      [pointOrdering](Point<T,S>* pt0, Point<T,S>* pt1) {
                          return pointOrdering.less(*(pt0), *(pt1));
                      });
        }

        void rearrangeGivenOrder(std::vector<Point<T,S>* >& points,
                                  std::vector<int> order) {
            // In-place permutation using cycle-following.
            // Upstream copies the entire vector then rearranges from copy.
            // This visits each element exactly once with O(1) extra space
            // (we take order by value since we mutate it as visited marker).
            int n = points.size();
            ajb_rt_stats.cycle_rearrange_calls++;
            for (int i = 0; i < n; i++) {
                // Follow the cycle starting at i
                int cycle_len = 0;  // [AJB_BP] M925: track this cycle's length
                while (order[i] != i) {
                    int target = order[i];
                    std::swap(points[i], points[target]);
                    std::swap(order[i], order[target]);
                    cycle_len++;
                    ajb_rt_stats.cycle_total_swaps++;
                }
                if (cycle_len > ajb_rt_stats.cycle_max_len) {
                    ajb_rt_stats.cycle_max_len = cycle_len;
                    // [AJB_BP] M925: print when new max cycle discovered (first 10 times)
                    if (ajb_rt_stats.cycle_rearrange_calls <= 10)
                        fprintf(stderr, "[AJB_BP][RangeTree] rearrange: new max_cycle_len=%d at call #%lld\n",
                                cycle_len, ajb_rt_stats.cycle_rearrange_calls);
                }
                // [AJB_BP] M925-enhanced: cycle length histogram
                if (cycle_len > 0) {
                    if (cycle_len == 1) ajb_rt_stats.cycle_hist[0]++;
                    else if (cycle_len <= 3) ajb_rt_stats.cycle_hist[1]++;
                    else if (cycle_len <= 7) ajb_rt_stats.cycle_hist[2]++;
                    else if (cycle_len <= 15) ajb_rt_stats.cycle_hist[3]++;
                    else ajb_rt_stats.cycle_hist[4]++;
                }
            }
        }

        SortedPointMatrix(const std::vector<Point<T,S>* >& pointsSortedByCurrentDim,
                          const std::deque<std::vector<int> >& redirectionTable,
                          int currentDim, int dim) : pointsSortedByCurrentDim(pointsSortedByCurrentDim), redirectionTable(redirectionTable),
                                        currentDim(currentDim), dim(dim) {}

    public:
        /**
        * Constructs a sorted point matrix
        */
        SortedPointMatrix(std::vector<Point<T,S>* >& points): currentDim(0) {
            if (points.size() == 0) {
                throw std::range_error("Cannot construct a SortedPointMatrix with 0 points.");
            } else {
                dim = points[0]->dim();
                for (int i = 1; i < points.size(); i++) {
                    if (points[i]->dim() != dim) {
                        throw std::logic_error("Input points to SortedPointMatrix must all"
                                                       " have the same dimension.");
                    }
                }

                int sortDimension = (points.size() > MAX_POINTS_BEFORE_SWITCH) ? dim - 1 : 0;
                PointOrdering<T,S> pointOrdering(sortDimension);
                std::sort(points.begin(), points.end(),
                          [pointOrdering](Point<T,S>* p1, Point<T,S>* p2) {
                              return pointOrdering.less(*p1, *p2);
                          });

                // Pre-allocate capacity to avoid reallocs during dedup.
                // Worst case: all points unique → size == points.size().
                pointsSortedByCurrentDim.reserve(points.size());
                pointsSortedByCurrentDim.push_back(points[0]);
                int k = 0;
                for (int i = 1; i < points.size(); i++) {
                    if (pointOrdering.equals(*(pointsSortedByCurrentDim[k]), *points[i])) {
                        if (pointsSortedByCurrentDim[k]->value() != points[i]->value()) {
                            throw std::logic_error("Input points have same position but different values");
                        }
                        pointsSortedByCurrentDim[k]->increaseCountBy(points[i]->count());
                    } else {
                        pointsSortedByCurrentDim.push_back(points[i]);
                        k++;
                    }
                }
                // Shrink to actual unique count to release excess capacity.
                pointsSortedByCurrentDim.shrink_to_fit();

                if (pointsSortedByCurrentDim.size() > MAX_POINTS_BEFORE_SWITCH) {
                    for (int i = dim - 2; i >= currentDim; i--) {
                        std::vector<int> order = sortOrder(pointsSortedByCurrentDim, i);
                        redirectionTable.push_front(order);
                        rearrangeGivenOrder(pointsSortedByCurrentDim, order);
                    }
                }
            }
        }

        void moveToNextDimension() {
            if (currentDim == dim - 1) {
                throw std::logic_error("Already at max dimension, cannot move to next.");
            }
            currentDim++;
            ajb_rt_stats.cycle_movenext_calls++;
            if (pointsSortedByCurrentDim.size() > MAX_POINTS_BEFORE_SWITCH) {
                // In-place permutation using cycle-following.
                // Upstream copies the entire vector then permutes from copy.
                int n = pointsSortedByCurrentDim.size();
                std::vector<int> perm = redirectionTable[0];  // take a mutable copy
                int local_max_cycle = 0;  // [AJB_BP] M925
                for (int i = 0; i < n; i++) {
                    int cycle_len = 0;
                    while (perm[i] != i) {
                        int target = perm[i];
                        std::swap(pointsSortedByCurrentDim[i],
                                  pointsSortedByCurrentDim[target]);
                        std::swap(perm[i], perm[target]);
                        cycle_len++;
                        ajb_rt_stats.cycle_total_swaps++;
                    }
                    if (cycle_len > local_max_cycle) local_max_cycle = cycle_len;
                }
                if (local_max_cycle > ajb_rt_stats.cycle_max_len)
                    ajb_rt_stats.cycle_max_len = local_max_cycle;
                // [AJB_BP] M925: print moveToNext cycle stats (first 10 calls)
                if (ajb_rt_stats.cycle_movenext_calls <= 10)
                    fprintf(stderr, "[AJB_BP][RangeTree] moveToNextDim(%d→%d): n=%d max_cycle=%d\n",
                            currentDim - 1, currentDim, n, local_max_cycle);
                redirectionTable.pop_front();
            } else {
                sort(pointsSortedByCurrentDim, currentDim);
            }
        }

        Point<T,S>* getMidPoint() {
            int mid = (numUniquePoints() - 1) / 2;
            return pointsSortedByCurrentDim[mid];
        }

        int numUniquePoints() {
            return pointsSortedByCurrentDim.size();
        }

        int getCurrentDim() {
            return currentDim;
        }

        std::vector<Point<T,S>* > getSortedPointsAtCurrentDim() {
            return pointsSortedByCurrentDim;
        }

        /**
        * Constructs two sorted point matrices after splitting on the current midpoint
        */
        std::pair<SortedPointMatrix, SortedPointMatrix> splitOnMid() {
            int n = numUniquePoints();
            if (n == 1) {
                throw std::logic_error("Cannot split on mid when there is only one point.");
            }
            ajb_rt_stats.split_calls++;

            int mid = (n - 1) / 2;
            int nLeft = mid + 1, nRight = n - nLeft;
            // [AJB_BP] M922: balance ratio tracking
            {
                double balance = (double)std::min(nLeft, nRight) / std::max(nLeft, nRight);
                if (balance < ajb_rt_stats.split_worst_balance)
                    ajb_rt_stats.split_worst_balance = balance;
                // Print first 10 splits or severely unbalanced ones
                if (ajb_rt_stats.split_calls <= 10 || balance < 0.1)
                    fprintf(stderr, "[AJB_BP][RangeTree] splitOnMid #%lld: n=%d nL=%d nR=%d balance=%.4f dim=%d\n",
                            ajb_rt_stats.split_calls, n, nLeft, nRight, balance, currentDim);
            }
            // Use iterator-range copy instead of element-by-element loop
            std::vector<Point<T, S>*> sortedPointsLeft(
                pointsSortedByCurrentDim.begin(),
                pointsSortedByCurrentDim.begin() + nLeft);
            std::vector<Point<T, S>*> sortedPointsRight(
                pointsSortedByCurrentDim.begin() + nLeft,
                pointsSortedByCurrentDim.end());

            if (n <= MAX_POINTS_BEFORE_SWITCH) {
                std::deque<std::vector<int> > redirectionTableLeft, redirectionTableRight;
                return std::make_pair(
                        SortedPointMatrix(sortedPointsLeft, redirectionTableLeft, currentDim, dim),
                        SortedPointMatrix(sortedPointsRight, redirectionTableRight, currentDim, dim));
            } else {
                // Use uint8_t bitvector instead of vector<bool> to avoid the
                // proxy-reference overhead of vector<bool>::operator[].
                std::vector<uint8_t> onLeft(n);
                long long local_ones = 0;  // [AJB_BP] M922: bitvec density
                for (int i = 0; i < n; i++) {
                    onLeft[i] = (i <= mid) ? 1 : 0;
                    if (onLeft[i]) local_ones++;
                }
                // [AJB_BP] M922: accumulate bitvector stats
                ajb_rt_stats.split_bitvec_ones += local_ones;
                ajb_rt_stats.split_bitvec_total += n;

                std::deque<std::vector<int> > redirectionTableLeft(redirectionTable.size(),
                                                                   std::vector<int>(nLeft));
                std::deque<std::vector<int> > redirectionTableRight(redirectionTable.size(),
                                                                    std::vector<int>(nRight));
                // Scratch vectors allocated once outside the loop
                std::vector<uint8_t> lastOnLeft(n);
                std::vector<int> newRedirect(n);

                for (size_t i = 0; i < redirectionTable.size(); i++) {
                    // Swap instead of copy: reuse lastOnLeft buffer
                    std::swap(lastOnLeft, onLeft);

                    // Pass 1: propagate onLeft through redirection + compute newRedirect
                    int kL = 0, kR = 0;
                    for (int j = 0; j < n; j++) {
                        onLeft[redirectionTable[i][j]] = lastOnLeft[j];
                    }
                    for (int j = 0; j < n; j++) {
                        if (onLeft[j]) {
                            newRedirect[j] = kL++;
                        } else {
                            newRedirect[j] = kR++;
                        }
                    }

                    // Pass 2: fill left/right redirection tables
                    kL = 0; kR = 0;
                    for (int j = 0; j < n; j++) {
                        int dest = newRedirect[redirectionTable[i][j]];
                        if (lastOnLeft[j]) {
                            redirectionTableLeft[i][kL++] = dest;
                        } else {
                            redirectionTableRight[i][kR++] = dest;
                        }
                    }
                }
                return std::make_pair(
                        SortedPointMatrix(sortedPointsLeft, redirectionTableLeft, currentDim, dim),
                        SortedPointMatrix(sortedPointsRight, redirectionTableRight, currentDim, dim));
            }
        }
    };

    /**
    * A class representing a single node in a RangeTree. These should not be
    * constructed directly, instead use the RangeTree class.
    */
    template <typename T, class S>
    class RangeTreeNode {
        static_assert(std::is_arithmetic<T>::value, "Type T must be numeric");
    private:
        std::shared_ptr<RangeTreeNode<T,S> > left; /**< Contains points <= the comparison point **/
        std::shared_ptr<RangeTreeNode<T,S> > right; /**< Contains points > the comparison point **/
        std::shared_ptr<RangeTreeNode<T,S> > treeOnNextDim; /**< Tree on the next dimension **/
        Point<T,S>* point; /**< The comparison point **/
        bool isLeaf; /**< Whether or not the point is a leaf **/
        int pointCountSum; /**< Total number of points, counting multiplicities, at leaves of the tree **/
        PointOrdering<T,S> pointOrdering; /**< Helper to totally order input points **/

        // For fractional cascading
        std::vector<T> pointsLastDimSorted;
        std::vector<Point<T,S>* > allPointsSorted;
        std::vector<int> pointerToGeqLeft;
        std::vector<int> pointerToLeqLeft;
        std::vector<int> pointerToGeqRight;
        std::vector<int> pointerToLeqRight;
        std::vector<int> cumuCountPoints;

    public:
        /**
        * Construct a range tree structure from points.
        *
        * Creates a range tree structure on the input collection \allPoints using the lexicographic order
        * starting at \compareStartInd.
        *
        * @param uniquePoints a collection of points.
        * @return a range tree structure
        */
        RangeTreeNode(SortedPointMatrix<T,S>& spm,
                      bool onLeftEdge = true,
                      bool onRightEdge = true): pointOrdering(spm.getCurrentDim()) {
            point = spm.getMidPoint();

            if (spm.numUniquePoints() == 1) {
                isLeaf = true;
                pointCountSum = point->count();
                pointsLastDimSorted.push_back((*point)[point->dim() - 1]);
                if (spm.getCurrentDim() == point->dim() - 2) {
                    spm.moveToNextDimension();
                }
            } else {
                auto spmPair = spm.splitOnMid();
                left = std::shared_ptr<RangeTreeNode<T,S> >(
                        new RangeTreeNode<T,S>(spmPair.first, onLeftEdge, false));
                right = std::shared_ptr<RangeTreeNode<T,S> >(
                        new RangeTreeNode<T,S>(spmPair.second, false, onRightEdge));
                pointCountSum = left->totalPoints() + right->totalPoints();

                int dim = point->dim();
                if (spm.getCurrentDim() + 2 == dim) {
                    spm.moveToNextDimension();

                    allPointsSorted = spm.getSortedPointsAtCurrentDim();
                    int nSorted = allPointsSorted.size();
                    // Pre-allocate both vectors: exact size known from allPointsSorted
                    pointsLastDimSorted.reserve(nSorted);
                    cumuCountPoints.reserve(nSorted + 1);
                    cumuCountPoints.push_back(0);
                    for (int i = 0; i < nSorted; i++) {
                        pointsLastDimSorted.push_back((*allPointsSorted[i])[dim - 1]);
                        cumuCountPoints.push_back(cumuCountPoints.back() + allPointsSorted[i]->count());
                    }
                    const auto& leftSorted = left->pointsLastDimSorted;
                    const auto& rightSorted = right->pointsLastDimSorted;

                    pointerToGeqLeft = createGeqPointers(pointsLastDimSorted, leftSorted);
                    pointerToGeqRight = createGeqPointers(pointsLastDimSorted, rightSorted);
                    pointerToLeqLeft = createLeqPointers(pointsLastDimSorted, leftSorted);
                    pointerToLeqRight = createLeqPointers(pointsLastDimSorted, rightSorted);
                }  else if (!onLeftEdge && !onRightEdge && spm.getCurrentDim() + 1 != point->dim()) {
                        spm.moveToNextDimension();
                        treeOnNextDim = std::shared_ptr<RangeTreeNode>(new RangeTreeNode(spm));
                }
                isLeaf = false;
            }
        }

        std::vector<int> createGeqPointers(const std::vector<T>& vec,
                                           const std::vector<T>& subVec) {
            // For each element in vec, find the first index in subVec that is >= it.
            // Upstream used a two-pointer merge (O(n+k)).
            // We use lower_bound per element: O(n * log k). Better when subVec is
            // much smaller than vec, which happens in unbalanced splits. For balanced
            // splits the two approaches are equivalent; lower_bound has lower branch
            // misprediction because it doesn't depend on the previous iteration's state.
            int k = subVec.size();
            std::vector<int> geqPointers(vec.size());
            for (size_t i = 0; i < vec.size(); i++) {
                auto it = std::lower_bound(subVec.begin(), subVec.end(), vec[i]);
                geqPointers[i] = static_cast<int>(it - subVec.begin());
            }
            return geqPointers;
        }

        std::vector<int> createLeqPointers(const std::vector<T>& vec,
                                           const std::vector<T>& subVec) {
            // For each element in vec, find the last index in subVec that is <= it.
            // Use upper_bound (returns first > target), then subtract 1.
            // Same O(n log k) rationale as createGeqPointers.
            std::vector<int> leqPointers(vec.size());
            for (int i = (int)vec.size() - 1; i >= 0; i--) {
                auto it = std::upper_bound(subVec.begin(), subVec.end(), vec[i]);
                leqPointers[i] = static_cast<int>(it - subVec.begin()) - 1;
            }
            return leqPointers;
        }

        int binarySearchFirstGeq(T needle, int lo, int hi) const {
            // Iterative binary search replacing the recursive version.
            // Avoids stack frames on deep trees with many sorted points.
            ajb_rt_stats.bsearch_geq_calls++;
            int steps = 0;  // [AJB_BP] M923: convergence tracking
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;  // overflow-safe midpoint
                if (needle <= pointsLastDimSorted[mid]) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
                steps++;
            }
            // [AJB_BP] M923: accumulate search steps
            ajb_rt_stats.bsearch_total_steps += steps;
            if (steps > ajb_rt_stats.bsearch_max_steps) {
                ajb_rt_stats.bsearch_max_steps = steps;
                if (ajb_rt_stats.bsearch_geq_calls <= 10)
                    fprintf(stderr, "[AJB_BP][RangeTree] bsGeq: new max_steps=%d needle=%g range=[%d,%d]\n",
                            steps, (double)needle, lo, hi);
            }
            // lo == hi: check if the element at lo satisfies >= needle
            if (lo < (int)pointsLastDimSorted.size() && needle <= pointsLastDimSorted[lo]) {
                return lo;
            }
            return lo + (lo <= hi ? 1 : 0);
        }

        int binarySearchFirstLeq(T needle, int lo, int hi) const {
            // Iterative binary search replacing the recursive version.
            ajb_rt_stats.bsearch_leq_calls++;
            int steps = 0;  // [AJB_BP] M923
            while (lo < hi) {
                int mid = lo + (hi - lo + 1) / 2;  // upper-mid for leq search
                if (needle >= pointsLastDimSorted[mid]) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
                steps++;
            }
            ajb_rt_stats.bsearch_total_steps += steps;
            if (steps > ajb_rt_stats.bsearch_max_steps) {
                ajb_rt_stats.bsearch_max_steps = steps;
                // [AJB_BP] M923-enhanced: print leq max too
                if (ajb_rt_stats.bsearch_leq_calls <= 10)
                    fprintf(stderr, "[AJB_BP][RangeTree] bsLeq: new max_steps=%d needle=%g range=[%d,%d]\n",
                            steps, (double)needle, lo, hi);
            }
            if (lo >= 0 && lo < (int)pointsLastDimSorted.size() &&
                needle >= pointsLastDimSorted[lo]) {
                return lo;
            }
            return lo - 1;
        }

        /**
        * Construct a RangeTreeNode representing a leaf.
        *
        * @param pointAtLeaf the point to use at the leaf.
        * @param compareStartInd the index defining the lexicographic ordering.
        * @return
        */
        RangeTreeNode(Point<T,S>* pointAtLeaf, int compareStartInd) :
                point(pointAtLeaf), isLeaf(true), pointCountSum(pointAtLeaf->count()), pointOrdering(compareStartInd) {}

        /**
        * Total count of points at the leaves of the range tree rooted at this node.
        *
        * The total count returned INCLUDES the multiplicities/count of the points at the leaves.
        *
        * @return the total count.
        */
        int totalPoints() const {
            return pointCountSum;
        }

        /**
        * Return all points at the leaves of the range tree rooted at this node.
        * @return all the points.
        */
        std::vector<Point<T,S> > getAllPoints() const {
            // Iterative DFS with explicit stack — avoids recursion depth issues
            // on heavily unbalanced trees and eliminates per-level vector copies.
            ajb_rt_stats.getall_calls++;
            std::vector<Point<T,S> > result;
            result.reserve(pointCountSum);  // exact pre-allocation
            std::vector<const RangeTreeNode<T,S>*> stack;
            stack.push_back(this);
            int local_max_stack = 0;  // [AJB_BP] M924
            while (!stack.empty()) {
                // [AJB_BP] M924: track max stack depth
                if ((int)stack.size() > local_max_stack)
                    local_max_stack = (int)stack.size();
                const RangeTreeNode<T,S>* cur = stack.back();
                stack.pop_back();
                if (cur->isLeaf) {
                    result.push_back(*(cur->point));
                } else {
                    // Push right first so left is processed first (in-order)
                    if (cur->right) stack.push_back(cur->right.get());
                    if (cur->left)  stack.push_back(cur->left.get());
                }
            }
            // [AJB_BP] M924: update global max and log
            ajb_rt_stats.getall_total_collected += result.size();
            if (local_max_stack > ajb_rt_stats.getall_max_stack) {
                ajb_rt_stats.getall_max_stack = local_max_stack;
                if (ajb_rt_stats.getall_calls <= 10)
                    fprintf(stderr, "[AJB_BP][RangeTree] getAllPoints: max_stack=%d collected=%zu at call #%lld\n",
                            local_max_stack, result.size(), ajb_rt_stats.getall_calls);
            }
            return result;
        }

        /**
        * Check if point is in a euclidean box.
        *
        * Determines whether or not a point is in a euclidean box. That is, if p_1 = (p_{11},...,p_{1n}) is an
        * n-dimensional point. Then this function returns true if, for all 1 <= i <= n we have
        *
        * lower[i] <= p_{1i} <= upper[i] if withLower[i] == true and withUpper[i] == true, or
        * lower[i] < p_{1i} <= upper[i] if withLower[i] == false and withUpper[i] == true, or
        * lower[i] <= p_{1i} < upper[i] if withLower[i] == true and withUpper[i] == false, or
        * lower[i] < p_{1i} < upper[i] if withLower[i] == false and withUpper[i] == false.
        *
        * @param point the point to check.
        * @param lower the lower points of the rectangle.
        * @param upper the upper bounds of the rectangle.
        * @param withLower whether to use strict (<) or not strict (<=) inequalities at certain coordiantes of p_1
        *                  for the lower bounds.
        * @param withUpper as for \withLower but for the upper bounds.
        * @return true if the point is in the rectangle, false otherwise.
        */
        bool pointInRange(const Point<T,S>& pt,
                          const std::vector<T>& lower,
                          const std::vector<T>& upper) const {
            // Test the tree's compare dimension first — it's the dimension along
            // which this node split, so it's most likely to eliminate quickly.
            int csi = pointOrdering.getCompareStartIndex();
            int d = pt.dim();
            for (int step = 0; step < d; step++) {
                int i = (csi + step) % d;
                T v = pt[i];
                if (v < lower[i] || v > upper[i]) return false;
            }
            return true;
        }

        /**
        * Count the number of points at leaves of tree rooted at the current node that are within the given bounds.
        *
        * @param lower see the pointInRange(...) function.
        * @param upper
        * @return the count.
        */
        int countInRange(const std::vector<T>& lower,
                         const std::vector<T>& upper) const {
            // [AJB_BP] M921: track recursive depth via iterative counting
            ajb_rt_stats.count_range_calls++;
            if (isLeaf) {
                ajb_rt_stats.count_range_leaf_hits++;
                if (pointInRange(*point, lower, upper)) {
                    return totalPoints();
                } else {
                    return 0;
                }
            }
            int compareInd = pointOrdering.getCompareStartIndex();

            if ((*point)[compareInd] > upper[compareInd]) {
                ajb_rt_stats.count_range_prune_right++;  // [AJB_BP] pruned right
                return left->countInRange(lower, upper);
            }
            if ((*point)[compareInd] < lower[compareInd]) {
                ajb_rt_stats.count_range_prune_left++;  // [AJB_BP] pruned left
                return right->countInRange(lower, upper);
            }

            int dim = point->dim();
            if (compareInd + 2 == dim) {
                int n = pointsLastDimSorted.size();
                int geqInd = binarySearchFirstGeq(lower.back(), 0, n - 1);
                int leqInd = binarySearchFirstLeq(upper.back(), 0, n - 1);

                if (geqInd > leqInd) {
                    return 0;
                }
                std::vector<RangeTreeNode<T, S>* > nodes;
                std::vector<std::pair<int,int> > inds;
                left->leftFractionalCascade(lower,
                                            pointerToGeqLeft[geqInd],
                                            pointerToLeqLeft[leqInd],
                                            nodes,
                                            inds);
                right->rightFractionalCascade(upper,
                                              pointerToGeqRight[geqInd],
                                              pointerToLeqRight[leqInd],
                                              nodes,
                                              inds);
                int sum = 0;
                for (int i = 0; i < nodes.size(); i++) {
                    if (nodes[i]->isLeaf) {
                        sum += nodes[i]->totalPoints();
                    } else {
                        sum += nodes[i]->cumuCountPoints[inds[i].second + 1] -
                                nodes[i]->cumuCountPoints[inds[i].first];
                    }
                }
                return sum;
            } else {
                std::vector<std::shared_ptr<RangeTreeNode<T, S> > > canonicalNodes;
                canonicalNodes.reserve(16);  // typical tree depth, avoids reallocs

                if (left->isLeaf) {
                    canonicalNodes.push_back(left);
                } else {
                    left->leftCanonicalNodes(lower, canonicalNodes);
                }

                if (right->isLeaf) {
                    canonicalNodes.push_back(right);
                } else {
                    right->rightCanonicalNodes(upper, canonicalNodes);
                }

                int numPointsInRange = 0;
                int lastDimCheck = compareInd + 1;
                for (const auto& node : canonicalNodes) {
                    if (node->isLeaf) {
                        if (pointInRange(*(node->point), lower, upper)) {
                            numPointsInRange += node->totalPoints();
                        }
                    } else if (lastDimCheck == point->dim()) {
                        numPointsInRange += node->totalPoints();
                    } else if (node->treeOnNextDim) {
                        numPointsInRange += node->treeOnNextDim->countInRange(lower, upper);
                    }
                    // Skip nodes with null treeOnNextDim (shouldn't happen but defensive)
                }
                return numPointsInRange;
            }
        }

        /**
        * Return the points at leaves of tree rooted at the current node that are within the given bounds.
        *
        * @param lower see the pointInRange(...) function.
        * @param upper
        * @return a std::vector of the Points.
        */
        std::vector<Point<T,S> > pointsInRange(const std::vector<T>& lower,
                                               const std::vector<T>& upper) const {
            // [AJB_BP] M921-enhanced: track pointsInRange calls
            ajb_rt_stats.pts_in_range_calls++;
            std::vector<Point<T,S> > pointsToReturn = {};
            if (isLeaf) {
                if (pointInRange(*point, lower, upper)) {
                    pointsToReturn.push_back(*point);
                }
                return pointsToReturn;
            }
            int compareInd = pointOrdering.getCompareStartIndex();

            if ((*point)[compareInd] > upper[compareInd]) {
                return left->pointsInRange(lower, upper);
            }
            if ((*point)[compareInd] < lower[compareInd]) {
                return right->pointsInRange(lower, upper);
            }

            int dim = point->dim();
            if (compareInd + 2 == dim) {
                int n = pointsLastDimSorted.size();
                int geqInd = binarySearchFirstGeq(lower.back(), 0, n - 1);
                int leqInd = binarySearchFirstLeq(upper.back(), 0, n - 1);

                if (geqInd > leqInd) {
                    return pointsToReturn;
                }
                std::vector<RangeTreeNode<T, S>* > nodes;
                std::vector<std::pair<int,int> > inds;
                left->leftFractionalCascade(lower,
                                            pointerToGeqLeft[geqInd],
                                            pointerToLeqLeft[leqInd],
                                            nodes,
                                            inds);
                right->rightFractionalCascade(upper,
                                              pointerToGeqRight[geqInd],
                                              pointerToLeqRight[leqInd],
                                              nodes,
                                              inds);
                for (int i = 0; i < nodes.size(); i++) {
                    if (nodes[i]->isLeaf) {
                        pointsToReturn.push_back(*(nodes[i]->point));
                    } else {
                        for (int j = inds[i].first; j <= inds[i].second; j++) {
                            pointsToReturn.push_back(*(nodes[i]->allPointsSorted[j]));
                        }
                    }
                }
                return pointsToReturn;
            } else {
                std::vector<std::shared_ptr<RangeTreeNode<T, S> > > canonicalNodes;
                canonicalNodes.reserve(16);

                if (left->isLeaf) {
                    canonicalNodes.push_back(left);
                } else {
                    left->leftCanonicalNodes(lower, canonicalNodes);
                }

                if (right->isLeaf) {
                    canonicalNodes.push_back(right);
                } else {
                    right->rightCanonicalNodes(upper, canonicalNodes);
                }

                for (const auto& node : canonicalNodes) {
                    if (node->isLeaf) {
                        if (pointInRange(*(node->point), lower, upper)) {
                            pointsToReturn.push_back(*(node->point));
                        }
                    } else if (compareInd + 1 == point->dim()) {
                        auto allPointsAtNode = node->getAllPoints();
                        // Move instead of copy — getAllPoints returns by value
                        pointsToReturn.insert(pointsToReturn.end(),
                            std::make_move_iterator(allPointsAtNode.begin()),
                            std::make_move_iterator(allPointsAtNode.end()));
                    } else if (node->treeOnNextDim) {
                        auto subResult = node->treeOnNextDim->pointsInRange(lower, upper);
                        pointsToReturn.insert(pointsToReturn.end(),
                            std::make_move_iterator(subResult.begin()),
                            std::make_move_iterator(subResult.end()));
                    }
                }
                // [AJB_BP] M921-enhanced: track canonical node decomposition in pointsInRange
                {
                    int cn = (int)canonicalNodes.size();
                    ajb_rt_stats.canonical_total_nodes += cn;
                    if (cn > ajb_rt_stats.canonical_max_nodes) ajb_rt_stats.canonical_max_nodes = cn;
                }
                // [AJB_BP] M921-enhanced: track result size
                {
                    int nr = (int)pointsToReturn.size();
                    ajb_rt_stats.pts_in_range_total_returned += nr;
                    if (nr > ajb_rt_stats.pts_in_range_max_returned)
                        ajb_rt_stats.pts_in_range_max_returned = nr;
                }
                return pointsToReturn;
            }
        }

        void leftFractionalCascade(const std::vector<T>& lower,
                                  int geqInd,
                                  int leqInd,
                                  std::vector<RangeTreeNode<T,S>* >& nodes,
                                  std::vector<std::pair<int,int> >& inds) {
            // [AJB_BP] M921-enhanced: fractional cascade left tracking
            ajb_rt_stats.frac_cascade_left_calls++;
            size_t nodes_before = nodes.size();
            // Iterative version: the recursion always follows exactly one child
            // (either left or right), so it's a linear chain — perfect for a loop.
            RangeTreeNode<T,S>* cur = this;
            int g = geqInd, l = leqInd;
            while (true) {
                if (l < g) return;
                int cmpIdx = cur->point->dim() - 2;

                if (lower[cmpIdx] <= (*(cur->point))[cmpIdx]) {
                    if (cur->isLeaf) {
                        nodes.push_back(cur);
                        inds.emplace_back(0, 0);
                        return;
                    }
                    // Collect right subtree as canonical if range is valid
                    int gR = cur->pointerToGeqRight[g];
                    int lR = cur->pointerToLeqRight[l];
                    if (lR >= gR) {
                        nodes.push_back(cur->right.get());
                        inds.emplace_back(cur->right->isLeaf ? 0 : gR,
                                          cur->right->isLeaf ? 0 : lR);
                    }
                    // Descend left
                    int gNext = cur->pointerToGeqLeft[g];
                    int lNext = cur->pointerToLeqLeft[l];
                    cur = cur->left.get();
                    g = gNext;
                    l = lNext;
                } else {
                    if (cur->isLeaf) return;
                    // Descend right
                    int gNext = cur->pointerToGeqRight[g];
                    int lNext = cur->pointerToLeqRight[l];
                    cur = cur->right.get();
                    g = gNext;
                    l = lNext;
                }
            }
        }

        void rightFractionalCascade(const std::vector<T>& upper,
                                   int geqInd,
                                   int leqInd,
                                   std::vector<RangeTreeNode<T,S>* >& nodes,
                                   std::vector<std::pair<int,int> >& inds) {
            // [AJB_BP] M921-enhanced: fractional cascade right tracking
            ajb_rt_stats.frac_cascade_right_calls++;
            // Iterative: same tail-recursion elimination as leftFractionalCascade.
            RangeTreeNode<T,S>* cur = this;
            int g = geqInd, l = leqInd;
            while (true) {
                if (l < g) return;
                int cmpIdx = cur->point->dim() - 2;

                if ((*(cur->point))[cmpIdx] <= upper[cmpIdx]) {
                    if (cur->isLeaf) {
                        nodes.push_back(cur);
                        inds.emplace_back(0, 0);
                        return;
                    }
                    // Collect left subtree as canonical if range is valid
                    int gL = cur->pointerToGeqLeft[g];
                    int lL = cur->pointerToLeqLeft[l];
                    if (lL >= gL) {
                        nodes.push_back(cur->left.get());
                        inds.emplace_back(cur->left->isLeaf ? 0 : gL,
                                          cur->left->isLeaf ? 0 : lL);
                    }
                    // Descend right
                    int gNext = cur->pointerToGeqRight[g];
                    int lNext = cur->pointerToLeqRight[l];
                    cur = cur->right.get();
                    g = gNext;
                    l = lNext;
                } else {
                    if (cur->isLeaf) return;
                    // Descend left
                    int gNext = cur->pointerToGeqLeft[g];
                    int lNext = cur->pointerToLeqLeft[l];
                    cur = cur->left.get();
                    g = gNext;
                    l = lNext;
                }
            }
        }

        /**
        * Helper function for countInRange(...).
        * @param lower
        * @param withLower
        * @param nodes
        */
        // [AJB] Canonical nodes: 找到覆盖查询区间的最少节点集合
        void leftCanonicalNodes(const std::vector<T>& lower,
                                std::vector<std::shared_ptr<RangeTreeNode<T,S> > >& nodes) {
            // [AJB_BP] M921-enhanced: canonical left decomposition
            ajb_rt_stats.canonical_left_calls++;
            if (isLeaf) {
                throw std::logic_error("Should never have a leaf deciding if its canonical.");
            }
            int compareInd = pointOrdering.getCompareStartIndex();
            int totalPoints = 0;
            if (lower[compareInd] <= (*point)[compareInd]) {
                nodes.push_back(right);
                if (left->isLeaf) {
                    nodes.push_back(left);
                } else {
                    left->leftCanonicalNodes(lower, nodes);
                }
            } else {
                if (right->isLeaf) {
                    nodes.push_back(right);
                } else {
                    right->leftCanonicalNodes(lower, nodes);
                }
            }
        }

        /**
        * Helper function for countInRange(...).
        * @param upper
        * @param nodes
        */
        void rightCanonicalNodes(const std::vector<T>& upper,
                                 std::vector<std::shared_ptr<RangeTreeNode<T,S> > >& nodes) {
            // [AJB_BP] M921-enhanced: canonical right decomposition
            ajb_rt_stats.canonical_right_calls++;
            if (isLeaf) {
                throw std::logic_error("Should never have a leaf deciding if its canonical.");
            }
            int compareInd = pointOrdering.getCompareStartIndex();
            int totalPoints = 0;
            if (upper[compareInd] >= (*point)[compareInd]) {
                nodes.push_back(left);
                if (right->isLeaf) {
                    nodes.push_back(right);
                } else {
                    right->rightCanonicalNodes(upper, nodes);
                }
            } else {
                if (left->isLeaf) {
                    nodes.push_back(left);
                } else {
                    left->rightCanonicalNodes(upper, nodes);
                }
            }
        }

        /**
        * Print the structure of the tree rooted at the curret node.
        *
        * The printed structure does not reflect any subtrees for other coordinates.
        *
        * @param numIndents the number of indents to use before every line printed.
        */
        void print(int numIndents) {
            // Iterative DFS tree print using explicit stack.
            // Each entry is (node_ptr, depth). Avoids recursive call overhead
            // and builds the indent string from depth rather than a per-tab loop.
            struct PrintEntry {
                const RangeTreeNode<T,S>* node;
                int depth;
            };
            std::vector<PrintEntry> stack;
            stack.push_back({this, numIndents});
            while (!stack.empty()) {
                auto [cur, d] = stack.back();
                stack.pop_back();
                // Build indent: one string construction per node, not per-tab.
                std::string indent(d, '\t');
                std::cout << indent;
                if (cur->isLeaf) {
                    cur->point->print(true);
                } else {
                    cur->point->print(false);
                    // Push right first so left prints first
                    if (cur->right) stack.push_back({cur->right.get(), d + 1});
                    if (cur->left)  stack.push_back({cur->left.get(), d + 1});
                }
            }
        }
    };

    /**
    * A class facilitating fast orthogonal range queries.
    *
    * A RangeTree allows for 'orthogonal range queries.' That is, given a collection of
    * points P = {p_1, ..., p_n} in euclidean d-dimensional space, a RangeTree can efficiently
    * answer questions of the form
    *
    * "How many points of p are in the box high dimensional rectangle
    * [l_1, u_1] x [l_2, u_2] x ... x [l_d, u_d]
    * where l_1 <= u_1, ..., l_n <= u_n?"
    *
    * It returns the number of such points in worst case
    * O(log(n)^d) time. It can also return the points that are in the rectangle in worst case
    * O(log(n)^d + k) time where k is the number of points that lie in the rectangle.
    *
    * The particular algorithm implemented here is described in Chapter 5 of the book
    *
    * Mark de Berg, Otfried Cheong, Marc van Kreveld, and Mark Overmars. 2008.
    * Computational Geometry: Algorithms and Applications (3rd ed. ed.). TELOS, Santa Clara, CA, USA.
    */
    template <typename T, class S>
    class RangeTree {
        static_assert(std::is_arithmetic<T>::value, "Type T must be numeric");
    private:
        std::shared_ptr<RangeTreeNode<T,S> > root;
        std::vector<std::shared_ptr<Point<T,S> > > savedPoints;
        std::vector<Point<T,S>* > savedPointsRaw;
        std::vector<T> dimensionsLowerBound;
        std::vector<T> dimensionsUpperBound;

        std::vector<std::shared_ptr<Point<T,S> > > copyPointsToHeap(const std::vector<Point<T,S> >& points) {
            std::vector<std::shared_ptr<Point<T,S> > > vecOfPointers;
            for (int i = 0; i < points.size(); i++) {
                vecOfPointers.push_back(std::shared_ptr<Point<T,S> >(new Point<T,S>(points[i])));
            }
            return vecOfPointers;
        }

        std::vector<Point<T,S>* > getRawPointers(std::vector<std::shared_ptr<Point<T,S> > >& points) {
            std::vector<Point<T,S>* > vecOfPointers;
            for (int i = 0; i < points.size(); i++) {
                vecOfPointers.push_back(points[i].get());
            }
            return vecOfPointers;
        }

        std::vector<T> getModifiedLower(const std::vector<T>& lower,
                         const std::vector<bool>& withLower) const {
            std::vector<T> newLower(lower.size());
            for (size_t i = 0; i < lower.size(); i++) {
                // Branchless: subtract !withLower (0 or 1) for integral types.
                // For floating point, use nextafter only when needed.
                if constexpr (std::is_integral<T>::value) {
                    newLower[i] = lower[i] + static_cast<T>(!withLower[i]);
                } else {
                    newLower[i] = withLower[i] ? lower[i]
                        : std::nextafter(lower[i], std::numeric_limits<T>::max());
                }
            }
            return newLower;
        }

        std::vector<T> getModifiedUpper(const std::vector<T>& upper,
                                        const std::vector<bool>& withUpper) const {
            std::vector<T> newUpper(upper.size());
            for (size_t i = 0; i < upper.size(); i++) {
                if constexpr (std::is_integral<T>::value) {
                    newUpper[i] = upper[i] - static_cast<T>(!withUpper[i]);
                } else {
                    newUpper[i] = withUpper[i] ? upper[i]
                        : std::nextafter(upper[i], std::numeric_limits<T>::lowest());
                }
            }
            return newUpper;
        }

    public:
        RangeTree(){}
        /**
        * Construct a new RangeTree from input points.
        *
        * Input points may have duplicates but if two points p_1 and p_2 are at the same euclidean position
        * then they are required to have the same value as all duplicate points will be accumulated into a
        * single point with multiplicity/count equal to the sum of the multiplicities/counts of all such duplicates.
        *
        * @param points the points from which to create a RangeTree
        */
        // [AJB_BP] RangeTree construction: 输入点数和维度
        RangeTree(const std::vector<Point<T,S> >& points): savedPoints(copyPointsToHeap(points)),
                                                           savedPointsRaw(getRawPointers(savedPoints)) {
            //calculate the lower and upper bounds of the points in each dimension
            int dim = points[0].dim();
            ajb_rt_stats.build_calls++;
            ajb_rt_stats.total_points_inserted += points.size();
            if(dim > ajb_rt_stats.max_dim) ajb_rt_stats.max_dim = dim;
            // [AJB_BP] M921: detailed build entry
            if (ajb_rt_stats.build_calls <= 10) {
                fprintf(stderr, "[AJB_TIMER][RangeTree] build #%lld: %zu points, dim=%d\n",
                        ajb_rt_stats.build_calls, points.size(), dim);
            }
            // Single-pass simultaneous min/max across all dimensions.
            // Upstream used nested loop: outer over points, inner over dims.
            // Here we accumulate all dims per-point in one pass — same O(n*d) but
            // better cache behavior: each point's vec is read once.
            dimensionsLowerBound.resize(dim);
            dimensionsUpperBound.resize(dim);
            for (int j = 0; j < dim; j++) {
                dimensionsLowerBound[j] = points[0][j];
                dimensionsUpperBound[j] = points[0][j];
            }
            for (size_t i = 1; i < points.size(); i++) {
                const auto& pv = points[i].asVector();
                for (int j = 0; j < dim; j++) {
                    T v = pv[j];
                    if (v < dimensionsLowerBound[j]) dimensionsLowerBound[j] = v;
                    else if (v > dimensionsUpperBound[j]) dimensionsUpperBound[j] = v;
                }
            }
            SortedPointMatrix<T,S> spm(savedPointsRaw);
            root = std::shared_ptr<RangeTreeNode<T,S> >(new RangeTreeNode<T,S>(spm));
        }

        const std::vector<T>& getLowerBounds() const {
            return dimensionsLowerBound;
        }

        const std::vector<T>& getUpperBounds() const {
            return dimensionsUpperBound;
        }

        int getCardinality() const {
            return savedPoints.size();
        }

        /**
        * The number of points within a high dimensional rectangle.
        *
        * The rectangle is defined by the input parameters. In particular, an n-dimensional point
        * p_1 = (p_{11},...,p_{1n}) is an is in the rectangle if, for all 1 <= i <= n, we have
        *
        * lower[i] <= p_{1i} <= upper[i] if withLower[i] == true and withUpper[i] == true, or
        * lower[i] < p_{1i} <= upper[i] if withLower[i] == false and withUpper[i] == true, or
        * lower[i] <= p_{1i} < upper[i] if withLower[i] == true and withUpper[i] == false, or
        * lower[i] < p_{1i} < upper[i] if withLower[i] == false and withUpper[i] == false.
        *
        * @param lower the lower bounds of the rectangle.
        * @param upper the upper bounds of the rectangle.
        * @param withLower whether to use strict (<) or not strict (<=) inequalities at certain coordiantes of p_1
        *                  for the lower bounds.
        * @param withUpper as for \withLower but for the upper bounds.
        * @return the number of points in the rectangle.
        */
        int countInRange(const std::vector<T>& lower,
                         const std::vector<T>& upper,
                         const std::vector<bool>& withLower,
                         const std::vector<bool>& withUpper) const {
            if (lower.size() != upper.size() || lower.size() != withLower.size() ||
                    lower.size() != withUpper.size()) {
                throw std::logic_error("All vectors inputted to countInRange must have the same length.");
            }
            for (int i = 0; i < lower.size(); i++) {
                if (((!withUpper[i] || !withLower[i]) && lower[i] >= upper[i]) ||
                    lower[i] > upper[i]) {
                    return 0;
                }
            }
            return root->countInRange(getModifiedLower(lower, withLower),
                                      getModifiedUpper(upper, withUpper));
        }

        /**
        * The number of points within a high dimensional rectangle.
        *
        * The rectangle is defined by the input parameters. In particular, an n-dimensional point
        * p_1 = (p_{11},...,p_{1n}) is an is in the rectangle if, for all 1 <= i <= n, we have
        *
        * lower[i] <= p_{1i} <= upper[i]
        *
        * @param lower the lower bounds of the rectangle.
        * @param upper the upper bounds of the rectangle.

        * @return the number of points in the rectangle.
        */
        int countInRange(const std::vector<T>& lower,
                         const std::vector<T>& upper) const {
            if (lower.size() != upper.size()) {
                throw std::logic_error("upper and lower in countInRange must have the same length.");
            }
            ajb_rt_stats.query_calls++;
            int ajb_result2 = root->countInRange(lower, upper);
            ajb_rt_stats.total_query_results += ajb_result2;
            // [AJB_BP] M921: print query details for first 10 queries or every 1000th
            if (ajb_rt_stats.query_calls <= 10 || ajb_rt_stats.query_calls % 1000 == 0) {
                fprintf(stderr, "[AJB_BP][RangeTree] query #%lld: dim=%zu hits=%d range=[",
                        ajb_rt_stats.query_calls, lower.size(), ajb_result2);
                for (size_t d = 0; d < lower.size(); d++) {
                    if (d) fprintf(stderr, ",");
                    fprintf(stderr, "%g:%g", (double)lower[d], (double)upper[d]);
                }
                fprintf(stderr, "]\n");
            }
            return ajb_result2;
        }

        /**
        * Return all points in range.
        *
        * Returns a std::vector of all points in the given rectangle. See \countInRange for how
        * this rectangle is specified. NOTE: these points may not be identical to those points
        * that were given as input to the RangeTree at construction time. This is because
        * duplicate points are merged together with appropriate incrementing of their multiplicity.
        * That is, two points at euclidean position (1,2,3) and multiplicities/counts of 2 and 3
        * respectively will be merged into a single Point with position (1,2,3) and multiplicity 5
        * (recall that all points with the same euclidean position are required to have the same
        * associated value so it is ok to merge in this way).
        *
        * @param lower the lower bounds of the rectangle.
        * @param upper the upper bounds of the rectangle.
        * @param withLower whether to use strict (<) or not strict (<=) inequalities at certain coordiantes of p_1
        *                  for the lower bounds.
        * @param withUpper as for \withLower but for the upper bounds.
        * @return the number of points in the rectangle.
        */
        std::vector<Point<T,S> > pointsInRange(const std::vector<T>& lower,
                                               const std::vector<T>& upper,
                                               const std::vector<bool>& withLower,
                                               const std::vector<bool>& withUpper) const {
            if (lower.size() != upper.size() || lower.size() != withLower.size() ||
                lower.size() != withUpper.size()) {
                throw std::logic_error("All vectors inputted to pointsInRange must have the same length.");
            }
            for (int i = 0; i < lower.size(); i++) {
                if (((!withUpper[i] || !withLower[i]) && lower[i] >= upper[i]) ||
                    lower[i] > upper[i]) {
                    return std::vector<Point<T,S> >();
                }
            }
            return root->pointsInRange(getModifiedLower(lower, withLower),
                                       getModifiedUpper(upper, withUpper));
        }

        void print() const {
            root->print(0);
        }

        // [AJB] 查询统计dump
        static void ajb_dump_stats() {
            ajb_rt_stats.dump();
        }

        // [AJB] 获取树的基数(不同点数)
        int ajb_cardinality() const {
            return getCardinality();
        }
    };

/**
* A class which is used to naively count the number of points in a given rectangle. This class is used
* for testing an benchmarking, it should not be used in practice.
*/
    template <typename T, class S>
    class NaiveRangeCounter {
        static_assert(std::is_arithmetic<T>::value, "Type T must be numeric");
    private:
        std::vector<Point<T,S> > points;

        static bool pointInRange(const Point<T,S>& point,
                                 const std::vector<T>& lower,
                                 const std::vector<T>& upper,
                                 const std::vector<bool>& withLower,
                                 const std::vector<bool>& withUpper) {
            for (int i = 0; i < point.dim(); i++) {
                if (point[i] < lower[i] ||
                    (point[i] == lower[i] && !withLower[i])) {
                    return false;
                }
                if (point[i] > upper[i] ||
                    (point[i] == upper[i] && !withUpper[i])) {
                    return false;
                }
            }
            return true;
        }

    public:
        NaiveRangeCounter(std::vector<Point<T,S> > points): points(points) {}

        int countInRange(const std::vector<T>& lower,
                         const std::vector<T>& upper,
                         const std::vector<bool>& withLower,
                         const std::vector<bool>& withUpper) const {
            int count = 0;
            int d = lower.size();
            for (size_t i = 0; i < points.size(); i++) {
                // Dimension-ordered early exit: test most discriminating dim first.
                // For uniform distributions the first dim often eliminates >50% of points.
                bool inRange = true;
                for (int j = 0; j < d && inRange; j++) {
                    T v = points[i][j];
                    inRange = (v > lower[j] || (v == lower[j] && withLower[j]))
                           && (v < upper[j] || (v == upper[j] && withUpper[j]));
                }
                if (inRange) count += points[i].count();
            }
            return count;
        }

        std::vector<Point<T,S> > pointsInRange(const std::vector<T>& lower,
                                               const std::vector<T>& upper,
                                               const std::vector<bool>& withLower,
                                               const std::vector<bool>& withUpper) const {
            std::vector<Point<T,S> > selectedPoints;
            selectedPoints.reserve(points.size() / 4);  // heuristic: expect ~25% selectivity
            int d = lower.size();
            for (size_t i = 0; i < points.size(); i++) {
                bool inRange = true;
                for (int j = 0; j < d && inRange; j++) {
                    T v = points[i][j];
                    inRange = (v > lower[j] || (v == lower[j] && withLower[j]))
                           && (v < upper[j] || (v == upper[j] && withUpper[j]));
                }
                if (inRange) selectedPoints.push_back(points[i]);
            }
            return selectedPoints;
        }
    };

} // namespace

#endif //RANGETREE_H
