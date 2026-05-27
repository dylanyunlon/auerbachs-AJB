using namespace std;
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
    unsigned long dim() const {
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
    T operator[](int index) const {
        if(index < 0 || index >= dim()) {
            throw out_of_range("[] access index for point is out of range.");
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
    bool operator==(const Point<T>& p) const {
        return vec == p.vec;
    }

    /**
     * Check for inequality.
     *
     * The opposite of ==.
     *
     * @param p some other point.
     * @return false if \p equals the current point, otherwise true.
     */
    bool operator!=(const Point<T>& p) const {
        return !((*this) == p);
    }

    bool operator < (const Point<T>& p) const {
        int siz = min(vec.size(), p.vec.size());
        for(int i = 0; i < siz; i++){
            if(vec[i] < p.vec[i]) return true;
            else if(vec[i] > p.vec[i]) return false;
        }
        return false;
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
    void print() const {
        cout << "(";
        for (int i = 0; i < dim() - 1; i++) {
            cout << (*this)[i] << ", ";
        }
        cout << (*this)[dim() - 1] << ") : " << cnt << endl;
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
    }

    int sumCnt(const Point<T> &pl, const Point<T> &pr) {
        vector<Point<int> >::iterator itl = lower_bound(points.begin(), points.end(), pl);
        vector<Point<int> >::iterator itr = upper_bound(points.begin(), points.end(), pr);
        if(itr == points.begin())return 0;
        else if(itl == points.begin())return (itr - 1)->cnt;
        else return (itr - 1)->cnt - (itl - 1)->cnt;
    }

    int sumCnt(const vector<Point<int> >::iterator &itl, const vector<Point<int> >::iterator &itr) {
        if(itr == points.begin())return 0;
        else if(itl == points.begin())return (itr - 1)->cnt;
        else return (itr - 1)->cnt - (itl - 1)->cnt;
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
        vector<T> vl, vr;
        for(int i = 0; i < bucket.size(); i++){
            vl.push_back(bucket[i].first);
            vr.push_back(bucket[i].second);
        }
        return count(Point<T>(vl), Point<T>(vr));
    }

    int countInRange(vector<T>& vl, vector<T>& vr) {
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

};