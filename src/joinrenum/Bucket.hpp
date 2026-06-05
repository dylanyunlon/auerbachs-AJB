// [AJB] Bucket: d维区间 [lowerBound, upperBound]
// splitDim = 第一个lower!=upper的维度, 指导分裂方向
// AGM = 这个区间内join结果数的上界(由Index通过RangeTree赋值)
#include <cstdio>
#include<iostream>
#include<vector>
using namespace std;

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

        // bool operator==(const Bucket& B) const {
        //     if(lowerBound.size() != B.getLowerBound().size() || upperBound.size() != B.getUpperBound().size()){
        //         cout << "Bucket size mismatch @ EQ" << endl;
        //         cout << lowerBound.size() << " != " << B.getLowerBound().size() << "||" << upperBound.size() << " != " << B.getUpperBound().size() << endl;
        //         return false;
        //     }
        //     return lowerBound == B.getLowerBound() && upperBound == B.getUpperBound();
        // }

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
            for(int i = 0; i < lowerBound.size(); i++){
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

        // const vector<pair<Bucket*, int> >& getChildren() const {
        //     return children;
        // }

        // void addChild(Bucket* child, int agm){
        //     children.push_back(make_pair(child, agm));
        // }

    // [AJB] structured dump — 用于parse_ajb_trace.py解析
    void ajb_dump(const char* label = "") const {
        fprintf(stderr, "[AJB_STATE][Bucket] %s d=%zu split=%d AGM=%lld range=[",
                label, lowerBound.size(), splitDim, AGM);
        for(size_t i = 0; i < lowerBound.size(); i++){
            if(i) fprintf(stderr, " ");
            fprintf(stderr, "%d..%d", lowerBound[i], upperBound[i]);
        }
        fprintf(stderr, "]\n");
    }

    // [AJB] volume: 区间的笛卡尔积大小, 用于评估分裂效率
    long long ajb_volume() const {
        long long vol = 1;
        for(size_t i = 0; i < lowerBound.size(); i++){
            vol *= (upperBound[i] - lowerBound[i] + 1);
            if(vol < 0) return -1; // overflow
        }
        return vol;
    }

    // [AJB] isLeaf: splitDim越界说明所有维度都是单点
    bool isLeaf() const { return splitDim >= (int)lowerBound.size(); }
};

