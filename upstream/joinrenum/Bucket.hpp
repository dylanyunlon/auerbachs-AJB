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
            while(splitDim < lowerBound.size() && lowerBound[splitDim] == upperBound[splitDim])splitDim++;
        }

        void updateSplitDim() {
            while(splitDim < lowerBound.size() && lowerBound[splitDim] == upperBound[splitDim])splitDim++;
        }

        const vector<int>& getLowerBound() const {
            return lowerBound;
        }

        const vector<int>& getUpperBound() const {
            return upperBound;
        }

        void reset(const vector<int> &newLowerBound, const vector<int> &newUpperBound, int newSplitDim = 0) {
            if(lowerBound.size() != newLowerBound.size() || upperBound.size() != newUpperBound.size()){
                cout << "Bucket size mismatch @ reset" << endl;
                lowerBound = newLowerBound;
                upperBound = newUpperBound;
            }
            copy(newLowerBound.begin(), newLowerBound.end(), lowerBound.begin());
            copy(newUpperBound.begin(), newUpperBound.end(), upperBound.begin());
            splitDim = newSplitDim;
            AGM = -1;
            // this->iters.clear();
            while(splitDim < lowerBound.size() && lowerBound[splitDim] == upperBound[splitDim])splitDim++;
            return;
        }

        void reset(const Bucket &B) {
            if(lowerBound.size() != B.lowerBound.size() || upperBound.size() != B.upperBound.size()){
                cout << "Bucket size mismatch @ reset" << endl;
                lowerBound = B.lowerBound;
                upperBound = B.upperBound;
            }
            copy(B.lowerBound.begin(), B.lowerBound.end(), lowerBound.begin());
            copy(B.lowerBound.begin(), B.upperBound.end(), upperBound.begin());
            splitDim = B.splitDim;
            AGM = -1;
            // this->iters.clear();
            while(splitDim < lowerBound.size() && lowerBound[splitDim] == upperBound[splitDim])splitDim++;
            return;
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
            while(splitDim < lowerBound.size() && lowerBound[splitDim] == upperBound[splitDim])splitDim++;
            return;
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
            if (lowerBound.size() != B.getLowerBound().size() || upperBound.size() != B.getUpperBound().size()) {
                cout << "Bucket size mismatch @ lessEQ" << endl;
                cout << lowerBound.size() << " != " << B.getLowerBound().size() << "||" << upperBound.size() << " != " << B.getUpperBound().size() << endl;
                return false;
            }
            if (lowerBound < B.getLowerBound()) return true;
            if (lowerBound > B.getLowerBound()) return false;
            return upperBound < B.getUpperBound();
        }

        Bucket replace(int lower, int upper) const {
            Bucket newBucket(lowerBound, upperBound, splitDim);
            newBucket.replaceSelf(lower, upper);
            return newBucket;
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
};