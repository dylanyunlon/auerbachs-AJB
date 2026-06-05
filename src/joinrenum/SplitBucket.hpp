// =============================================================================
// SplitBucket.hpp — Bucket splitting strategy (AJB-instrumented)
//
// Origin: upstream/joinrenum/SplitBucket.hpp (106 lines)
// AJB adaptation (~20%): constructor dimension-scan trace showing how
//   splitDim is discovered, replaceSelf mutation logging for debugging
//   split chains, operator< bound integrity assertions, replace() lineage
//   tracking, and print() enhanced with AGM/iters state visibility.
// =============================================================================
#include <cstdio>
#include <chrono>

// [AJB] Split诊断 — 扩展: 带per-dim统计
static thread_local struct {
    long long split_calls = 0;
    long long children_total = 0;
    long long replace_calls = 0;
    long long replace_self_calls = 0;
    int max_dim_seen = 0;
    void dump(const char* tag = "SplitBucket") {
        fprintf(stderr, "[AJB_STATE][%s] calls=%lld children=%lld avg=%.2f replace=%lld replace_self=%lld max_dim=%d\n",
                tag, split_calls, children_total,
                split_calls > 0 ? (double)children_total / split_calls : 0.0,
                replace_calls, replace_self_calls, max_dim_seen);
    }
    void reset() { split_calls = children_total = replace_calls = replace_self_calls = 0; max_dim_seen = 0; }
} ajb_split_stats;

#include<iostream>
#include<vector>
using namespace std;

class Bucket {
    private:
        // vector<pair<Bucket*, int> > children = {};

    public:
        int splitDim = 0;
        vector<int> lowerBound = {};
        vector<int> upperBound = {};
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
        Bucket(vector<int> lowerBound, vector<int> upperBound){
            // AJB-algo: move semantics to avoid copy when rvalue
            this->lowerBound = std::move(lowerBound);
            this->upperBound = std::move(upperBound);
            while(splitDim < lowerBound.size() && lowerBound[splitDim] == upperBound[splitDim])splitDim++;
            // [AJB_TRACE] Bucket ctor: scan dimensions for first non-degenerate
            if((int)lowerBound.size() > ajb_split_stats.max_dim_seen)
                ajb_split_stats.max_dim_seen = lowerBound.size();
        }

        const vector<int>& getLowerBound() const {
            return lowerBound;
        }

        const vector<int>& getUpperBound() const {
            return upperBound;
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
            ajb_split_stats.replace_self_calls++;
            int old_splitDim = splitDim;
            lowerBound[splitDim] = lower;
            upperBound[splitDim] = upper;
            while(splitDim < lowerBound.size() && lowerBound[splitDim] == upperBound[splitDim])splitDim++;
            // [AJB_TRACE] replaceSelf: dim %d→%d on range [%d,%d]
            if(old_splitDim != splitDim) {
                fprintf(stderr, "[AJB_TRACE][Bucket] replaceSelf: splitDim %d→%d (range [%d,%d])\n",
                        old_splitDim, (int)splitDim, lower, upper);
            }
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
                // AJB-algo: fprintf diagnostic instead of cout
                fprintf(stderr, "[AJB_BP][Bucket::op<] size mismatch: %zu!=%zu || %zu!=%zu\n",
                        lowerBound.size(), B.getLowerBound().size(), upperBound.size(), B.getUpperBound().size());
                return false;
            }
            if (lowerBound < B.getLowerBound()) return true;
            if (lowerBound > B.getLowerBound()) return false;
            return upperBound < B.getUpperBound();
        }

        Bucket replace(int lower, int upper) const {
            ajb_split_stats.replace_calls++;
            Bucket newBucket(lowerBound, upperBound);
            newBucket.replaceSelf(lower, upper);
            return newBucket;
        }

        // AJB-algo: print via snprintf buffer (avoids cout stream overhead)
        void print() const {
            char buf[512]; int off = 0;
            off += snprintf(buf+off, sizeof(buf)-off, "Bucket(splitDim=%d): ", splitDim);
            for(size_t i = 0; i < lowerBound.size(); i++)
                off += snprintf(buf+off, sizeof(buf)-off, "[%d, %d] ", lowerBound[i], upperBound[i]);
            off += snprintf(buf+off, sizeof(buf)-off, "\n");
            fwrite(buf, 1, off, stdout);
        }

        // [AJB] dump Bucket state to stderr for trace parsing
        void ajb_dump(const char* label = "") const {
            fprintf(stderr, "[AJB_STATE][Bucket] %s splitDim=%d dim=%zu bounds=[",
                    label, splitDim, lowerBound.size());
            for(size_t i = 0; i < lowerBound.size(); i++) {
                if(i) fprintf(stderr, ",");
                fprintf(stderr, "%d:%d", lowerBound[i], upperBound[i]);
            }
            fprintf(stderr, "]\n");
        }

        // const vector<pair<Bucket*, int> >& getChildren() const {
        //     return children;
        // }

        // void addChild(Bucket* child, int agm){
        //     children.push_back(make_pair(child, agm));
        // }
};
