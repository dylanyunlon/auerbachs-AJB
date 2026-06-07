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
    // [AJB_BP] M930: volume tracking
    double total_volume_before = 0.0;
    double total_volume_after = 0.0;
    double min_child_volume = 1e18;
    double max_child_volume = 0.0;
    long long volume_measurements = 0;
    // [AJB_BP] M931: splitDim dimension frequency histogram (up to 16 dims)
    long long dim_freq[16] = {};
    // [AJB_BP] M932: replaceSelf dim-change tracking
    long long dim_change_count = 0;
    void dump(const char* tag = "SplitBucket") {
        fprintf(stderr, "[AJB_STATE][%s] calls=%lld children=%lld avg=%.2f replace=%lld replace_self=%lld max_dim=%d\n",
                tag, split_calls, children_total,
                split_calls > 0 ? (double)children_total / split_calls : 0.0,
                replace_calls, replace_self_calls, max_dim_seen);
        fprintf(stderr, "[AJB_STATE][%s] volume: measurements=%lld before=%.1f after=%.1f ratio=%.4f\n",
                tag, volume_measurements, total_volume_before, total_volume_after,
                total_volume_before > 0 ? total_volume_after / total_volume_before : 0.0);
        fprintf(stderr, "[AJB_STATE][%s] child_volume: min=%.1f max=%.1f\n",
                tag, min_child_volume, max_child_volume);
        fprintf(stderr, "[AJB_STATE][%s] dim_freq: [", tag);
        for(int d = 0; d < 16 && d <= max_dim_seen; d++) {
            if(d) fprintf(stderr, ",");
            fprintf(stderr, "d%d=%lld", d, dim_freq[d]);
        }
        fprintf(stderr, "] dim_changes=%lld\n", dim_change_count);
    }
    void reset() {
        split_calls = children_total = replace_calls = replace_self_calls = 0; max_dim_seen = 0;
        total_volume_before = total_volume_after = 0.0;
        min_child_volume = 1e18; max_child_volume = 0.0; volume_measurements = 0;
        for(int i = 0; i < 16; i++) dim_freq[i] = 0;
        dim_change_count = 0;
    }
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
            while(splitDim < this->lowerBound.size() && this->lowerBound[splitDim] == this->upperBound[splitDim])splitDim++;
            // [AJB_TRACE] Bucket ctor: scan dimensions for first non-degenerate
            if((int)this->lowerBound.size() > ajb_split_stats.max_dim_seen)
                ajb_split_stats.max_dim_seen = this->lowerBound.size();
            // [AJB_BP] M930: compute bucket volume
            {
                double vol = 1.0;
                for(size_t d = 0; d < this->lowerBound.size(); d++) {
                    vol *= (double)(this->upperBound[d] - this->lowerBound[d] + 1);
                    if(vol > 1e15) { vol = 1e15; break; }
                }
                ajb_split_stats.volume_measurements++;
                ajb_split_stats.total_volume_after += vol;
                if(vol < ajb_split_stats.min_child_volume) ajb_split_stats.min_child_volume = vol;
                if(vol > ajb_split_stats.max_child_volume) ajb_split_stats.max_child_volume = vol;
            }
            // [AJB_BP] M931: splitDim frequency
            if(splitDim < 16) ajb_split_stats.dim_freq[splitDim]++;
            // [AJB_BP] M931: log first 10 constructions
            if(ajb_split_stats.volume_measurements <= 10) {
                fprintf(stderr, "[AJB_BP][Bucket] ctor: dim=%zu splitDim=%d bounds=[",
                        this->lowerBound.size(), splitDim);
                for(size_t d = 0; d < this->lowerBound.size(); d++) {
                    if(d) fprintf(stderr, ",");
                    fprintf(stderr, "%d:%d", this->lowerBound[d], this->upperBound[d]);
                }
                fprintf(stderr, "]\n");
            }
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
            // [AJB_BP] M930: compute volume before replacement
            double vol_before = 1.0;
            for(size_t d = 0; d < lowerBound.size(); d++) {
                vol_before *= (double)(upperBound[d] - lowerBound[d] + 1);
                if(vol_before > 1e15) { vol_before = 1e15; break; }
            }
            ajb_split_stats.total_volume_before += vol_before;

            lowerBound[splitDim] = lower;
            upperBound[splitDim] = upper;
            while(splitDim < lowerBound.size() && lowerBound[splitDim] == upperBound[splitDim])splitDim++;

            // [AJB_BP] M930: compute volume after replacement
            double vol_after = 1.0;
            for(size_t d = 0; d < lowerBound.size(); d++) {
                vol_after *= (double)(upperBound[d] - lowerBound[d] + 1);
                if(vol_after > 1e15) { vol_after = 1e15; break; }
            }
            ajb_split_stats.total_volume_after += vol_after;

            // [AJB_TRACE] replaceSelf: dim %d→%d on range [%d,%d]
            if(old_splitDim != splitDim) {
                ajb_split_stats.dim_change_count++;
                fprintf(stderr, "[AJB_TRACE][Bucket] replaceSelf: splitDim %d→%d (range [%d,%d]) vol %.0f→%.0f\n",
                        old_splitDim, (int)splitDim, lower, upper, vol_before, vol_after);
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
