#include<bits/stdc++.h>
#include "CountOracle.hpp"
#include "Bucket.hpp"
#include "BucketPool.hpp"
using namespace std;
int main() {
    BucketPool pool;

    // upstream: 3 allocs, 1 free, 2 copies
    int id0 = pool.newBucket({0,0}, {1,1});
    cout << id0 << endl;
    int id1 = pool.newBucket({0,0}, {2,2});
    cout << id1 << endl;
    int id2 = pool.newBucket({0,0}, {3,3});
    cout << id2 << endl;
    pool.free(1);
    int id3 = pool.newCopy(pool[0]);
    cout << id3 << endl;
    int id4 = pool.newCopy(pool[0]);
    cout << id4 << endl;
    pool[0].reset({1,1}, {4,4});
    pool[0].upperBound[0] = 1;
    pool[0].updateSplitDim();

    // --- splitDim validation: manual dimension scan ---
    // upstream: just called updateSplitDim and trusted it
    // changed: scan all dimensions to find which has max range,
    //   verify updateSplitDim picked the same one
    {
        auto lb = pool[0].getLowerBound();
        auto ub = pool[0].getUpperBound();
        int manual_best_dim = 0;
        int max_span = ub[0] - lb[0];
        for(int d = 1; d < (int)lb.size(); d++) {
            int span = ub[d] - lb[d];
            if(span > max_span) {
                max_span = span;
                manual_best_dim = d;
            }
        }
        int actual = pool[0].getSplitDim();
        if(actual != manual_best_dim) {
            cout << "splitDim disagree: actual=" << actual
                 << " manual=" << manual_best_dim << endl;
        }
    }

    // --- iteration order: reverse then forward ---
    // upstream: for(i=0; i<4) pool[i].print + getSplitDim
    // changed: reverse pass first — exercises random access pattern
    for(int i = 3; i >= 0; i--) {
        pool[i].print();
        cout << pool[i].getSplitDim() << endl;
    }

    // --- randomized alloc/free stress ---
    // upstream: fixed sequence only
    // changed: 16 random alloc/free operations to stress the free-list
    srand(7);
    vector<int> live = {0, 2, 3, 4};  // currently live after upstream sequence
    for(int round = 0; round < 16; round++) {
        if(rand() % 2 == 0 && !live.empty()) {
            // free a random live slot
            int pick = rand() % (int)live.size();
            pool.free(live[pick]);
            live.erase(live.begin() + pick);
        } else {
            // alloc a new bucket
            int lo = rand() % 5;
            int hi = lo + 1 + rand() % 5;
            int nid = pool.newBucket({lo, lo}, {hi, hi});
            live.push_back(nid);
        }
    }
    // verify all live slots are accessible
    for(int id : live) {
        pool[id].print();
    }
    cout << "Final live count: " << live.size() << endl;

    return 0;
}
