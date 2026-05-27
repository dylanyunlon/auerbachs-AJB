#include<bits/stdc++.h>
#include "CountOracle.hpp"
#include "Bucket.hpp"
#include "BucketPool.hpp"
using namespace std;
int main() {
    BucketPool pool;
    cout << pool.newBucket({0,0}, {1,1}) << endl;
    cout << pool.newBucket({0,0}, {2,2}) << endl;
    cout << pool.newBucket({0,0}, {3,3}) << endl;
    pool.free(1);
    cout << pool.newCopy(pool[0]) << endl;
    cout << pool.newCopy(pool[0]) << endl;
    pool[0].reset({1,1}, {4,4});
    pool[0].upperBound[0] = 1;
    pool[0].updateSplitDim();
    for(int i = 0; i < 4; i++) {
        pool[i].print();
        cout << pool[i].getSplitDim() << endl;
    }
    return 0;
}