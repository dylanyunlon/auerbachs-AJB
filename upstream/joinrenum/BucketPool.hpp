class BucketPool {
    private:
    vector<Bucket> pool;
    stack<int> emptypos;

    public:
    Bucket & operator [](const int idx) {
        return pool[idx];
    }

    const Bucket & operator [](const int idx) const {
        return pool[idx];
    }
    
    int newBucket(vector<int> lowerBound, vector<int> upperBound, int splitDim = 0) {
        if(emptypos.empty()){
            pool.emplace_back(lowerBound, upperBound, splitDim);
            return pool.size() - 1;
        } else {
            int pos = emptypos.top();
            emptypos.pop();
            pool[pos].reset(lowerBound, upperBound, splitDim);
            return pos;
        }
    }

    int newCopy(const Bucket &B) {
        if(emptypos.empty()) {
            pool.emplace_back(B);
            return pool.size() - 1;
        } else {
            int pos = emptypos.top();
            emptypos.pop();
            pool[pos].reset(B.lowerBound, B.upperBound, B.splitDim);
            return pos;
        }
    }

    int poolSize() {
        return pool.size();
    }

    int fragmentSize() {
        return emptypos.size();
    }

    void free(int idx) {
        if(idx < 0 || idx >= pool.size()) return;
        emptypos.push(idx);
    }
};