#include<bits/stdc++.h>
#define TBD 0
#define MAX_INT 0x7fffffff
#define MAX_DATA 2000000000
using namespace std;
vector<vector<int> > matrix;
int cntLoop1 = 0, cntLoop2 = 0;
int cntF1 = 0, cntF2 = 0;


double F(const vector<int> && pos) {
    long long ans = 1;
    for(int a : pos) ans *= a;
    return sqrt(ans);
}

double F(const vector<int> & pos) {
    long long ans = 1;
    for(int a : pos) ans *= a;
    return sqrt(ans);
}

void gendata(int m, int n, int p) {
    matrix.resize(m);
    for (int i = 0; i < m; i++) {
        matrix[i].resize(n);
        for (int j = 0; j < n; j++) {
            matrix[i][j] = ((rand() << 15) + rand()) % p;
        }
        sort(matrix[i].begin(), matrix[i].end());
    }
    return;
}

void printdata() {
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

vector<int> getpos(const vector<pair<vector<int>::iterator, vector<int>::iterator> > &iters, int target) {
    vector<int> pos(iters.size());
    for (int i = 0; i < iters.size(); i++) {
        pos[i] = lower_bound(iters[i].first, iters[i].second, target) - iters[i].first;
    }
    return pos;
}

void getpos(const vector<pair<vector<int>::iterator, vector<int>::iterator> > &iters, int target, vector<int> &pos) {
    for (int i = 0; i < iters.size(); i++) {
        pos[i] = lower_bound(iters[i].first, iters[i].second, target) - iters[i].first;
    }
}

int BinarySearch(const vector<pair<vector<int>::iterator, vector<int>::iterator> > &iters, const int target) {
    int MIN = *iters[0].first, MAX = *(iters[0].second - 1);
    for(int i = 1; i < iters.size(); i++) {
        MIN = min(MIN, *iters[i].first);
        MAX = max(MAX, *(iters[i].second - 1));
    }
    long long l = MIN, r = MAX, mid, res;
    vector<int> pos(iters.size());
    while(l <= r) {
        cntLoop2++;
        mid = (l + r) >> 1;
        // pos = 
        int ans = F(getpos(iters, mid));
        cntF2++;
        if(ans < target) {
            res = mid;
            l = mid + 1;
        }
        else {
            r = mid - 1;
        }
    }
    return res;
}


int MultiHeadBinarySearch(const vector<pair<vector<int>::iterator, vector<int>::iterator> > &iters, const int target) {
    vector<pair<vector<int>::iterator, vector<int>::iterator> > bounds = iters;
    vector<vector<int>::iterator> itermid(iters.size());
    vector<int> pos(iters.size());
    vector<int> tmppos(iters.size());
    for(int i = 0; i < iters.size(); i++) {
        itermid[i] = iters[i].first + (iters[i].second - iters[i].first) / 2;
        pos[i] = itermid[i] - iters[i].first;
    }
    int mini, maxi, cnt = 0;
    double res;
    while(cnt < iters.size()) {
        mini = -1, maxi = -1;
        for(int i = 0; i < iters.size(); i++){
            if(bounds[i].second - bounds[i].first <= 1) continue;
            if(mini == -1 || *itermid[i] < *itermid[mini]) mini = i;
            if(maxi == -1 || *itermid[i] > *itermid[maxi]) maxi = i;
        }
        res = F(pos);
        if(res <= target) {
            bounds[mini].first = itermid[mini];
            if(bounds[mini].second - bounds[mini].first <= 1) {
                getpos(iters, *bounds[mini].first + 1, tmppos);
                if(F(tmppos) > target) return *bounds[mini].first;
                else pos[mini] = bounds[mini].second - iters[mini].first;
                cnt++;
            }
            else {
                itermid[mini] = bounds[mini].first + (bounds[mini].second - bounds[mini].first) / 2;
                pos[mini] = itermid[mini] - iters[mini].first;
            }
        }
        else {
            bounds[maxi].second = itermid[maxi];
            if(bounds[maxi].second - bounds[maxi].first <= 1) {
                getpos(iters, *bounds[maxi].first + 1, tmppos);
                if(F(tmppos) > target) return *bounds[maxi].first;
                else pos[maxi] = bounds[maxi].second - iters[maxi].first;
                cnt++;
            }
            else {
                itermid[maxi] = bounds[maxi].first + (bounds[maxi].second - bounds[maxi].first) / 2;
                pos[maxi] = itermid[maxi] - iters[maxi].first;
            }
        }
    }
    int ans = MAX_INT;
    for(int i = 0; i < iters.size(); i++) {
        if(bounds[i].second != iters[i].second) ans = min(ans, *bounds[i].second);
    }
    return ans;
}


int main() {
    int m = 2, n = 1000000, p = MAX_DATA + 1;
    srand(time(0));
    gendata(m, n, p);
    vector<pair<vector<int>::iterator, vector<int>::iterator> > iters(m);
    vector<int> leftit(m, 0), rightit(m, n);
    for (int i = 0; i < m; i++) {
        iters[i].first = matrix[i].begin();
        iters[i].second = matrix[i].end();
    }
    int res, correct = 0, t;
    double now, nxt;
    int TestTimes = 100000;
    vector<int> vec(TestTimes);
    for(int i = 0; i < TestTimes; i++) {
        vec[i] = rand() % (1000010);
        // cout << vec[i] << " ";
    }
    // return 0;


    clock_t start, end;

    
    start = clock();
    for(int i = 0; i < TestTimes; i++) {
        res = MultiHeadBinarySearch(iters, vec[i]);
        // now = F(getpos(iters, res));
        // nxt = F(getpos(iters, res + 1));
        // if(now > vec[i] || (nxt <= vec[i] && res < MAX_INT)) cout << "Error: " << vec[i] << " " << res << " " << now << " " << nxt << endl;
        // else correct++;
    }
    cout << "Correct: " << correct << endl;
    end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "Time: " << duration << "s" << endl;

    start = clock();
    for(int i = 0; i < TestTimes; i++) {
        res = BinarySearch(iters, vec[i]);
    }
    end = clock();
    duration = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "Time: " << duration << "s" << endl;
    cout << "#Calling F: " << cntF1 << " , " << cntF2 << endl;
    cout << "#Loop: " << cntLoop1 << " , " << cntLoop2 << endl;
    return 0;
}



        // if(*itermid[mini] < MIN) {
        //     bounds[mini].first = itermid[mini];
        //     if(bounds[mini].second - bounds[mini].first <= 1) {
        //         if(F(getpos(iters, *bounds[mini].first + 1)) > target)
        //             return *bounds[mini].first;
        //         else pos[mini] = bounds[mini].second - iters[mini].first;
        //         cnt++;
        //     }
        //     else {
        //         itermid[mini] = bounds[mini].first + (bounds[mini].second - bounds[mini].first) / 2;
        //         pos[mini] = itermid[mini] - iters[mini].first;
        //     }
        //     continue;
        // }
        // if(*itermid[maxi] > MAX) {
        //     bounds[maxi].second = itermid[maxi];
        //     if(bounds[maxi].second - bounds[maxi].first <= 1) {
        //         if(F(getpos(iters, *bounds[maxi].first + 1)) > target)
        //             return *bounds[maxi].first;
        //         else pos[maxi] = bounds[maxi].second - iters[maxi].first;
        //         cnt++;
        //     }
        //     else {
        //         itermid[maxi] = bounds[maxi].first + (bounds[maxi].second - bounds[maxi].first) / 2;
        //         pos[maxi] = itermid[maxi] - iters[maxi].first;
        //     }
        //     continue;
        // }
        // cntF1++;