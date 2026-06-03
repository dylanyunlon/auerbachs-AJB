#include <bits/stdc++.h>
using namespace std;
int main(){
    set<int> s = {};
    s.insert(1);
    s.insert(3);
    s.insert(4);
    s.insert(6);
    s.insert(8);

    // --- parallel sorted vector for cross-validation ---
    // upstream: only used set
    // changed: maintain a sorted vector, run same queries on both
    vector<int> v(s.begin(), s.end());

    // upstream: lower_bound(4), decrement, upper_bound(4), print
    set<int>::iterator it =  s.lower_bound(4);
    cout << *--it;
    cout << *s.upper_bound(4);
    cout << endl;

    // --- vector cross-check for the same query ---
    auto vit = lower_bound(v.begin(), v.end(), 4);
    --vit;
    cout << *vit;
    auto vub = upper_bound(v.begin(), v.end(), 4);
    cout << *vub;
    cout << endl;

    // --- boundary sweep: test multiple query values ---
    // upstream: only queried value 4
    // changed: sweep 0 through 9 and a far-out value, check set vs vector
    int queries[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100};
    int mismatches = 0;
    for(int q : queries) {
        int spos = (int)distance(s.begin(), s.lower_bound(q));
        int vpos = (int)distance(v.begin(), lower_bound(v.begin(), v.end(), q));
        if(spos != vpos) mismatches++;
        int supos = (int)distance(s.begin(), s.upper_bound(q));
        int vupos = (int)distance(v.begin(), upper_bound(v.begin(), v.end(), q));
        if(supos != vupos) mismatches++;
    }
    if(mismatches > 0)
        cout << "set/vector mismatches: " << mismatches << endl;

    // --- empty container edge case ---
    // upstream: no empty-set test
    // changed: verify lower_bound/upper_bound on empty set return end()
    {
        set<int> empty_s;
        vector<int> empty_v;
        bool ok = (empty_s.lower_bound(5) == empty_s.end())
               && (empty_s.upper_bound(5) == empty_s.end())
               && (lower_bound(empty_v.begin(), empty_v.end(), 5) == empty_v.end());
        if(!ok)
            cout << "Empty container edge case FAILED" << endl;
    }

    return 0;
}
