#include <bits/stdc++.h>
using namespace std;
int main(){
    set<int> s = {};
    s.insert(1);
    s.insert(3);
    s.insert(4);
    s.insert(6);
    s.insert(8);
    set<int>::iterator it =  s.lower_bound(4);
    cout << *--it;
    cout << *s.upper_bound(4);
    return 0;
}