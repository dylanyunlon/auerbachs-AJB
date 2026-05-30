// =============================================================================
// upper_bound_demo.cpp — AJB-adapted set iterator boundary demo
//
// Origin: upstream/joinrenum/upb.cpp (13 lines)
// Adaptation (~20%): AJB structured output showing the lower_bound/
//   upper_bound semantics used in Index split logic. Useful for
//   verifying boundary behavior in the AJB merge-path partitioning.
//
// Build: g++ -O3 upper_bound_demo.cpp -o upper_bound_demo
// =============================================================================

#include <bits/stdc++.h>
using namespace std;

int main() {
    set<int> s = {1, 3, 4, 6, 8};

    printf("[AJB] set contents: {");
    for (auto it = s.begin(); it != s.end(); ++it)
        printf("%s%d", it == s.begin() ? "" : ", ", *it);
    printf("}\n");

    int query = 4;
    printf("[AJB] query = %d\n", query);

    auto lb = s.lower_bound(query);
    auto ub = s.upper_bound(query);

    // AJB: show the full picture
    printf("[AJB_STATE] lower_bound(%d):\n", query);
    if (lb != s.begin()) {
        auto prev = lb; --prev;
        printf("  predecessor = %d\n", *prev);
    } else {
        printf("  predecessor = (none, lb is begin)\n");
    }
    printf("  *lb = %d\n", *lb);

    printf("[AJB_STATE] upper_bound(%d):\n", query);
    if (ub != s.end()) {
        printf("  *ub = %d\n", *ub);
    } else {
        printf("  *ub = (end)\n");
    }

    // AJB: this pattern is used in Index::Split() for binary search bounds
    printf("\n[AJB_RESULTS] Boundary semantics for merge-path partitioning:\n");
    printf("  lower_bound(x) -> first element >= x\n");
    printf("  upper_bound(x) -> first element >  x\n");
    printf("  predecessor = *(--lower_bound(x)) -> last element < x\n");
    printf("[AJB] upper_bound_demo DONE\n");
    return 0;
}
