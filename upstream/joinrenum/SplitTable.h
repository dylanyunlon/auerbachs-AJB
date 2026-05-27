//
// Created by shai.zeevi on 04/06/2019.
//

#ifndef RANDOMORDERENUMERATION_SPLITTABLE_H
#define RANDOMORDERENUMERATION_SPLITTABLE_H

#include "iostream"
#include <functional>
#include <unordered_map>
#include "Table.h"

typedef long long ll;

using namespace std;

//KeyParcel is the key by which it is split - we'll use it to constitutes an incomplete parcel which
//encompasses the intersection of the atom attr. and the parent attr.

template<typename KeyParcel, typename Parcel, typename KeyParcelHash = hash<KeyParcel>, typename KeyParcelEqual = equal_to<KeyParcel>,
        typename ParcelHash = hash<Parcel>, typename ParcelEqual = equal_to<Parcel>>
struct SplitTable {
    unordered_map<KeyParcel, Table<Parcel, ParcelHash, ParcelEqual>, KeyParcelHash, KeyParcelEqual> buckets;

    SplitTable() = default;

    inline void push_back(const KeyParcel& key, const Row<Parcel>& value) {
//        if (buckets.find(key) == buckets.end()) {
//            buckets[key] = Table<Parcel, ParcelHash, ParcelEqual>();
//        }
        buckets[key].push_back(value);
    }

//  KeyParcel is the key, Parcel is the value
    inline void loadFromTable(const Table<Parcel, ParcelHash, ParcelEqual> &table) { /*unified table containing the values*/
        buckets.clear();
        for (const Row<Parcel> &row : table.data) { //traverse the unified table
            push_back(row.parcel.template to<KeyParcel>() /*key extraction*/, row); //put the tuple in its bucket
        }
    }

    inline long long getCummulativeWeight() {
        ll cumWeight = 0;

        for(auto& b : buckets) {
            cumWeight += b.second.totalWeight;
        }

        return cumWeight;
    }

    void print() const {
        for (auto& p : buckets) {
            cout << "key = [\n";
            p.first.print();
            cout << "]:" << endl;
            p.second.print();
            cout << endl;
        }
    }

};


#endif //RANDOMORDERENUMERATION_SPLITTABLE_H
