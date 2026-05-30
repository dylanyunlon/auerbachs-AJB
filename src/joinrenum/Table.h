//
// Created by shai.zeevi on 04/06/2019.
//

#ifndef RANDOMORDERENUMERATION_TABLE_H
#define RANDOMORDERENUMERATION_TABLE_H

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <assert.h>
#include "iostream"
// #include "RangeTree.hpp"
#include "CountOracle.hpp"
#include <chrono>
typedef long long ll;
// namespace RT = RangeTree;

using namespace std;

//give the reader a chance to update his structure on each addition
typedef bool (*SelectionPredicate)(void* arg, void* parcel /*cast to appropriate parcel*/);

///
/// \tparam Parcel/ PParcel
/// Requirements from all Parcel-like typenames:
///
/// if it belongs to an actual db table:
/// - static function with prototype "Parcel fromLine(const string& s)"
///     construct a Parcel from a line read from the file
///

/// - member template function with prototype "PParcel to<PParcel>()"
///     the function should cast Parcel to PParcel or throw if not applicable
///     implement via specializations ONLY
///     it is needed to cast to the columns joined with the parent
///
/// - specialization for std::equal_to
/// - specialization for std::hash
/// - copy ctor

//Parcel == current table data
//KeyParcel == intersection with parent
template<typename Parcel>
struct Row {
    Parcel parcel;

    //int card;

    int weight;

    Parcel getParcel() {
        return parcel;
    }

    Row(const Parcel &p/*, int c*/, int w) : parcel(p)/*, card(c)*/, weight(w) {};

    //Row() : card(-1), weight(-1) {};

    void print() const {
        cout << "parcel = ";
        parcel.print();
        cout << ", weight = " << weight << /* ", card = " << card <<*/ endl;
    }

};

//on this table we run Yannakakis - then we split it
template<typename Parcel, typename ParcelHash = hash<Parcel>, typename ParcelEqual = equal_to<Parcel>>
struct Table {
    vector<Row<Parcel>> data;

    // RT::RangeTree<int, bool> rt;
    CountOracle<int> rt;

    ll totalWeight = 0;

    vector<ll> weightPrefixSum;

#if defined(INVERSE_MAPPING) || defined(INVERTED_ACCESS)
    //sort of the inverse mapping for the prefixSum
    unordered_map<Parcel, ll> weightOffsetMap;
#endif

    inline void push_back(const Row<Parcel> &t) {
        data.push_back(t);
        //totalWeight += t.weight*t.card;
        //weightPrefixSum.push_back(weightPrefixSum.back() + t.weight*t.card);
    }

    inline void push_back(const Parcel &t/*, int card = 1*/, int weight = 1) {
        data.emplace_back(t/*, card*/, weight);
        //totalWeight += weight*card;
        //weightPrefixSum.push_back(weightPrefixSum.back() + weight*card);
    }

    /*inline void loadFromFile_simple(const string &filename, int numLines = 0) {
        unordered_set<Parcel, ParcelHash, ParcelEqual> parcelSet;
        parcelSet.reserve(numLines);

        ifstream file(filename);
        assert(!file.fail());

        string line;
        while (getline(file, line)) {
            Parcel curr = Parcel::from(line);
            parcelSet.insert(curr);
        }

        data.reserve(parcelSet.size());
        for (auto &p : parcelSet) {
            data.emplace_back(p, 1); //card we've collected, weight is 1
        }
    }*/

    inline void loadFromFile(const string &filename, int numLines = 0, vector<int> columns = {} /*,
                             SelectionPredicate predicate = nullptr, void *arg = nullptr*/) {

        unordered_set<Parcel, ParcelHash, ParcelEqual> parcelSet;
        parcelSet.reserve(numLines);

        ifstream file;
        file.open(filename);
        assert(!file.fail());

        string line;
        while (getline(file, line)) {
            Parcel curr = Parcel::from(line, columns);
            // curr.print();
            // if(++cnt > 100)break;
            /*if (predicate == nullptr || predicate(arg, (void *) &curr)) {
                parcelSet.insert(curr); //only insert if it passes the filter (or it doesn't exists)
            }*/
            parcelSet.insert(curr);
        }

        data.reserve(parcelSet.size());
        // vector<RT::Point<int, bool>> list;
        vector<Point<int>> list;
        for (auto &p : parcelSet) {
            // RT::Point<int, bool> a(p.toTuple(), 1);
            Point<int> a(p.toTuple());
            list.push_back(a);
        }

        // print the time
        auto start = std::chrono::high_resolution_clock::now();
        rt = CountOracle<int>(list);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cout << "Time to build the Index: " << elapsed.count() << " s\n";
        
        for (auto &p : parcelSet) {
            data.emplace_back(p, 1); //weight is 1
        }
    }

    int cardinality() const {
        return data.size();
    }

    int count(vector<int> &lower, vector<int> &upper){
        return rt.countInRange(lower, upper);
    }

    const vector<int>& getLowerBounds() const {
        return rt.getLowerBounds();
    }

    const vector<int>& getUpperBounds() const {
        return rt.getUpperBounds();
    }

    void print() const {
        cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        for(const Row<Parcel>& row : data) {
            row.print();
        }

        cout << "total weight = " << totalWeight << endl;
        cout << "weightPrefixSum: [";
        for(int i = 0; i < weightPrefixSum.size(); i++) {
            cout << ((i > 0) ? ", " : "") << weightPrefixSum[i];
        }
        cout << "]" << endl;
        cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    }

    // does not preserve anything (weights, prefixSum, totalWeight)
    Table select(SelectionPredicate predicate = nullptr, void* arg = nullptr) {
        Table newTable;

        for(Row<Parcel>& row : data) {
            if (predicate == nullptr || predicate(arg, (void *) &(row.parcel))) {
                newTable.data.emplace_back(row.parcel, 1);
            }
        }


        return newTable;
    }


#ifdef PROJECTION
    template<typename Projected_Parcel>
    Table<Projected_Parcel> project() {
        unordered_set<Projected_Parcel> newRows;
        Table<Projected_Parcel> projectedTable;

        for(auto r : data) {
            /*projection does not conserve weights*/
            Projected_Parcel p = Projected_Parcel::from(r.parcel);
            if(newRows.find(p) == newRows.end()) {
                newRows.insert(p);
                projectedTable.push_back(p);
            }
        }

        return move(projectedTable);
    }
#endif

};

#endif //RANDOMORDERENUMERATION_TABLE_H
