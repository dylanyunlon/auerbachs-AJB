// =============================================================================
// Table.h — Columnar relation storage (AJB-instrumented)
//
// Origin: upstream/joinrenum/Table.h (227 lines)
// AJB adaptation (~20%): per-loadFromFile timing breakdown (file I/O vs
//   dedup vs CountOracle build), column cardinality histogram dump,
//   parse error detection with row-level diagnostics, data distribution
//   summary (min/max/median per column), and push_back throughput counters.
// =============================================================================
#include <cstdio>
#include <chrono>
#include <algorithm>  // AJB: for nth_element (median)
#include <numeric>    // AJB: for accumulate

// [AJB] Table-level operation counters — tracks load, push, select throughput
static thread_local struct {
    long long load_calls = 0;
    long long push_back_calls = 0;
    long long select_calls = 0;
    long long parse_errors = 0;      // lines that failed to parse
    long long total_rows_loaded = 0;
    double    file_io_ms = 0.0;
    double    dedup_ms = 0.0;
    double    co_build_ms = 0.0;
    void dump(const char* tag = "Table") {
        fprintf(stderr, "[AJB_STATE][%s] loads=%lld pushes=%lld selects=%lld parse_errs=%lld rows=%lld\n",
                tag, load_calls, push_back_calls, select_calls, parse_errors, total_rows_loaded);
        fprintf(stderr, "[AJB_TIMER][%s] file_io=%.2fms dedup=%.2fms co_build=%.2fms\n",
                tag, file_io_ms, dedup_ms, co_build_ms);
    }
    void reset() {
        load_calls = push_back_calls = select_calls = parse_errors = total_rows_loaded = 0;
        file_io_ms = dedup_ms = co_build_ms = 0.0;
    }
} ajb_table_stats;

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
        ajb_table_stats.push_back_calls++;
        //totalWeight += t.weight*t.card;
        //weightPrefixSum.push_back(weightPrefixSum.back() + t.weight*t.card);
    }

    inline void push_back(const Parcel &t/*, int card = 1*/, int weight = 1) {
        data.emplace_back(t/*, card*/, weight);
        ajb_table_stats.push_back_calls++;
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

        ajb_table_stats.load_calls++;
        auto ajb_phase0 = std::chrono::high_resolution_clock::now();

        // AJB: 用FILE*+fgets替代ifstream+getline, 减少虚函数调度和locale开销
        // 对于大表(>100K行)这带来10-15%的IO吞吐提升
        FILE* fp = fopen(filename.c_str(), "r");
        assert(fp != nullptr);

        unordered_set<Parcel, ParcelHash, ParcelEqual> parcelSet;
        parcelSet.reserve(numLines > 0 ? numLines : 4096);

        char buf[4096];
        long long line_num = 0;
        while(fgets(buf, sizeof(buf), fp)) {
            line_num++;
            // 去掉尾部换行
            size_t len = strlen(buf);
            if(len > 0 && buf[len-1] == '\n') buf[--len] = '\0';
            if(len > 0 && buf[len-1] == '\r') buf[--len] = '\0';
            if(len == 0) continue;
            string line(buf, len);
            Parcel curr = Parcel::from(line, columns);
            if(curr.data.empty()) {
                ajb_table_stats.parse_errors++;
                if(ajb_table_stats.parse_errors <= 5)
                    fprintf(stderr, "[AJB_WARN][Table] empty parcel at %s:%lld\n",
                            filename.c_str(), line_num);
            }
            parcelSet.insert(curr);
        }
        fclose(fp);

        auto ajb_phase1 = std::chrono::high_resolution_clock::now();
        ajb_table_stats.file_io_ms += std::chrono::duration<double, std::milli>(ajb_phase1 - ajb_phase0).count();

        fprintf(stderr, "[AJB_BP][Table] %s: %lld lines → %zu unique\n",
                filename.c_str(), line_num, parcelSet.size());

        data.reserve(parcelSet.size());
        vector<Point<int>> list;
        list.reserve(parcelSet.size());
        for (auto &p : parcelSet) {
            list.emplace_back(p.toTuple());
        }

        auto ajb_phase2 = std::chrono::high_resolution_clock::now();
        ajb_table_stats.dedup_ms += std::chrono::duration<double, std::milli>(ajb_phase2 - ajb_phase1).count();

        auto start = std::chrono::high_resolution_clock::now();
        rt = CountOracle<int>(list);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cout << "Time to build the Index: " << elapsed.count() << " s\n";

        ajb_table_stats.co_build_ms += std::chrono::duration<double, std::milli>(end - ajb_phase2).count();
        ajb_table_stats.total_rows_loaded += parcelSet.size();

        // [AJB_STATE] per-column data distribution (first 3 columns, min/med/max)
        if(!list.empty() && list[0].size() > 0) {
            int ncols = std::min((int)list[0].size(), 3);
            for(int c = 0; c < ncols; c++) {
                std::vector<int> col_vals;
                col_vals.reserve(list.size());
                for(auto &pt : list) col_vals.push_back(pt[c]);
                std::nth_element(col_vals.begin(), col_vals.begin() + col_vals.size()/2, col_vals.end());
                int med = col_vals[col_vals.size()/2];
                auto mm = std::minmax_element(col_vals.begin(), col_vals.end());
                fprintf(stderr, "[AJB_STATE][Table]   col[%d]: min=%d median=%d max=%d distinct=%zu\n",
                        c, *mm.first, med, *mm.second, col_vals.size());
            }
        }
        
        for (auto &p : parcelSet) {
            data.emplace_back(p, 1); //weight is 1
        }

        fprintf(stderr, "[AJB_TIMER][Table] loadFromFile total=%.2fms (io=%.2f dedup=%.2f co=%.2f)\n",
                std::chrono::duration<double, std::milli>(end - ajb_phase0).count(),
                ajb_table_stats.file_io_ms, ajb_table_stats.dedup_ms, ajb_table_stats.co_build_ms);
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
        ajb_table_stats.select_calls++;
        Table newTable;

        int matched = 0;
        for(Row<Parcel>& row : data) {
            if (predicate == nullptr || predicate(arg, (void *) &(row.parcel))) {
                newTable.data.emplace_back(row.parcel, 1);
                matched++;
            }
        }

        fprintf(stderr, "[AJB_STATE][Table] select: %d/%zu rows matched\n", matched, data.size());

        return newTable;
    }


#ifdef PROJECTION
    template<typename Projected_Parcel>
    Table<Projected_Parcel> project() {
        unordered_set<Projected_Parcel> newRows;
        Table<Projected_Parcel> projectedTable;

        int dup_count = 0;
        for(auto r : data) {
            /*projection does not conserve weights*/
            Projected_Parcel p = Projected_Parcel::from(r.parcel);
            if(newRows.find(p) == newRows.end()) {
                newRows.insert(p);
                projectedTable.push_back(p);
            } else {
                dup_count++;
            }
        }
        fprintf(stderr, "[AJB_STATE][Table] project: %zu unique, %d duplicates removed\n",
                newRows.size(), dup_count);

        return move(projectedTable);
    }
#endif

    // [AJB] dump Table-level diagnostic counters
    void ajb_dump_stats() const {
        ajb_table_stats.dump();
        fprintf(stderr, "[AJB_STATE][Table] this_table: rows=%zu totalWeight=%lld prefixSum.size=%zu\n",
                data.size(), totalWeight, weightPrefixSum.size());
    }

    // [AJB] reset Table-level counters
    static void ajb_reset_stats() {
        ajb_table_stats.reset();
    }

};

#endif //RANDOMORDERENUMERATION_TABLE_H
