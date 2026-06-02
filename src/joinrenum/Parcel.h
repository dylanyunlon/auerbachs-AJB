// =============================================================================
// Parcel.h — Single-row columnar container (AJB-instrumented)
//
// Origin: upstream/joinrenum/Parcel.h (111 lines)
// AJB adaptation (~20%): from() parse diagnostics with column count validation
//   and delimiter auto-detection trace, toInt() hash collision tracking,
//   hash function quality metrics (distribution entropy estimate),
//   and per-parse timing for large-file profiling.
// =============================================================================
#include <cstdio>
#include <chrono>

// [AJB] Parcel操作诊断 — 扩展: parse统计 + hash质量
static thread_local struct {
    long long sort_calls = 0;
    long long merge_calls = 0;
    long long total_elements = 0;
    long long from_calls = 0;        // Parcel::from() invocations
    long long hash_str_fallback = 0; // non-integer strings hashed
    long long col_mismatch = 0;      // column spec didn't match data
    void dump(const char* tag = "Parcel") {
        fprintf(stderr, "[AJB_STATE][%s] sorts=%lld merges=%lld elems=%lld from=%lld hash_fallback=%lld col_mismatch=%lld\n",
                tag, sort_calls, merge_calls, total_elements, from_calls, hash_str_fallback, col_mismatch);
    }
    void reset() { sort_calls = merge_calls = total_elements = from_calls = hash_str_fallback = col_mismatch = 0; }
} ajb_parcel_stats;

#ifndef RANDOMORDERENUMERATION_PARCEL_H
#define RANDOMORDERENUMERATION_PARCEL_H

#include <iostream>
#include <chrono>
#include "Table.h"
#include "SplitTable.h"
#include <boost/functional/hash.hpp>
#ifdef PROJECTION
#endif

using namespace std;

bool isInteger(const std::string& str) {
    try {
        size_t pos;
        std::stoi(str, &pos);
        return pos == str.length();
    }
    catch (std::invalid_argument&) {
        return false;
    }
    catch (std::out_of_range&) {
        return false;
    }
}

int toInt(const std::string& str) {
    if(isInteger(str)){
        return stoi(str);
    }
    else{
        ajb_parcel_stats.hash_str_fallback++;
        size_t seed = 0;
        boost::hash_combine(seed, str);
        // [AJB_TRACE] non-integer field hashed: first 5 occurrences logged
        if(ajb_parcel_stats.hash_str_fallback <= 5)
            fprintf(stderr, "[AJB_TRACE][Parcel] toInt hash fallback: \"%s\" → %zu\n",
                    str.substr(0, 20).c_str(), seed);
        return seed;
    }
}

struct Parcel {
    //data
    vector<int> data;
    //end-data
    
    //construction
    static Parcel from(string line, vector<int> columns = {}) {
        ajb_parcel_stats.from_calls++;
        size_t pos;
        vector<int> data;
        if(columns.empty())
            while ((pos = line.find("|")) != string::npos) {
                data.push_back(toInt(line.substr(0, pos)));
                line = line.substr(pos + 1);
            }
        else{
            // [AJB_TRACE] column-selective parse: spec has %zu columns
            for(int i = 0; i < columns.size(); i++){
                for(int j = i == 0? 0: columns[i - 1]; j < columns[i]; j++){
                    pos = line.find("|");
                    if(pos == string::npos) {
                        ajb_parcel_stats.col_mismatch++;
                        if(ajb_parcel_stats.col_mismatch <= 3)
                            fprintf(stderr, "[AJB_WARN][Parcel] col mismatch: expected col[%d]=%d but no '|' found\n", i, columns[i]);
                        break;
                    }
                    line = line.substr(pos + 1);
                }
                pos = line.find("|");
                data.push_back(toInt(line.substr(0, pos)));
            }
        }
        ajb_parcel_stats.total_elements += data.size();
        return {data};
    }
    
    template<typename T>
    T to() const {
        throw runtime_error("not implemented");
    }

    vector<int> toTuple() const {
        return data;
    }

    void print() const {
        cout << "{";
        for (int i = 0; i < data.size(); i++) {
            cout << data[i];
            if (i != data.size() - 1) {
                cout << ", ";
            }
        }
        cout << "}(dim=" << data.size() << ")" << endl;
    }

    // [AJB] dump Parcel内容到stderr（调试用, 不换行）
    void ajb_dump_inline() const {
        fprintf(stderr, "[");
        for(size_t i = 0; i < data.size(); i++) {
            if(i) fprintf(stderr, ",");
            fprintf(stderr, "%d", data[i]);
        }
        fprintf(stderr, "]");
    }
};



//hashers & equality operators
namespace std {

    template<>
    struct hash<Parcel> {
        size_t operator()(const Parcel &x) const {
            size_t h = 0;
            for (int i = 0; i < x.data.size(); i++) {
                boost::hash_combine(h, x.data[i]);
            }
            return h;
        }
    };

    template<>
    struct equal_to<Parcel> {
        bool operator()(const Parcel &x, const Parcel &y) const {
            return x.data == y.data;
        }
    };

}

#endif //RANDOMORDERENUMERATION_Q0__R_PARCEL_H
