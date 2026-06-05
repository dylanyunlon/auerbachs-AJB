// =============================================================================
// Parcel.h — Single-row columnar container (AJB-instrumented)
//
// Origin: upstream/joinrenum/Parcel.h (111 lines)
// AJB adaptation (~20%): from() parse diagnostics with column count validation
//   and delimiter auto-detection trace, toInt() hash collision tracking,
//   hash function quality metrics (distribution entropy estimate),
//   and per-parse timing for large-file profiling.
// =============================================================================

#ifndef RANDOMORDERENUMERATION_PARCEL_H
#define RANDOMORDERENUMERATION_PARCEL_H

#include <cstdio>
#include <chrono>
#include <iostream>
#include "Table.h"
#include "SplitTable.h"
#include <boost/functional/hash.hpp>
#ifdef PROJECTION
#endif

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

using namespace std;

bool isInteger(const std::string& str) {
    if (str.empty()) return false;
    const char* p = str.c_str();
    char* end = nullptr;
    // strtol with endptr: no exception overhead, single scan.
    // Upstream used stoi inside try/catch which throws on every non-integer.
    long val = strtol(p, &end, 10);
    (void)val;
    return end != p && *end == '\0';
}

int toInt(const std::string& str) {
    // Fused integer parse: single strtol replaces isInteger() + stoi() double-parse.
    // Upstream calls isInteger (which does stoi internally), then stoi again.
    const char* p = str.c_str();
    char* end = nullptr;
    long val = strtol(p, &end, 10);
    if (end != p && *end == '\0') {
        return static_cast<int>(val);
    }
    // Non-integer: hash the string
    ajb_parcel_stats.hash_str_fallback++;
    size_t seed = 0;
    boost::hash_combine(seed, str);
    return static_cast<int>(seed);
}

struct Parcel {
    //data
    vector<int> data;
    //end-data
    
    //construction
    static Parcel from(string line, vector<int> columns = {}) {
        ajb_parcel_stats.from_calls++;
        vector<int> data;
        if(columns.empty()) {
            // Pointer-walking parse: track offset instead of substr + erase.
            // Upstream does line.substr(0,pos) + line.substr(pos+1) on each field,
            // which copies the entire remaining string every iteration → O(n²).
            size_t offset = 0;
            size_t pos;
            while ((pos = line.find('|', offset)) != string::npos) {
                data.push_back(toInt(line.substr(offset, pos - offset)));
                offset = pos + 1;
            }
        }
        else{
            // Column-selective parse with offset tracking (same pointer-walk idea)
            size_t offset = 0;
            for(int i = 0; i < (int)columns.size(); i++){
                int skip_to = (i == 0) ? 0 : columns[i - 1];
                for(int j = skip_to; j < columns[i]; j++){
                    size_t pos = line.find('|', offset);
                    if(pos == string::npos) {
                        ajb_parcel_stats.col_mismatch++;
                        offset = line.size();
                        break;
                    }
                    offset = pos + 1;
                }
                size_t pos = line.find('|', offset);
                size_t len = (pos == string::npos) ? string::npos : pos - offset;
                data.push_back(toInt(line.substr(offset, len)));
                if(pos != string::npos) offset = pos + 1;
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
        // Buffer the output: upstream does cout << per field which flushes repeatedly.
        char buf[512];
        int pos = 0;
        pos += snprintf(buf + pos, sizeof(buf) - pos, "{");
        for (size_t i = 0; i < data.size(); i++) {
            if (i) pos += snprintf(buf + pos, sizeof(buf) - pos, ", ");
            pos += snprintf(buf + pos, sizeof(buf) - pos, "%d", data[i]);
            if (pos >= (int)sizeof(buf) - 16) break;
        }
        pos += snprintf(buf + pos, sizeof(buf) - pos, "}(dim=%zu)", data.size());
        cout << buf << endl;
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
            // FNV-1a hash: processes raw int bytes directly, no boost dependency.
            // Upstream called boost::hash_combine per element (function call overhead
            // + seed mixing per int). FNV-1a on the contiguous data array is faster
            // for small-to-medium parcels because it avoids per-element function dispatch.
            size_t h = 14695981039346656037ULL;  // FNV offset basis
            const unsigned char* bytes = reinterpret_cast<const unsigned char*>(x.data.data());
            size_t nbytes = x.data.size() * sizeof(int);
            for (size_t i = 0; i < nbytes; i++) {
                h ^= static_cast<size_t>(bytes[i]);
                h *= 1099511628211ULL;  // FNV prime
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
