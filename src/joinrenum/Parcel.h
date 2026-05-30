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
        size_t seed = 0;
        boost::hash_combine(seed, str);
        return seed;
    }
}

struct Parcel {
    //data
    vector<int> data;
    //end-data
    
    //construction
    static Parcel from(string line, vector<int> columns = {}) {
        size_t pos;
        vector<int> data;
        if(columns.empty())
            while ((pos = line.find("|")) != string::npos) {
                data.push_back(toInt(line.substr(0, pos)));
                line = line.substr(pos + 1);
            }
        else{
            for(int i = 0; i < columns.size(); i++){
                for(int j = i == 0? 0: columns[i - 1]; j < columns[i]; j++){
                    pos = line.find("|");
                    line = line.substr(pos + 1);
                }
                pos = line.find("|");
                data.push_back(toInt(line.substr(0, pos)));
            }
        }
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
        cout << "}" << endl;
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