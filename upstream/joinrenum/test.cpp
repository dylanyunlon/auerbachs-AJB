#include <iostream>
#include <random>
#include <vector>
#include "Table.h"
#include "Parcel.h"
#include "Index.hpp"
#include "ReadConfig.hpp"
#include "BanPickTree.hpp"
using namespace std;


void printInfo(Index &idx) {
    
    cout << "Cache Hit of SplitBucket: " << idx.cntCacheHit << " Total Call: " << idx.cntTotalCall << endl;

    cout << "Total AGM Call: " << idx.cntAGMCall << endl;
    cout << "Total AGM Time: " << idx.totalAGMTime << endl;
    cout << "Total Count Oracle Time: " << idx.totalCountOracleTime << endl;
    cout << "Total Split Time: " << idx.totalSplitTime << endl;
    cout << "Total Split Call: " << idx.cntSplitCall << endl;
    cout << "Total Cache Hit Time: " << idx.totalCacheHitTime << endl;
    return;
}

int main() {
    // Table<Parcel> tbl;
    unordered_map<string, string> filenames = readFilenames("db/filenames.txt");
    unordered_map<string, int> numlines = readNumLines("db/numlines.txt");
    unordered_map<string, vector<string> > relations = readRelations("db/relations.txt");
    
    unordered_map<string, vector<string> >::iterator it = relations.begin();
    vector<string> query_rels;
    vector<vector<string> > query_vars;
    while(it != relations.end()) {
        query_rels.push_back(it->first);
        query_vars.push_back(it->second);
        it++;
    }
    for (int i = 0; i < query_rels.size(); i++) {
        cout << query_rels[i] << ": ";
        for (int j = 0; j < query_vars[i].size(); j++) {
            cout << query_vars[i][j] << " ";
        }
        cout << endl;
    }

    // Query q(query_rels, query_vars);
    Query q({"R1", "R2", "R3"}, {{"A", "B"}, {"B", "C"}, {"A", "C"}});
    // Query q({"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"}, {{"x1", "x2"}, {"x2", "x3"}, {"x1", "x3"}, {"x3", "x4"}, {"x4", "x5"}, {"x5", "x6"}, {"x4", "x6"}, {"x1", "x5"}, {"x2", "x6"}});
    // Query q({"R1", "R2", "R3", "R4", "R5", "R6"}, {{"P", "Q", "R"}, {"Q", "S", "T"}, {"R", "T", "U"}, {"P", "S", "V"}, {"U", "V", "W"}, {"W", "P", "Q"}});
    // Query q({"R1", "R2", "R3", "R4"}, {{"A", "B", "C", "D"}, {"B", "C", "E", "F"}, {"C", "D", "F", "G"}, {"B", "D", "E", "G"}});

    // Query q({"L1", "L2", "O1", "O2", "C1", "C2", "S"},
    // {{"ok1", "pk"},
    //  {"ok2", "pk"},
    //  {"ok1", "ck1"},
    //  {"ok2", "ck2"},
    //  {"ck1", "nk"},
    //  {"ck2", "nk"},
    //  {"sk", "nk"}});
    Index idx(q);
    idx.preProcessing(relations, filenames, numlines);
    cout << "Variables: ";
    for(int i = 0; i < q.getVarNames().size(); i++) {
        cout << q.getVarNames()[i] << " ";
    }
    cout << endl;
    // idx.getFullBucket().print();
    // for(int i = 1; i < 20; i++) {
    //     vector<int> res = idx.sampleUntilSuccess();
    //     cout << "Sample " << i << ": ";
    //     for(int j = 0; j < res.size(); j++) {
    //         cout << res[j] << " ";
    //     }
    //     cout << endl;
    // }
    // idx.printBucketInfo(idx.getFullBucket());
    // idx.printBucketTree(idx.getFullBucket());


    int cntsuccess = 0, cnt = 0;
    // for(int i = 1; i <= idx.AGM(); i++) {
    //         pair<bool, vector<int> > res = idx.randomAccess(idx.getFullBucket(), i);
    //         cout << i << ": ";
    //         cout << res.first << "::";
    //         for(int j = 0; j < res.second.size(); j++) {
    //             cout << res.second[j] << ",";
    //         }
    //         cout << endl;
    //         if(!res.first) {
    //             cntfail++;
    //             i = res.second[1];
    //         }
    //         else cntsuccess++;
    // }

    // vector<int> cars = idx.getCar(idx.getFullBucket());
    // cout << endl;
    // if(freopen("res/res_q1_bmitu.txt", "w", stdout) == NULL)cout << "WRITEERR" << endl;
    int step = 20;
    cout << idx.AGM() << endl;
    //////////////////////////////REnum-BMITU
    BanPickTree bp(idx.AGM());
    if(freopen("res/result.txt", "w", stdout) == NULL)cout << "WRITEERR" << endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double last_percentage = 0;
    while(bp.remaining()){
        cnt++;
        int s = bp.pick();
        pair<bool, vector<int> > res = idx.randomAccess_opt(idx.getFullBucket(), s);
        if(res.first){
            cntsuccess++;
            if(cntsuccess < step || cntsuccess % step == 0){
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            cout << cntsuccess << ", " << cnt << ", " << bp.remaining() << ", " << bp.getPercentage() << ", " << elapsed.count() << endl;
        }
            if(cntsuccess % 500 == 0)printInfo(idx);

        }        
        if(res.first) bp.ban(s,s);

        else bp.ban(res.second[0], res.second[1]);
        // double done = bp.getPercentage();
        // if(int(done * 100) % 10 == 0 && int(done * 100) != int(last_percentage*100)){
        //     last_percentage = done;
        //     end = std::chrono::high_resolution_clock::now();
        //     elapsed = end - start;
        //     cout << bp.getPercentage() << ", " << elapsed.count() << endl;
        //     last_percentage = done;
        // }
    }


    // ////////////////////////////////REnum
    // int N = idx.AGM();
    // vector<int> A(N,0);
    // if(freopen("res/res_q1_renum.txt", "w", stdout) == NULL)cout << "WRITEERR" << endl;
    // random_device rd;
    // mt19937 gen(rd());
    // auto start = std::chrono::high_resolution_clock::now();
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // int pos, j;
    // for(int i = 1; i <= N; i++){
    //     cnt++;
    //     uniform_int_distribution<> distr(i, N);
    //     j = distr(gen);
    //     if(A[j] > 0)pos = A[j];
    //     else pos = j;
    //     if(A[i] > 0)A[j] = A[i];
    //     else A[j] = i;
    //     pair<bool, vector<int> > res = idx.randomAccess(idx.getFullBucket(), pos);
    //     if(res.first){
    //         cntsuccess++;
    //         end = std::chrono::high_resolution_clock::now();
    //         elapsed = end - start;
    //         cout << cntsuccess << ", " << cnt << ", " << elapsed.count() << endl;
    //     }        
    // }

    //////////////////////////////Sample
    // set<vector<int> > S;
    // if(freopen("res/res_q2_sample.txt", "w", stdout) == NULL)cout << "WRITEERR" << endl;
    // while(true) {
    //     cnt++;
    //     vector<int> s = idx.sampleUntilSuccess();
    //     if(S.find(s) != S.end()) continue;
    //     S.insert(s);
    //     cntsuccess++;
    //     if(cntsuccess < step || cntsuccess % step == 0){
    //         end = std::chrono::high_resolution_clock::now();
    //         elapsed = end - start;
    //         cout << cntsuccess << ", " << cnt << ", " << elapsed.count() << endl;
    //     }
    // }
    
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << cntsuccess << ", " << cnt << ", " << bp.remaining() << ", " << bp.getPercentage() << ", " << elapsed.count() << endl;

    printInfo(idx);

    // idx.printBucketTree(idx.getFullBucket());
    // cout << cntsuccess + 1 << ", " << cnt << ", " << elapsed.count() << endl;
    // cout << "Success: " << cntsuccess << " Total: " << cnt << endl;
    // cout << "Total: " << bp.getTotal() << endl;
    return 0;
}
