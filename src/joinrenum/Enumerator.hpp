#include "BanPickTree.hpp"
#include "RRAccessTree.hpp"
#include <sys/resource.h>
#include <ctime>
using namespace std;

size_t getMemoryUsage() {
    std::ifstream stat_stream("/proc/self/status");
    std::string line;
    size_t memory = 0;

    while (std::getline(stat_stream, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::string mem_str = line.substr(6);
            memory = std::stoul(mem_str);
            break;
        }
    }
    return memory; // in KB
}

class Enumerator {

private:
public:
    int option = 3;
    /*
        0: REnum,
        1: REnum_L(Larger),
        2: REnum_M(Merge),
        3: REnum_B(Batch),
        4: REnum_HC(HalfCache),
        5: REnum_HC_Pool (MTI on Nocache levels)
        6: REnum_HC_Pool_basic (basic LTI on Nocache levels)
    */
    bool treeflag = true; // true: use TU-S
    RRAccessTree access_tree;
    BanPickTree bp;

    Enumerator(
        unordered_map<string, vector<string> > relations,
        unordered_map<string, string> filenames,
        unordered_map<string, int> numlines) :
        access_tree(relations, filenames, numlines, treeflag),
        // bp(min(access_tree.AGM, access_tree.idx.jt.treeUpp(access_tree.idx.FB))) {}   
        bp(access_tree.AGM) {access_tree.idx.treeflag = treeflag;}
    void random_enumerate() {
        double totalRRAccessTime = 0;
        int cntsuccess = 0, cnt = 0, step = 20;
        clock_t start = clock();
        clock_t end;
        double elapsed = 0;
        double last_percentage = 0;
        long long s;
        bool res;
        struct rusage r_usage;
        while(bp.remaining()){
            // cout << "REMAINING: " << bp.remaining() << endl;
            cnt++;
            s = bp.pick();
            // auto startRRAccess = std::chrono::high_resolution_clock::now();
            switch(option) {
                case 0: res = access_tree.RRAccess(s); break;
                case 1: res = access_tree.RRAccess_LTI(s); break;
                case 2: res = access_tree.RRAccess_MTI(s); break;
                case 3: res = access_tree.RRAccess_BTI(s); break;
                case 4: res = access_tree.RRAccess_HalfCache(s); break;
                case 5: res = access_tree.RRAccess_HalfCache_Pool(s); break;
                case 6: res = access_tree.RRAccess_HalfCache_Pool_basic(s); break;
                default: res = access_tree.RRAccess_BTI(s); break;
            }
            // res = access_tree.RRAccess_BTI(s);
            // auto endRRAccess = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> elapsedRRAccess = endRRAccess - startRRAccess;
            // totalRRAccessTime += elapsedRRAccess.count();
            if(res){
                // cout << "(";
                // for(int i = 0; i < res.second.size(); i++) {
                //     cout << res.second[i] << ",";
                // }
                // cout << ")" << endl;
                
                cntsuccess++;
                // if(cntsuccess == 77610){
                if(cntsuccess <= 20 || cntsuccess % 10000 == 0){
                end = clock();
                elapsed = double(end - start) / CLOCKS_PER_SEC;
                getrusage(RUSAGE_SELF, &r_usage);
                cout << cntsuccess << ", " << cnt << ", " << bp.remaining() << ", " << bp.getPercentage() << ", " << elapsed  << ", "<< r_usage.ru_maxrss/1024 << "MB, " << access_tree.idx.totalrrtreenode << endl;
                }
            }
            // if(cnt % 100 == 0){
            //     end = clock();
            //     elapsed = double(end - start) / CLOCKS_PER_SEC;
            //     cout << cntsuccess << ", " << cnt << ", " << bp.remaining() << ", " << bp.getPercentage() << ", " << elapsed << endl;
            //     }
            if(res) bp.ban(s,s);
            for(int i = 0; i < access_tree.numti; i++) {
                bp.ban(access_tree.trivialIntervals[i].first, access_tree.trivialIntervals[i].second);
            }
            // else bp.ban(access_tree.trivialInterval.first, access_tree.trivialInterval.second);
        }
        end = clock();
        elapsed = double(end - start) / CLOCKS_PER_SEC;
        cout << cntsuccess << ", " << cnt << ", " << bp.remaining() << ", " << bp.getPercentage() << ", " << elapsed << endl;
        cout << "Total RRAccess Time: " << totalRRAccessTime << endl;
    }

};