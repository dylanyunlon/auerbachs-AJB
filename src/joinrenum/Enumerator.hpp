#include "BanPickTree.hpp"
#include "RRAccessTree.hpp"
#include <sys/resource.h>
#include <ctime>
#include <chrono>
using namespace std;

// [AJB] Enumerator诊断: 追踪enumerate过程中的成功率、ban效率、内存增长
static thread_local struct {
    long long total_attempts = 0;
    long long total_success  = 0;
    long long total_bans     = 0;
    double    last_hit_rate  = 0.0;
    double    peak_rss_mb    = 0.0;
    void dump(const char* tag = "Enumerator") {
        fprintf(stderr, "[AJB_STATE][%s] attempts=%lld success=%lld bans=%lld hit_rate=%.4f peak_rss=%.1fMB\n",
                tag, total_attempts, total_success, total_bans, last_hit_rate, peak_rss_mb);
    }
    void reset() { total_attempts = total_success = total_bans = 0; last_hit_rate = peak_rss_mb = 0.0; }
} ajb_enum_stats;

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
        bp(access_tree.AGM) {
        access_tree.idx.treeflag = treeflag;
        // [AJB_BP] Enumerator ready: AGM + option + treeflag = 枚举的三个关键参数
        fprintf(stderr, "[AJB_BP][Enumerator] constructed: AGM=%lld option=%d treeflag=%d\n",
                access_tree.AGM, option, (int)treeflag);
        fprintf(stderr, "[AJB_STATE][Enumerator] BanPickTree range=[1, %lld]\n", access_tree.AGM);
    }

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
        auto ajb_wall_start = std::chrono::steady_clock::now();
        fprintf(stderr, "[AJB_BP][Enumerator] enumerate start: option=%d AGM=%lld\n",
                option, access_tree.AGM);
        while(bp.remaining()){
            // cout << "REMAINING: " << bp.remaining() << endl;
            cnt++;
            ajb_enum_stats.total_attempts++;
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
                ajb_enum_stats.total_success++;
                // if(cntsuccess == 77610){
                if(cntsuccess <= 20 || cntsuccess % 10000 == 0){
                end = clock();
                elapsed = double(end - start) / CLOCKS_PER_SEC;
                getrusage(RUSAGE_SELF, &r_usage);
                double rss_mb = r_usage.ru_maxrss / 1024.0;
                if(rss_mb > ajb_enum_stats.peak_rss_mb) ajb_enum_stats.peak_rss_mb = rss_mb;
                cout << cntsuccess << ", " << cnt << ", " << bp.remaining() << ", " << bp.getPercentage() << ", " << elapsed  << ", "<< r_usage.ru_maxrss/1024 << "MB, " << access_tree.idx.totalrrtreenode << endl;
                // [AJB_TRACE] 阶段性进度: hit_rate在这里能看出算法效率
                fprintf(stderr, "[AJB_TRACE][Enumerator] progress: success=%d/%d remaining=%lld pct=%.4f rss=%.0fMB rrtreenode=%d\n",
                        cntsuccess, cnt, bp.remaining(), bp.getPercentage(), rss_mb, access_tree.idx.totalrrtreenode);
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
                ajb_enum_stats.total_bans++;
            }
            // else bp.ban(access_tree.trivialInterval.first, access_tree.trivialInterval.second);
        }
        end = clock();
        elapsed = double(end - start) / CLOCKS_PER_SEC;
        auto ajb_wall_end = std::chrono::steady_clock::now();
        double wall_sec = std::chrono::duration<double>(ajb_wall_end - ajb_wall_start).count();
        cout << cntsuccess << ", " << cnt << ", " << bp.remaining() << ", " << bp.getPercentage() << ", " << elapsed << endl;
        cout << "Total RRAccess Time: " << totalRRAccessTime << endl;
        // [AJB_TIMER] final summary: 成功数/尝试数/ban数/wall time
        ajb_enum_stats.last_hit_rate = cnt > 0 ? (double)cntsuccess / cnt : 0.0;
        fprintf(stderr, "[AJB_TIMER][Enumerator] done: success=%d attempts=%d hit_rate=%.6f cpu=%.3fs wall=%.3fs\n",
                cntsuccess, cnt, ajb_enum_stats.last_hit_rate, elapsed, wall_sec);
        fprintf(stderr, "[AJB_STATE][Enumerator] total_bans=%lld peak_rss=%.1fMB\n",
                ajb_enum_stats.total_bans, ajb_enum_stats.peak_rss_mb);
    }

};
