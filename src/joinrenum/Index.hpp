// =============================================================================
// Index.hpp — Core join index: manages RangeTree/CountOracle per relation
//             (AJB-instrumented)
//
// Origin: upstream/joinrenum/Index.hpp (935 lines)
// AJB adaptation (~20%): per-function entry/exit breakpoints with line-locator
//   tags, MHBS convergence trace (iteration count per call), split chain
//   depth tracking, preProcessing per-relation timing breakdown,
//   setAGMandIters bound-range dump, and sample()/randomAccess() path trace.
// M916: HyperLogLog cardinality estimation, split dim selection trace
// =============================================================================
#include <cstdio>
#include <chrono>
#include <random>  // AJB: for sample() method

// [AJB] Index运行时诊断计数器 — 扩展版
static thread_local struct {
    long long preprocess_calls = 0;
    long long agm_calls = 0;
    long long set_agm_calls = 0;
    long long split_calls = 0;
    long long split_bs_calls = 0;       // splitBucket_BS variant
    long long mhbs_calls = 0;
    long long mhbs_total_iters = 0;     // MHBS内部循环总次数
    long long mhbs_max_iters = 0;       // 单次MHBS最大循环次数
    long long split_pool_calls = 0;
    double preprocess_ms = 0;
    double agm_total_ms = 0;
    double split_total_ms = 0;
    double mhbs_total_ms = 0;
    int max_split_children = 0;
    int max_split_depth = 0;            // splitBucket递归最大深度
    void dump(const char* tag = "Index") {
#ifdef AJB_DEBUG
        fprintf(stderr, "[AJB_STATE][%s] preprocess=%lld agm=%lld set_agm=%lld split=%lld split_bs=%lld mhbs=%lld\n",
                tag, preprocess_calls, agm_calls, set_agm_calls, split_calls, split_bs_calls, mhbs_calls);
        fprintf(stderr, "[AJB_STATE][%s] mhbs_iters=%lld mhbs_max_single=%lld split_pool=%lld\n",
                tag, mhbs_total_iters, mhbs_max_iters, split_pool_calls);
        fprintf(stderr, "[AJB_TIMER][%s] preprocess=%.1fms agm=%.1fms split=%.1fms mhbs=%.1fms\n",
                tag, preprocess_ms, agm_total_ms, split_total_ms, mhbs_total_ms);
        fprintf(stderr, "[AJB_STATE][%s] max_split_children=%d max_split_depth=%d\n",
                tag, max_split_children, max_split_depth);
#endif
    }
    void reset() {
        preprocess_calls = agm_calls = set_agm_calls = split_calls = split_bs_calls = mhbs_calls = 0;
        mhbs_total_iters = mhbs_max_iters = split_pool_calls = 0;
        preprocess_ms = agm_total_ms = split_total_ms = mhbs_total_ms = 0;
        max_split_children = max_split_depth = 0;
    }
} ajb_idx_stats;

#include "Table.h"
#include "AGM.hpp"
#include "Parcel.h"
#include "Bucket.hpp"
#include "JoinTree.hpp"
#include "BucketPool.hpp"
#include <random>
// #include "MHBS.hpp"
using namespace std;

class Index {
    private:
        // AGM上界: 将浮点AGM值转换为整数上界
        // upstream在8处重复写 ceil(res)-res<1e-5 ? ceil(res) : (long long)res
        // 这里统一为单一语义: "最紧整数上界"
        // 算法区别: epsilon判断从1e-5收紧到1e-9, 避免大基数下的误判
        static inline long long agm_upper(double val) {
            double c = ceil(val);
            return (c - val < 1e-9) ? static_cast<long long>(c)
                                     : static_cast<long long>(val);
        }
        
    public:
        Query q;
        JoinTree jt;
        Bucket FB;
        vector<vector<int> > R;
        vector<Table<Parcel> > tables;
        vector<vector<vector<int> > > data;
        vector<vector<long long> > treeBound;
        vector<vector<int> > varPos; // varPos[i][j] = the position of the j-th variable in the i-th relation, -1 if not found
        vector<vector<uint8_t> > mask;  // AJB-algo: avoid vector<bool> proxy reference
        vector<vector<int> > rels;
        vector<int> cardinalities;
        vector<pair<vector<int>::iterator, vector<int>::iterator> > vecIters;
        bool treeflag = false;
        // vector<int> attVal;
        // vector<vector<Point<int> >::iterator> beginIters;
        int cntCacheHit = 0;
        int cntTotalCall = 0;
        int cntAGMCall = 0;
        int cntSplitCall = 0;
        int cntBSCall = 0;
        int totalrrtreenode = 0;
        double totalAGMTime = 0;
        double totalCountOracleTime = 0;
        double totalSplitTime = 0;
        double totalCacheHitTime = 0;
        double totalBoundPrepareTime = 0;
        std::mt19937 gen{std::random_device{}()};  // AJB: persistent RNG

        Index() {};

        Index(Query q, bool treeflag = false) : q(q), treeflag(treeflag) {};
        // [AJB_BP] Index constructed with query

        // Index(
        //     const unordered_map<string, vector<string> >& relations,
        //     const unordered_map<string, string>& filenames,
        //     const unordered_map<string, int>& numLines) {
        //     // parse vector<string> relationNames, vector<vector<string> > relations from relations
        //     vector<string> relationNames;
        //     vector<vector<string> > relationVars;
        //     for(unordered_map<string, vector<string> >::const_iterator it = relations.begin(); it != relations.end(); it++) {
        //         relationNames.push_back(it->first);
        //         relationVars.push_back(it->second);
        //     }
        //     q = Query(relationNames, relationVars);
        //     preProcessing(relations, filenames, numLines);
        // }
        void getpos(const vector<pair<vector<int>::iterator, vector<int>::iterator> > &iters, const vector<pair<vector<int>::iterator, vector<int>::iterator> > &bounds, int splitDim, const int t, vector<int> &pos) {
            for (int i = 0; i < iters.size(); i++) {
                if(!mask[splitDim][i]) pos[i] = iters[i].second - iters[i].first;
                else pos[i] = lower_bound(bounds[i].first, bounds[i].second, t) - iters[i].first;
            }
        }

        int MultiHeadBinarySearch(const vector<pair<vector<int>::iterator, vector<int>::iterator> > &iters, int splitDim, const long long target) {
            vector<pair<vector<int>::iterator, vector<int>::iterator> > bounds = iters;
            ajb_idx_stats.mhbs_calls++;
            vector<vector<int>::iterator> itermid(iters.size());
            vector<int> pos(iters.size());
            vector<int> tmppos(iters.size());
            // AJB: indexbounds用于方向缩窄——当relation已收敛时跳过min/max扫描
            vector<pair<int, int> > indexbounds;
        indexbounds.reserve(iters.size());  // AJB-algo: pre-reserve
            const size_t nrels = iters.size();
            int mini, maxi, cnt = 0;
            long long upp;
            double res;
            for(int i = 0; i < iters.size(); i++) {
                if(!mask[splitDim][i]) {
                    pos[i] = iters[i].second - iters[i].first;
                    cnt++;
                }
                else if(iters[i].second - iters[i].first <= 1) {
                    getpos(iters, bounds, splitDim, *iters[i].first + 1, tmppos);
                    res = q.AGM(tmppos);
                    upp = agm_upper(res);
                    if(upp > target) return *iters[i].first;
                    else pos[i] = iters[i].second - iters[i].first;
                    cnt++;
                }
                else{
                    itermid[i] = iters[i].first + (iters[i].second - iters[i].first) / 2;
                    pos[i] = itermid[i] - iters[i].first;
                }
            }
            while(cnt < iters.size()) {
                mini = -1, maxi = -1;
                for(size_t i : rels[splitDim]){
                    if(bounds[i].second - bounds[i].first <= 1) continue;
                    if(mini == -1 || *itermid[i] < *itermid[mini]) mini = i;
                    if(maxi == -1 || *itermid[i] > *itermid[maxi]) maxi = i;
                }
                res = q.AGM(pos);
                upp = agm_upper(res);
                if(upp <= target) {
                    bounds[mini].first = itermid[mini];
                    if(bounds[mini].second - bounds[mini].first <= 1) {
                        if(*bounds[mini].first == *bounds[mini].second) return *bounds[mini].first;
                        getpos(iters, bounds, splitDim, *bounds[mini].first + 1, tmppos);
                        res = q.AGM(tmppos);
                        upp = agm_upper(res);
                        if(upp > target) return *bounds[mini].first;
                        else pos[mini] = bounds[mini].second - iters[mini].first;
                        cnt++;
                    }
                    else {
                        itermid[mini] = bounds[mini].first + (bounds[mini].second - bounds[mini].first) / 2;
                        pos[mini] = itermid[mini] - iters[mini].first;
                    }
                }
                else {
                    bounds[maxi].second = itermid[maxi];
                    if(bounds[maxi].second - bounds[maxi].first <= 1) {
                        if(*bounds[maxi].first == *bounds[maxi].second) return *bounds[maxi].first;
                        getpos(iters, bounds, splitDim, *bounds[maxi].first + 1, tmppos);
                        res = q.AGM(tmppos);
                        upp = agm_upper(res);
                        if(upp > target) return *bounds[maxi].first;
                        else pos[maxi] = bounds[maxi].second - iters[maxi].first;
                        cnt++;
                    }
                    else {
                        itermid[maxi] = bounds[maxi].first + (bounds[maxi].second - bounds[maxi].first) / 2;
                        pos[maxi] = itermid[maxi] - iters[maxi].first;
                    }
                }
            }
            int ans = 2147483647;
            for(int i = 0; i < iters.size(); i++) {
                if(bounds[i].second != iters[i].second) ans = min(ans, *bounds[i].second);
            }
            return ans;
        }

        void getpos(const vector<pair<int, int> > &iters, const vector<pair<int, int> > &bounds, int splitDim, const int t, vector<int> &pos) {
            for (int i = 0; i < iters.size(); i++) {
                if(!mask[splitDim][i]) pos[i] = iters[i].second - iters[i].first;
                else {
                    // 用lower_bound查找t在data[i][varPos[i][splitDim]][bounds[i].first, bounds[i].second)中的位置
                    auto &vec = data[i][varPos[i][splitDim]];
                    pos[i] = std::lower_bound(vec.begin() + bounds[i].first, vec.begin() + bounds[i].second, t) - (vec.begin() + iters[i].first);
                }
            }
        }

        int MultiHeadBinarySearch(const vector<pair<int, int> > &iters, int splitDim, const long long target) {
            vector<pair<int, int> > bounds = iters;
            ajb_idx_stats.mhbs_calls++;
            vector<int> itermid(iters.size());
            vector<int> pos(iters.size());
            vector<int> tmppos(iters.size());
            // AJB算法改写: 预缓存每个relation在splitDim上的列指针
            // upstream每次循环做 (*splitCol[i])[idx] (三层下标)
            // 缓存后变成 splitCol[i][idx] (一层下标)
            vector<const vector<int>*> splitCol(iters.size(), nullptr);
            for(size_t i = 0; i < iters.size(); i++) {
                if(mask[splitDim][i] && varPos[i][splitDim] >= 0)
                    splitCol[i] = &data[i][varPos[i][splitDim]];
            }
            int mini, maxi, cnt = 0;
            long long upp;
            double res;
            for(int i = 0; i < iters.size(); i++) {
                if(!mask[splitDim][i]) {
                    pos[i] = iters[i].second - iters[i].first;
                    cnt++;
                }
                else if(iters[i].second - iters[i].first <= 1) {
                    getpos(iters, bounds, splitDim, (*splitCol[i])[iters[i].first] + 1, tmppos);
                    res = q.AGM(tmppos);
                    upp = agm_upper(res);
                    if(treeflag && splitDim < jt.countRels.size()) upp = min(upp, treeUpp(iters, tmppos, jt.countRels[splitDim]));
            // if(B.splitDim < jt.countRels.size())B.AGM = min(B.AGM, treeUpp(B.iters, jt.countRels[B.splitDim]));
                    if(upp > target) return (*splitCol[i])[iters[i].first];
                    else pos[i] = iters[i].second - iters[i].first;
                    cnt++;
                }
                else{
                    itermid[i] = iters[i].first + (iters[i].second - iters[i].first) / 2;
                    pos[i] = itermid[i] - iters[i].first;
                }
            }
            // AJB: 活跃relation集合——只扫描还在二分中的relation
            vector<size_t> activeRels;
            activeRels.reserve(rels[splitDim].size());
            for (size_t ri : rels[splitDim]) {
                if (bounds[ri].second - bounds[ri].first > 1) activeRels.push_back(ri);
            }
            while(cnt < iters.size()) {
                mini = -1, maxi = -1;
                for(size_t i : activeRels){
                    if(bounds[i].second - bounds[i].first <= 1) continue;
                    if(mini == -1 || (*splitCol[i])[itermid[i]] < (*splitCol[mini])[itermid[mini]]) mini = i;
                    if(maxi == -1 || (*splitCol[i])[itermid[i]] > (*splitCol[maxi])[itermid[maxi]]) maxi = i;
                }
                res = q.AGM(pos);
                upp = agm_upper(res);
                if(treeflag && splitDim < jt.countRels.size()) upp = min(upp, treeUpp(iters, pos, jt.countRels[splitDim]));
                if(upp <= target) {
                    bounds[mini].first = itermid[mini];
                    if(bounds[mini].second - bounds[mini].first <= 1) {
                        // fprintf(stderr, "[AJB_BP][Index] output line 0\n"); // was: cout
                        if(bounds[mini].second < (*splitCol[mini]).size() && (*splitCol[mini])[bounds[mini].first] == (*splitCol[mini])[bounds[mini].second])
                            return (*splitCol[mini])[bounds[mini].first];
                        getpos(iters, bounds, splitDim, (*splitCol[mini])[bounds[mini].first] + 1, tmppos);
                        res = q.AGM(tmppos);
                        upp = agm_upper(res);
                        if(treeflag && splitDim < jt.countRels.size()) upp = min(upp, treeUpp(iters, tmppos, jt.countRels[splitDim]));
                        if(upp > target) return (*splitCol[mini])[bounds[mini].first];
                        else pos[mini] = bounds[mini].second - iters[mini].first;
                        cnt++;
                    }
                    else {
                        itermid[mini] = bounds[mini].first + (bounds[mini].second - bounds[mini].first) / 2;
                        pos[mini] = itermid[mini] - iters[mini].first;
                    }
                }
                else {
                    bounds[maxi].second = itermid[maxi];
                    if(bounds[maxi].second - bounds[maxi].first <= 1) {
                        if(bounds[maxi].second < (*splitCol[maxi]).size() && (*splitCol[maxi])[bounds[maxi].first] == (*splitCol[maxi])[bounds[maxi].second])
                            return (*splitCol[maxi])[bounds[maxi].first];
                        getpos(iters, bounds, splitDim, (*splitCol[maxi])[bounds[maxi].first] + 1, tmppos);
                        res = q.AGM(tmppos);
                        upp = agm_upper(res);
                        if(treeflag && splitDim < jt.countRels.size()) upp = min(upp, treeUpp(iters, tmppos, jt.countRels[splitDim]));
                        if(upp > target) return (*splitCol[maxi])[bounds[maxi].first];
                        else pos[maxi] = bounds[maxi].second - iters[maxi].first;
                        cnt++;
                    }
                    else {
                        itermid[maxi] = bounds[maxi].first + (bounds[maxi].second - bounds[maxi].first) / 2;
                        pos[maxi] = itermid[maxi] - iters[maxi].first;
                    }
                }
            }
            int ans = 2147483647;
            for(int i = 0; i < iters.size(); i++) {
                if(bounds[i].second != iters[i].second) ans = min(ans, (*splitCol[i])[bounds[i].second]);
            }
            return ans;
        }

        void preProcessing(const unordered_map<string, vector<string> >& relations, const unordered_map<string, string>& filenames, const unordered_map<string, int>& numLines) {
            for(size_t i = 0; i < q.getRelNames().size(); i++) {
            ajb_idx_stats.preprocess_calls++;
            auto ajb_pp_t0 = std::chrono::steady_clock::now();
            fprintf(stderr, "[AJB_BP][Index] preProcessing start: %zu relations\n", relations.size());
                string relName = q.getRelNames()[i];
                Table<Parcel> tbl;
                vector<int> columns;
                // AJB算法改写: 用unordered_map做列名→位置的O(1)查找
                // upstream: 对每个变量线性扫描relation的列名 O(vars*cols)
                // 改为: 先建索引 O(cols), 再查 O(vars)
                const auto& relCols = relations.at(relName);
                unordered_map<string, int> colIndex;
                colIndex.reserve(relCols.size());
                for(size_t k = 0; k < relCols.size(); k++) {
                    colIndex[relCols[k]] = k;
                }
                for(size_t j = 0; j < q.getRelVars()[i].size(); j++) {
                    auto it = colIndex.find(q.getRelVars()[i][j]);
                    if (it != colIndex.end()) columns.push_back(it->second);
                }
                tbl.loadFromFile(filenames.at(relName), numLines.at(relName), columns);
                tables.push_back(tbl);
            auto ajb_pp_t1 = std::chrono::steady_clock::now();
            ajb_idx_stats.preprocess_ms = std::chrono::duration<double, std::milli>(ajb_pp_t1 - ajb_pp_t0).count();
            fprintf(stderr, "[AJB_TIMER][Index] preProcessing: %.1fms, %zu tables built\n",
                    ajb_idx_stats.preprocess_ms, tables.size());
            fprintf(stderr, "[AJB_STATE][Index] varnum=%d relations=%zu FB.AGM=%lld\n",
                    q.getVarNumber(), R.size(), FB.AGM);
            }
            jt = JoinTree(q, getCountOracles());
            R = q.getRelations();
            int varnum = q.getVarNumber();
            vector<int> lowerBound(varnum, 2147483647);
            vector<int> upperBound(varnum, -2147483648);
            // AJB算法改写: 缓存每个table的bounds引用——getLowerBounds/getUpperBounds
            // 可能是虚函数调用, 在relation多时避免重复dispatch
            for(size_t i = 0; i < R.size(); i++) {
                const auto& lb = tables[i].getLowerBounds();
                const auto& ub = tables[i].getUpperBounds();
                for(size_t j = 0; j < R[i].size(); j++) {
                    int var = R[i][j];
                    if (lb[j] < lowerBound[var]) lowerBound[var] = lb[j];
                    if (ub[j] > upperBound[var]) upperBound[var] = ub[j];
                }
            }
            FB = {lowerBound, upperBound};
            // for(size_t i = 0; i < tables.size(); i++) {
            //     beginIters.push_back(tables[i].rt.points.begin());
            // }
            // set<int> attValSet;
            // for(size_t i = 0; i < tables[0].rt.points.size(); i++) {
            //     attValSet.insert(tables[0].rt.points[i][0]);
            // }
            // for(auto it = attValSet.begin(); it != attValSet.end(); it++) {
            //     attVal.push_back(*it);
            // }
            // store the points in column
            data.resize(tables.size());
            treeBound.resize(tables.size());
            cardinalities.resize(tables.size());
            vecIters.resize(tables.size());
            varPos.resize(tables.size(), vector<int>(q.getVarNumber(), 0));
            mask.resize(q.getVarNumber(), vector<uint8_t>(tables.size(), 0));
            rels.resize(q.getVarNumber(), {});
            for(size_t i = 0; i < data.size(); i++) {
                const auto& rels_i = q.getRelations()[i];
                const auto& pts = tables[i].rt.points;
                data[i].resize(rels_i.size());
                treeBound[i].resize(pts.size() + 1);
                for(size_t j = 0; j < data[i].size(); j++) {
                    varPos[i][rels_i[j]] = j;
                    mask[rels_i[j]][i] = true;
                    rels[rels_i[j]].push_back(i);
                    // AJB算法改写: reserve + push_back代替resize + 逐元素赋值
                    // 对大表减少一次默认初始化遍历
                    data[i][j].reserve(pts.size());
                    for(size_t k = 0; k < pts.size(); k++) {
                        data[i][j].push_back(pts[k][j]);
                    }
                }
                // AJB算法改写: treeBound前缀和用std::partial_sum
                // upstream: 手写循环 treeBound[i][j] = points[j-1].cnt
                // 这里先收集cnt到vector, 再用partial_sum生成前缀和
                treeBound[i][0] = 0;
                if (pts.size() > 0) {
                    vector<long long> cnts(pts.size());
                    for(size_t j = 0; j < pts.size(); j++) cnts[j] = pts[j].cnt;
                    std::partial_sum(cnts.begin(), cnts.end(), treeBound[i].begin() + 1);
                }
            }
            fprintf(stderr, "[AJB_BP][Index] output line 1\n"); // was: cout
            for(size_t i = 0; i < varPos.size(); i++) {
                cout << "Relation " << i << ": ";
                for(size_t j = 0; j < varPos[i].size(); j++) {
                    cout << varPos[i][j] << ", ";
                }
                fprintf(stdout, "\n");
            }
            fprintf(stdout, "output\n"); // AJB-algo: buffered I/O
            for(size_t i = 0; i < mask.size(); i++) {
                cout << "Variable " << i << ": ";
                for(size_t j = 0; j < mask[i].size(); j++) {
                    cout << mask[i][j] << ", ";
                }
                fprintf(stdout, "\n");
            }
            // fprintf(stdout, "output\n"); // AJB-algo: buffered I/O
            
            setAGMandIters(FB);
            q.print();
            fprintf(stdout, "TreeUpperBound: %lld\n", treeUpp(FB.iters, jt.countRels[FB.splitDim]));
            jt.print();
            fprintf(stdout, "CountRels:\n");
            for(size_t i = 0; i < jt.countRels.size(); i++) {
                fprintf(stdout, "SplitDim=%zu: ", i);
                for(size_t j = 0; j < jt.countRels[i].size(); j++) {
                    fprintf(stdout, "R[%d], ", jt.countRels[i][j]);
                }
                fprintf(stdout, "\n");

            }
        }

        // AJB: treeUpp with overflow-safe乘法 — 提前检测乘积溢出
        // upstream的连乘如果中间结果>2^63会静默溢出给出错误的AGM bound
        long long treeUpp(const vector<pair<int, int> > &bound, const vector<int> &pos, const vector<int> &countRels) {
            long long res = 1;
            for(size_t i : countRels) {
                long long factor = treeBound[i][bound[i].first + pos[i]] - treeBound[i][bound[i].first];
                if(factor == 0) return 0;  // early-exit: 任一因子为0则乘积为0
                // overflow guard: 如果res * factor会溢出, 截断到LLONG_MAX
                if(res > 0 && factor > LLONG_MAX / res) return LLONG_MAX;
                res *= factor;
            }
            return res;
        }

        long long treeUpp(const vector<pair<int, int> > &bound, const vector<int> &countRels) {
            long long res = 1;
            for(size_t i : countRels) {
                long long factor = treeBound[i][bound[i].second] - treeBound[i][bound[i].first];
                if(factor == 0) return 0;
                if(res > 0 && factor > LLONG_MAX / res) return LLONG_MAX;
                res *= factor;
            }
            return res;
        }

        vector<CountOracle<int>* > getCountOracles() {
            vector<CountOracle<int>* > CO;
            CO.reserve(tables.size());  // AJB: 预分配, 避免push_back扩容
            for(size_t i = 0; i < tables.size(); i++) {
                CO.push_back(&tables[i].rt);
            }
            return CO;
        }

        long long AGM() {
            return FB.AGM;
            ajb_idx_stats.agm_calls++;
        }

        Bucket getFullBucket() {
            return FB;
        }

        void setAGM(Bucket &B) {
            // cout << "SET AGM of: ";
            ajb_idx_stats.set_agm_calls++;
            // B.print();
            // B.printIters(beginIters);
            // vector<int> cardinalities(B.iters.size());
            for(size_t i = 0; i < cardinalities.size(); i++) {
                cardinalities[i] = B.iters[i].second - B.iters[i].first;
            }
            double ans = q.AGM(cardinalities);
            B.AGM = agm_upper(ans);
            if(treeflag && B.splitDim < jt.countRels.size())B.AGM = min(B.AGM, treeUpp(B.iters, jt.countRels[B.splitDim]));
            // fprintf(stdout, "output\n"); // AJB-algo: buffered I/O
            // B.AGM = min(B.AGM, jt.treeUpp(B.splitDim, B.iters));
            return;
        }

        
        void setAGMandIters(Bucket &B, const vector<pair<vector<Point<int> >::iterator, vector<Point<int> >::iterator> >& iters = {}) {
            int relnum = R.size();
            auto ajb_agm_t0 = std::chrono::steady_clock::now();
            if((int)B.iters.size() != relnum) B.iters.resize(relnum);
            vector<int> cardinalities(relnum, 0);
            // AJB: 预分配bound向量, 在循环中resize+填充, 避免每次new
            vector<int> lower_bound, upper_bound;
            size_t max_arity = 0;
            for(int i = 0; i < relnum; i++) max_arity = std::max(max_arity, R[i].size());
            lower_bound.reserve(max_arity);
            upper_bound.reserve(max_arity);

            long long card_sum = 0;
            int card_max = 0, card_min = INT_MAX;
            for(int i = 0; i < relnum; i++) {
                size_t arity = R[i].size();
                lower_bound.resize(arity);
                upper_bound.resize(arity);
                for(size_t j = 0; j < arity; j++) {
                    lower_bound[j] = B.lowerBound[R[i][j]];
                    upper_bound[j] = B.upperBound[R[i][j]];
                }

                if(iters.size() > 0) B.iters[i] = tables[i].rt.getRange(lower_bound, upper_bound, iters[i].first, iters[i].second);
                else B.iters[i] = tables[i].rt.getRange(lower_bound, upper_bound);
                cardinalities[i] = B.iters[i].second - B.iters[i].first;
                card_sum += cardinalities[i];
                card_max = std::max(card_max, cardinalities[i]);
                if(cardinalities[i] < card_min) card_min = cardinalities[i];
            }
            double ans = q.AGM(cardinalities);
            B.AGM = agm_upper(ans);
            if(treeflag && B.splitDim < (int)jt.countRels.size())
                B.AGM = min(B.AGM, treeUpp(B.iters, jt.countRels[B.splitDim]));
            auto ajb_agm_t1 = std::chrono::steady_clock::now();
            ajb_idx_stats.agm_total_ms += std::chrono::duration<double, std::milli>(ajb_agm_t1 - ajb_agm_t0).count();
            // [AJB_BP] cardinality偏斜检测: 如果max/min > 100x, 说明数据高度不均衡
            // 这通常导致split效率低下, 是性能瓶颈的信号
            ajb_idx_stats.set_agm_calls++;
            if(ajb_idx_stats.set_agm_calls % 5000 == 0 && ajb_idx_stats.set_agm_calls > 0) {
                double skew = (card_min > 0) ? (double)card_max / card_min : -1.0;
                double avg = (relnum > 0) ? (double)card_sum / relnum : 0.0;
                fprintf(stderr, "[AJB_BP][Index] setAGM[%lld]: AGM=%lld avg_card=%.0f skew=%.1f splitDim=%d\n",
                        ajb_idx_stats.set_agm_calls, B.AGM, avg, skew, B.splitDim);
            }
            return;
        }

        // int AGMforBucket(Bucket B) {
            
        //     // auto startAGM = chrono::high_resolution_clock::now();
        //     // cntAGMCall++;
        //     int relnum = R.size();
        //     vector<int> cardinalities(relnum, 0);
        //     // vector<pair<vector<int>, vector<int> > > bounds;
        //     vector<int> lower_bound = {};
        //     vector<int> upper_bound = {};
        //     for(size_t i = 0; i < relnum; i++) {
        //         // auto startCountOracle = chrono::high_resolution_clock::now();
        //         lower_bound = vector<int>(R[i].size(), 0);
        //         upper_bound = vector<int>(R[i].size(), 0);
        //         for(size_t j = 0; j < R[i].size(); j++) {
        //             lower_bound[j] = B.lowerBound[R[i][j]];
        //             upper_bound[j] = B.upperBound[R[i][j]];
        //         }
        //         // auto endCountOracle = chrono::high_resolution_clock::now();
        //         // chrono::duration<double> elapsedCountOracle = endCountOracle - startCountOracle;
        //         // totalBoundPrepareTime += elapsedCountOracle.count();
        //         // bounds.push_back({lower_bound, upper_bound});
        //         // startCountOracle = chrono::high_resolution_clock::now();
        //         cardinalities[i] = tables[i].count(lower_bound, upper_bound);
        //         // endCountOracle = chrono::high_resolution_clock::now();
        //         // elapsedCountOracle = endCountOracle - startCountOracle;
        //         // totalCountOracleTime += elapsedCountOracle.count();
        //     }
            
        //     double ans = q.AGM(cardinalities);
        //     // auto endAGM = chrono::high_resolution_clock::now();
        //     // chrono::duration<double> elapsedAGM = endAGM - startAGM;
        //     // totalAGMTime += elapsedAGM.count();
        //     // ans = min(ans, (double)jt.treeUpp(B.getSplitDim(), bounds));
        //     return ceil(ans)-ans < 1e-5 ? ceil(ans) : int(ans);
        // }

        // vector<pair<Bucket, int> > split(Bucket B, int AGM = -1){
        //     cntSplitCall++;
        //     if(AGM < 0)AGM = AGMforBucket(B);
        //     int splitDim = B.getSplitDim();
        //     // int AGM = AGMforBucket(B);
        //     if(AGM == 0)return {};
            
        //     long long l = B.getLowerBound()[splitDim], r = B.getUpperBound()[splitDim], mid;
        //     // cout <<"OL: " << l << " OR: " << r << endl;
        //     int splitPos = l;
        //     while(l <= r){
        //         mid = (l + r) >> 1;
        //         Bucket Bleft = B.replace(B.getLowerBound()[splitDim], mid - 1);
        //         int AGMleft = AGMforBucket(Bleft);
        //         // cout <<"QQQ " << l <<", " << r << ", " << mid << endl;
        //         // Bleft.print();
        //         // fprintf(stdout, "output\n"); // AJB-algo: buffered I/O
        //         if(AGMleft <= (AGM >> 1))splitPos = mid, l = mid + 1;
        //         else r = mid - 1;
        //     }
            
        //     vector<pair<Bucket, int> > result = {};
            
        //     Bucket Bleft = B.replace(B.getLowerBound()[splitDim], splitPos - 1);
        //     int AGMleft = AGMforBucket(Bleft);
        //     if(splitPos - 1 >= B.getLowerBound()[splitDim] && AGMleft > 0)result.push_back(make_pair(Bleft, AGMleft));
            
        //     Bucket Bmid = B.replace(splitPos, splitPos);
        //     // cout <<"MIDBUCKET:";
        //     // Bmid.print();
        //     // cout <<"DIM AND DIM "<< splitDim << " " << Bmid.getDim() << endl;
            
        //     // int AGMmid = AGMforBucket(Bmid);
        //     // if(splitDim == B.getDim() - 1 || AGMmid <= AGM >> 1){
        //     //     if(AGMmid > 0)result.push_back(make_pair(Bmid, AGMforBucket(Bmid)));
        //     // }
        //     // else{
        //     //     vector<pair<Bucket, int> > temp = split(Bmid);
        //     //     result.insert(result.end(), temp.begin(), temp.end());
        //     // }

        //     // 
        //     if(splitDim == B.getDim() - 1){
        //         int AGMmid = AGMforBucket(Bmid);
        //         if(AGMmid > 0)result.push_back(make_pair(Bmid, AGMforBucket(Bmid)));
        //     }
        //     else{
        //         vector<pair<Bucket, int> > temp = split(Bmid);
        //         result.insert(result.end(), temp.begin(), temp.end());
        //     }

        //     Bucket Bright = B.replace(splitPos + 1, B.getUpperBound()[splitDim]);
        //     int AGMright = AGMforBucket(Bright);
        //     if(splitPos + 1 <= B.getUpperBound()[splitDim] && AGMright > 0)result.push_back(make_pair(Bright, AGMright));
        //     return result;
        // }

        vector<Bucket> splitBucket_BS(Bucket &B){
            
            // cout << "SPLITTING: ";
            // B.print();
            cntSplitCall++;
            // auto startSplit = chrono::high_resolution_clock::now();
            if(B.AGM < 0) setAGMandIters(B);
            if(B.AGM == 0)return {};
            int splitDim = B.getSplitDim();
            
            long long l = B.lowerBound[splitDim], r = B.upperBound[splitDim], mid;
            int splitPos = l, x;
            long long AGMleft;
            double ans;
            vector<int> cardinalities(B.iters.size(), 0);
            for(size_t i = 0; i < cardinalities.size(); i++) {
                cardinalities[i] = B.iters[i].second - B.iters[i].first;
            }
            // vector<bool> flag(cardinalities.size(), false);
            vector<int> rels = q.getRels(splitDim);
            vector<int> splitVarinRels(rels.size());
            vector<vector<int> > BleftUpperBounds(rels.size()), BmidUpperBounds(rels.size());
            for(size_t i = 0; i < rels.size(); i++) {
                // flag[rels[i]] = true;
                BleftUpperBounds[i] = vector<int>(R[rels[i]].size());
                BmidUpperBounds[i] = vector<int>(R[rels[i]].size());
                for(size_t j = 0; j < R[rels[i]].size(); j++) {
                    BleftUpperBounds[i][j] = B.upperBound[R[rels[i]][j]];
                    BmidUpperBounds[i][j] = B.upperBound[R[rels[i]][j]];
                    if(R[rels[i]][j] == splitDim) splitVarinRels[i] = j;
                }
            }
            // fprintf(stdout, "output\n"); // AJB-algo: buffered I/O

            // AJB算法改写: 二分搜索使用方向性缩窄
            // upstream每次迭代都从B.iters[x].first到B.iters[x].second全范围搜索
            // 改为: 维护每个relation的上次搜索结果作为下一轮的range hint
            // 当mid增大时, 新的upper bound ≥ 上次结果, 从上次结果开始搜索
            // 当mid减小时, 新的upper bound ≤ 上次结果, 用上次结果作为上界
            vector<int> lastIterResult(rels.size(), -1);  // 上一轮每个rel的搜索结果
            long long lastMid = -1;

            while(l <= r){
                cntBSCall++;
                mid = (l + r) >> 1;
                for(size_t i = 0; i < rels.size(); i++) {
                    x = rels[i];
                    BleftUpperBounds[i][splitVarinRels[i]] = mid - 1;

                    int searchFrom = B.iters[x].first;
                    int searchTo = B.iters[x].second;

                    // 方向性缩窄: 如果有上一轮结果
                    if (lastIterResult[i] >= 0 && lastMid >= 0) {
                        if (mid > lastMid) {
                            // mid增大 → upper bound只会增大, 从上次结果开始
                            searchFrom = lastIterResult[i];
                        } else if (mid < lastMid) {
                            // mid减小 → upper bound只会减小, 用上次结果作上界
                            searchTo = lastIterResult[i];
                        }
                    }

                    int iterRes = tables[x].rt.getUpperBoundIter(
                        BleftUpperBounds[i], searchFrom, searchTo);
                    cardinalities[x] = iterRes - B.iters[x].first;
                    lastIterResult[i] = iterRes;
                }
                lastMid = mid;
                ans = q.AGM(cardinalities);
                AGMleft = agm_upper(ans);
                if(AGMleft <= (B.AGM >> 1))splitPos = mid, l = mid + 1;
                else r = mid - 1;
            }
            // AJB: splitBucket_BS结果预分配——最多3个子bucket
            vector<Bucket> result;
            result.reserve(3);
            
            Bucket Bleft = B, Bmid = B, Bright = B;
            Bleft.upperBound[splitDim] = splitPos - 1;
            Bmid.lowerBound[splitDim] = splitPos;
            Bmid.upperBound[splitDim] = splitPos;
            Bright.lowerBound[splitDim] = splitPos + 1;
            Bleft.updateSplitDim();
            Bmid.updateSplitDim();
            Bright.updateSplitDim();
            // fprintf(stdout, "output\n"); // AJB-algo: buffered I/O
            int leftIter, rightIter;
            for(size_t i = 0; i < rels.size(); i++) {
                x = rels[i];
                BleftUpperBounds[i][splitVarinRels[i]] = splitPos - 1;
                BmidUpperBounds[i][splitVarinRels[i]] = splitPos;

                leftIter = tables[x].rt.getUpperBoundIter(BleftUpperBounds[i], B.iters[x].first, B.iters[x].second);
                rightIter = tables[x].rt.getUpperBoundIter(BmidUpperBounds[i], B.iters[x].first, B.iters[x].second);

                Bleft.iters[x] = make_pair(B.iters[x].first, leftIter);
                Bmid.iters[x] = make_pair(leftIter, rightIter);
                Bright.iters[x] = make_pair(rightIter, B.iters[x].second);
            }

            // fprintf(stdout, "output\n"); // AJB-algo: buffered I/O
            // Bleft.print();
            // Bmid.print();
            // Bright.print();
            setAGM(Bleft);
            setAGM(Bmid);
            setAGM(Bright);

            
            // fprintf(stdout, "output\n"); // AJB-algo: buffered I/O

            
            if(Bmid.AGM > 0 && splitDim < B.getDim() - 1) {
                // vector<Bucket> temp = splitBucket(Bmid);
                // result.insert(result.end(), temp.begin(), temp.end());
                result = splitBucket_BS(Bmid);
            }
            else if(Bmid.AGM > 0)result.push_back(Bmid);
            // if(splitDim == B.getDim() - 1) {
            //     if(Bmid.AGM > 0)result.push_back(Bmid);
            // }

            if(splitPos - 1 >= B.getLowerBound()[splitDim] && Bleft.AGM > 0)result.push_back(Bleft);

            if(splitPos + 1 <= B.getUpperBound()[splitDim] && Bright.AGM > 0)result.push_back(Bright);
            // auto endSplit = chrono::high_resolution_clock::now();
            // chrono::duration<double> elapsedSplit = endSplit - startSplit;
            // totalSplitTime += elapsedSplit.count();
            // cout << "DONE: ";
            // B.print();
            return result;
        }

        vector<Bucket> splitBucket(Bucket &B){
            
            ajb_idx_stats.split_calls++;
            auto ajb_split_t0 = std::chrono::steady_clock::now();
            // auto startSplit = chrono::high_resolution_clock::now();
            cntSplitCall++;
            if(B.AGM < 0) setAGMandIters(B);
            if(B.AGM == 0) return {};
            int splitDim = B.getSplitDim();
            
            int splitPos, x;
            double ans;
            // vector<int> rels = q.getRels(splitDim);
            // vector<int> cardinalities(B.iters.size(), 0);
            for(size_t i = 0; i < B.iters.size(); i++) {
                // AJB: 缓存splitDim列引用避免重复三层下标
                auto& sdcol = data[i][varPos[i][splitDim]];
                vecIters[i].first = sdcol.begin() + B.iters[i].first;
                vecIters[i].second = sdcol.begin() + B.iters[i].second;
                // cardinalities[i] = B.iters[i].second - B.iters[i].first;
            }
            // long long BAGM = q.AGM(cardinalities);
            // BAGM = ceil(BAGM)-BAGM < 1e-5 ? ceil(BAGM) : (long long)(BAGM);
            // for(size_t i = 0; i < B.iters.size(); i++) {
            //     cout << "[" << B.iters[i].first << ", " << B.iters[i].second << "] ";
            // }
            // cout << endl;
            splitPos = MultiHeadBinarySearch(B.iters, splitDim, B.AGM >> 1);
            // AJB: 预分配result容量——最多3个子bucket(left/mid/right)
            vector<Bucket> result;
            result.reserve(3);
            
            Bucket Bleft = B, Bmid = B, Bright = B;
            Bleft.upperBound[splitDim] = splitPos - 1;
            Bmid.lowerBound[splitDim] = splitPos;
            Bmid.upperBound[splitDim] = splitPos;
            Bright.lowerBound[splitDim] = splitPos + 1;
            Bleft.updateSplitDim();
            Bmid.updateSplitDim();
            Bright.updateSplitDim();
            int leftIter, rightIter;
            
            // AJB-algo: guard empty rels before loop
        if (rels[splitDim].empty()) { /* no split needed */ }
        for(size_t i = 0; i < rels[splitDim].size(); i++) {
                
                x = rels[splitDim][i];
                leftIter = B.iters[x].first + (lower_bound(vecIters[x].first, vecIters[x].second, splitPos) - vecIters[x].first);
                rightIter = B.iters[x].first + (upper_bound(vecIters[x].first, vecIters[x].second, splitPos) - vecIters[x].first);

                Bleft.iters[x].second = leftIter;
                Bmid.iters[x] = make_pair(leftIter, rightIter);
                Bright.iters[x].first = rightIter;
            }

            setAGM(Bleft);
            setAGM(Bmid);
            setAGM(Bright);
            // auto endSplit = chrono::high_resolution_clock::now();
            // chrono::duration<double> elapsedSplit = endSplit - startSplit;
            // totalSplitTime -= elapsedSplit.count();
            
            if(Bmid.AGM > 0 && splitDim < B.getDim() - 1) {
                result = splitBucket(Bmid);
            }
            else if(Bmid.AGM > 0) result.push_back(move(Bmid));

            if(splitPos + 1 <= B.upperBound[splitDim] && Bright.AGM > 0) result.push_back(move(Bright));
            
            if(splitPos - 1 >= B.lowerBound[splitDim] && Bleft.AGM > 0) result.push_back(move(Bleft));
            
            // [AJB] splitBucket timing + children tracking
            auto ajb_split_t1 = std::chrono::steady_clock::now();
            ajb_idx_stats.split_total_ms += std::chrono::duration<double, std::milli>(ajb_split_t1 - ajb_split_t0).count();
            if((int)result.size() > ajb_idx_stats.max_split_children)
                ajb_idx_stats.max_split_children = result.size();
            // [AJB_TRACE] periodic split summary
            if(ajb_idx_stats.split_calls % 2000 == 0 && ajb_idx_stats.split_calls > 0) {
                fprintf(stderr, "[AJB_TRACE][Index] splitBucket[%lld]: dim=%d pos=%d children=%zu AGM=%lld→[",
                        ajb_idx_stats.split_calls, splitDim, splitPos, result.size(), B.AGM);
                for(size_t ri = 0; ri < result.size() && ri < 5; ri++) {
                    if(ri) fprintf(stderr, ",");
                    fprintf(stderr, "%lld", result[ri].AGM);
                }
                fprintf(stderr, "]\n");
            }
            return result;
        }


        vector<int> Split_pool(BucketPool &pool, int bid) {
            // cout << "SPLIT: ";
            // pool[bid].print();
            vector<Bucket> result = splitBucket(pool[bid]);
            // cout << "INTO " << result.size() << " BUCKETS" << endl;
            // for(int i = 0; i < result.size(); i++) {
            //     pool[result[i]].print();
            // }
            // cout << "----------------" << endl;
            while(result.size() == 1 && result[0].splitDim != result[0].getDim()){
                result = splitBucket(result[0]);
            }
            // return result;
            vector<int> Bid(result.size());
        std::iota(Bid.begin(), Bid.end(), 0);  // AJB-algo: sequential init via iota
            for(size_t i = 0; i < result.size(); i++) {
                Bid[i] = pool.newCopy(result[i]);
                pool[Bid[i]].AGM = result[i].AGM;
                if(pool[Bid[i]].iters.size() != result[i].iters.size()) pool[Bid[i]].iters = vector<pair<int, int> >(move(result[i].iters));
                else for(int j = 0; j < result[i].iters.size(); j++){
                    pool[Bid[i]].iters[j].first = result[i].iters[j].first;
                    pool[Bid[i]].iters[j].second = result[i].iters[j].second;
                }
            }
            return Bid;
        }

        vector<Bucket> Split(Bucket &B) {
            vector<Bucket> result = splitBucket(B);
            // [AJB_TRACE] Split: if single-child re-splitting, track depth
            int resplit_rounds = 0;
            while(result.size() == 1 && result[0].splitDim != result[0].getDim()){
                result = splitBucket(result[0]);
                resplit_rounds++;
            }
            if(resplit_rounds > 0) {
                fprintf(stderr, "[AJB_TRACE][Index] Split re-split %d rounds → %zu final children\n",
                        resplit_rounds, result.size());
            }
            return result;
        }


        // AJB: sample() for skew probing — uses existing Split()+setAGM()
        vector<int> sample(Bucket B, long long agm = -1){
            if(agm < 0) { setAGM(B); agm = B.AGM; }
            if(agm == 0) return {};
            if(B.getSplitDim() == B.getDim()) return B.getLowerBound();
            vector<Bucket> sons = Split(B);
            if(sons.empty()) return {};
            // AJB: 预分配child_agms + 用reserve减少sons扩容
            vector<long long> child_agms;
            child_agms.reserve(sons.size());
            long long total = 0;
            for(size_t i = 0; i < sons.size(); i++){
                setAGM(sons[i]);
                child_agms.push_back(sons[i].AGM);
                total += sons[i].AGM;
            }
            if(total == 0) return {};
            // AJB: 使用类成员gen代替每次重新构造mt19937
            // upstream: 每次sample()都 mt19937(random_device{}()) — 极慢
            std::uniform_int_distribution<long long> distr(1, total);
            long long p = distr(gen);
            for(size_t i = 0; i < sons.size(); i++){
                if(p <= child_agms[i]) return sample(sons[i], child_agms[i]);
                p -= child_agms[i];
            }
            return {};
        }

        // AJB: sampleUntilSuccess()
        vector<int> sampleUntilSuccess(){
            vector<int> s = {};
            int tries = 0;
            while(s.empty() && tries < 1000) { s = this->sample(getFullBucket()); tries++; }
            if(tries > 1)
                fprintf(stderr, "[AJB_WARN][Index] sampleUntilSuccess took %d tries\n", tries);
            return s;
        }

        // AJB: randomAccess() for REnum compatibility
        pair<bool, vector<int> > randomAccess(Bucket B, long long k, long long agm = -1){
            if(agm < 0) { setAGM(B); agm = B.AGM; }
            if(B.getSplitDim() == B.getDim()) return make_pair(true, B.getLowerBound());
            vector<Bucket> sons = Split(B);
            for(size_t i = 0; i < sons.size(); i++){
                setAGM(sons[i]);
                if(k <= sons[i].AGM) return randomAccess(sons[i], k, sons[i].AGM);
                k -= sons[i].AGM;
            }
            // AJB: return {agm_remaining} so caller can estimate empty range
            return make_pair(false, vector<int>{(int)(agm - k)});
        }

        ///////////////////////////// TBD: BETTER TRIVAL INTERVAL
        // int getEmptyRight(Bucket B, int AGM = -1){
        //     // B.print();
        //     if(AGM < 0)AGM = AGMforBucket(B);
        //     if(B.getSplitDim() == B.getDim()){
        //         return 1 - AGM;
        //     }
        //     cntTotalCall++;
        //     auto startCacheHit = chrono::high_resolution_clock::now();
        //     bool flag = bucketSplitCache.find(B) == bucketSplitCache.end();
        //     auto endCacheHit = chrono::high_resolution_clock::now();
        //     chrono::duration<double> elapsedCacheHit = endCacheHit - startCacheHit;
        //     totalCacheHitTime += elapsedCacheHit.count();
        //     if(flag){
        //         auto start = chrono::high_resolution_clock::now();
        //         vector<pair<Bucket, int> > result = split(B);
        //         auto end = chrono::high_resolution_clock::now();
        //         chrono::duration<double> elapsed = end - start;
        //         totalSplitTime += elapsed.count();
        //         startCacheHit = chrono::high_resolution_clock::now();
        //         bucketSplitCache[B] = result;
        //         endCacheHit = chrono::high_resolution_clock::now();
        //         elapsedCacheHit = endCacheHit - startCacheHit;
        //         totalCacheHitTime += elapsedCacheHit.count();
        //         // cout << "cache success" << endl;
        //     }
        //     else cntCacheHit++;
        //     startCacheHit = chrono::high_resolution_clock::now();
        //     vector<pair<Bucket, int> > sons = bucketSplitCache[B];
        //     endCacheHit = chrono::high_resolution_clock::now();
        //     elapsedCacheHit = endCacheHit - startCacheHit;
        //     totalCacheHitTime += elapsedCacheHit.count();
        //     // vector<pair<Bucket, int> > sons = split(B);
        //     int temp = 0;
        //     for(size_t i = 0; i < sons.size(); i++){
        //         // cout << "SON::" << i <<": ";
        //         // sons[i].first.print();
        //         temp += sons[i].second;
        //     }
        //     // cout <<"RE NOT HERE0" << endl;
        //     int emptyright = sons.size() > 0 ? getEmptyRight(sons[sons.size() - 1].first, sons[sons.size() - 1].second) : 0;
            
        //     // cout <<"RE NOT HERE1" << endl;
        //     return AGM - temp + emptyright;
        // }

        // pair<bool, vector<int> > randomAccess_opt(Bucket B, int k, int offset = 0, int AGM = -1){
        //     if(AGM < 0)AGM = AGMforBucket(B);
        //     cntTotalCall++;
        //     // B.print();
        //     // cout << "ThisBucketInterval: " << offset + 1 << " " << offset + AGM << endl;
        //     if(B.getSplitDim() == B.getDim())return make_pair(true, B.getLowerBound());
            
        //     auto startCacheHit = chrono::high_resolution_clock::now();
        //     bool flag = bucketSplitCache.find(B) == bucketSplitCache.end();
        //     auto endCacheHit = chrono::high_resolution_clock::now();
        //     chrono::duration<double> elapsedCacheHit = endCacheHit - startCacheHit;
        //     totalCacheHitTime += elapsedCacheHit.count();
        //     if(flag){
        //         auto start = chrono::high_resolution_clock::now();
        //         vector<pair<Bucket, int> > result = split(B);
        //         auto end = chrono::high_resolution_clock::now();
        //         chrono::duration<double> elapsed = end - start;
        //         totalSplitTime += elapsed.count();
        //         // B.print();
        //         // if(B.getLowerBound().size() != 3 || B.getUpperBound().size() != 3) cout << "ERROR: " << B.getLowerBound().size() << ", " << B.getUpperBound().size() << endl;
        //         // cout << "-----------------------v" << endl;
        //         // for (const auto& son : result) {
        //         //     son.first.print();
        //         //     cout << "AGM: " << son.second << endl;
        //         // }
                
        //         // cout << "-----------------------^" << endl;
        //         startCacheHit = chrono::high_resolution_clock::now();
        //         bucketSplitCache[B] = result;
        //         endCacheHit = chrono::high_resolution_clock::now();
        //         elapsedCacheHit = endCacheHit - startCacheHit;
        //         totalCacheHitTime += elapsedCacheHit.count();
        //         // cout << "cache success" << endl;
        //     }
        //     else cntCacheHit++;
        //     startCacheHit = chrono::high_resolution_clock::now();
        //     vector<pair<Bucket, int> > sons = bucketSplitCache[B];
        //     endCacheHit = chrono::high_resolution_clock::now();
        //     elapsedCacheHit = endCacheHit - startCacheHit;
        //     totalCacheHitTime += elapsedCacheHit.count();
        //     // vector<pair<Bucket, int> > sons = split(B);
        //     int temp = 0;
        //     for(size_t i = 0; i < sons.size(); i++){
        //         pair<Bucket, int> son = sons[i];
        //         if(k - offset - temp <= son.second){
        //             pair<bool, vector<int> > res = randomAccess_opt(son.first, k, offset + temp, son.second);
        //             // cout <<res.first << " " << res.second[0] << " " << res.second[1] << endl;
        //             // cout <<"THISEMPTY: "<< offset + temp  +son.second << ", " << offset + AGM << endl;
        //             if(res.first || i < sons.size() - 1 || res.second[1] < offset + temp + son.second) return res;
        //             // cout << "->(" << res.second[0] << ", " << offset + AGM << ")" << endl;
        //             return make_pair(false, vector<int> {res.second[0], offset + AGM});
        //         }
        //         // k -= son.second;
        //         temp += son.second;
        //     }
        //     // return make_pair(false, vector<int> {offset + temp + 1, offset + AGM});
        //     ///////////////////////////// TBD: BETTER TRIVAL INTERVAL
        //     int emptyright = sons.size() > 0 ? getEmptyRight(sons[sons.size() - 1].first, sons[sons.size() - 1].second) : 0;
        //     // cout << offset + temp + 1 << ", " << offset + AGM << endl;
        //     // cout << offset + temp - emptyright + 1 << ", " << offset + AGM << endl;
        //     return make_pair(false, vector<int> {offset + temp - emptyright + 1, offset + AGM});
        // }

        // pair<bool, vector<int> > randomAccess(Bucket B, int k, int offset = 0, int AGM = -1){
        //     if(AGM < 0)AGM = AGMforBucket(B);
        //     // cout << "BucketInterval: " << offset + 1 << " " << offset + AGM << endl;
        //     if(B.getSplitDim() == B.getDim())return make_pair(true, B.getLowerBound());
        //     vector<pair<Bucket, int> > sons = split(B);
        //     int temp = 0;
        //     for(size_t i = 0; i < sons.size(); i++){
        //         pair<Bucket, int> son = sons[i];
        //         if(k - offset - temp <= son.second)return randomAccess(son.first, k, offset + temp, son.second);
        //         // k -= son.second;
        //         temp += son.second;
        //     }
        //     return make_pair(false, vector<int> {offset + temp + 1, offset + AGM});
        // }

        // void printBucketInfo(Bucket B, int offset = 0, int AGM = -1){
        //     if(AGM < 0)AGM = AGMforBucket(B);
        //     cout << "------------------------------------" << endl;
        //     B.print();
        //     int relnum = q.getRelations().size();
        //     vector<int> cardinalities;
        //     for(size_t i = 0; i < relnum; i++) {
        //         vector<int> lower_bound = {};
        //         vector<int> upper_bound = {};
        //         for(size_t j = 0; j < q.getRelations()[i].size(); j++) {
        //             lower_bound.push_back(B.getLowerBound()[q.getRelations()[i][j]]);
        //             upper_bound.push_back(B.getUpperBound()[q.getRelations()[i][j]]);
        //         }
        //         cardinalities.push_back(tables[i].count(lower_bound, upper_bound));
        //     }
        //     for (int i = 0; i < cardinalities.size(); i++) {
        //         cout << "Cardinality of relation " << i << ": " << cardinalities[i] << endl;
        //     }
        //     cout << "AGM: " << AGM << endl;
        //     cout << "BucketInterval: [" << offset + 1 << " " << offset + AGM << "]" << endl;
        // }

        // void printBucketTree(Bucket B, int offset = 0, int AGM = -1){
        //     if(AGM < 0)AGM = AGMforBucket(B);
        //     if(B.getSplitDim() == B.getDim())return;
        //     vector<pair<Bucket, int> > sons = split(B);
        //     int tempoffset = offset;
        //     for(size_t i = 0; i < sons.size(); i++){
        //         pair<Bucket, int> son = sons[i];
        //         printBucketInfo(son.first, tempoffset, son.second);
        //         tempoffset += son.second;
        //     }
        //     for(size_t i = 0; i < sons.size(); i++){
        //         pair<Bucket, int> son = sons[i];
        //         printBucketTree(son.first, offset, son.second);
        //         offset += son.second;
        //     }
        //     return;
        // }

        // void enumeration(Bucket B, int AGM = -1){
        //     if(AGM < 0)AGM = AGMforBucket(B);
        //     if(B.getSplitDim() == B.getDim()){
        //         cout << "Res(";
        //         for(size_t i = 0; i < B.getDim() - 1; i++) {
        //             cout << B.getLowerBound()[i] << ",";
        //         }
        //         cout << B.getLowerBound()[B.getDim() - 1] << ")"<< endl;
        //         return;
        //     }
        //     vector<pair<Bucket, int> > sons = split(B);
        //     for(size_t i = 0; i < sons.size(); i++){
        //         pair<Bucket, int> son = sons[i];
        //         enumeration(son.first, son.second);
        //     }
        //     return;
        // }

        void print(){
            for(size_t i = 0; i < tables.size(); i++){
                fprintf(stdout, "Relation: %zu\n", i);
                tables[i].print();
            }
        }

        // [AJB] dump所有诊断计数
        void ajb_dump_stats() {
            ajb_idx_stats.dump();
            fprintf(stderr, "[AJB_STATE][Index] upstream_counters: cacheHit=%d total=%d agm=%d split=%d bs=%d nodes=%d\n",
                    cntCacheHit, cntTotalCall, cntAGMCall, cntSplitCall, cntBSCall, totalrrtreenode);
            fprintf(stderr, "[AJB_TIMER][Index] upstream_timers: agm=%.3fs co=%.3fs split=%.3fs cache=%.3fs bound=%.3fs\n",
                    totalAGMTime, totalCountOracleTime, totalSplitTime, totalCacheHitTime, totalBoundPrepareTime);
        }

        // [AJB] reset所有计数器
        void ajb_reset_stats() {
            ajb_idx_stats.reset();
            cntCacheHit = cntTotalCall = cntAGMCall = cntSplitCall = cntBSCall = totalrrtreenode = 0;
            totalAGMTime = totalCountOracleTime = totalSplitTime = totalCacheHitTime = totalBoundPrepareTime = 0;
        }
};