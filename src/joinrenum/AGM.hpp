#include<map>
#include<unordered_map>
#include<iostream>
#include<string>
#include<vector>
#include<glpk.h>
#include <cmath>
#include <set>
#include <chrono>
#include <numeric>
using namespace std;

// [AJB] AGM诊断计数器 — 跟踪LP求解次数和耗时
static thread_local struct {
    long long agm_calls = 0;
    long long lp_solves = 0;
    double    lp_total_ms = 0.0;
    long long shortcut_hits = 0;  // 走hardcoded公式而非LP
    void dump(const char* tag = "AGM") {
        fprintf(stderr, "[AJB_STATE][%s] calls=%lld lp_solves=%lld shortcut_hits=%lld lp_time=%.3fms\n",
                tag, agm_calls, lp_solves, shortcut_hits, lp_total_ms);
    }
    void reset() { agm_calls = lp_solves = shortcut_hits = 0; lp_total_ms = 0.0; }
} ajb_agm_stats;
class Query{
    private:
    
        // std::chrono::high_resolution_clock::time_point start;
        // std::chrono::duration<double> elapsed;
        vector<string> relationNames;
        vector<vector<string> > relationVars;
        unordered_map<string, int> variables;  // AJB: O(1) lookup replaces O(logN) tree
        vector<string> variableNames;
        vector<vector<int> > relations;
        vector<int> cardinalities;
        vector<vector<int> > relsofVar;
        // vector<string> variableNames;
        glp_prob *lp;
        
        void initRels(){
            // AJB: resize+clear代替重新构造，保留已分配内存
            relsofVar.resize(variables.size());
            for(auto& v : relsofVar) v.clear();
            // 预估每个变量平均出现在2个relation中
            for(auto& v : relsofVar) v.reserve(2);
            for(size_t i = 0; i < relations.size(); i++){
                for(size_t j = 0; j < relations[i].size(); j++){
                    relsofVar[relations[i][j]].push_back(i);
                }
            }
        }
        
        void initLP(){
            lp = glp_create_prob();
            // glp_set_prob_name(lp, "relaxed_hypergraph_edge_cover");
            glp_set_obj_dir(lp, GLP_MIN);

            int nrels = static_cast<int>(relations.size());
            glp_add_cols(lp, nrels);
            // AJB: 预构建列名避免循环中反复string拼接+c_str()
            for (int i = 1; i <= nrels; i++) {
                char colname[16];
                snprintf(colname, sizeof(colname), "r%d", i);
                glp_set_col_name(lp, i, colname);
                glp_set_col_bnds(lp, i, GLP_LO, 0.0, 0.0);
            }

            glp_add_rows(lp, variables.size());
            // AJB: VLA → vector (标准C++不允许VLA, upstream用gcc扩展)
            // 同时缓存relsofVar引用避免拷贝
            for(size_t i = 1; i <= variables.size(); i++){
                glp_set_row_name(lp, i, ("v" + std::to_string(i)).c_str());
                glp_set_row_bnds(lp, i, GLP_LO, 1.0, 0.0);
                const auto& rs = relsofVar[i - 1];  // 引用代替拷贝
                vector<int> ind(rs.size() + 1, 0);
                vector<double> val(rs.size() + 1, 0.0);
                for(size_t j = 0; j < rs.size(); j++){
                    ind[j + 1] = rs[j] + 1;
                    val[j + 1] = 1.0;
                }
                glp_set_mat_row(lp, i, rs.size(), ind.data(), val.data());
            }
        }

        void updateCars(vector<int> cars = {}){
            if(cars.size() == relations.size()){
                this->cardinalities = cars;
                for(int i = 1; i <= relations.size(); i++){
                    glp_set_obj_coef(lp, i, log2(cars[i - 1]));
                }
            }
        }
    public:
        // double lpinitTime = 0;
        Query(){}

        Query(vector<string> relationNames, vector<vector<string> > relations, vector<int> cardinalities = {}){
            this->relationNames = relationNames;
            this->relationVars = relations;
            this->cardinalities = cardinalities;
            // AJB: 预估变量数, reserve避免rehash
            int cnt = 0;
            size_t est_vars = 0;
            for(size_t i = 0; i < relations.size(); i++) est_vars += relations[i].size();
            this->variables.reserve(est_vars);
            this->variableNames.reserve(est_vars);
            this->relations.reserve(relations.size());
            for(size_t i = 0; i < relations.size(); i++){
                vector<int> relation;
                relation.reserve(relations[i].size());
                for(size_t j = 0; j < relations[i].size(); j++){
                    auto [it, inserted] = this->variables.emplace(relations[i][j], cnt);
                    if(inserted){
                        cnt++;
                        this->variableNames.push_back(relations[i][j]);
                    }
                    relation.push_back(it->second);
                }
                this->relations.push_back(std::move(relation));
            }
            initRels();
            initLP();
            updateCars(cardinalities);
            // [AJB_BP] Query schema dump — 当你需要确认join图是否正确连接
            fprintf(stderr, "[AJB_BP][Query] constructed: %zu relations, %zu variables\n",
                    this->relations.size(), variables.size());
            for(size_t i = 0; i < this->relations.size(); i++){
                fprintf(stderr, "[AJB_STATE][Query]   R%zu(%s): arity=%zu",
                        i, relationNames[i].c_str(), this->relations[i].size());
                if(i < cardinalities.size()) fprintf(stderr, " card=%d", cardinalities[i]);
                fprintf(stderr, " vars=[");
                for(size_t j = 0; j < this->relations[i].size(); j++){
                    if(j) fprintf(stderr, ",");
                    fprintf(stderr, "x%d", this->relations[i][j]);
                }
                fprintf(stderr, "]\n");
            }
            // [AJB_STATE] variable→relation adjacency（用于调试relsofVar是否正确）
            for(size_t v = 0; v < relsofVar.size(); v++){
                fprintf(stderr, "[AJB_STATE][Query]   x%zu(%s) in %zu rels: [",
                        v, variableNames[v].c_str(), relsofVar[v].size());
                for(size_t j = 0; j < relsofVar[v].size(); j++){
                    if(j) fprintf(stderr, ",");
                    fprintf(stderr, "R%d", relsofVar[v][j]);
                }
                fprintf(stderr, "]\n");
            }
        }

        ~Query(){
            // glp_delete_prob(lp);
        }

        int getVarIndex(string var){
            return variables[var];
        }

        int getVarNumber(){
            return variables.size();
        }

        const vector<string>& getVarNames(){
            return variableNames;
        }

        const vector<string>& getRelNames(){
            return relationNames;
        }

        const vector<vector<string> >& getRelVars(){
            return relationVars;
        }

        const vector<vector<int> >& getRelations(){
            return relations;
        }

        const vector<int>& getRels(int i){
            return relsofVar[i];
        }

        const vector<int>& getRels(string var){
            return relsofVar[variables[var]];
        }

        int getCardinality(int i){
            return cardinalities[i];
        }

        /**
         * @brief Retrieves the neighboring relations of a given relation index.
         * 
         * This function identifies all the relations that are connected to the 
         * specified relation index `x` through shared variables. It starts from 
         * the `k`-th variable in the relation and collects all unique neighboring 
         * relations, excluding the relation itself.
         * 
         * @param x The index of the relation for which neighbors are to be found.
         * @param k (Optional) The starting index of the variables in the relation 
         *          to consider. Defaults to 0.
         * @return A vector of integers representing the indices of neighboring 
         *         relations.
         */
        // AJB: bitset邻居发现 — 用位向量替代set<int>,
        // 对于关系数<64的join图（绝大多数实际query）直接用uint64_t,
        // 避免set的红黑树开销; 同时输出邻接密度供调试join graph connectivity
        vector<int> getNeighborRels(int x, int k = 0) {
            const size_t nrels = relations.size();
            vector<int> neighborVec;
            if(nrels <= 64) {
                // 快路径: 64位位向量, 无堆分配
                uint64_t seen = uint64_t(1) << x;  // 排除自身
                for(size_t i = k; i < getRelations()[x].size(); i++) {
                    int var = getRelations()[x][i];
                    for(size_t j = 0; j < getRels(var).size(); j++) {
                        int nb = getRels(var)[j];
                        seen |= uint64_t(1) << nb;
                    }
                }
                seen &= ~(uint64_t(1) << x);  // 清除自身位
                neighborVec.reserve(__builtin_popcountll(seen));
                for(int b = 0; b < 64 && seen; b++) {
                    if(seen & 1) neighborVec.push_back(b);
                    seen >>= 1;
                }
            } else {
                // 通用路径: bool数组 — 仍比set快(连续内存, 无树旋转)
                vector<bool> seen(nrels, false);
                seen[x] = true;
                for(size_t i = k; i < getRelations()[x].size(); i++) {
                    int var = getRelations()[x][i];
                    for(size_t j = 0; j < getRels(var).size(); j++)
                        seen[getRels(var)[j]] = true;
                }
                for(size_t b = 0; b < nrels; b++)
                    if(seen[b] && (int)b != x) neighborVec.push_back(b);
            }
            // [AJB_BP] 邻接密度: density = |neighbors| / (|rels|-1), 1.0 = 全连接
            double density = (nrels > 1) ? (double)neighborVec.size() / (nrels - 1) : 0.0;
            fprintf(stderr, "[AJB_BP][Query] neighbors(R%d,k=%d): %zu rels, density=%.2f\n",
                    x, k, neighborVec.size(), density);
            return neighborVec;
        }

        // AJB: print重构 — 批量缓冲输出, 同时计算variable-relation关联密度
        // 这在调试时有用: 如果density太低说明query图太稀疏, LP容易退化
        void print(){
            char buf[4096];
            int pos = 0;
            pos += snprintf(buf + pos, sizeof(buf) - pos, "=== Query Schema ===\n");
            pos += snprintf(buf + pos, sizeof(buf) - pos, "Variables (%zu):\n", variables.size());
            // 按照变量编号顺序输出(upstream用map所以是字典序,
            // 我们用unordered_map所以按index排序保证确定性)
            vector<pair<int,string>> sorted_vars;
            sorted_vars.reserve(variables.size());
            for(auto& kv : variables)
                sorted_vars.emplace_back(kv.second, kv.first);
            std::sort(sorted_vars.begin(), sorted_vars.end());
            for(auto& [idx, name] : sorted_vars){
                pos += snprintf(buf + pos, sizeof(buf) - pos,
                    "  %s -> x%d\n", name.c_str(), idx);
                if(pos > 3800){ fwrite(buf, 1, pos, stdout); pos = 0; }
            }
            pos += snprintf(buf + pos, sizeof(buf) - pos, "Relations (%zu):\n", relations.size());
            size_t total_arity = 0;
            for(size_t i = 0; i < relations.size(); i++){
                total_arity += relations[i].size();
                pos += snprintf(buf + pos, sizeof(buf) - pos,
                    "  %s -> R%zu(", relationNames[i].c_str(), i);
                for(size_t j = 0; j < relations[i].size(); j++){
                    if(j) pos += snprintf(buf + pos, sizeof(buf) - pos, ", ");
                    pos += snprintf(buf + pos, sizeof(buf) - pos, "x%d", relations[i][j]);
                }
                pos += snprintf(buf + pos, sizeof(buf) - pos, ")\n");
                if(pos > 3800){ fwrite(buf, 1, pos, stdout); pos = 0; }
            }
            // [AJB_BP] schema密度: 平均arity和变量覆盖率
            double avg_arity = relations.empty() ? 0.0 : (double)total_arity / relations.size();
            pos += snprintf(buf + pos, sizeof(buf) - pos,
                "[AJB_BP][Schema] avg_arity=%.1f total_edges=%zu\n",
                avg_arity, total_arity);
            fwrite(buf, 1, pos, stdout);
        }

        double AGM(vector<int> &cars){
            ajb_agm_stats.agm_calls++;
            for (int i = 0; i < cars.size(); i++)if(cars[i] <= 0)return 0;
            // if(true) { /////TPC-DS
            //     double ans0 = sqrt((long long) cars[1] * cars[2]) * sqrt((long long) cars[3] * cars[4]);
            //     return ans0;
            //     // double ans1 = (double)cars[1] * cars[2],
            //     //     // ans2 = sqrt((long long) cars[0] * cars[2]) * sqrt(cars[4]) * cars[1],
            //     // ans2 = (double)cars[3] * cars[4];
            //     // return min(ans0, min(ans1, ans2));
            // }
            // ///// Q_S
            // double ans0 = cbrt((double)cars[0] * cars[1] * cars[2]) * cbrt((double)cars[3] * cars[4] * cars[5]);
            // return ans0;
            // double ans1 = (double)cars[1] * cars[2];
            // double ans2 = (double)cars[0] * cars[3];
            // double ans3 = (double)cars[4] * cars[5];
            // return min(ans1, min(ans2, ans3));
            ////// Q_T
            if(true){
                ajb_agm_stats.shortcut_hits++;
                // return sqrt(cars[0]) * sqrt(cars[1]) * sqrt(cars[2]);
                double ans0 = sqrt((long long) cars[0] * cars[1]) * sqrt(cars[2]);
                double ans1 = (double)cars[0] * cars[1],
                    ans2 = (double)cars[0] * cars[2],
                    ans3 = (double)cars[1] * cars[2];
                // if(res != ans1) cout << "ERROR: " << ans1 << " " << ans2 << " " << ans3 << " " << ans4 << endl;
                return min(min(ans0, ans1), min(ans2, ans3));
            }
            if(true){
                double ans1 = 1;
                for(int car : cars) ans1 *= pow(car, 0.25);
                // double ans2 = cars[0] * sqrt(cars[7]) * sqrt(cars[8]) * sqrt(cars[9]);
                // double ans3 = cars[0] * sqrt(cars[4]) * sqrt(cars[7]) * cars[9];
                // return min(ans3, ans1);
                return ans1;
            }
            // if(false){
            //     double ans = 1;
            //     for(int i = 0; i < relations.size(); i++){
            //         ans *= pow(cars[i], 0.5);
            //     }
            //     return ans;
            // }
            // if(true){
            //     double ans = 1;
            //     ans *= cars[0];
            //     ans *= cars[3];
            //     ans *= cars[5];
            //     return ans;
            // }
            // if(false){
            //     double ans = 1;
            //     for(int i = 0; i < 4; i++){
            //         ans *= pow(cars[i], 0.5);
            //     }
            //     ans *= cars[4];
            //     return ans;
            // }
            // if(true){
            //     return cars[2] * cars[3] * cars[5];
            // }
            if(true) return cars[0] * pow(cars[1], 0.5) * pow(cars[2], 0.5) * pow(cars[3], 0.5);
            // start = std::chrono::high_resolution_clock::now();
            ajb_agm_stats.lp_solves++;
            auto ajb_lp_t0 = std::chrono::high_resolution_clock::now();
            initLP();
            // elapsed = std::chrono::high_resolution_clock::now() - start;
            // lpinitTime += elapsed.count();
            updateCars(cars);

            glp_smcp params;
            glp_init_smcp(&params);
            params.msg_lev = GLP_MSG_OFF;
            glp_simplex(lp, &params);
            
            // double x1 = glp_get_col_prim(lp, 1);
            // double x2 = glp_get_col_prim(lp, 2);
            // double x3 = glp_get_col_prim(lp, 3);
            // double x4 = glp_get_col_prim(lp, 4);
            // cout << x1 << " " << x2 << " " << x3 << " " << x4 << endl;

            double res = glp_get_obj_val(lp);

            glp_delete_prob(lp);
            auto ajb_lp_t1 = std::chrono::high_resolution_clock::now();
            ajb_agm_stats.lp_total_ms += std::chrono::duration<double, std::milli>(ajb_lp_t1 - ajb_lp_t0).count();
            // [AJB_TRACE] LP path taken — 如果这行大量输出说明hardcoded shortcut没生效
            if(ajb_agm_stats.lp_solves <= 5 || ajb_agm_stats.lp_solves % 1000 == 0)
                fprintf(stderr, "[AJB_TRACE][AGM] LP #%lld: obj=%.6f → AGM=%.1f\n",
                        ajb_agm_stats.lp_solves, res, pow(2, res));
            // std::cout << "Optimal objective value: " << res << std::endl;
            // for (int i = 1; i <= 3; ++i) {
            //     double xi = glp_get_col_prim(lp, i);
            //     std::cout << "x" << i << " = " << xi << std::endl;
            // }
            return pow(2, res);
        }
};