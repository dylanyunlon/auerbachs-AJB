#include<map>
#include<iostream>
#include<string>
#include<vector>
#include<glpk.h>
#include <cmath>
using namespace std;
class Query{
    private:
    
        // std::chrono::high_resolution_clock::time_point start;
        // std::chrono::duration<double> elapsed;
        vector<string> relationNames;
        vector<vector<string> > relationVars;
        map<string, int> variables;
        vector<string> variableNames;
        vector<vector<int> > relations;
        vector<int> cardinalities;
        vector<vector<int> > relsofVar;
        // vector<string> variableNames;
        glp_prob *lp;
        
        void initRels(){
            relsofVar = vector<vector<int> >(variables.size(), vector<int>());
            for(int i = 0; i < relations.size(); i++){
                for(int j = 0; j < relations[i].size(); j++){
                    relsofVar[relations[i][j]].push_back(i);
                }
            }
        }
        
        void initLP(){
            lp = glp_create_prob();
            // glp_set_prob_name(lp, "relaxed_hypergraph_edge_cover");
            glp_set_obj_dir(lp, GLP_MIN);

            glp_add_cols(lp, relations.size());
            for (int i = 1; i <= relations.size(); i++) {
                glp_set_col_name(lp, i, ("r" + std::to_string(i)).c_str());
                glp_set_col_bnds(lp, i, GLP_LO, 0.0, 0.0);
            }

            glp_add_rows(lp, variables.size());
            for(int i = 1; i <= variables.size(); i++){
                glp_set_row_name(lp, i, ("v" + std::to_string(i)).c_str());
                glp_set_row_bnds(lp, i, GLP_LO, 1.0, 0.0);
                vector<int> rs = relsofVar[i - 1];
                int ind[rs.size() + 1] = {0};
                double val[rs.size() + 1] = {0};
                for(int j = 0; j < rs.size(); j++){
                    ind[j + 1] = rs[j] + 1;
                    val[j + 1] = 1.0;
                }
                glp_set_mat_row(lp, i, rs.size(), ind, val);
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
            int cnt = 0;
            for(int i = 0; i < relations.size(); i++){
                vector<int> relation;
                for(int j = 0; j < relations[i].size(); j++){
                    if(this->variables.find(relations[i][j]) == this->variables.end()){
                        this->variables[relations[i][j]] = cnt++;
                        this->variableNames.push_back(relations[i][j]);
                    }
                    relation.push_back(this->variables[relations[i][j]]);
                }
                this->relations.push_back(relation);
            }
            initRels();
            initLP();
            updateCars(cardinalities);
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
        vector<int> getNeighborRels(int x, int k = 0) {
            set<int> neighbors;
            int var, neighbor;
            for(int i = k; i < getRelations()[x].size(); i++) {
                var = getRelations()[x][i]; // get the variable index
                for(int j = 0; j < getRels(var).size(); j++) {
                    neighbor = getRels(var)[j]; // get the relation index
                    if(neighbor != x) neighbors.insert(neighbor);
                }
            }
            // convert set to vector
            vector<int> neighborVec(neighbors.begin(), neighbors.end());
            return neighborVec; // return the vector of neighbors
        }

        void print(){
            cout << "Variables: " << endl;
            for(auto it = variables.begin(); it != variables.end(); it++){
                cout << it->first << "--Rename-> x" << it->second << endl;
            }
            cout << "Relations: " << endl;
            for(int i = 0; i < relations.size(); i++){
                cout << relationNames[i] << "--Rename-> ";
                cout << "R" << i <<"(";
                for(int j = 0; j < relations[i].size(); j++){
                    cout << "x" << relations[i][j];
                    if(j + 1 < relations[i].size()) cout << ", ";
                }
                cout << ")" << endl;
            }
        
        }

        double AGM(vector<int> &cars){
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
            // std::cout << "Optimal objective value: " << res << std::endl;
            // for (int i = 1; i <= 3; ++i) {
            //     double xi = glp_get_col_prim(lp, i);
            //     std::cout << "x" << i << " = " << xi << std::endl;
            // }
            return pow(2, res);
        }
};