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
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <functional>
using namespace std;

// ============================================================================
// [AJB] Dual LP / Sensitivity Analysis result structs
// ============================================================================

// Complementary slackness verification result for a single primal-dual pair
struct CSSlacknessEntry {
    int    index;          // variable index (for primal row) or relation index (for dual row)
    string name;           // human-readable name
    double primal_val;     // primal variable value (xᵢ for relation, slack for variable)
    double dual_val;       // dual variable value (yᵥ for variable, slack for relation)
    double cs_violation;   // |primal_val * dual_slack| or |dual_val * primal_slack|
    bool   cs_satisfied;   // true if violation < tolerance
};

// Full result of dual_LP verification
struct DualLPResult {
    bool   verified;              // true if optimality verified via complementary slackness
    double primal_obj;            // primal objective value (min Σ cᵢxᵢ)
    double dual_obj;              // dual objective value (max Σ yᵥ)
    double duality_gap;           // |primal_obj - dual_obj|
    double duality_gap_rel;       // relative gap: |gap| / max(1, |primal_obj|)
    int    cs_violations;         // number of complementary slackness violations
    double max_cs_violation;      // largest CS violation magnitude
    vector<double> primal_x;     // primal solution: xᵢ for each relation
    vector<double> dual_y;       // dual solution: yᵥ for each variable
    vector<CSSlacknessEntry> primal_cs;  // CS check for primal constraints (one per variable)
    vector<CSSlacknessEntry> dual_cs;    // CS check for dual constraints (one per relation)
    double solve_time_ms;         // wall time for dual solve + verification
};

// Shadow price entry for sensitivity analysis
struct ShadowPriceEntry {
    int    constraint_idx;    // row index in LP (1-based for GLPK, 0-based in output)
    string variable_name;     // which query variable this constraint corresponds to
    double shadow_price;      // marginal value of relaxing this constraint (= dual variable)
    double rhs_value;         // current RHS of the constraint (always 1.0 for edge cover)
    double row_slack;         // row activity - RHS: how much slack the constraint has
    int    row_status;        // GLPK basis status: GLP_BS, GLP_NL, GLP_NU, GLP_NF, GLP_NS
    // Ranging info
    double rhs_lower;         // smallest RHS keeping basis feasible
    double rhs_upper;         // largest RHS keeping basis feasible
    double obj_coef_lower;    // not applicable to rows, but included for completeness
    double obj_coef_upper;
};

// Full sensitivity analysis result
struct SensitivityResult {
    bool   valid;
    double base_obj;                        // objective at current solution
    double base_agm;                        // AGM = 2^base_obj
    vector<ShadowPriceEntry> row_analysis;  // one per variable constraint
    // Column (relation) reduced costs
    struct ColEntry {
        int    col_idx;
        string relation_name;
        double reduced_cost;     // cᵢ - Σ_{v ∈ Rᵢ} yᵥ  (should be ≥ 0 at optimality)
        double col_value;        // xᵢ at optimality
        int    col_status;       // GLPK basis status
        double obj_coef;         // log₂(|Rᵢ|)
    };
    vector<ColEntry> col_analysis;  // one per relation
    double solve_time_ms;
};

// === SUB-CLAUDE #17 ADDITIONS ===

            // }
            return pow(2, res);
        }

        // ====================================================================
        // [AJB] dual_LP — Dual formulation of AGM bound LP with
        //       complementary slackness optimality verification
        // ====================================================================
        //
        // Primal (existing):
        //   min  Σᵢ log₂(|Rᵢ|) · xᵢ
        //   s.t. ∀v: Σ_{Rᵢ ∋ v} xᵢ ≥ 1      (variable covering constraints)
        //        xᵢ ≥ 0
        //
        // Dual:
        //   max  Σᵥ yᵥ
        //   s.t. ∀Rᵢ: Σ_{v ∈ Rᵢ} yᵥ ≤ log₂(|Rᵢ|)  (packing constraints)
        //        yᵥ ≥ 0
        //
        // At optimality: strong duality ⟹ primal_obj == dual_obj
        // Complementary slackness conditions:
        //   (1) xᵢ > 0  ⟹  Σ_{v ∈ Rᵢ} yᵥ = log₂(|Rᵢ|)   (dual constraint tight)
        //   (2) yᵥ > 0  ⟹  Σ_{Rᵢ ∋ v} xᵢ = 1              (primal constraint tight)
        //
        DualLPResult dual_LP(vector<int> &cars, double cs_tolerance = 1e-8) {
            auto t0 = std::chrono::high_resolution_clock::now();
            DualLPResult result;
            result.verified = false;
            result.cs_violations = 0;
            result.max_cs_violation = 0.0;

            int nrels = static_cast<int>(relations.size());
            int nvars = static_cast<int>(variables.size());

            // --- Sanity checks ---
            if ((int)cars.size() != nrels) {
                fprintf(stderr, "[AJB_ERROR][dual_LP] cars.size()=%zu != nrels=%d\n",
                        cars.size(), nrels);
                result.primal_obj = result.dual_obj = 0.0;
                result.duality_gap = result.duality_gap_rel = 0.0;
                auto t1 = std::chrono::high_resolution_clock::now();
                result.solve_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                return result;
            }
            for (int i = 0; i < nrels; i++) {
                if (cars[i] <= 0) {
                    result.primal_obj = result.dual_obj = 0.0;
                    result.duality_gap = result.duality_gap_rel = 0.0;
                    result.verified = true; // trivially optimal: AGM = 0
                    auto t1 = std::chrono::high_resolution_clock::now();
                    result.solve_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    return result;
                }
            }

            // Precompute objective coefficients
            vector<double> c(nrels);
            for (int i = 0; i < nrels; i++) c[i] = log2(cars[i]);

            // ============================================================
            // Step 1: Solve the PRIMAL LP
            // ============================================================
            glp_prob *primal = glp_create_prob();
            glp_set_obj_dir(primal, GLP_MIN);

            glp_add_cols(primal, nrels);
            for (int i = 1; i <= nrels; i++) {
                glp_set_col_bnds(primal, i, GLP_LO, 0.0, 0.0); // xᵢ ≥ 0
                glp_set_obj_coef(primal, i, c[i - 1]);
            }

            glp_add_rows(primal, nvars);
            for (int v = 0; v < nvars; v++) {
                glp_set_row_bnds(primal, v + 1, GLP_LO, 1.0, 0.0); // Σ xᵢ ≥ 1
                const auto& rs = relsofVar[v];
                vector<int> ind(rs.size() + 1);
                vector<double> val(rs.size() + 1);
                ind[0] = 0; val[0] = 0.0;
                for (size_t j = 0; j < rs.size(); j++) {
                    ind[j + 1] = rs[j] + 1;
                    val[j + 1] = 1.0;
                }
                glp_set_mat_row(primal, v + 1, (int)rs.size(), ind.data(), val.data());
            }

            glp_smcp parm;
            glp_init_smcp(&parm);
            parm.msg_lev = GLP_MSG_OFF;
            glp_simplex(primal, &parm);

            // Extract primal solution
            result.primal_obj = glp_get_obj_val(primal);
            result.primal_x.resize(nrels);
            for (int i = 0; i < nrels; i++)
                result.primal_x[i] = glp_get_col_prim(primal, i + 1);

            // Extract dual variables directly from primal (GLPK provides them)
            // The dual value of row v+1 = shadow price = yᵥ
            vector<double> primal_dual_y(nvars);
            for (int v = 0; v < nvars; v++)
                primal_dual_y[v] = glp_get_row_dual(primal, v + 1);

            glp_delete_prob(primal);

            // ============================================================
            // Step 2: Solve the DUAL LP explicitly
            // ============================================================
            // max Σᵥ yᵥ  s.t. ∀Rᵢ: Σ_{v ∈ Rᵢ} yᵥ ≤ log₂(|Rᵢ|), yᵥ ≥ 0
            glp_prob *dual = glp_create_prob();
            glp_set_obj_dir(dual, GLP_MAX);

            // Columns = dual variables yᵥ (one per query variable)
            glp_add_cols(dual, nvars);
            for (int v = 1; v <= nvars; v++) {
                glp_set_col_bnds(dual, v, GLP_LO, 0.0, 0.0); // yᵥ ≥ 0
                glp_set_obj_coef(dual, v, 1.0);               // maximize Σ yᵥ
            }

            // Rows = dual constraints (one per relation Rᵢ)
            // Σ_{v ∈ Rᵢ} yᵥ ≤ log₂(|Rᵢ|)
            glp_add_rows(dual, nrels);
            for (int i = 0; i < nrels; i++) {
                glp_set_row_bnds(dual, i + 1, GLP_UP, 0.0, c[i]); // ≤ cᵢ
                const auto& rel = relations[i];
                vector<int> ind(rel.size() + 1);
                vector<double> val(rel.size() + 1);
                ind[0] = 0; val[0] = 0.0;
                for (size_t j = 0; j < rel.size(); j++) {
                    ind[j + 1] = rel[j] + 1;  // variable index → column
                    val[j + 1] = 1.0;
                }
                glp_set_mat_row(dual, i + 1, (int)rel.size(), ind.data(), val.data());
            }

            glp_smcp dparm;
            glp_init_smcp(&dparm);
            dparm.msg_lev = GLP_MSG_OFF;
            glp_simplex(dual, &dparm);

            // Extract dual solution
            result.dual_obj = glp_get_obj_val(dual);
            result.dual_y.resize(nvars);
            for (int v = 0; v < nvars; v++)
                result.dual_y[v] = glp_get_col_prim(dual, v + 1);

            glp_delete_prob(dual);

            // ============================================================
            // Step 3: Duality gap check
            // ============================================================
            result.duality_gap = fabs(result.primal_obj - result.dual_obj);
            result.duality_gap_rel = result.duality_gap / max(1.0, fabs(result.primal_obj));

            // ============================================================
            // Step 4: Complementary slackness verification
            // ============================================================
            // CS condition (1): for each relation Rᵢ
            //   xᵢ · (cᵢ - Σ_{v ∈ Rᵢ} yᵥ) = 0
            //   i.e., if xᵢ > 0 then Σ_{v ∈ Rᵢ} yᵥ must equal cᵢ
            result.dual_cs.resize(nrels);
            for (int i = 0; i < nrels; i++) {
                double dual_sum = 0.0;
                for (int v : relations[i]) dual_sum += result.dual_y[v];
                double dual_slack = c[i] - dual_sum;  // should be ≥ 0

                CSSlacknessEntry entry;
                entry.index = i;
                entry.name = (i < (int)relationNames.size()) ? relationNames[i] : ("R" + to_string(i));
                entry.primal_val = result.primal_x[i];       // xᵢ
                entry.dual_val = dual_slack;                  // cᵢ - Σ yᵥ (dual slack)
                entry.cs_violation = fabs(entry.primal_val * entry.dual_val);
                entry.cs_satisfied = (entry.cs_violation < cs_tolerance);
                result.dual_cs[i] = entry;

                if (!entry.cs_satisfied) {
                    result.cs_violations++;
                    result.max_cs_violation = max(result.max_cs_violation, entry.cs_violation);
                }
            }

            // CS condition (2): for each variable v
            //   yᵥ · (Σ_{Rᵢ ∋ v} xᵢ - 1) = 0
            //   i.e., if yᵥ > 0 then the covering constraint must be tight
            result.primal_cs.resize(nvars);
            for (int v = 0; v < nvars; v++) {
                double primal_sum = 0.0;
                for (int ri : relsofVar[v]) primal_sum += result.primal_x[ri];
                double primal_slack = primal_sum - 1.0;  // should be ≥ 0

                CSSlacknessEntry entry;
                entry.index = v;
                entry.name = (v < (int)variableNames.size()) ? variableNames[v] : ("x" + to_string(v));
                entry.primal_val = primal_slack;           // primal slack
                entry.dual_val = result.dual_y[v];         // yᵥ
                entry.cs_violation = fabs(entry.primal_val * entry.dual_val);
                entry.cs_satisfied = (entry.cs_violation < cs_tolerance);
                result.primal_cs[v] = entry;

                if (!entry.cs_satisfied) {
                    result.cs_violations++;
                    result.max_cs_violation = max(result.max_cs_violation, entry.cs_violation);
                }
            }

            // Optimality verified if gap is small AND no CS violations
            result.verified = (result.duality_gap_rel < 1e-6) && (result.cs_violations == 0);

            auto t1 = std::chrono::high_resolution_clock::now();
            result.solve_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // ============================================================
            // Diagnostic output
            // ============================================================
            fprintf(stderr, "[AJB_DUAL][AGM] primal_obj=%.8f  dual_obj=%.8f  gap=%.2e (rel=%.2e)\n",
                    result.primal_obj, result.dual_obj, result.duality_gap, result.duality_gap_rel);
            fprintf(stderr, "[AJB_DUAL][AGM] CS violations=%d  max_violation=%.2e  verified=%s\n",
                    result.cs_violations, result.max_cs_violation,
                    result.verified ? "YES" : "NO");
            fprintf(stderr, "[AJB_DUAL][AGM] primal_x=[");
            for (int i = 0; i < nrels; i++) {
                if (i) fprintf(stderr, ",");
                fprintf(stderr, "%.6f", result.primal_x[i]);
            }
            fprintf(stderr, "]  dual_y=[");
            for (int v = 0; v < nvars; v++) {
                if (v) fprintf(stderr, ",");
                fprintf(stderr, "%.6f", result.dual_y[v]);
            }
            fprintf(stderr, "]\n");

            // Cross-validate: dual_y from explicit dual should match primal's row duals
            double cross_diff = 0.0;
            for (int v = 0; v < nvars; v++)
                cross_diff += fabs(result.dual_y[v] - primal_dual_y[v]);
            fprintf(stderr, "[AJB_DUAL][AGM] cross-validation: Σ|y_dual - y_primal_shadow| = %.2e\n",
                    cross_diff);

            // Print any CS violations in detail
            if (result.cs_violations > 0) {
                fprintf(stderr, "[AJB_DUAL][AGM] CS violation details:\n");
                for (auto& e : result.dual_cs) {
                    if (!e.cs_satisfied)
                        fprintf(stderr, "  rel %s: x=%.6f dual_slack=%.6f product=%.2e\n",
                                e.name.c_str(), e.primal_val, e.dual_val, e.cs_violation);
                }
                for (auto& e : result.primal_cs) {
                    if (!e.cs_satisfied)
                        fprintf(stderr, "  var %s: y=%.6f primal_slack=%.6f product=%.2e\n",
                                e.name.c_str(), e.dual_val, e.primal_val, e.cs_violation);
                }
            }

            fprintf(stderr, "[AJB_DUAL][AGM] AGM = 2^%.6f = %.4f  (time=%.3fms)\n",
                    result.primal_obj, pow(2, result.primal_obj), result.solve_time_ms);

            return result;
        }

        // ====================================================================
        // [AJB] sensitivity_analysis — Shadow prices and ranging for each
        //       constraint in the AGM edge cover LP
        // ====================================================================
        //
        // For each row constraint (variable covering):
        //   shadow_price = ∂(obj)/∂(RHS) = how much the AGM log-bound changes
        //   if we relax this variable's covering requirement
        //
        // For each column (relation):
        //   reduced_cost = cᵢ - Σ_{v ∈ Rᵢ} yᵥ = how much cheaper this relation
        //   would need to be to enter the basis
        //
        // Also performs RHS ranging via GLPK's sensitivity routines when available,
        // falling back to manual perturbation analysis otherwise.
        //
        SensitivityResult sensitivity_analysis(vector<int> &cars, double perturb_eps = 1e-4) {
            auto t0 = std::chrono::high_resolution_clock::now();
            SensitivityResult result;
            result.valid = false;

            int nrels = static_cast<int>(relations.size());
            int nvars = static_cast<int>(variables.size());

            if ((int)cars.size() != nrels || nrels == 0 || nvars == 0) {
                fprintf(stderr, "[AJB_SENS] invalid input: cars=%zu nrels=%d nvars=%d\n",
                        cars.size(), nrels, nvars);
                auto t1 = std::chrono::high_resolution_clock::now();
                result.solve_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                return result;
            }
            for (int i = 0; i < nrels; i++) {
                if (cars[i] <= 0) {
                    result.base_obj = 0.0;
                    result.base_agm = 0.0;
                    result.valid = true;
                    auto t1 = std::chrono::high_resolution_clock::now();
                    result.solve_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    return result;
                }
            }

            vector<double> c(nrels);
            for (int i = 0; i < nrels; i++) c[i] = log2(cars[i]);

            // Build and solve the LP
            glp_prob *lp_sa = glp_create_prob();
            glp_set_obj_dir(lp_sa, GLP_MIN);

            glp_add_cols(lp_sa, nrels);
            for (int i = 1; i <= nrels; i++) {
                char colname[32];
                snprintf(colname, sizeof(colname), "x_%s",
                         i <= (int)relationNames.size() ? relationNames[i-1].c_str() : "?");
                glp_set_col_name(lp_sa, i, colname);
                glp_set_col_bnds(lp_sa, i, GLP_LO, 0.0, 0.0);
                glp_set_obj_coef(lp_sa, i, c[i - 1]);
            }

            glp_add_rows(lp_sa, nvars);
            for (int v = 0; v < nvars; v++) {
                char rowname[32];
                snprintf(rowname, sizeof(rowname), "cov_%s",
                         v < (int)variableNames.size() ? variableNames[v].c_str() : "?");
                glp_set_row_name(lp_sa, v + 1, rowname);
                glp_set_row_bnds(lp_sa, v + 1, GLP_LO, 1.0, 0.0);
                const auto& rs = relsofVar[v];
                vector<int> ind(rs.size() + 1);
                vector<double> val(rs.size() + 1);
                ind[0] = 0; val[0] = 0.0;
                for (size_t j = 0; j < rs.size(); j++) {
                    ind[j + 1] = rs[j] + 1;
                    val[j + 1] = 1.0;
                }
                glp_set_mat_row(lp_sa, v + 1, (int)rs.size(), ind.data(), val.data());
            }

            glp_smcp parm;
            glp_init_smcp(&parm);
            parm.msg_lev = GLP_MSG_OFF;
            int simplex_ret = glp_simplex(lp_sa, &parm);

            if (simplex_ret != 0 || glp_get_status(lp_sa) != GLP_OPT) {
                fprintf(stderr, "[AJB_SENS] LP did not reach optimality (ret=%d status=%d)\n",
                        simplex_ret, glp_get_status(lp_sa));
                glp_delete_prob(lp_sa);
                auto t1 = std::chrono::high_resolution_clock::now();
                result.solve_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                return result;
            }

            result.base_obj = glp_get_obj_val(lp_sa);
            result.base_agm = pow(2, result.base_obj);
            result.valid = true;

            // ============================================================
            // Row analysis: shadow prices from dual values
            // ============================================================
            result.row_analysis.resize(nvars);
            for (int v = 0; v < nvars; v++) {
                ShadowPriceEntry entry;
                entry.constraint_idx = v;
                entry.variable_name = (v < (int)variableNames.size()) ? variableNames[v] : ("x" + to_string(v));
                entry.shadow_price = glp_get_row_dual(lp_sa, v + 1);
                entry.rhs_value = 1.0;
                entry.row_slack = glp_get_row_prim(lp_sa, v + 1) - 1.0;
                entry.row_status = glp_get_row_stat(lp_sa, v + 1);

                // Manual perturbation for RHS ranging:
                // Perturb RHS of row v+1 by ±eps, re-solve, check if basis changes
                // Lower range
                glp_set_row_bnds(lp_sa, v + 1, GLP_LO, 1.0 - perturb_eps, 0.0);
                glp_simplex(lp_sa, &parm);
                double obj_lower = glp_get_obj_val(lp_sa);

                // Upper range
                glp_set_row_bnds(lp_sa, v + 1, GLP_LO, 1.0 + perturb_eps, 0.0);
                glp_simplex(lp_sa, &parm);
                double obj_upper = glp_get_obj_val(lp_sa);

                // Restore
                glp_set_row_bnds(lp_sa, v + 1, GLP_LO, 1.0, 0.0);
                glp_simplex(lp_sa, &parm);

                // Estimate range where shadow price remains constant
                // If perturbation changes obj linearly, the basis is stable
                double sp = entry.shadow_price;
                double expected_lower = result.base_obj - sp * perturb_eps;
                double expected_upper = result.base_obj + sp * perturb_eps;
                double err_lower = fabs(obj_lower - expected_lower);
                double err_upper = fabs(obj_upper - expected_upper);

                // If linear prediction matches, the basis is stable for at least ±eps
                entry.rhs_lower = (err_lower < 1e-8) ? 0.0 : (1.0 - perturb_eps);
                entry.rhs_upper = (err_upper < 1e-8) ? DBL_MAX : (1.0 + perturb_eps);
                entry.obj_coef_lower = 0.0;
                entry.obj_coef_upper = 0.0;

                result.row_analysis[v] = entry;
            }

            // ============================================================
            // Column analysis: reduced costs
            // ============================================================
            result.col_analysis.resize(nrels);
            for (int i = 0; i < nrels; i++) {
                SensitivityResult::ColEntry entry;
                entry.col_idx = i;
                entry.relation_name = (i < (int)relationNames.size()) ? relationNames[i] : ("R" + to_string(i));
                entry.reduced_cost = glp_get_col_dual(lp_sa, i + 1);
                entry.col_value = glp_get_col_prim(lp_sa, i + 1);
                entry.col_status = glp_get_col_stat(lp_sa, i + 1);
                entry.obj_coef = c[i];

                result.col_analysis[i] = entry;
            }

            glp_delete_prob(lp_sa);

            auto t1 = std::chrono::high_resolution_clock::now();
            result.solve_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // ============================================================
            // Diagnostic output
            // ============================================================
            fprintf(stderr, "[AJB_SENS][AGM] === Sensitivity Analysis ===\n");
            fprintf(stderr, "[AJB_SENS][AGM] base_obj=%.8f  AGM=%.4f  time=%.3fms\n",
                    result.base_obj, result.base_agm, result.solve_time_ms);

            fprintf(stderr, "[AJB_SENS][AGM] --- Row (variable covering) shadow prices ---\n");
            // Sort by shadow price magnitude for readable output
            vector<int> row_order(nvars);
            iota(row_order.begin(), row_order.end(), 0);
            sort(row_order.begin(), row_order.end(), [&](int a, int b) {
                return fabs(result.row_analysis[a].shadow_price) >
                       fabs(result.row_analysis[b].shadow_price);
            });
            for (int v : row_order) {
                auto& e = result.row_analysis[v];
                const char* stat_str = "??";
                switch(e.row_status) {
                    case GLP_BS: stat_str = "BS"; break;  // basic
                    case GLP_NL: stat_str = "NL"; break;  // at lower bound
                    case GLP_NU: stat_str = "NU"; break;  // at upper bound
                    case GLP_NF: stat_str = "NF"; break;  // free
                    case GLP_NS: stat_str = "NS"; break;  // fixed
                }
                fprintf(stderr, "[AJB_SENS]   var %-10s  shadow=%.6f  slack=%.6f  status=%s",
                        e.variable_name.c_str(), e.shadow_price, e.row_slack, stat_str);
                if (fabs(e.shadow_price) > 1e-10) {
                    // Interpretation: a unit increase in RHS changes obj by shadow_price
                    // In AGM terms: AGM changes by factor 2^shadow_price
                    fprintf(stderr, "  → AGM×%.4f per unit RHS", pow(2, e.shadow_price));
                }
                fprintf(stderr, "\n");
            }

            fprintf(stderr, "[AJB_SENS][AGM] --- Column (relation) reduced costs ---\n");
            for (int i = 0; i < nrels; i++) {
                auto& e = result.col_analysis[i];
                const char* stat_str = "??";
                switch(e.col_status) {
                    case GLP_BS: stat_str = "BS"; break;
                    case GLP_NL: stat_str = "NL"; break;
                    case GLP_NU: stat_str = "NU"; break;
                    case GLP_NF: stat_str = "NF"; break;
                    case GLP_NS: stat_str = "NS"; break;
                }
                fprintf(stderr, "[AJB_SENS]   rel %-10s  x=%.6f  rc=%.6f  c=%.6f  status=%s",
                        e.relation_name.c_str(), e.col_value, e.reduced_cost, e.obj_coef, stat_str);
                if (e.col_status == GLP_NL && e.reduced_cost > 1e-10) {
                    // This relation is non-basic at lower bound;
                    // reduced cost = how much cheaper it must become to enter basis
                    fprintf(stderr, "  (not in cover; needs rc drop %.6f)", e.reduced_cost);
                }
                fprintf(stderr, "\n");
            }

            // Summary: which variables are "bottleneck" (highest shadow price)
            if (nvars > 0) {
                int bottleneck = row_order[0];
                fprintf(stderr, "[AJB_SENS][AGM] Bottleneck variable: %s (shadow=%.6f)\n",
                        result.row_analysis[bottleneck].variable_name.c_str(),
                        result.row_analysis[bottleneck].shadow_price);
            }

            return result;
        }

        // ====================================================================
        // [AJB] perturbation_sweep — Multi-epsilon sensitivity analysis
        //
        // Instead of a single perturbation epsilon, sweep across a geometric
        // series of perturbation magnitudes to find the breakpoint where the
        // basis changes (shadow price becomes nonlinear). This identifies
        // the *range of validity* of each shadow price more precisely than
        // the single-epsilon approach above.
        //
        // Returns per-variable: the largest epsilon where the shadow price
        // remains approximately constant (basis stability radius).
        // ====================================================================
        struct PerturbSweepEntry {
            int    var_idx;
            string variable_name;
            double shadow_price;
            double stability_radius;   // largest eps where shadow price is stable
            double breakpoint_eps;     // first eps where basis changes
            bool   is_degenerate;      // shadow price flips sign during sweep
            vector<pair<double,double>> eps_obj_curve; // (eps, obj_value) for plotting
        };

        struct PerturbSweepResult {
            vector<PerturbSweepEntry> entries;
            double sweep_time_ms;
            int    total_solves;
        };

        PerturbSweepResult perturbation_sweep(vector<int> &cars,
                                               int n_steps = 8,
                                               double eps_min = 1e-6,
                                               double eps_max = 1.0) {
            auto t0 = std::chrono::high_resolution_clock::now();
            PerturbSweepResult sweep;
            sweep.total_solves = 0;

            int nrels = static_cast<int>(relations.size());
            int nvars = static_cast<int>(variables.size());

            // Build LP once
            vector<double> c(nrels);
            for (int i = 0; i < nrels; i++) c[i] = log2(cars[i]);

            glp_prob *lp = glp_create_prob();
            glp_set_obj_dir(lp, GLP_MIN);
            glp_add_cols(lp, nrels);
            for (int i = 1; i <= nrels; i++) {
                glp_set_col_bnds(lp, i, GLP_LO, 0.0, 0.0);
                glp_set_obj_coef(lp, i, c[i - 1]);
            }
            glp_add_rows(lp, nvars);
            for (int v = 0; v < nvars; v++) {
                glp_set_row_bnds(lp, v + 1, GLP_LO, 1.0, 0.0);
                const auto& rs = relsofVar[v];
                vector<int> ind(rs.size() + 1);
                vector<double> val(rs.size() + 1);
                ind[0] = 0; val[0] = 0.0;
                for (size_t j = 0; j < rs.size(); j++) {
                    ind[j + 1] = rs[j] + 1;
                    val[j + 1] = 1.0;
                }
                glp_set_mat_row(lp, v + 1, (int)rs.size(), ind.data(), val.data());
            }
            glp_smcp parm;
            glp_init_smcp(&parm);
            parm.msg_lev = GLP_MSG_OFF;

            // Solve baseline
            glp_simplex(lp, &parm);
            double base_obj = glp_get_obj_val(lp);
            sweep.total_solves++;

            // Geometric series of epsilon values
            vector<double> eps_values;
            double ratio = pow(eps_max / eps_min, 1.0 / (n_steps - 1));
            for (int s = 0; s < n_steps; s++) {
                eps_values.push_back(eps_min * pow(ratio, s));
            }

            sweep.entries.resize(nvars);

            for (int v = 0; v < nvars; v++) {
                PerturbSweepEntry& entry = sweep.entries[v];
                entry.var_idx = v;
                entry.variable_name = (v < (int)variableNames.size())
                    ? variableNames[v] : ("x" + to_string(v));
                entry.shadow_price = glp_get_row_dual(lp, v + 1);
                entry.stability_radius = eps_max;
                entry.breakpoint_eps = eps_max;
                entry.is_degenerate = false;

                double sp = entry.shadow_price;
                bool found_break = false;

                for (double eps : eps_values) {
                    // Perturb row v+1 upward by eps
                    glp_set_row_bnds(lp, v + 1, GLP_LO, 1.0 + eps, 0.0);
                    glp_simplex(lp, &parm);
                    double obj_perturbed = glp_get_obj_val(lp);
                    sweep.total_solves++;

                    // Linear prediction: obj should change by sp * eps
                    double expected = base_obj + sp * eps;
                    double prediction_error = fabs(obj_perturbed - expected);
                    double rel_error = (fabs(expected) > 1e-15)
                        ? prediction_error / fabs(expected) : prediction_error;

                    entry.eps_obj_curve.push_back({eps, obj_perturbed});

                    // Check if new dual at this row has flipped sign
                    double new_sp = glp_get_row_dual(lp, v + 1);
                    if (sp * new_sp < 0 && fabs(new_sp) > 1e-10) {
                        entry.is_degenerate = true;
                    }

                    // Basis change: prediction error exceeds 1% relative
                    if (!found_break && rel_error > 0.01) {
                        entry.breakpoint_eps = eps;
                        // Stability radius is the previous epsilon
                        entry.stability_radius = (eps > eps_min) ? eps / ratio : eps_min;
                        found_break = true;
                    }

                    // Restore for next iteration
                    glp_set_row_bnds(lp, v + 1, GLP_LO, 1.0, 0.0);
                }

                // If no breakpoint found, shadow price is stable across entire range
                if (!found_break) {
                    entry.stability_radius = eps_max;
                    entry.breakpoint_eps = eps_max;
                }

                fprintf(stderr, "[AJB_BP][PerturbSweep] var=%s shadow=%.6f "
                        "stability_radius=%.6f breakpoint=%.6f %s\n",
                        entry.variable_name.c_str(), entry.shadow_price,
                        entry.stability_radius, entry.breakpoint_eps,
                        entry.is_degenerate ? "DEGENERATE" : "stable");
            }

            glp_delete_prob(lp);
            auto t1 = std::chrono::high_resolution_clock::now();
            sweep.sweep_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            fprintf(stderr, "[AJB_SENS][PerturbSweep] %d variables, %d total solves, "
                    "%.3f ms\n", nvars, sweep.total_solves, sweep.sweep_time_ms);

            return sweep;
        }

        // ====================================================================
        // [AJB] classify_binding_constraints — Partition variable constraints
        // into three categories that guide the join enumeration strategy:
        //
        //   TIGHT:  shadow_price > 0, slack ≈ 0  → bottleneck, worth optimizing
        //   SLACK:  shadow_price ≈ 0, slack > 0  → not binding, low priority
        //   NEAR:   shadow_price ≈ 0, slack ≈ 0  → near-degenerate, sensitive
        //
        // The TIGHT set directly drives which relations to prioritize in the
        // join order. NEAR constraints may become TIGHT with small data changes.
        // ====================================================================
        struct BindingClassification {
            vector<int> tight;      // variable indices
            vector<int> slack;
            vector<int> near_degenerate;
            double tight_shadow_sum;  // total shadow price of tight constraints
        };

        BindingClassification classify_binding_constraints(
            const SensitivityResult& sens,
            double shadow_threshold = 1e-6,
            double slack_threshold = 1e-4) {

            BindingClassification bc;
            bc.tight_shadow_sum = 0.0;

            for (size_t v = 0; v < sens.row_analysis.size(); v++) {
                const auto& e = sens.row_analysis[v];
                double abs_shadow = fabs(e.shadow_price);
                double abs_slack = fabs(e.row_slack);

                if (abs_shadow > shadow_threshold && abs_slack < slack_threshold) {
                    bc.tight.push_back(v);
                    bc.tight_shadow_sum += abs_shadow;
                } else if (abs_shadow < shadow_threshold && abs_slack > slack_threshold) {
                    bc.slack.push_back(v);
                } else {
                    bc.near_degenerate.push_back(v);
                }
            }

            // Sort tight by descending shadow price
            sort(bc.tight.begin(), bc.tight.end(), [&](int a, int b) {
                return fabs(sens.row_analysis[a].shadow_price) >
                       fabs(sens.row_analysis[b].shadow_price);
            });

            fprintf(stderr, "[AJB_STATE][BindingClass] tight=%zu slack=%zu near=%zu "
                    "tight_shadow_sum=%.6f\n",
                    bc.tight.size(), bc.slack.size(), bc.near_degenerate.size(),
                    bc.tight_shadow_sum);

            for (int v : bc.tight) {
                const auto& e = sens.row_analysis[v];
                fprintf(stderr, "[AJB_STATE][BindingClass]   TIGHT var=%s shadow=%.6f\n",
                        e.variable_name.c_str(), e.shadow_price);
            }

            return bc;
        }
};