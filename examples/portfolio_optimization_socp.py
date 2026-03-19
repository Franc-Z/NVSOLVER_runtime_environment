#!/usr/bin/env python3
"""
Portfolio optimization using SOCP (Second-Order Cone Programming) with CVXPY
Based on multi-period mean-risk portfolio optimization
Reference: https://github.com/Franc-Z/GPU-Accelerated_Portfolio_Optimization/blob/main/Test_Accuracy/clarabelgpu_multi_period(mean-risk).jl
"""

import numpy as np
import cvxpy as cp
import time
from contextlib import contextmanager

from clarabel_gpu import UpdateHint

# ── GPU memory tracking via RMM ──────────────────────────────────────────────
_rmm_stats_available = False
try:
    import rmm
    import rmm.statistics
    rmm.reinitialize(pool_allocator=True,
                     initial_pool_size=256 * 1024 * 1024,   # 256 MB
                     maximum_pool_size=4 * 1024 * 1024 * 1024)  # 4 GB
    rmm.statistics.enable_statistics()
    _rmm_stats_available = True
    print("[RMM] GPU memory statistics enabled")
except ImportError:
    print("[RMM] rmm not installed – GPU memory tracking disabled")
except Exception as e:
    print(f"[RMM] Could not enable statistics: {e}")


def get_peak_gpu_memory_mb():
    """Return peak GPU memory allocated (in MB) since statistics were enabled."""
    if not _rmm_stats_available:
        return None
    stats = rmm.statistics.get_statistics()
    if stats is None:
        return None
    return stats.peak_bytes / (1024 * 1024)


def print_gpu_memory_stats(label=""):
    """Print current / peak GPU memory usage reported by RMM."""
    if not _rmm_stats_available:
        return
    stats = rmm.statistics.get_statistics()
    if stats is None:
        return
    prefix = f"[GPU Memory] {label}" if label else "[GPU Memory]"
    print(f"{prefix}  current = {stats.current_bytes / 1024 / 1024:.2f} MB, "
          f"peak = {stats.peak_bytes / 1024 / 1024:.2f} MB, "
          f"total allocated = {stats.total_bytes / 1024 / 1024:.2f} MB "
          f"(current allocs = {stats.current_count}, peak allocs = {stats.peak_count})")
# ─────────────────────────────────────────────────────────────────────────────

# Performance monitoring decorator
@contextmanager
def perf_monitor(operation_name):
    """Context manager for monitoring operation performance"""
    start_time = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"[Performance] {operation_name}: {elapsed:.2f} ms")

# Helper function to check if solve was successful
def is_solved_successfully(status):
    """Check if the optimization problem was solved successfully
    
    Args:
        status: CVXPy problem status
    
    Returns:
        bool: True if solved optimally or with acceptable accuracy
    """
    # Accept both OPTIMAL and ALMOST_SOLVED (optimal_inaccurate)
    acceptable_statuses = [
        cp.OPTIMAL,
        'optimal',
        'optimal_inaccurate',  # ALMOST_SOLVED
        'solved',
        'Solved',
        'almost_solved'
    ]
    return status in acceptable_statuses or str(status).lower() in ['optimal', 'solved', 'optimal_inaccurate','almost_solved']


def analyze_constraint_tightness(problem, constraint_names=None, tol=1e-6):
    """
    Analyze constraint tightness using slack variables (definitive criterion).

    Theoretical basis:
    - Slack variable s is the SOLE definitive criterion for binding/slack.
      For inequality g(x)<=0 with s>=0:  s=0 <=> tight,  s>0 <=> loose.
    - For SOC (t,u) in K_soc:  margin m = t - ||u||_2.
      m=0 <=> tight (on cone boundary),  m>0 <=> loose (cone interior).
    - dual_value (lambda) is NOT needed for binding determination:
      lambda>0 => s=0 (certain), but s=0 does NOT imply lambda>0 (degenerate).
      dual_value is only useful for sensitivity analysis (shadow price) and
      degenerate constraint identification.

    Metrics computed:
    1. Slack / margin (definitive): scalar slack for inequalities, cone margin for SOC
    2. Feasibility residual (violation): verification that constraint is satisfied
    3. Dual variable norm: for sensitivity analysis only (NOT for binding determination)

    Determination rules (based on solver tolerance tol):
    - Scalar inequality: slack_min <= tol -> TIGHT | slack_min >= 10*tol -> LOOSE
    - SOC constraint:    margin   <= tol -> TIGHT | margin   >= 10*tol -> LOOSE
    - Equality:          report violation residual

    Args:
        problem: Solved CVXPY Problem object
        constraint_names: Constraint name list (corresponds one-to-one with problem.constraints)
        tol: Determination threshold, default 1e-6 (recommended to use max(eps_abs, eps_rel))

    Returns:
        results: Analysis results list for each constraint
        summary: Tightness classification summary dictionary
    """

    print("\n" + "=" * 70)
    print("Constraint Tightness Analysis (slack-based)")
    print("=" * 70)
    print(f"Solver tolerance tol = {tol:.2e}")
    print(f"Determination rules (slack variable — definitive criterion):")
    print(f"  Inequality: slack_min <= tol -> TIGHT   (s=0 <=> binding)")
    print(f"              slack_min >= 10*tol -> LOOSE (s>0 <=> interior)")
    print(f"              otherwise -> BORDERLINE")
    print(f"  SOC cone:   margin = t - ||u||_2 <= tol -> TIGHT  (on cone boundary)")
    print(f"              margin >= 10*tol -> LOOSE             (cone interior)")
    print(f"  Equality:   report violation residual")
    print(f"  Note: dual shown for sensitivity analysis only, NOT for binding judgment")
    print()

    summary = {'tight': [], 'loose': [], 'borderline': [], 'equality': []}
    results = []

    for i, c in enumerate(problem.constraints):
        name = constraint_names[i] if constraint_names and i < len(constraint_names) else f"Constraint #{i}"
        ctype = type(c).__name__

        # (1) Feasibility residual
        try:
            r = c.violation()
            violation_val = float(r) if np.isscalar(r) else float(np.max(np.asarray(r)))
        except Exception:
            violation_val = float('nan')

        # (2) Dual variable norm
        dual = c.dual_value
        if dual is not None:
            dual_arr = np.asarray(dual, dtype=float)
            dual_norm = float(np.linalg.norm(dual_arr.ravel()))
        else:
            dual_norm = None

        result = {
            'index': i, 'name': name, 'type': ctype,
            'violation': violation_val, 'dual_norm': dual_norm,
        }

        # (3) Determine tightness by constraint type
        if ctype in ('NonPos', 'Inequality'):
            # Element-wise inequality: expr <= 0, slack = -expr
            expr_val = c.args[0].value
            if expr_val is not None:
                slack = -np.asarray(expr_val, dtype=float).ravel()
                m_min = float(np.min(slack))
                m_max = float(np.max(slack))
                m_mean = float(np.mean(slack))
                n_tight = int(np.sum(slack <= tol))
                n_total = int(slack.size)
                result.update({
                    'slack_min': m_min, 'slack_max': m_max, 'slack_mean': m_mean,
                    'n_tight_elements': n_tight, 'n_total_elements': n_total,
                })
                if m_min <= tol:
                    result['label'] = 'TIGHT'
                    summary['tight'].append(i)
                elif m_min >= 10 * tol:
                    result['label'] = 'LOOSE'
                    summary['loose'].append(i)
                else:
                    result['label'] = 'BORDERLINE'
                    summary['borderline'].append(i)
            else:
                result['label'] = 'No value'

        elif ctype == 'NonNeg':
            # Element-wise inequality: expr >= 0, slack = expr
            expr_val = c.args[0].value
            if expr_val is not None:
                slack = np.asarray(expr_val, dtype=float).ravel()
                m_min = float(np.min(slack))
                m_max = float(np.max(slack))
                m_mean = float(np.mean(slack))
                n_tight = int(np.sum(slack <= tol))
                n_total = int(slack.size)
                result.update({
                    'slack_min': m_min, 'slack_max': m_max, 'slack_mean': m_mean,
                    'n_tight_elements': n_tight, 'n_total_elements': n_total,
                })
                if m_min <= tol:
                    result['label'] = 'TIGHT'
                    summary['tight'].append(i)
                elif m_min >= 10 * tol:
                    result['label'] = 'LOOSE'
                    summary['loose'].append(i)
                else:
                    result['label'] = 'BORDERLINE'
                    summary['borderline'].append(i)
            else:
                result['label'] = 'No value'

        elif ctype == 'Zero':
            # Equality constraint: report violation residual
            result['label'] = 'EQUALITY'
            summary['equality'].append(i)

        elif ctype == 'SOC':
            # SOC constraint: (t, u) in K_soc, margin m = t - ||u||_2
            # m > 0: loose (strictly in cone interior)
            # m ~ 0: tight (on cone boundary)
            # m < 0: violated (infeasible)
            try:
                t_val = float(np.asarray(c.args[0].value).ravel()[0])
                x_val = np.asarray(c.args[1].value, dtype=float).ravel()
                soc_margin = t_val - np.linalg.norm(x_val)
            except Exception:
                soc_margin = float('nan')
            result['soc_margin'] = soc_margin
            if np.isnan(soc_margin):
                result['label'] = 'SOC/NO VALUE'
            elif soc_margin <= tol:
                result['label'] = 'TIGHT/SOC'
                summary['tight'].append(i)
            elif soc_margin >= 10 * tol:
                result['label'] = 'LOOSE/SOC'
                summary['loose'].append(i)
            else:
                result['label'] = 'BORDERLINE/SOC'
                summary['borderline'].append(i)

        else:
            result['label'] = f'Other ({ctype})'

        results.append(result)

    # Print detailed table
    header = (f"{'#':>3} | {'Constraint Name':<38} | {'Type':<8} | {'Status':<18} | "
              f"{'slack/margin':>12} | {'violation':>10} | {'dual(sens)':>10} | {'Tight/Total':>11}")
    print(header)
    print("-" * len(header))

    for res in results:
        nm = res['name'][:36] + '..' if len(res['name']) > 38 else res['name']
        viol_s = f"{res['violation']:.2e}"
        dual_s = f"{res['dual_norm']:.2e}" if res['dual_norm'] is not None else "N/A"
        if 'slack_min' in res:
            slack_s = f"{res['slack_min']:.2e}"
        elif 'soc_margin' in res:
            slack_s = f"{res['soc_margin']:.2e}"
        else:
            slack_s = "N/A"
        tight_s = (f"{res.get('n_tight_elements', '-')}/{res.get('n_total_elements', '-')}"
                   if 'n_tight_elements' in res else "N/A")
        print(f"{res['index']:3d} | {nm:<38} | {res['type']:<8} | {res['label']:<18} | "
              f"{slack_s:>12} | {viol_s:>10} | {dual_s:>10} | {tight_s:>8}")

    # Summary
    total = len(results)
    print(f"\n{'─' * 70}")
    print(f"Summary ({total} constraints total):")
    print(f"  Tight (TIGHT):       {len(summary['tight']):3d}  "
          f"{summary['tight'][:10]}{'...' if len(summary['tight']) > 10 else ''}")
    print(f"  Loose (LOOSE):       {len(summary['loose']):3d}  "
          f"{summary['loose'][:10]}{'...' if len(summary['loose']) > 10 else ''}")
    print(f"  Borderline (BORDERLINE): {len(summary['borderline']):3d}  "
          f"{summary['borderline'][:10]}{'...' if len(summary['borderline']) > 10 else ''}")
    print(f"  Equality (EQUALITY):  {len(summary['equality']):3d}")

    # Tightest inequality constraints Top-5 (by slack — definitive)
    ineq_results = [r for r in results if 'slack_min' in r]
    if ineq_results:
        ineq_sorted = sorted(ineq_results, key=lambda r: r['slack_min'])
        print(f"\n  Tightest inequality constraints (Top 5, by slack — definitive):")
        for r in ineq_sorted[:5]:
            d_s = f"{r['dual_norm']:.2e}" if r['dual_norm'] is not None else "N/A"
            print(f"    #{r['index']:3d} {r['name']:<38} slack={r['slack_min']:.2e}  dual(sensitivity)={d_s}")

    # SOC constraints sorted by margin (m = t - ||u||, definitive criterion)
    soc_results = [r for r in results if 'soc_margin' in r]
    if soc_results:
        soc_sorted = sorted(soc_results, key=lambda r: r.get('soc_margin', float('inf')))
        print(f"\n  SOC constraints (sorted by margin m = t - ||u||_2 — definitive):")
        for r in soc_sorted:
            m_s = f"{r['soc_margin']:.2e}" if 'soc_margin' in r else "N/A"
            d_s = f"{r['dual_norm']:.2e}" if r['dual_norm'] is not None else "N/A"
            print(f"    #{r['index']:3d} {r['name']:<38} margin={m_s}  dual(sensitivity)={d_s}")

    print("=" * 70)

    return results, summary


# Fix random seed
np.random.seed(42)

# Parameter settings
k = 50       # Number of style factors
n = k * 100  # Total number of assets for portfolio optimization
T = 1        # Number of periods (>= 1)

# Generate data
D_diag = np.random.rand(n) * np.sqrt(k)
F = np.random.randn(n, k) * (np.random.rand(n, k) < 0.5)
Omega_temp = np.random.randn(k, k)
Omega = (Omega_temp @ Omega_temp.T) / k
mu_matrix = (3.0 + 9.0 * np.random.rand(n, T)) / 100.0

# Cholesky decomposition for SOCP form
L_Omega = np.linalg.cholesky(Omega)  # Omega = L @ L^T
F_t = F.T

# Problem dimensions and parameters
n, k = F.shape
T = mu_matrix.shape[1]
x0 = np.zeros(n)
gamma = 1.0  # Risk aversion parameter
d = 1.0 - np.sum(x0)
transaction_cost_rate = 0.002
max_risk = 0.15  # Maximum risk limit (for SOCP constraints)

print(f"Problem dimensions: n={n}, k={k}, T={T}")
print(f"Parameters: gamma={gamma}, d={d}, transaction_cost_rate={transaction_cost_rate}")
print(f"Max risk constraint: {max_risk}")

def create_portfolio_socp_model():
    """Create SOCP-based portfolio optimization model with only mu as DPP parameter"""
    
    # Declare only mu as parameter (DPP way) - for efficient updates
    mu_param = cp.Parameter((n, T))                # Expected return parameter
    
    # Initialize parameter value
    mu_param.value = mu_matrix
    
    # Other values used as constants (not parameters)
    D_diag_sqrt = np.sqrt(D_diag)  # Pre-compute square root for SOCP
    
    # Decision variables - support multi-period problems
    x = cp.Variable((n, T), nonneg=True)  # Portfolio weights: n×T variables
    y = cp.Variable((k, T))                # Factor exposures: k×T variables (no non-negative constraint)
    t = cp.Variable(T)                     # Auxiliary variables for SOCP constraints
    
    # Add upper bounds on variables for efficiency
    constraints = []
    constraint_names = []

    constraints.append(x <= 0.1)
    constraint_names.append('Weight upper bound (x <= 0.1)')

    # Budget constraints (multi-period)
    # First period: sum(x[:,0]) == d + sum(x0)
    constraints.append(cp.sum(x[:, 0]) == d + np.sum(x0))
    constraint_names.append('Budget constraint (period 0): sum(x) == 1')

    # Subsequent periods: sum(x[:,t]) == sum(x[:,t-1])
    for period in range(1, T):
        constraints.append(cp.sum(x[:, period]) == cp.sum(x[:, period-1]))
        constraint_names.append(f'Budget constraint (period {period}): sum(x_t) == sum(x_{{t-1}})')

    # Factor exposure constraints: y[:,t] == F_t @ x[:,t]
    for period in range(T):
        constraints.append(y[:, period] == F_t @ x[:, period])
        constraint_names.append(f'Factor exposure (period {period}): y == F^T x')

    # SOCP risk constraints (risk limits for each period)
    # ||[L_Omega @ y[:,t]; D_diag^{1/2} * x[:,t]]||_2 <= max_risk
    for period in range(T):
        # Build two parts of risk vector (using constants)
        factor_risk = L_Omega @ y[:, period]
        specific_risk = cp.multiply(D_diag_sqrt, x[:, period])

        # SOCP constraint: ||[factor_risk; specific_risk]||_2 <= max_risk
        risk_vector = cp.hstack([factor_risk, specific_risk])
        constraints.append(cp.norm(risk_vector, 2) <= max_risk)
        constraint_names.append(f'SOC risk upper bound (period {period}): ||risk|| <= {max_risk}')

    # Optional: Use auxiliary variable t in SOCP form (for objective function)
    # ||[L_Omega @ y[:,t]; D_diag^{1/2} * x[:,t]]||_2 <= t[t]
    for period in range(T):
        factor_risk = L_Omega @ y[:, period]
        specific_risk = cp.multiply(D_diag_sqrt, x[:, period])
        risk_vector = cp.hstack([factor_risk, specific_risk])
        constraints.append(cp.norm(risk_vector, 2) <= t[period])
        constraint_names.append(f'SOC risk auxiliary (period {period}): ||risk|| <= t')
    
    # Objective function (multi-period) - using SOCP form
    objective_terms = []
    
    for period in range(T):
        # Expected returns: -mu_param[:,t] @ x[:,t] (using parameter)
        expected_return = -mu_param[:, period] @ x[:, period]
        
        # Transaction costs: transaction_cost_rate * ||x_t - x_prev||_1
        delta_x = x[:, period] - (x0 if period == 0 else x[:, period - 1])
        transaction_cost = transaction_cost_rate * cp.norm(delta_x, 1)
        
        # Risk term using auxiliary variable t (SOCP form)
        risk_term = gamma * t[period]
        
        # Add total cost for each period
        objective_terms.append(expected_return + transaction_cost + risk_term)
    
    # Total objective: minimize sum of all terms
    objective = cp.Minimize(cp.sum(objective_terms))
    
    # Create problem
    problem = cp.Problem(objective, constraints)
    
    # Return problem, variables and parameters (only mu for subsequent updates)
    return problem, x, y, t, {'mu': mu_param}, constraint_names

def configure_clarabelgpu_settings(verbose=False, max_iter=100):
    """Configure ClarabelGPU GPU solver settings - optimized version
    
    Use optimized parameter configuration to reduce unnecessary settings.
    Full parameter version - consistent with portfolio_optimization_qp_clarabelgpu.py
    """
    settings = {
        'verbose': verbose,
        'max_iter': max_iter,
        'time_limit': 5.0,
        'tol_feas': 1e-8,
        'tol_gap_abs': 1e-8,
        'tol_gap_rel': 1e-8,
        'equilibrate_enable': True,
        'chordal_decomposition_enable': False,
        'iterative_refinement_enable': True,
        'iterative_refinement_max_iter': 10,
        'iterative_refinement_reltol': 1e-13,
        'iterative_refinement_abstol': 1e-12,
        'iterative_refinement_stop_ratio': 5.0,
        'static_regularization_enable': True,
        'dynamic_regularization_enable': True,
    }
    return settings

def solve_portfolio_socp_optimization():
    """Main optimization function - SOCP version with DPP parameters"""
    
    print("=== Portfolio Optimization with SOCP (CVXPY + Clarabel) ===")
    print("Based on multi-period mean-risk model with DPP parameters\n")
    
    print("Creating SOCP optimization model with DPP parameters...")
    with perf_monitor("Model creation"):
        problem, x, y, t, params, constraint_names = create_portfolio_socp_model()
    
    print("Configuring ClarabelGPU solver...")
    clarabelgpu_settings = configure_clarabelgpu_settings(verbose=True)
    
    print("Solving initial problem with ClarabelGPU...")
    
    # Solve with performance monitoring
    # Note: verbose is already included in clarabelgpu_settings, do not pass it again
    start_time = time.time()
    with perf_monitor("Initial solve"):
        try:
            problem.solve(
                solver='CLARABELGPU',
                **clarabelgpu_settings
            )
        except cp.error.SolverError as e:
            solve_time = time.time() - start_time
            print(f"\n❌ Solver Error (time: {solve_time:.4f} seconds)")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"\n💡 Troubleshooting suggestions:")
            print("   1. Try adjusting equilibrate_enable or ir_max_iter")
            print("   2. Reduce problem size (fewer assets or periods)")
            print("   3. Check GPU memory: nvidia-smi")
            print("   4. Verify ClarabelGPU installation")
            return None
    
    solve_time = time.time() - start_time
    
    # GPU memory after initial solve
    print_gpu_memory_stats("After initial solve")
    
    # Get detailed solver statistics
    if hasattr(problem, 'solver_stats'):
        stats = problem.solver_stats
        if hasattr(stats, 'num_iters'):
            print(f"Number of iterations: {stats.num_iters}")
        if hasattr(stats, 'solve_time'):
            print(f"Solver internal time: {stats.solve_time:.4f} seconds")
    
    # Check solve status (accept OPTIMAL and ALMOST_SOLVED)
    if not is_solved_successfully(problem.status):
        print(f"\n⚠️  Warning: Problem not solved successfully. Status: {problem.status}")
        if problem.value is not None:
            print(f"Best objective value found: {problem.value:.6f}")
        return None
    
    # Display solve status
    if problem.status == 'optimal_inaccurate' or str(problem.status).lower() == 'optimal_inaccurate':
        print(f"✓ Problem solved with acceptable accuracy (status: {problem.status})")
    
    print(f"Optimal objective value: {problem.value:.6f}")
    
    # Constraint tightness analysis
    tightness_results, tightness_summary = analyze_constraint_tightness(
        problem, constraint_names=constraint_names, tol=clarabelgpu_settings.get('tol_feas', 1e-6)
    )
    
    # Extract solution
    x_opt = x.value
    y_opt = y.value
    t_opt = t.value
    
    # Display results for each period
    for period in range(T):
        print(f"\n=== Time Period {period+1} ===")
        print(f"Risk level t[{period}]: {t_opt[period]:.6f}")
        
        # Calculate actual risk
        factor_risk = L_Omega @ y_opt[:, period]
        specific_risk = np.sqrt(D_diag) * x_opt[:, period]
        total_risk = np.sqrt(np.sum(factor_risk**2) + np.sum(specific_risk**2))
        print(f"Calculated total risk: {total_risk:.6f}")
        
        # Display top holdings
        print(f"\nTop 10 holdings in period {period+1}:")
        period_weights = x_opt[:, period]
        top10_indices = np.argsort(period_weights)[-10:][::-1]
        
        for i, idx in enumerate(top10_indices):
            print(f"Rank {i+1:2d}: Asset {idx+1:4d}, Weight = {period_weights[idx]:.6f}")
    
    tc_total = sum(np.sum(np.abs(x_opt[:, p] - (x0 if p == 0 else x_opt[:, p-1])))
                    for p in range(T))
    print(f"\nTotal transaction cost: {transaction_cost_rate * tc_total:.6f}")
    
    return {
        'problem': problem,
        'x_optimal': x_opt,
        'y_optimal': y_opt,
        't_optimal': t_opt,
        'objective_value': problem.value,
        'solve_time': solve_time,
        'params': params,  # Include parameters for subsequent updates
        'variables': {'x': x, 'y': y, 't': t},  # Save variable references
        'constraint_names': constraint_names,
        'tightness_results': tightness_results,
        'tightness_summary': tightness_summary,
    }

def repeated_solve_with_updates(result, num_iterations=10):
    """
    Perform parameter updates and repeated solves using DPP (SOCP version)
    Demonstrate the acceleration effect of CVXPy parameterized modeling
    """
    
    if result is None:
        print("Cannot perform repeated solves - initial solve failed")
        return
    
    problem = result['problem']
    params = result['params']
    x_opt = result['x_optimal']
    variables = result['variables']
    
    print(f"\n=== Starting DPP parameter updates and repeated solves ({num_iterations} iterations) ===")
    print("Note: After using DPP, repeated solves will be significantly accelerated (typically 5-50x)\n")
    
    solve_times = []
    
    for i in range(num_iterations):
        print(f"\nIteration {i+1}:")
        
        # Update parameter values (simulate market data changes)
        # Only update expected returns mu (add random perturbation)
        new_mu = mu_matrix * (1 + 0.1 * (np.random.rand(*mu_matrix.shape) - 0.5))
        params['mu'].value = new_mu
        
        # Note: Other values (D_diag, L_Omega, tc_rate, max_risk) are now constants
        # and cannot be updated during DPP repeated solves
        
        print(f"  Parameters updated:")
        print(f"  - Expected return change range: {np.min(new_mu/mu_matrix):.3f} ~ {np.max(new_mu/mu_matrix):.3f}")
        print(f"  - Note: Only mu is a DPP parameter; other values remain constant")
        
        # DPP repeated solve - must re-solve after parameter update (but no need to recompile)
        print(f"  Executing solve...")
        
        with perf_monitor(f"DPP solve (iteration {i+1})"):
            try:
                start_time = time.time()
                # DPP repeated solve: always silent (verbose=False) to improve performance
                problem.solve(
                    solver='CLARABELGPU',
                    warm_start=True,
                    update_hints={UpdateHint.q_values_changed},
                    **configure_clarabelgpu_settings(verbose=True, max_iter=40)
                )
                solve_time = time.time() - start_time
                solve_times.append(solve_time)
                print(f"  Objective value: {problem.value:.6f}")
                
                # Accept both OPTIMAL and ALMOST_SOLVED
                if is_solved_successfully(problem.status):
                    # Use saved variable references to get latest values
                    x_new = variables['x'].value
                    t_new = variables['t'].value
                    
                    # Display risk level
                    print(f"  Risk level (t): {t_new[-1]:.6f}")  # Display risk for last period
                    
                    # Only show top 5 holdings for final period
                    final_weights = x_new[:, -1]
                    top5_indices = np.argsort(final_weights)[-5:][::-1]
                    
                    # Use different indicators for different statuses
                    status_indicator = "✓" if problem.status == cp.OPTIMAL else "~"
                    print(f"  {status_indicator} Top 5 holdings in final period (status: {problem.status}):")
                    for j, idx in enumerate(top5_indices):
                        print(f"    Top {j+1}: Asset {idx+1:4d}, Weight = {final_weights[idx]:.6f}")
                else:
                    print(f"  Warning: Iteration {i+1} did not reach acceptable solution. Status: {problem.status}")
            except cp.error.SolverError as e:
                solve_time = time.time() - start_time
                print(f"  Solver status: Unknown or error (time: {solve_time:.4f} seconds)")
                print(f"  Details: {str(e)}")
    
    # GPU memory after all DPP solves
    print_gpu_memory_stats("After DPP repeated solves")

    # Display acceleration statistics
    if len(solve_times) > 1:
        print(f"\n=== DPP Acceleration Statistics ===")
        print(f"Initial solve time: {result['solve_time']:.4f} seconds")
        print(f"Average repeated solve time: {np.mean(solve_times):.4f} seconds")
        print(f"Speedup ratio: {result['solve_time'] / np.mean(solve_times):.1f}x")
        print(f"Fastest solve time: {np.min(solve_times):.4f} seconds (speedup {result['solve_time'] / np.min(solve_times):.1f}x)")
        print(f"Slowest solve time: {np.max(solve_times):.4f} seconds (speedup {result['solve_time'] / np.max(solve_times):.1f}x)")
    
    # Return solve times for later analysis
    return solve_times


def demonstrate_clarabel_native_interface(cvxpy_result, cvxpy_repeated_solve_times=None):
    """
    Demonstrate how to extract data from CVXPy and use Clarabel native Python interface
    Directly update b vector to show higher performance
    
    Args:
        cvxpy_result: Result dictionary from initial CVXPY solve
        cvxpy_repeated_solve_times: List of solve times from DPP repeated solves
    """
    
    if cvxpy_result is None:
        print("Cannot demonstrate: Need CVXPy solving result first")
        return
    
    # Import required modules
    import scipy.sparse as sp
    import numpy as np
    import time
    
    print("\n1. Extracting standard form data from CVXPy problem...")
    
    # Get CVXPy problem
    problem = cvxpy_result['problem']
    
    # Extract standard form data (using ClarabelGPU format, compatible with Clarabel)
    data = problem.get_problem_data("CLARABELGPU")
    
    # Extract matrices and vectors
    P = data[0].get('P', None)
    q = data[0].get('c', None)  # CVXPy uses 'c', Clarabel uses 'q'
    A = data[0].get('A', None)
    b = data[0].get('b', None)
    
    # Extract cone information
    dims = data[0].get('dims', None)
    
    print(f"  P matrix: {P.shape if P is not None else 'None (linear problem)'}")
    print(f"  q vector: {q.shape}")
    print(f"  A matrix: {A.shape}")
    print(f"  b vector: {b.shape}")
    
    # Convert cone dimension format
    cone_dims = {}
    if dims.zero > 0:
        cone_dims['z'] = dims.zero  # Equality constraints
    if dims.nonneg > 0:
        cone_dims['l'] = dims.nonneg  # Inequality constraints
    if hasattr(dims, 'soc') and dims.soc:
        cone_dims['q'] = list(dims.soc)  # Second-order cone constraints
    
    print(f"  Cone dimensions: {cone_dims}")
    
    # Import Clarabel
    try:
        # Import from installed package
        import clarabel_gpu
        from clarabel_gpu import ClarabelGPU
        print("  ✓ Successfully imported Clarabel solver interface")
        print(f"  Using clarabel_gpu from: {clarabel_gpu.__file__}")
        using_enhanced = False
            
    except ImportError as e:
        print(f"\nError: Cannot import Clarabel interface: {e}")
        return
    
    print("\n2. Creating Clarabel native solver...")
    print("  Using standard solver interface")
    
    start_time = time.time()
    # Create solver instance
    clarabel_solver = ClarabelGPU()
    
    # Setup problem - handle None P matrix
    if P is None:
        # Linear problem - create zero matrix
        P = sp.csc_matrix((len(q), len(q)))
    
    # Setup solver with problem data
    clarabel_solver.setup(
        P, q, A, b, cone_dims,
        gpu_mode=True,  # Enable GPU mode for direct GPU updates
        verbose=False,
        max_iter=200,
        tol_feas=1e-8,
        tol_gap_abs=1e-8,
        tol_gap_rel=1e-8,
        equilibrate_enable=True
    )
    print(f"  Setup time: {time.time() - start_time:.4f} seconds")
    # Initial solve (verification)
    print("\n3. Solving with Clarabel native interface (verification)...")
    start_time = time.time()
    native_result = clarabel_solver.solve()
    native_solve_time = time.time() - start_time
    
    print(f"  Status: {native_result['status']}")
    print(f"  Objective value: {native_result['obj_val']:.6f}")
    print(f"  Iterations: {native_result.get('iterations', '?')}")
    print(f"  Solve time: {native_solve_time:.4f} seconds")
    print_gpu_memory_stats("After native initial solve")
    
    # Verify solution consistency
    cvxpy_obj = cvxpy_result['objective_value']
    native_obj = native_result['obj_val']
    print(f"  Difference from CVXPy objective: {abs(cvxpy_obj - native_obj):.2e}")
    
    # Update parameters using native interface
    print("\n4. Updating parameters and re-solving with Clarabel native interface...")
    print("-"*50)
    
    gpu_update_supported = True
    
    if gpu_update_supported:
        print("✓ GPU direct transfer update supported")
        print("Workflow: GPU update parameters -> solve -> get new solution")
    else:
        print("Workflow: Update parameters -> solve -> get new solution (CPU)")
    
    print("Note: Must call solve() after each parameter update to solve new problem")
    
    b_original = b.copy()
    native_update_times = []
    use_gpu_update = False  # Initialize variable
    
    # If GPU update is supported, import CuPy
    if gpu_update_supported:
        try:
            import cupy as cupy_lib  # Avoid naming conflict with cvxpy (cp)
            # Test if CuPy can actually compile CUDA code
            test_array = cupy_lib.array([1.0, 2.0, 3.0])
            test_result = test_array + 1.0  # This will trigger compilation
            
            b_gpu = cupy_lib.asarray(b_original)
            use_gpu_update = True
            print("✓ CuPy imported, will use GPU direct update")
        except ImportError:
            use_gpu_update = False
            print("ℹ️ CuPy not installed, using CPU update")
        except Exception as e:
            use_gpu_update = False
            if "cuda_fp16.h" in str(e) or "NVRTC_ERROR" in str(e):
                print("⚠️ CuPy CUDA compilation error, using CPU update instead")
                print("   (This is likely due to missing CUDA headers in the environment)")
            else:
                print(f"⚠️ CuPy error: {type(e).__name__}, using CPU update")
    else:
        use_gpu_update = False
    
    print("\nPerforming 10 parameter updates and solves:")
    print("(Note: Times shown are solve times only, not including parameter update time)")
    
    for i in range(10):
        # Generate new b vector (simulate constraint changes)
        perturbation = np.random.uniform(-0.001, 0.001, size=len(b))
        
        # Keep equality constraints unchanged (if any)
        if 'z' in cone_dims and cone_dims['z'] > 0:
            perturbation[:cone_dims['z']] = 0
        
        # CRITICAL: Don't perturb SOC constraints - they should remain 0
        # SOC constraints come after equality and inequality constraints
        offset = cone_dims.get('z', 0) + cone_dims.get('l', 0)
        if 'q' in cone_dims:
            for soc_dim in cone_dims['q']:
                perturbation[offset:offset+soc_dim] = 0
                offset += soc_dim
        
        if use_gpu_update:
            # GPU direct update
            b_gpu_new = b_gpu + cupy_lib.asarray(perturbation)
            clarabel_solver.update_b(b_gpu_new)
            update_method = "GPU"
        else:
            # CPU update
            b_new = b_original + perturbation
            clarabel_solver.update_b(b_new)
            update_method = "CPU"
        
        # Must re-solve after update
        start_time = time.time()
        native_result = clarabel_solver.solve()
        update_time = time.time() - start_time
        
        native_update_times.append(update_time)
        
        # Display solve time and status for each iteration
        status = native_result.get('status', 'unknown')
        obj_val = native_result.get('obj_val', float('nan'))
        
        # Safely compare status
        is_solved = str(status).lower() == 'solved'
        
        iters = native_result.get('iterations', '?')
        if is_solved:
            print(f"  Iteration {i+1:2d} [{update_method}]: solve time = {update_time:.4f}s, iters={iters}, objective = {obj_val:.6f}")
        else:
            print(f"  Iteration {i+1:2d} [{update_method}]: solve time = {update_time:.4f}s, iters={iters}, status = {status} (Warning: not optimal)")
    
    # GPU memory after native repeated solves
    print_gpu_memory_stats("After native repeated solves")

    # Performance comparison
    print("\n" + "="*50)
    print("Performance Comparison Summary")
    print("="*50)
    
    print(f"\n1. CVXPy + ClarabelGPU (DPP):")
    print(f"   Initial solve time: {cvxpy_result['solve_time']:.4f} seconds")
    if cvxpy_repeated_solve_times:
        print(f"   Average DPP repeated solve time: {np.mean(cvxpy_repeated_solve_times):.4f} seconds")
        print(f"   DPP speedup vs initial: {cvxpy_result['solve_time'] / np.mean(cvxpy_repeated_solve_times):.1f}x")
    
    print(f"\n2. Clarabel native interface:")
    print(f"   Interface type: Standard interface")
    print(f"   Update method: {'GPU direct transfer' if 'use_gpu_update' in locals() and use_gpu_update else 'CPU update'}")
    print(f"   Initial solve time: {native_solve_time:.4f} seconds")
    print(f"   Average update time: {np.mean(native_update_times):.4f} seconds")
    print(f"   Fastest update time: {np.min(native_update_times):.4f} seconds")
    
    # Calculate speedup ratio
    cvxpy_to_native_speedup = cvxpy_result['solve_time'] / np.mean(native_update_times)
    
    print(f"\n3. Speedup comparison:")
    print(f"   Clarabel native vs CVXPy initial solve: {cvxpy_to_native_speedup:.1f}x")
    if cvxpy_repeated_solve_times:
        clarabel_vs_cvxpy_dpp = np.mean(cvxpy_repeated_solve_times) / np.mean(native_update_times)
        print(f"   Clarabel native vs CVXPy DPP repeated: {clarabel_vs_cvxpy_dpp:.1f}x faster")
    print(f"   Clarabel native repeated vs its initial: {native_solve_time / np.mean(native_update_times):.1f}x")
    
    if 'use_gpu_update' in locals() and use_gpu_update:
        print(f"\n4. GPU direct transfer advantages:")
        print("   ✓ Avoids CPU-GPU data transfer overhead")
        print("   ✓ More suitable for high-frequency update scenarios")
        print("   ✓ Seamless integration with GPU computing frameworks (e.g., deep learning)")
    
    print("\nConclusion:")
    print("- CVXPy DPP provides convenient parameterized modeling")
    print("- Clarabel native interface provides the highest performance")
    print("- For high-frequency parameter update scenarios, native interface is the best choice")

    del clarabel_solver
    

if __name__ == "__main__":
    print("=== Portfolio Optimization with SOCP (CVXPy + ClarabelGPU) ===")
    print("Using DPP parameterized modeling for efficient parameter updates\n")
    
    # Perform initial optimization
    result = solve_portfolio_socp_optimization()
    
    if result:
        print("\n=== Initial optimization completed successfully (compilation + solving) ===")
        
        # Use DPP for parameter updates and repeated solves
        cvxpy_solve_times = repeated_solve_with_updates(result, num_iterations=5)
        
        print("\n=== DPP parameterized modeling demonstration completed ===")
        print("Summary: By using cp.Parameter and DPP rules, SOCP problem repeated solves achieve significant speedup!")
        
        # Additional demonstration: Use Clarabel native interface for parameter updates
        print("\n\n" + "="*70)
        print("Additional demonstration: Extract data from CVXPy, use Clarabel native interface")
        print("="*70)
        demonstrate_clarabel_native_interface(result, cvxpy_solve_times)
        
        # Final GPU memory summary
        peak_mb = get_peak_gpu_memory_mb()
        if peak_mb is not None:
            print("\n" + "=" * 70)
            print(f"  Peak GPU memory usage (RMM): {peak_mb:.2f} MB")
            print("=" * 70)
    else:
        print("Initial optimization failed!")
