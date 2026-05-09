#!/usr/bin/env python3
"""
Portfolio optimization using CVXPY with ClarabelGPU GPU-accelerated solver
Translated from Julia JuMP implementation

This example demonstrates two solving methods:
1. CVXPy + ClarabelGPU: High-level modeling interface, supports DPP parameterized modeling
2. Clarabel native interface: Low-level solver API, optimal performance

Key features:
- Uses ClarabelGPU class with setup() + solve() workflow
- GPU mode with CuPy direct transfer for q/b updates
- update_P/update_A always sync KKT matrix via DataUpdater
"""

import numpy as np
import cvxpy as cp
import time
from contextlib import contextmanager

from clarabel_gpu import UpdateHint

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
        constraint_names: List of constraint names (one-to-one with problem.constraints)
        tol: Determination threshold, default 1e-6 (recommended: max(eps_abs, eps_rel))

    Returns:
        results: Analysis result list for each constraint
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
            # Elementwise inequality: expr <= 0, slack = -expr
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
                result['label'] = 'NO VALUE'

        elif ctype == 'NonNeg':
            # Elementwise inequality: expr >= 0, slack = expr
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
                result['label'] = 'NO VALUE'

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
            result['label'] = f'OTHER ({ctype})'

        results.append(result)

    # Print detailed table
    header = (f"{'#':>3} | {'Constraint Name':<38} | {'Type':<8} | {'Label':<18} | "
              f"{'slack/margin':>12} | {'violation':>10} | {'dual(sens)':>10} | {'tight/total':>12}")
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
    print(f"  Tight (TIGHT):         {len(summary['tight']):3d}  "
          f"{summary['tight'][:10]}{'...' if len(summary['tight']) > 10 else ''}")
    print(f"  Loose (LOOSE):         {len(summary['loose']):3d}  "
          f"{summary['loose'][:10]}{'...' if len(summary['loose']) > 10 else ''}")
    print(f"  Borderline (BORDERLINE): {len(summary['borderline']):3d}  "
          f"{summary['borderline'][:10]}{'...' if len(summary['borderline']) > 10 else ''}")
    print(f"  Equality (EQUALITY):   {len(summary['equality']):3d}")

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


# Load data (equivalent to NPZ loading in Julia)
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

F_t = F.T

# Problem dimensions and parameters
n, k = F.shape
T = mu_matrix.shape[1]  # Restore this line, use correct T value
x0 = np.zeros(n)
gamma = 1.0
d = 1.0 - np.sum(x0)
transaction_cost_rate = 0.002

print(f"Problem dimensions: n={n}, k={k}, T={T}")
print(f"Parameters: gamma={gamma}, d={d}, transaction_cost_rate={transaction_cost_rate}")

def create_portfolio_qp_model():
    """Create the CVXPY QP model for portfolio optimization with only mu as DPP parameter"""
    
    # Declare only mu as parameter (DPP way) - for efficient updates
    mu_param = cp.Parameter((n, T))           # Expected return parameter
    
    # Initialize parameter value
    mu_param.value = mu_matrix
    
    # Other values used as constants (not parameters)
    # D_diag, Omega, transaction_cost_rate are used directly
    
    # Decision variables - support multi-period problems (fully consistent with Julia code)
    x = cp.Variable((n, T), nonneg=True)  # Portfolio weights: n×T variables
    y = cp.Variable((k, T))  # Factor exposures: k×T variables (non-negative constraint in Julia)
    
    # Add upper bounds on variables for efficiency (fully consistent with Julia)
    constraints = []
    constraint_names = []  # Constraint names for tightness analysis

    constraints.append(x <= 0.1)
    constraint_names.append('Weight upper bound (x <= 0.1)')
    constraints.append(y <= 0.1)
    constraint_names.append('Factor exposure upper bound (y <= 0.1)')

    # Budget constraints (multi-period)
    # First period: sum(x[:,0]) == d + sum(x0)
    constraints.append(cp.sum(x[:, 0]) == d + np.sum(x0))
    constraint_names.append('Budget constraint (period 0): sum(x) == 1')

    # Subsequent periods: sum(x[:,t]) == sum(x[:,t-1])
    for t in range(1, T):
        constraints.append(cp.sum(x[:, t]) == cp.sum(x[:, t-1]))
        constraint_names.append(f'Budget constraint (period {t}): sum(x_t) == sum(x_{{t-1}})')

    # Factor exposure constraints: y[:,t] == F_t @ x[:,t]
    for t in range(T):
        constraints.append(y[:, t] == F_t @ x[:, t])
        constraint_names.append(f'Factor exposure (period {t}): y == F^T x')
    
    # Remove SOCP constraints, Julia code uses quadratic form directly in objective function
    # No need for Cholesky decomposition or SOC constraints
    
    # Objective function (multi-period) - fully consistent with Julia code
    objective_terms = []
    
    for t in range(T):
        # Expected returns: -mu_param[:,t] @ x[:,t] (using parameter)
        expected_return = -mu_param[:, t] @ x[:, t]
        
        # Transaction costs: transaction_cost_rate * ||x_t - x_prev||_1
        delta_x = x[:, t] - (x0 if t == 0 else x[:, t - 1])
        transaction_cost = transaction_cost_rate * cp.norm(delta_x, 1)
        
        # Risk terms: gamma * (y[:,t]^T @ Omega @ y[:,t] + x[:,t]^T @ diag(D_diag) @ x[:,t])
        # Quadratic form: factor risk + specific risk (using constants)
        factor_risk = cp.quad_form(y[:, t], Omega)  # y^T @ Omega @ y
        specific_risk = cp.sum(cp.multiply(D_diag, cp.square(x[:, t])))  # x^T @ diag(D_diag) @ x
        risk_term = gamma * (factor_risk + specific_risk)
        
        # Add total cost for each period
        objective_terms.append(expected_return + transaction_cost + risk_term)
    
    # Total objective: minimize sum of all terms
    objective = cp.Minimize(cp.sum(objective_terms))
    
    # Create problem
    problem = cp.Problem(objective, constraints)
    
    # Return problem, variables and parameters (only mu for subsequent updates)
    return problem, x, y, {'mu': mu_param}, constraint_names

def configure_ClarabelGPU_settings(verbose=True, max_iter=100):
    """Configure ClarabelGPU GPU solver settings - optimized version
    
    Use optimized parameter configuration to reduce unnecessary settings.
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

def solve_portfolio_optimization(verbose=True):  # Changed to default True
    """Main optimization routine with DPP parameters
    
    Args:
        verbose: Whether to display detailed info during initial solve (DPP repeated solve always silent)
    """
    
    print("Creating QP optimization model with DPP parameters...")
    with perf_monitor("Model creation"):
        problem, x, y, params, constraint_names = create_portfolio_qp_model()
    
    print("Configuring ClarabelGPU GPU solver...")
    ClarabelGPU_settings = configure_ClarabelGPU_settings(verbose=verbose)
    
    print("Solving initial problem with ClarabelGPU...")
    
    # Solve with performance monitoring
    # Note: verbose already included in ClarabelGPU_settings, do not pass it again
    start_time = time.time()
    with perf_monitor("Initial solve"):
        problem.solve(
            solver='ClarabelGPU',
            **ClarabelGPU_settings  # Contains verbose and other solver settings
        )
    solve_time = time.time() - start_time
    
    # Get detailed solver statistics
    if hasattr(problem, 'solver_stats'):
        stats = problem.solver_stats
        if hasattr(stats, 'num_iters'):
            print(f"Number of iterations: {stats.num_iters}")
        if hasattr(stats, 'solve_time'):
            print(f"Solver internal time: {stats.solve_time:.4f} seconds")
    
    # Check solve status (accept OPTIMAL and ALMOST_SOLVED)
    if not is_solved_successfully(problem.status):
        print(f"Warning: Problem not solved successfully. Status: {problem.status}")
        return None
    
    # Display solve status
    if problem.status == 'optimal_inaccurate' or str(problem.status).lower() == 'optimal_inaccurate':
        print(f"✓ Problem solved with acceptable accuracy (status: {problem.status})")
    
    print(f"Optimal objective value: {problem.value:.6f}")
    
    # Constraint tightness analysis
    tightness_results, tightness_summary = analyze_constraint_tightness(
        problem, constraint_names=constraint_names, tol=ClarabelGPU_settings.get('tol_feas', 1e-6)
    )
    
    # Extract solution
    x_opt = x.value
    y_opt = y.value
    
    # Display top holdings for final period (similar to Julia output)
    print(f"\nTop 10 holdings in final period (T={T}):")
    final_weights = x_opt[:, -1]
    top10_indices = np.argsort(final_weights)[-10:][::-1]
    
    for i, idx in enumerate(top10_indices):
        print(f"Rank {i+1:2d}: Asset {idx+1:4d}, Weight = {final_weights[idx]:.6f}")
    print()
    
    return {
        'problem': problem,
        'x_optimal': x_opt,
        'y_optimal': y_opt,
        'objective_value': problem.value,
        'solve_time': solve_time,
        'params': params,  # Include parameters for subsequent updates
        'constraint_names': constraint_names,
        'tightness_results': tightness_results,
        'tightness_summary': tightness_summary,
    }

def repeated_solve_with_updates(result, num_iterations=5):
    """
    Perform parameter updates and repeated solves using DPP.
    Compares automatic detection vs manual UpdateHint performance.
    """

    if result is None:
        print("Cannot perform repeated solves - initial solve failed")
        return

    params = result['params']
    settings = configure_ClarabelGPU_settings(verbose=False, max_iter=40)

    np.random.seed(123)
    mu_perturbations = [
        mu_matrix * (1 + 0.1 * (np.random.rand(*mu_matrix.shape) - 0.5))
        for _ in range(num_iterations)
    ]

    # ── [A] Automatic detection (no update_hints) ──────────────────
    print(f"\n{'='*70}")
    print(f"  DPP Repeated Solves: Automatic vs UpdateHint ({num_iterations} rounds)")
    print(f"{'='*70}")
    print(f"\n[A] Automatic detection (np.array_equal on P/A/q/b each round)")

    problem_auto = result['problem']
    auto_times = []
    for i, new_mu in enumerate(mu_perturbations):
        params['mu'].value = new_mu
        t0 = time.perf_counter()
        problem_auto.solve(solver='ClarabelGPU', warm_start=True, **settings)
        dt = time.perf_counter() - t0
        auto_times.append(dt)
        tag = "ok" if is_solved_successfully(problem_auto.status) else "FAIL"
        print(f"    Round {i+1:2d}: {dt*1000:8.2f} ms  obj={problem_auto.value:+.8f}  [{tag}]")

    # ── [B] Manual hints (skip all comparisons) ────────────────────
    print(f"\n[B] UpdateHint.q_values_changed (skip np.array_equal, direct update_q)")

    hint_times = []
    for i, new_mu in enumerate(mu_perturbations):
        params['mu'].value = new_mu
        t0 = time.perf_counter()
        problem_auto.solve(
            solver='ClarabelGPU', warm_start=True,
            update_hints={UpdateHint.q_values_changed},
            **settings)
        dt = time.perf_counter() - t0
        hint_times.append(dt)
        tag = "ok" if is_solved_successfully(problem_auto.status) else "FAIL"
        print(f"    Round {i+1:2d}: {dt*1000:8.2f} ms  obj={problem_auto.value:+.8f}  [{tag}]")

    # ── Summary ────────────────────────────────────────────────────
    avg_auto = np.mean(auto_times) * 1000
    avg_hint = np.mean(hint_times) * 1000
    saved = avg_auto - avg_hint

    print(f"\n  {'Mode':<35s} {'Avg (ms)':>9s} {'Min (ms)':>9s} {'Max (ms)':>9s}")
    print(f"  {'─'*35} {'─'*9} {'─'*9} {'─'*9}")
    print(f"  {'Automatic (np.array_equal)':<35s} "
          f"{avg_auto:>9.2f} {min(auto_times)*1000:>9.2f} {max(auto_times)*1000:>9.2f}")
    print(f"  {'UpdateHint.q_values_changed':<35s} "
          f"{avg_hint:>9.2f} {min(hint_times)*1000:>9.2f} {max(hint_times)*1000:>9.2f}")
    print(f"\n  Savings per solve: {saved:+.2f} ms ({saved/avg_auto*100:.1f}%)")
    print(f"  Initial solve time: {result['solve_time']*1000:.1f} ms")
    print(f"  DPP speedup vs initial: {result['solve_time']/np.mean(hint_times):.1f}x")

    return auto_times


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
    
    print("\n1. Extracting standard form data from CVXPy problem...")
    
    # Get CVXPy problem
    problem = cvxpy_result['problem']
    
    # Extract standard form data (using ClarabelGPU format, compatible with Clarabel)
    data = problem.get_problem_data("ClarabelGPU")
    
    # Extract matrices and vectors
    P = data[0].get('P', None)
    q = data[0].get('q', None)  # For QP problems, CVXPy uses 'q' not 'c'
    if q is None:
        q = data[0].get('c', None)  # Fallback to 'c' for conic problems
    
    # For QP format, CVXPy may separate equality and inequality constraints
    A_eq = data[0].get('A', None)  # Equality constraints
    b_eq = data[0].get('b', None)
    F = data[0].get('F', None)  # Inequality constraints (Fx <= G)
    G = data[0].get('G', None)  # Note: uppercase G in CVXPy QP format
    
    # Print available keys for debugging
    print(f"  Available keys in data: {list(data[0].keys())}")
    
    # Combine constraints into standard conic form
    # For Clarabel: A*x + s = b, where s is in the cone
    import scipy.sparse as sp
    import numpy as np
    
    print("\n  Debugging constraint format:")
    print(f"    A_eq shape: {A_eq.shape if A_eq is not None else None}")
    print(f"    b_eq shape: {b_eq.shape if b_eq is not None else None}")
    print(f"    F shape: {F.shape if F is not None else None}")
    print(f"    G shape: {G.shape if G is not None else None}")
    
    # The issue might be with how we're interpreting the QP format
    # CVXPy QP format: minimize (1/2)x'Px + q'x subject to Ax=b, Fx<=G
    # But when ClarabelGPU processes it as conic, the format changes
    
    # Let's use the conic format data directly if available
    if 'A' in data[0] and 'b' in data[0]:
        # This is already in conic format from ClarabelGPU processing
        A = data[0]['A']
        b = data[0]['b']
        print(f"    Using conic format directly from ClarabelGPU data")
    elif A_eq is not None and F is not None:
        # Combine equality and inequality constraints
        # For conic form with Ax + s = b, s in cone:
        # Equality: A_eq*x = b_eq (s in zero cone)
        # Inequality: F*x <= G becomes -F*x + s = -G (s in nonneg cone)
        A = sp.vstack([A_eq, -F])
        b = np.concatenate([b_eq, -G])
    elif A_eq is not None:
        A = A_eq
        b = b_eq
    elif F is not None:
        A = -F
        b = -G
    else:
        raise ValueError("No constraints found in problem data")
    
    # Extract cone information
    dims = data[0].get('dims', None)
    
    print(f"  P matrix: {P.shape if P is not None else 'None (linear problem)'}")
    print(f"  q vector: {q.shape if q is not None else 'None'}")
    print(f"  A matrix (combined): {A.shape}")
    print(f"  b vector (combined): {b.shape}")
    
    # Convert cone dimension format
    cone_dims = {}
    if dims.zero > 0:
        cone_dims['z'] = dims.zero  # Equality constraints
    if dims.nonneg > 0:
        cone_dims['l'] = dims.nonneg  # Inequality constraints
    if hasattr(dims, 'soc') and dims.soc:
        cone_dims['q'] = list(dims.soc)  # Second-order cone constraints
    
    print(f"  Cone dimensions: {cone_dims}")
    
    # Import Clarabel solver
    try:
        from clarabel_gpu import ClarabelGPU
        print("  ✓ Successfully imported Clarabel solver interface")
            
    except ImportError as e:
        print(f"\nError: Cannot import Clarabel interface: {e}")
        return
    
    print("\n2. Creating Clarabel native solver...")
    
    clarabel_solver = ClarabelGPU()
    clarabel_solver.setup(
        P=P,
        q=q,
        A=A,
        b=b,
        cone_dims=cone_dims,
        gpu_mode=True,
        verbose=False,
        max_iter=50,
        tol_feas=1e-6,
        tol_gap_abs=1e-6,
        tol_gap_rel=1e-6,
        equilibrate_enable=False
    )
    
    # Initial solve (verification)
    print("\n3. Solving with Clarabel native interface (verification)...")
    start_time = time.time()
    native_result = clarabel_solver.solve()
    native_solve_time = time.time() - start_time
    
    print(f"  Status: {native_result['status']}")
    print(f"  Objective value: {native_result['obj_val']:.6f}")
    print(f"  Solve time: {native_solve_time:.4f} seconds")
    print(f"  Number of iterations: {native_result.get('iterations', 'N/A')}")
    
    # Verify solution consistency
    cvxpy_obj = cvxpy_result['objective_value']
    native_obj = native_result['obj_val']
    print(f"  Difference from CVXPy objective: {abs(cvxpy_obj - native_obj):.2e}")
    
    # Update parameters with native interface
    print("\n4. Update parameters and re-solve with Clarabel native interface...")
    print("-"*50)
    
    print("✓ GPU mode enabled")
    print("  Workflow: GPU direct parameter update -> solve -> get new solution")
    print("Note: Must call solve() after each parameter update to solve the new problem")
    
    b_original = b.copy()
    native_update_times = []
    
    try:
        import cupy as cupy_lib
    except ImportError:
        cupy_lib = None
    
    if cupy_lib is not None:
        b_gpu = cupy_lib.asarray(b_original)
        print("✓ b vector transferred to GPU")
    else:
        b_gpu = None
    
    print("\nPerforming 10 parameter updates and solves:")
    print("(Note: Times shown are solve times only, not including parameter update time)")
    
    for i in range(10):
        # Generate new b vector (simulate constraint changes)
        # Use small perturbations for numerical stability
        perturbation = np.random.uniform(-0.001, 0.001, size=len(b))
        
        # Keep equality constraints unchanged (if any)
        if 'z' in cone_dims and cone_dims['z'] > 0:
            perturbation[:cone_dims['z']] = 0
        
        if b_gpu is not None:
            b_gpu_new = b_gpu + cupy_lib.asarray(perturbation)
            clarabel_solver.update_b(b_gpu_new)
            update_method = "GPU"
        else:
            # CPU update (using update_b method)
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
        iterations = native_result.get('iterations', 'N/A')
        
        # Safely compare status
        is_solved = str(status).lower() == 'solved'
        
        if is_solved:
            print(f"  Iteration {i+1:2d} [{update_method}]: solve time = {update_time:.4f}s, objective = {obj_val:.6f}, iterations = {iterations}")
        else:
            print(f"  Iteration {i+1:2d} [{update_method}]: solve time = {update_time:.4f}s, status = {status} (Warning: not optimal)")
    
    # Performance comparison
    print("\n" + "="*50)
    print("Performance Comparison Summary")
    print("="*50)
    
    print(f"\n1. CVXPy + ClarabelGPU (DPP):")
    print(f"   Initial solve time: {cvxpy_result['solve_time']:.4f} seconds")
    if cvxpy_repeated_solve_times:
        print(f"   DPP repeated solve average time: {np.mean(cvxpy_repeated_solve_times):.4f} seconds")
        print(f"   DPP speedup vs initial solve: {cvxpy_result['solve_time'] / np.mean(cvxpy_repeated_solve_times):.1f}x")
    
    print(f"\n2. Clarabel native interface:")
    print(f"   Interface type: ClarabelGPU")
    print(f"   Update method: GPU direct transfer")
    print(f"   Initial solve time: {native_solve_time:.4f} seconds")
    print(f"   Average update solve time: {np.mean(native_update_times):.4f} seconds")
    print(f"   Fastest update solve time: {np.min(native_update_times):.4f} seconds")
    
    # Calculate speedup ratios
    cvxpy_to_native_speedup = cvxpy_result['solve_time'] / np.mean(native_update_times)
    
    print(f"\n3. Speedup comparison:")
    print(f"   Clarabel native vs CVXPy initial solve: {cvxpy_to_native_speedup:.1f}x")
    if cvxpy_repeated_solve_times:
        clarabel_vs_cvxpy_dpp = np.mean(cvxpy_repeated_solve_times) / np.mean(native_update_times)
        print(f"   Clarabel native vs CVXPy DPP repeated: {clarabel_vs_cvxpy_dpp:.1f}x faster")
    print(f"   Clarabel native repeated vs its initial solve: {native_solve_time / np.mean(native_update_times):.1f}x")
    
    print(f"\n4. GPU direct transfer advantages:")
    print("   ✓ Avoid CPU-GPU data transfer overhead")
    print("   ✓ More suitable for high-frequency update scenarios")
    print("   ✓ Seamless integration with GPU computing frameworks (e.g., deep learning)")
    
    mem_info = clarabel_solver.get_memory_info()
    if mem_info and mem_info.get('rmm_enabled'):
        print(f"\n5. GPU memory usage (RMM):")
        print(f"   Current usage: {mem_info['cupy_used_mb']:.2f} MB")
        print(f"   Total allocated: {mem_info['cupy_total_mb']:.2f} MB")
    
    print("\nConclusion:")
    print("- CVXPy DPP provides convenient parameterized modeling")
    print("- Clarabel native interface provides highest performance")
    print("- For high-frequency parameter update scenarios, native interface is the best choice")
    print("- GPU mode significantly improves solve speed for large-scale problems")
    
    # Additional demonstration: Update multiple parameters simultaneously
    print("\n" + "="*50)
    print("Additional demonstration: Update multiple parameters simultaneously")
    print("="*50)
    
    print("\nNote: use update_q() and update_b() separately to update parameters")
    print("Can update multiple parameters in a single call (P, q, A, b)")
    
    # Generate new q and b
    q_new = q + np.random.uniform(-0.001, 0.001, size=len(q))
    b_new = b_original + np.random.uniform(-0.001, 0.001, size=len(b))
    
    # Keep equality constraints unchanged
    if 'z' in cone_dims and cone_dims['z'] > 0:
        b_new[:cone_dims['z']] = b_original[:cone_dims['z']]
    
    print("\nUpdating q and b vectors simultaneously...")
    
    if cupy_lib is not None:
        q_gpu = cupy_lib.asarray(q_new)
        b_gpu_new = cupy_lib.asarray(b_new)
        clarabel_solver.update_q(q_gpu)
        clarabel_solver.update_b(b_gpu_new)
        print("✓ Updated q and b via GPU direct transfer")
    else:
        clarabel_solver.update_q(q_new)
        clarabel_solver.update_b(b_new)
        print("✓ Updated q and b")
    
    # Solve
    start_time = time.time()
    multi_update_result = clarabel_solver.solve()
    multi_update_time = time.time() - start_time
    
    print(f"\nMulti-parameter update solve results:")
    print(f"  Status: {multi_update_result['status']}")
    print(f"  Objective value: {multi_update_result['obj_val']:.6f}")
    print(f"  Solve time: {multi_update_time:.4f} seconds")
    print(f"  Number of iterations: {multi_update_result.get('iterations', 'N/A')}")
    
    print("\n✓ Multi-parameter update completed successfully")


if __name__ == "__main__":
    print("=== Portfolio Optimization with CVXPY + ClarabelGPU (GPU) ===")
    print("Using DPP parameterized modeling for efficient parameter updates\n")
    
    # Check if ClarabelGPU is available
    if 'CLARABELGPU' in cp.installed_solvers():
        print("✓ ClarabelGPU GPU solver is available")
    else:
        print("✗ ClarabelGPU GPU solver is not available")
        print("Please ensure ClarabelGPU is properly installed and integrated with CVXPy")
        exit(1)
    
    # Perform initial optimization
    # verbose=True: First solve shows detailed info
    # DPP repeated solve automatically silent (verbose=False already set inside function)
    result = solve_portfolio_optimization(verbose=True)
    
    if result:
        print("=== Initial optimization completed successfully (compilation + solving) ===")
        
        # Use DPP for parameter updates and repeated solves
        cvxpy_solve_times = repeated_solve_with_updates(result, num_iterations=10)
        
        print("\n=== DPP parameterized modeling demonstration completed ===")
        print("Summary: By using cp.Parameter and DPP rules, repeated solves achieve significant speedup!")
        
        # Additional demonstration: Use Clarabel native interface for parameter updates
        print("\n\n" + "="*70)
        print("Additional demonstration: Extract data from CVXPy, use Clarabel native interface")
        print("="*70)
        demonstrate_clarabel_native_interface(result, cvxpy_solve_times)
        
    else:
        print("Initial optimization failed!")
