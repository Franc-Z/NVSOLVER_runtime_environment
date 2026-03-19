#!/usr/bin/env python3
"""
Portfolio optimization using Power Cones — MOSEK vs CLARABELGPU benchmark.

Replaces the SOC ||risk||_2 <= t constraint with a p-norm constraint (p != 2),
which CVXPY decomposes into Power Cone constraints internally.

Usage:
  conda activate cufolio
  python3 portfolio_optimization_powcone_bench.py [--n=5000] [--p=1.5] [--repeats=5]
"""

import numpy as np
import cvxpy as cp
import time
import sys


def parse_args():
    args = {'n_assets': 5000, 'p_norm': 1.5, 'repeats': 5, 'k_factors': 50}
    for a in sys.argv[1:]:
        if a.startswith('--n='):
            args['n_assets'] = int(a.split('=')[1])
        elif a.startswith('--p='):
            args['p_norm'] = float(a.split('=')[1])
        elif a.startswith('--repeats='):
            args['repeats'] = int(a.split('=')[1])
        elif a.startswith('--k='):
            args['k_factors'] = int(a.split('=')[1])
    return args


def generate_data(n, k, T=1, seed=42):
    np.random.seed(seed)
    D_diag = np.random.rand(n) * np.sqrt(k)
    F = np.random.randn(n, k) * (np.random.rand(n, k) < 0.5)
    Omega_temp = np.random.randn(k, k)
    Omega = (Omega_temp @ Omega_temp.T) / k
    L_Omega = np.linalg.cholesky(Omega)
    mu_matrix = (3.0 + 9.0 * np.random.rand(n, T)) / 100.0
    return {
        'n': n, 'k': k, 'T': T,
        'D_diag': D_diag, 'F': F, 'L_Omega': L_Omega,
        'mu_matrix': mu_matrix, 'F_t': F.T,
    }


def create_powcone_model(data, p_norm, gamma=1.0, tc_rate=0.002, max_risk=0.15):
    """Create portfolio model with p-norm risk constraint (Power Cone formulation)."""
    n, k, T = data['n'], data['k'], data['T']
    L_Omega = data['L_Omega']
    D_diag_sqrt = np.sqrt(data['D_diag'])
    F_t = data['F_t']

    mu_param = cp.Parameter((n, T))
    mu_param.value = data['mu_matrix']

    x = cp.Variable((n, T), nonneg=True)
    y = cp.Variable((k, T))
    z = cp.Variable((n, T))
    t_risk = cp.Variable(T)

    x0 = np.zeros(n)
    constraints = []

    constraints.append(x <= 0.1)
    constraints.append(z[:, 0] >= x[:, 0] - x0)
    constraints.append(z[:, 0] >= x0 - x[:, 0])

    for period in range(1, T):
        constraints.append(z[:, period] >= x[:, period] - x[:, period - 1])
        constraints.append(z[:, period] >= x[:, period - 1] - x[:, period])

    constraints.append(cp.sum(x[:, 0]) == 1.0)
    for period in range(1, T):
        constraints.append(cp.sum(x[:, period]) == cp.sum(x[:, period - 1]))

    for period in range(T):
        constraints.append(y[:, period] == F_t @ x[:, period])

    # p-norm risk constraint: ||risk||_p <= max_risk  (Power Cone)
    for period in range(T):
        factor_risk = L_Omega @ y[:, period]
        specific_risk = cp.multiply(D_diag_sqrt, x[:, period])
        risk_vector = cp.hstack([factor_risk, specific_risk])
        constraints.append(cp.pnorm(risk_vector, p_norm) <= max_risk)

    # p-norm in objective via auxiliary: ||risk||_p <= t_risk
    for period in range(T):
        factor_risk = L_Omega @ y[:, period]
        specific_risk = cp.multiply(D_diag_sqrt, x[:, period])
        risk_vector = cp.hstack([factor_risk, specific_risk])
        constraints.append(cp.pnorm(risk_vector, p_norm) <= t_risk[period])

    obj_terms = []
    for period in range(T):
        ret = -mu_param[:, period] @ x[:, period]
        tc = tc_rate * cp.sum(z[:, period])
        risk = gamma * t_risk[period]
        obj_terms.append(ret + tc + risk)

    problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
    return problem, {'mu': mu_param}, {'x': x, 'y': y, 'z': z, 't': t_risk}


def solve_and_bench(problem, solver_name, settings, repeats, params, data, mu_sequence=None):
    """Solve problem `repeats` times, return cold time + warm times + objectives."""
    times = []
    objs = []

    for i in range(repeats):
        if mu_sequence is not None:
            params['mu'].value = mu_sequence[i]
        elif i > 0:
            new_mu = data['mu_matrix'] * (1 + 0.05 * (np.random.rand(*data['mu_matrix'].shape) - 0.5))
            params['mu'].value = new_mu

        t0 = time.perf_counter()
        try:
            problem.solve(solver=solver_name, **settings)
        except Exception as e:
            times.append(float('nan'))
            objs.append(float('nan'))
            print(f"    Run {i+1}: ERROR — {e}")
            continue
        t1 = time.perf_counter()

        times.append(t1 - t0)
        obj = problem.value if problem.value is not None else float('nan')
        objs.append(obj)

    return times, objs


def main():
    args = parse_args()
    n = args['n_assets']
    k = args['k_factors']
    p = args['p_norm']
    repeats = args['repeats']

    print("=" * 80)
    print(f"  PowerCone Portfolio Benchmark: MOSEK vs CLARABELGPU")
    print(f"  Assets n={n}, Factors k={k}, p-norm p={p}, Repeats={repeats}")
    print("=" * 80)

    data = generate_data(n, k)
    print(f"\nData generated: n={n}, k={k}, risk_vector_dim={k + n}")

    # Pre-generate all mu perturbations so both solvers use identical parameters
    mu_sequence = [data['mu_matrix'].copy()]
    rng = np.random.RandomState(123)
    for _ in range(repeats - 1):
        mu_sequence.append(
            data['mu_matrix'] * (1 + 0.05 * (rng.rand(*data['mu_matrix'].shape) - 0.5))
        )

    # ── MOSEK ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"  MOSEK  (p-norm p={p})")
    print(f"{'─' * 80}")

    prob_m, params_m, vars_m = create_powcone_model(data, p)

    mosek_settings = {
        'verbose': False,
    }

    print(f"  Compiling + solving {repeats} times...")
    m_times, m_objs = solve_and_bench(prob_m, 'MOSEK', mosek_settings, repeats, params_m, data, mu_sequence)

    for i, (t_val, o) in enumerate(zip(m_times, m_objs)):
        tag = "cold" if i == 0 else "warm"
        print(f"    Run {i+1} ({tag}): {t_val*1000:10.2f} ms   obj={o:+.10e}")

    # ── ClarabelGPU ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"  ClarabelGPU  (p-norm p={p})")
    print(f"{'─' * 80}")

    prob_c, params_c, vars_c = create_powcone_model(data, p)

    ClarabelGPU_settings = {
        'verbose': False,
        'max_iter': 100,
        'tol_feas': 1e-8,
        'tol_gap_abs': 1e-8,
        'tol_gap_rel': 1e-8,
        'equilibrate_enable': True,
        'iterative_refinement_enable': True,
    }

    print(f"  Compiling + solving {repeats} times...")
    c_times, c_objs = solve_and_bench(prob_c, 'CLARABELGPU', ClarabelGPU_settings, repeats, params_c, data, mu_sequence)

    for i, (t_val, o) in enumerate(zip(c_times, c_objs)):
        tag = "cold" if i == 0 else "warm"
        print(f"    Run {i+1} ({tag}): {t_val*1000:10.2f} ms   obj={o:+.10e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  Summary")
    print(f"{'=' * 80}")

    valid_m = [(t_val, o) for t_val, o in zip(m_times, m_objs) if not np.isnan(t_val)]
    valid_c = [(t_val, o) for t_val, o in zip(c_times, c_objs) if not np.isnan(t_val)]

    if len(valid_m) > 1 and len(valid_c) > 1:
        m_cold = valid_m[0][0]
        c_cold = valid_c[0][0]
        m_warm = [t_val for t_val, _ in valid_m[1:]]
        c_warm = [t_val for t_val, _ in valid_c[1:]]
        m_warm_avg = np.mean(m_warm)
        c_warm_avg = np.mean(c_warm)
        m_warm_std = np.std(m_warm)
        c_warm_std = np.std(c_warm)

        print(f"\n  {'':30s} {'MOSEK':>15s}   {'ClarabelGPU':>15s}")
        print(f"  {'─'*30} {'─'*15}   {'─'*15}")
        print(f"  {'Cold start (run 1)':30s} {m_cold*1000:12.2f} ms   {c_cold*1000:12.2f} ms")
        print(f"  {'Warm avg (run 2~N)':30s} {m_warm_avg*1000:12.2f} ms   {c_warm_avg*1000:12.2f} ms")
        print(f"  {'Warm std':30s} {m_warm_std*1000:12.2f} ms   {c_warm_std*1000:12.2f} ms")
        print(f"  {'Warm min':30s} {np.min(m_warm)*1000:12.2f} ms   {np.min(c_warm)*1000:12.2f} ms")
        print(f"  {'Warm max':30s} {np.max(m_warm)*1000:12.2f} ms   {np.max(c_warm)*1000:12.2f} ms")

        ratio = m_warm_avg / c_warm_avg if c_warm_avg > 0 else float('inf')
        if ratio > 1:
            print(f"\n  Warm solve: ClarabelGPU is {ratio:.2f}x faster than MOSEK")
        else:
            print(f"\n  Warm solve: MOSEK is {1/ratio:.2f}x faster than ClarabelGPU")

        # Accuracy comparison (last run)
        m_obj_last = valid_m[-1][1]
        c_obj_last = valid_c[-1][1]
        if not np.isnan(m_obj_last) and not np.isnan(c_obj_last):
            abs_diff = abs(m_obj_last - c_obj_last)
            rel_diff = abs_diff / max(abs(m_obj_last), 1e-10)
            print(f"\n  Objective (last warm run):")
            print(f"    MOSEK:    {m_obj_last:+.12e}")
            print(f"    ClarabelGPU: {c_obj_last:+.12e}")
            print(f"    Rel diff: {rel_diff:.4e}")

        # Per-run accuracy comparison (same mu for both solvers)
        print(f"\n  Per-run objective comparison (same parameters):")
        print(f"  {'Run':>5s}  {'MOSEK obj':>18s}  {'ClarabelGPU obj':>18s}  {'Rel diff':>10s}")
        for i in range(min(len(valid_m), len(valid_c))):
            mo_i = valid_m[i][1]
            co_i = valid_c[i][1]
            if not np.isnan(mo_i) and not np.isnan(co_i):
                rd = abs(mo_i - co_i) / max(abs(mo_i), 1e-10)
                tag = "cold" if i == 0 else "warm"
                print(f"  {i+1:3d} ({tag:4s}) {mo_i:+.10e}  {co_i:+.10e}  {rd:.4e}")
            else:
                print(f"  {i+1:3d}        {'N/A':>18s}  {'N/A':>18s}")
    else:
        print("\n  Insufficient valid results for comparison.")

    # Cone stats
    print(f"\n  Problem cones (ClarabelGPU):")
    try:
        pdata = prob_c.get_problem_data('ClarabelGPU')
        dims = pdata[0].get('dims', None)
        if dims:
            if dims.zero > 0:
                print(f"    ZeroCone:        {dims.zero}")
            if dims.nonneg > 0:
                print(f"    NonnegativeCone: {dims.nonneg}")
            if hasattr(dims, 'soc') and dims.soc:
                soc_list = list(dims.soc)
                print(f"    SOC:             {len(soc_list)} cones, dims={sorted(set(soc_list))}")
            if hasattr(dims, 'exp') and dims.exp:
                print(f"    ExpCone:         {dims.exp}")
            if hasattr(dims, 'p3d') and dims.p3d:
                unique_alphas = sorted(set(round(a, 6) for a in dims.p3d))
                print(f"    PowerCone3D:     {len(dims.p3d)} cones, alphas={unique_alphas}")
            if hasattr(dims, 'psd') and dims.psd:
                print(f"    PSD:             {list(dims.psd)}")
    except Exception:
        pass

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
