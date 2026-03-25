"""
ClarabelGPU CVXPY conic solver interface.

Wraps the clarabel_gpu Cython/CUDA backend as a standard CVXPY
ConicSolver, following the same architecture as the official
CLARABEL solver interface (clarabel_conif.py).
"""

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver

from clarabel_gpu import UpdateHint


def dims_to_solver_cones(cone_dims):
    """Convert CVXPY ConeDims to clarabel_gpu cone objects.

    Constraint row ordering follows the CVXPY/SCS convention:
        Zero -> Nonneg -> SOC -> PSD -> Exp -> Power
    """
    import clarabel_gpu

    cones = []

    if cone_dims.zero > 0:
        cones.append(clarabel_gpu.ZeroConeT(cone_dims.zero))

    if cone_dims.nonneg > 0:
        cones.append(clarabel_gpu.NonnegativeConeT(cone_dims.nonneg))

    for dim in cone_dims.soc:
        cones.append(clarabel_gpu.SecondOrderConeT(dim))

    for dim in cone_dims.psd:
        cones.append(clarabel_gpu.PSDTriangleConeT(dim))

    for _ in range(cone_dims.exp):
        cones.append(clarabel_gpu.ExponentialConeT())

    for alpha in cone_dims.p3d:
        cones.append(clarabel_gpu.PowerConeT(float(alpha)))

    return cones


def triu_to_full(upper_tri, n):
    """Expand n*(n+1)/2 upper-triangular svec to full n*n matrix.

    Off-diagonal elements are scaled by 1/sqrt(2).  Column-major layout.
    """
    full = np.zeros((n, n))
    full[np.tril_indices(n)] = upper_tri
    full += full.T
    full[np.diag_indices(n)] /= 2
    full[np.tril_indices(n, k=-1)] /= np.sqrt(2)
    full[np.triu_indices(n, k=1)] /= np.sqrt(2)
    return np.reshape(full, n * n, order="F")


class ClarabelGPU(ConicSolver):
    """GPU-accelerated Clarabel solver interface for CVXPY.

    Supports LP, QP, SOCP, SDP, EXP, and POW problems via the
    clarabel_gpu Cython/CUDA backend.
    """

    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [
        SOC, ExpCone, PowCone3D, PSD,
    ]
    EXP_CONE_ORDER = [0, 1, 2]

    # clarabel_gpu._convert_status() returns lowercase strings.
    STATUS_MAP = {
        "solved": s.OPTIMAL,
        "primal_infeasible": s.INFEASIBLE,
        "dual_infeasible": s.UNBOUNDED,
        "almost_solved": s.OPTIMAL_INACCURATE,
        "almost_primal_infeasible": s.INFEASIBLE_INACCURATE,
        "almost_dual_infeasible": s.UNBOUNDED_INACCURATE,
        "max_iterations": s.USER_LIMIT,
        "max_time": s.USER_LIMIT,
        "numerical_error": s.SOLVER_ERROR,
        "insufficient_progress": s.SOLVER_ERROR,
    }

    # Official Clarabel long-form setting names -> backend short names.
    _SETTING_ALIASES = {
        "static_regularization_enable": "sr_enable",
        "static_regularization_eps": "sr_constant",
        "static_regularization_proportional": "sr_proportional",
        "dynamic_regularization_enable": "dr_enable",
        "dynamic_regularization_eps": "dr_eps",
        "dynamic_regularization_delta": "dr_delta",
        "iterative_refinement_enable": "ir_enable",
        "iterative_refinement_reltol": "ir_reltol",
        "iterative_refinement_abstol": "ir_abstol",
        "iterative_refinement_max_iter": "ir_max_iter",
        "iterative_refinement_stop_ratio": "ir_stop_ratio",
        "chordal_decomposition_enable": "chordal_decomposition_enable",
        "equilibrate_enable": "equilibrate_enable",
        "equilibrate_max_iter": "equilibrate_max_iter",
        "equilibrate_min_scaling": "equilibrate_min_scaling",
        "equilibrate_max_scaling": "equilibrate_max_scaling",
        "warm_start_enable": "warm_start_enable",
    }

    # CVXPY-internal options that must not be forwarded to the backend.
    _META_OPTS = frozenset({"use_quad_obj", "update_hints"})

    def name(self):
        return "CLARABELGPU"

    def import_solver(self) -> None:
        import clarabel_gpu  # noqa: F401

    def supports_quad_obj(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # PSD support (identical to the official CLARABEL implementation)
    # ------------------------------------------------------------------

    @staticmethod
    def psd_format_mat(constr):
        """Return a linear operator for PSD constraint coefficients.

        Clarabel expects PSD constraints on the upper-triangular part
        of the variable matrix with symmetric sqrt(2) scaling.
        """
        rows = cols = constr.expr.shape[0]
        entries = rows * (cols + 1) // 2

        row_arr = np.arange(0, entries)

        upper_diag_indices = np.triu_indices(rows)
        col_arr = np.sort(np.ravel_multi_index(
            upper_diag_indices, (rows, cols), order='F'))

        val_arr = np.zeros((rows, cols))
        val_arr[upper_diag_indices] = np.sqrt(2)
        np.fill_diagonal(val_arr, 1.0)
        val_arr = np.ravel(val_arr, order='F')
        val_arr = val_arr[np.nonzero(val_arr)]

        shape = (entries, rows * cols)
        scaled_upper_tri = sp.csc_array((val_arr, (row_arr, col_arr)), shape)

        idx = np.arange(rows * cols)
        val_symm = 0.5 * np.ones(2 * rows * cols)
        K = idx.reshape((rows, cols))
        row_symm = np.append(idx, np.ravel(K, order='F'))
        col_symm = np.append(idx, np.ravel(K.T, order='F'))
        symm_matrix = sp.csc_array((val_symm, (row_symm, col_symm)))

        return scaled_upper_tri @ symm_matrix

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """Extract the dual value for *constraint* starting at *offset*.

        PSD constraints are stored internally in svec (scaled
        upper-triangular) form and need expansion to a full matrix.
        """
        if isinstance(constraint, PSD):
            dim = constraint.shape[0]
            upper_tri_dim = dim * (dim + 1) >> 1
            new_offset = offset + upper_tri_dim
            upper_tri = result_vec[offset:new_offset]
            full = triu_to_full(upper_tri, dim)
            return full, new_offset
        else:
            return utilities.extract_dual_value(result_vec, offset, constraint)

    # ------------------------------------------------------------------
    # Core solve / invert
    # ------------------------------------------------------------------

    def solve_via_data(self, data, warm_start, verbose, solver_opts,
                       solver_cache=None):
        """Solve the conic problem via the clarabel_gpu backend.

        Parameters
        ----------
        data : dict
            Problem data produced by ``apply()``.
        warm_start : bool
            If True and a cached solver with matching structure exists
            in *solver_cache*, reuse it (update data in-place).
        verbose : bool
            CVXPY-level verbosity flag.
        solver_opts : dict
            User-provided solver options.  Both official Clarabel
            long-form names and backend short names are accepted.
        solver_cache : dict or None
            Per-problem cache managed by CVXPY.

        Returns
        -------
        dict
            Raw result dictionary from ``clarabel_gpu.ClarabelGPU.solve()``.
        """
        import clarabel_gpu

        solver_opts = solver_opts or {}

        c = data[s.C]
        b = data[s.B]
        A = data[s.A]
        P = data.get(s.P)
        dims = data[ConicSolver.DIMS]

        if not isinstance(c, np.ndarray):
            c = np.asarray(c, dtype=np.float64)
        if not isinstance(b, np.ndarray):
            b = np.asarray(b, dtype=np.float64)

        n = A.shape[1]
        m = A.shape[0]

        if P is not None:
            if not isinstance(P, sp.csr_matrix):
                P = P.tocsr()
        else:
            P = sp.csr_matrix((n, n), dtype=np.float64)

        if not isinstance(A, sp.csr_matrix):
            A = A.tocsr()

        settings = self._translate_settings(solver_opts, verbose)
        current_key = self._structure_key(dims, n, m)

        solver = self._try_cached_solver(
            solver_cache, warm_start, current_key,
            P, c, A, b, dims, settings,
        )

        if solver is None:
            cones = dims_to_solver_cones(dims)
            solver = clarabel_gpu.ClarabelGPU()
            solver.setup(P, c, A, b, cones, **settings)

        result = solver.solve()

        if solver_cache is not None:
            solver_cache[self.name()] = {
                "solver": solver,
                "structure_key": current_key,
                "A_indptr": A.indptr.copy(),
                "A_indices": A.indices.copy(),
                "A_data": A.data.copy(),
                "P_indptr": P.indptr.copy() if P.nnz > 0 else None,
                "P_indices": P.indices.copy() if P.nnz > 0 else None,
                "P_data": P.data.copy() if P.nnz > 0 else None,
            }

        return result

    def invert(self, solution, inverse_data):
        """Map the backend result back to CVXPY solution objects."""
        attr = {}
        status = self.STATUS_MAP.get(
            str(solution.get("status", "")), s.SOLVER_ERROR)
        attr[s.SOLVE_TIME] = solution.get("solve_time", 0)
        attr[s.NUM_ITERS] = solution.get("iterations", 0)

        if status in s.SOLUTION_PRESENT:
            primal_val = solution.get("obj_val")
            opt_val = (primal_val + inverse_data[s.OFFSET]
                       if primal_val is not None else None)

            x_sol = solution.get("x")
            if x_sol is not None and not isinstance(x_sol, np.ndarray):
                x_sol = np.asarray(x_sol.get()) if hasattr(x_sol, 'get') else np.asarray(x_sol)
            primal_vars = {
                inverse_data[ClarabelGPU.VAR_ID]: x_sol,
            }

            z = solution.get("z")
            if z is not None:
                if not isinstance(z, np.ndarray):
                    z = np.asarray(z.get()) if hasattr(z, 'get') else np.asarray(z)
                dims = inverse_data[ConicSolver.DIMS]
                eq_dual_vars = utilities.get_dual_values(
                    z[:dims.zero],
                    self.extract_dual_value,
                    inverse_data[ClarabelGPU.EQ_CONSTR],
                )
                ineq_dual_vars = utilities.get_dual_values(
                    z[dims.zero:],
                    self.extract_dual_value,
                    inverse_data[ClarabelGPU.NEQ_CONSTR],
                )
                dual_vars = {}
                dual_vars.update(eq_dual_vars)
                dual_vars.update(ineq_dual_vars)
            else:
                dual_vars = {}

            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def cite(self, data):
        return (
            "@inproceedings{Goulart_2024,\n"
            "    author  = {Goulart, P. and Chen, Y.},\n"
            "    title   = {Clarabel: An interior-point solver for conic\n"
            "               programs with quadratic objectives},\n"
            "    year    = {2024},\n"
            "    note    = {GPU-accelerated backend via clarabel\\_gpu}\n"
            "}\n"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _translate_settings(self, solver_opts, verbose):
        """Map user-provided solver options to backend setting names.

        Accepts both official Clarabel long-form names (e.g.
        ``static_regularization_enable``) and backend short names
        (e.g. ``sr_enable``).  Long-form names are mapped to short
        names; unrecognised keys are passed through unchanged so that
        the backend can report errors for truly invalid options.
        """
        settings = {}
        for key, val in solver_opts.items():
            if key in self._META_OPTS:
                continue
            backend_key = self._SETTING_ALIASES.get(key, key)
            settings[backend_key] = val
        settings.setdefault("verbose", verbose)
        settings.setdefault("warm_start_enable", True)
        return settings

    @staticmethod
    def _try_cached_solver(solver_cache, warm_start, current_key,
                           P, c, A, b, dims, settings):
        """Return a cached solver updated with new data, or None.

        If ``settings`` contains an ``"update_hints"`` key whose value
        is a set of :class:`UpdateHint` members, the function uses those
        hints to decide which updates to perform — skipping all
        ``np.array_equal`` comparisons.  When the set contains
        ``UpdateHint.automatic`` (or ``"update_hints"`` is absent),
        the original comparison-based detection is used.

        Three-way detection (automatic mode):

        Case 1 — Sparsity pattern changed (A.indptr or A.indices differ):
            Call ``rebuild()`` on the cached solver.  This destroys the
            internal Solver but preserves the elimination-tree cache
            (shared_ptr), so the new ANALYSIS injects the cached tree
            and skips REORDERING (~50-80% ANALYSIS savings).

        Case 2 — Only numerical values changed (same indptr/indices):
            Scatter new P/A values into the existing KKT matrix.
            ANALYSIS is skipped entirely; only REFACTORIZATION is needed.

        Case 3 — Matrices completely unchanged (same data):
            Only q and b are updated; P/A scatter is skipped.
        """
        hints = settings.pop("update_hints", None)

        if not warm_start or solver_cache is None:
            return None
        cache_entry = solver_cache.get("CLARABELGPU")
        if cache_entry is None:
            return None
        cached_solver = cache_entry.get("solver")
        cached_key = cache_entry.get("structure_key")
        if cached_solver is None or cached_key != current_key:
            return None

        # ----------------------------------------------------------
        # Fast path: caller-supplied hints (skip np.array_equal)
        # ----------------------------------------------------------
        if hints is not None and UpdateHint.automatic not in hints:
            pattern_changed = (UpdateHint.P_pattern_changed in hints
                               or UpdateHint.A_pattern_changed in hints)
            if pattern_changed:
                cones = dims_to_solver_cones(dims)
                cached_solver.rebuild(P, c, A, b, cones, **settings)
                return cached_solver

            cached_solver.set_verbose(settings.get("verbose", False))
            if UpdateHint.P_values_changed in hints:
                cached_solver.update_P(P)
            if UpdateHint.q_values_changed in hints:
                cached_solver.update_q(c)
            if UpdateHint.A_values_changed in hints:
                cached_solver.update_A(A)
            if UpdateHint.b_values_changed in hints:
                cached_solver.update_b(b)
            return cached_solver

        # ----------------------------------------------------------
        # Automatic detection (original comparison-based logic)
        # ----------------------------------------------------------
        cached_A_indptr = cache_entry.get("A_indptr")
        cached_A_indices = cache_entry.get("A_indices")

        if cached_A_indptr is None or cached_A_indices is None:
            return None

        a_pattern_same = (A.nnz == len(cached_A_indices)
                          and np.array_equal(A.indptr, cached_A_indptr)
                          and np.array_equal(A.indices, cached_A_indices))

        p_pattern_same = True
        if P.nnz > 0:
            cached_P_indptr = cache_entry.get("P_indptr")
            cached_P_indices = cache_entry.get("P_indices")
            if cached_P_indptr is None or cached_P_indices is None:
                p_pattern_same = False
            else:
                p_pattern_same = (P.nnz == len(cached_P_indices)
                                  and np.array_equal(P.indptr, cached_P_indptr)
                                  and np.array_equal(P.indices, cached_P_indices))

        if not a_pattern_same or not p_pattern_same:
            cones = dims_to_solver_cones(dims)
            cached_solver.rebuild(P, c, A, b, cones, **settings)
            return cached_solver

        cached_solver.set_verbose(settings.get("verbose", False))

        cached_P_data = cache_entry.get("P_data")
        cached_A_data = cache_entry.get("A_data")

        if cached_P_data is None or not np.array_equal(P.data, cached_P_data):
            cached_solver.update_P(P)

        cached_solver.update_q(c)

        if cached_A_data is None or not np.array_equal(A.data, cached_A_data):
            cached_solver.update_A(A)

        cached_solver.update_b(b)
        return cached_solver

    @staticmethod
    def _structure_key(dims, n, m):
        """Hashable key that identifies the problem structure."""
        return (
            n, m, dims.zero, dims.nonneg,
            tuple(dims.soc) if dims.soc else (),
            tuple(dims.psd) if dims.psd else (),
            dims.exp or 0,
            tuple(dims.p3d) if dims.p3d else (),
        )
