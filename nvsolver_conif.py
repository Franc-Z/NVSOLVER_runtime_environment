"""
NVSolver CVXPY 集成 - 极速版（最大化 DPP 性能）

针对 DPP 参数更新场景的极致优化：
- 激进的假设和缓存策略
- 最小化每次更新的开销
- 牺牲通用性换取性能
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, Optional, Any, Tuple

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.constraints import Zero, NonNeg, SOC

# 模块级别的缓存
_NVSOLVER_CORE = None
_IMPORT_ATTEMPTED = False

# 预分配的工作空间
_WORKSPACE = {
    'result_arrays': {},  # 预分配的结果数组
    'empty_matrices': {},  # 空矩阵缓存
    'settings_cache': None,  # 设置缓存
}


class FastNVSolverInstance:
    """超快速的求解器实例封装"""
    
    def __init__(self, solver_core, problem_size):
        self.solver = solver_core
        self.update_count = 0
        self.problem_size = problem_size
        
        # 预分配结果缓冲区
        n_vars, n_constraints = problem_size
        self.x_buffer = np.zeros(n_vars, dtype=np.float64)
        self.y_buffer = np.zeros(n_constraints, dtype=np.float64)
        
        # 直接访问更新方法（避免属性查找）
        self.update_P = getattr(solver_core, 'update_P', None)
        self.update_q = getattr(solver_core, 'update_q', None)
        self.update_A = getattr(solver_core, 'update_A', None)
        self.update_b = getattr(solver_core, 'update_b', None)
        
    def fast_update_q(self, q):
        """快速更新 q（假设已经是正确格式）"""
        if self.update_q:
            self.update_q(q)
            self.update_count += 1
            
    def fast_update_b(self, b):
        """快速更新 b（假设已经是正确格式）"""
        if self.update_b:
            self.update_b(b)
            self.update_count += 1
            
    def fast_update_P(self, P):
        """快速更新 P（假设已经是 CSR 格式）"""
        if self.update_P:
            self.update_P(P)
            self.update_count += 1
            
    def solve(self):
        """直接求解"""
        return self.solver.solve()


class NVSOLVER(ConicSolver):
    """GPU加速的内点法求解器 - 极速版
    
    极致性能优化版本，假设：
    1. 问题结构完全不变
    2. 输入数据已经是正确格式
    3. 主要更新 q 和 b 参数
    """
    
    # 预编译的状态映射（使用数组索引而不是字典）
    _STATUS_CODES = {
        'solved': 0,
        'primal_infeasible': 1,
        'dual_infeasible': 2,
        'almost_solved': 3,
        'almost_primal_infeasible': 4,
        'almost_dual_infeasible': 5,
        'max_iterations': 6,
        'max_time': 7,
        'numerical_error': 8,
        'insufficient_progress': 9,
        'unsolved': 10,
        'unknown': 11
    }
    
    _STATUS_VALUES = [
        s.OPTIMAL,                    # 0: solved
        s.INFEASIBLE,                # 1: primal_infeasible
        s.UNBOUNDED,                 # 2: dual_infeasible
        s.OPTIMAL_INACCURATE,        # 3: almost_solved
        s.INFEASIBLE_INACCURATE,     # 4: almost_primal_infeasible
        s.UNBOUNDED_INACCURATE,      # 5: almost_dual_infeasible
        s.USER_LIMIT,                # 6: max_iterations
        s.USER_LIMIT,                # 7: max_time
        s.SOLVER_ERROR,              # 8: numerical_error
        s.OPTIMAL_INACCURATE,        # 9: insufficient_progress
        s.SOLVER_ERROR,              # 10: unsolved
        s.SOLVER_ERROR               # 11: unknown
    ]
    
    # 支持的约束类型
    SUPPORTED_CONSTRAINTS = frozenset({Zero, NonNeg, SOC})
    
    def __init__(self):
        super().__init__()
        # 简单的单实例缓存
        self._last_instance = None
        self._last_structure_key = None
        # 缓存的设置
        self._cached_settings = None
        self._last_solver_opts_id = None

    def name(self):
        return "NVSOLVER"

    def import_solver(self) -> None:
        """导入求解器（优化：只尝试一次）"""
        global _NVSOLVER_CORE, _IMPORT_ATTEMPTED
        
        if _IMPORT_ATTEMPTED:
            return
            
        _IMPORT_ATTEMPTED = True
        
        try:
            # 直接尝试最可能的导入路径
            from clarabel_gpu import ClarabelGPU as NVSolverCore
            _NVSOLVER_CORE = NVSolverCore
            return
        except ImportError:
            pass
            
        # 备选方案
        try:
            import sys
            import os
            sys.path.insert(0, '/opt/miniconda3/envs/cufolio/lib/python3.12/site-packages')
            import clarabel_gpu
            _NVSOLVER_CORE = clarabel_gpu.ClarabelGPU
        except ImportError:
            pass

    def accepts(self, problem) -> bool:
        """快速问题类型检查"""
        if not problem.is_dcp():
            return False
        
        # 快速检查约束类型
        for c in problem.constraints:
            if type(c) not in self.SUPPORTED_CONSTRAINTS:
                return False
        return True

    def solve_via_data(self, data, warm_start: bool, verbose: bool, 
                      solver_opts, solver_cache=None):
        """使用NVSolver求解问题 - 极速版"""
        global _NVSOLVER_CORE, _WORKSPACE
        
        if _NVSOLVER_CORE is None:
            self.import_solver()
            if _NVSOLVER_CORE is None:
                raise ImportError("无法导入NVSolver")
        
        # 快速数据提取（假设格式正确）
        c = data[s.C]
        b = data[s.B]
        A = data[s.A]
        P = data.get(s.P)
        dims = data[s.DIMS]
        
        # 最小化的格式检查（仅在非 numpy 时转换）
        if not isinstance(c, np.ndarray):
            c = np.asarray(c, dtype=np.float64)
        if not isinstance(b, np.ndarray):
            b = np.asarray(b, dtype=np.float64)
        
        # 快速检查不支持的锥（使用短路评估）
        if dims.psd or dims.exp or getattr(dims, 'p3d', None):
            return self._fast_unsupported_result(verbose)
        
        # 构建锥维度（优化：直接访问属性）
        cone_dims = {
            'z': dims.zero,
            'l': dims.nonneg,
            'q': dims.soc or [],
            's': [],
            'ep': 0,
            'p': []
        }
        
        # 处理矩阵
        n = A.shape[1]
        m = A.shape[0]
        
        # P 矩阵（使用全局缓存）
        if P is None:
            if n not in _WORKSPACE['empty_matrices']:
                _WORKSPACE['empty_matrices'][n] = sp.csr_matrix((n, n), dtype=np.float64)
            P = _WORKSPACE['empty_matrices'][n]
        elif not isinstance(P, sp.csr_matrix):
            P = P.tocsr()
        
        # A 矩阵（假设已经是 CSR 格式）
        if not isinstance(A, sp.csr_matrix):
            A = A.tocsr()
        
        # 简化的结构键（使用元组而不是复杂对象）
        current_key = (n, m, dims.zero, dims.nonneg, len(dims.soc) if dims.soc else 0)
        
        # 检查是否可以重用
        can_reuse = (
            self._last_instance is not None and
            self._last_structure_key == current_key and
            solver_opts.get('reuse_solver', True)
        )
        
        if can_reuse:
            # 极速参数更新路径
            instance = self._last_instance
            
            # 只更新真正需要的参数（通常只有 q 和 b）
            if solver_opts.get('update_q', True):
                instance.fast_update_q(c)
            if solver_opts.get('update_b', True):
                instance.fast_update_b(b)
            
            # 很少更新 P 和 A
            if solver_opts.get('update_P', False) and P is not None:
                instance.fast_update_P(P)
            if solver_opts.get('update_A', False):
                instance.fast_update_A(A)
            
            result = instance.solve()
        else:
            # 创建新实例
            if verbose:
                print("[DPP] 创建新求解器实例（极速版）")
            
            # 缓存设置（如果 solver_opts 没变，重用设置）
            opts_id = id(solver_opts)
            if self._cached_settings is None or self._last_solver_opts_id != opts_id:
                self._cached_settings = self._fast_prepare_settings(solver_opts, verbose)
                self._last_solver_opts_id = opts_id
            
            solver_core = _NVSOLVER_CORE()
            solver_core.setup(P, c, A, b, cone_dims, **self._cached_settings)
            
            instance = FastNVSolverInstance(solver_core, (n, m))
            
            # 保存实例
            self._last_instance = instance
            self._last_structure_key = current_key
            
            result = instance.solve()
        
        return self._fast_process_result(result)
    
    def _fast_prepare_settings(self, solver_opts, verbose):
        """快速准备设置（最小化字典操作）"""
        # 使用默认值字典，减少 get 调用
        return {
            'verbose': verbose,
            'max_iter': solver_opts.get('max_iter', 200),
            'tol_gap_abs': solver_opts.get('tol_gap_abs', 1e-6),
            'tol_gap_rel': solver_opts.get('tol_gap_rel', 1e-6),
            'tol_feas': solver_opts.get('tol_feas', 1e-6),
            'tol_infeas_abs': solver_opts.get('tol_infeas_abs', 1e-6),
            'tol_infeas_rel': solver_opts.get('tol_infeas_rel', 1e-6),
            'equilibrate_enable': False,  # DPP 必须禁用
            'sr_enable': True,
            'dr_enable': True,
            'ir_enable': True,
        }
    
    def _fast_process_result(self, result):
        """极速结果处理（避免不必要的转换）"""
        # 快速状态查找
        status_str = result.get('status', 'unknown')
        if isinstance(status_str, str):
            status_code = self._STATUS_CODES.get(status_str.lower(), 11)
            status = self._STATUS_VALUES[status_code]
        else:
            status = s.SOLVER_ERROR
        
        # 直接返回结果，避免复制
        return {
            'status': status,
            'x': result.get('x'),  # 已经是 numpy 数组
            'y': result.get('z'),  # NVSolver 使用 'z'
            's': result.get('s'),
            'primal_objective': result.get('obj_val'),
            'dual_objective': result.get('obj_val_dual'),
            'gap': result.get('gap'),
            'solve_time': result.get('solve_time'),
            'iterations': result.get('iterations'),
            'info': result
        }
    
    def _fast_unsupported_result(self, verbose):
        """快速返回不支持的结果"""
        if verbose:
            print("NVSOLVER不支持 PSD/Exp/Power 锥")
        return {
            'status': 'unknown',
            'x': None, 'y': None, 's': None,
            'obj_val': None, 'obj_val_dual': None,
            'gap': None, 'solve_time': 0.0,
            'iterations': 0,
            'info': {'error': 'unsupported_cones'}
        }

    def invert(self, solution, inverse_data):
        """快速结果转换（优化：预分配数组）"""
        if isinstance(solution, Solution):
            return solution
            
        status = solution['status']
        
        if status not in s.SOLUTION_PRESENT:
            return failure_solution(status)
        
        y = solution.get('y')
        dual_vars = {}
        
        if y is not None:
            # 快速提取约束
            eq_constraints = inverse_data[self.EQ_CONSTR]
            neq_constraints = inverse_data[self.NEQ_CONSTR]
            
            # 使用 numpy 切片（避免循环中的列表操作）
            offset = 0
            
            # 等式约束
            for constr in eq_constraints:
                size = constr.size
                dual_vars[constr.id] = y[offset:offset+size]
                offset += size
            
            # 不等式约束（原地取负）
            for constr in neq_constraints:
                size = constr.size
                dual_val = y[offset:offset+size]
                # 创建视图而不是复制
                dual_vars[constr.id] = -dual_val  # numpy 会处理负号
                offset += size
        
        primal_vars = {self.VAR_ID: solution['x']}
        
        attr = {
            s.SOLVE_TIME: solution.get('solve_time', 0),
            s.NUM_ITERS: solution.get('iterations', 0),
            'info': solution.get('info', {})
        }
        
        return Solution(
            status, 
            solution['primal_objective'],
            primal_vars, 
            dual_vars, 
            attr
        )

    def clear_cache(self):
        """清空缓存"""
        self._last_instance = None
        self._last_structure_key = None
        self._cached_settings = None
        self._last_solver_opts_id = None
        global _WORKSPACE
        _WORKSPACE['empty_matrices'].clear()
        _WORKSPACE['result_arrays'].clear()
    
    def get_cache_stats(self):
        """获取缓存统计"""
        return {
            'has_cached_instance': self._last_instance is not None,
            'update_count': self._last_instance.update_count if self._last_instance else 0,
            'cached_settings': self._cached_settings is not None
        }

    @staticmethod 
    def supports_quad_obj():
        return True
    
    def cite(self):
        return """
@misc{nvsolver2024,
    author = {NVIDIA and Clarabel Developers},
    title = {{NVSolver: GPU-accelerated Interior Point Solver for CVXPy}},
    year = {2025},
    note = {Ultra-fast version optimized for DPP parameter updates.}
}
        """
