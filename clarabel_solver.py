"""
High-level Python interface for Clarabel GPU solver

提供了更符合Python习惯的API接口，作为底层Cython扩展的封装。
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional, Dict, Union, List, Tuple
import warnings

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

# Import the Cython extension
try:
    # Try package import first
    from clarabel_gpu import ClarabelGPU as _ClarabelGPU
except ImportError:
    try:
        # Try direct import from parent directory
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
        from clarabel_gpu import ClarabelGPU as _ClarabelGPU
    except ImportError as e:
        raise ImportError(
            f"无法导入 clarabel_gpu 扩展模块。\n"
            f"错误: {e}\n"
            f"请确保已编译 Clarabel GPU 扩展：\n"
            f"  cd /clarabel-clean\n"
            f"  ./build_all.sh\n"
        )

class ClarabelSolver:
    """
    高层Python接口，封装Clarabel GPU求解器
    
    支持以下问题类型：
    - 二次规划 (QP)
    - 二阶锥规划 (SOCP)
    - 线性规划 (LP)
    
    标准形式：
        minimize    0.5 * x' * P * x + q' * x
        subject to  A * x + s = b
                    s in K
                    
    其中 K 是零锥、非负锥和二阶锥的笛卡尔积。
    
    注意：此类是对底层 ClarabelGPU (Cython扩展) 的封装，提供了：
    - 更友好的API
    - 参数验证
    - 错误处理
    - 文档说明
    """
    
    def __init__(self, 
                 P: Optional[Union[np.ndarray, sp.spmatrix]] = None,
                 q: Optional[np.ndarray] = None,
                 A: Optional[Union[np.ndarray, sp.spmatrix]] = None,
                 b: Optional[np.ndarray] = None,
                 cone_dims: Optional[Dict[str, Union[int, List[int]]]] = None,
                 gpu_mode: bool = False,
                 **settings):
        """
        初始化 Clarabel GPU 求解器
        
        Parameters
        ----------
        P : array_like or sparse matrix, optional
            二次目标函数矩阵（必须对称半正定）
            如果为 None，则目标函数为线性
        q : array_like
            线性目标向量
        A : array_like or sparse matrix
            约束矩阵
        b : array_like
            约束向量
        cone_dims : dict
            锥约束维度字典：
            - 'z': int, 零锥维度（等式约束）
            - 'l': int, 非负锥维度（不等式约束 >= 0）
            - 'q': list of int, 二阶锥维度列表
        gpu_mode : bool, default=False
            是否启用GPU模式（返回CuPy数组）
            - True: 解向量保持在GPU上（CuPy数组），使用RMM管理显存
            - False: 解向量在CPU上（NumPy数组）
        **settings : keyword arguments
            求解器设置：
            - verbose : bool, 是否输出详细信息
            - max_iter : int, 最大迭代次数 (默认: 200)
            - tol_gap_abs : float, 绝对间隙容差 (默认: 1e-8)
            - tol_gap_rel : float, 相对间隙容差 (默认: 1e-8)
            - tol_feas : float, 可行性容差 (默认: 1e-8)
            - equilibrate_enable : bool, 是否启用平衡 (默认: True)
            - sr_enable : bool, 静态正则化 (默认: True)
            - dr_enable : bool, 动态正则化 (默认: True)
            - ir_enable : bool, 迭代细化 (默认: True)
            
        Examples
        --------
        >>> import numpy as np
        >>> import scipy.sparse as sp
        >>> from clarabel_solver import ClarabelSolver
        >>> 
        >>> # 简单QP问题: min 0.5*x'*I*x + q'*x, s.t. x >= 0
        >>> n = 10
        >>> P = sp.eye(n, format='csr')
        >>> q = np.random.randn(n)
        >>> A = sp.eye(n, format='csr')
        >>> b = np.zeros(n)
        >>> cone_dims = {'l': n}  # 非负约束
        >>> 
        >>> # CPU模式
        >>> solver = ClarabelSolver(P, q, A, b, cone_dims)
        >>> result = solver.solve()
        >>> print(f"状态: {result['status']}, 目标值: {result['obj_val']:.6f}")
        >>> 
        >>> # GPU模式（需要CuPy）
        >>> solver_gpu = ClarabelSolver(P, q, A, b, cone_dims, gpu_mode=True)
        >>> result_gpu = solver_gpu.solve()
        """
        # 创建底层Cython求解器实例
        self._solver = _ClarabelGPU()
        
        # 存储问题数据（延迟到setup时使用）
        self._problem_data = {
            'P': P,
            'q': q,
            'A': A,
            'b': b,
            'cone_dims': cone_dims
        }
        self._gpu_mode = gpu_mode
        self._settings = settings
        self._setup_called = False
        
        # 如果所有参数都提供了，自动调用setup
        if all(x is not None for x in [q, A, b, cone_dims]):
            self.setup()
        
    def setup(self):
        """
        设置优化问题
        
        如果在构造函数中提供了所有参数，会自动调用此方法。
        否则，需要先调用此方法再调用solve()。
        
        Raises
        ------
        ValueError
            如果参数验证失败
        """
        if self._setup_called:
            warnings.warn("setup() 已经被调用过，忽略重复调用。")
            return
            
        # 验证输入参数
        self._validate_inputs()
        
        # 调用底层Cython扩展的setup方法
        self._solver.setup(
            self._problem_data['P'],
            self._problem_data['q'],
            self._problem_data['A'],
            self._problem_data['b'],
            self._problem_data['cone_dims'],
            gpu_mode=self._gpu_mode,
            **self._settings
        )
        
        self._setup_called = True
        
    def solve(self) -> Dict:
        """
        求解优化问题
        
        Returns
        -------
        dict
            解字典，包含：
            - 'status' : str
                求解状态：'solved', 'primal_infeasible', 'dual_infeasible',
                'max_iterations', 'numerical_error' 等
            - 'x' : ndarray or cupy.ndarray
                原始解向量（如果gpu_mode=True则为CuPy数组）
            - 'z' : ndarray or cupy.ndarray
                对偶解向量（拉格朗日乘子，如果gpu_mode=True则为CuPy数组）
            - 's' : ndarray or cupy.ndarray
                松弛变量（如果gpu_mode=True则为CuPy数组）
            - 'obj_val' : float
                原始目标值
            - 'obj_val_dual' : float
                对偶目标值
            - 'solve_time' : float
                求解时间（秒）
            - 'iterations' : int
                迭代次数
            - 'r_prim' : float
                原始残差
            - 'r_dual' : float
                对偶残差
                
        Raises
        ------
        RuntimeError
            如果setup()尚未调用或求解失败
            
        Examples
        --------
        >>> solver = ClarabelSolver(P, q, A, b, cone_dims)
        >>> result = solver.solve()
        >>> print(f"状态: {result['status']}")
        >>> print(f"目标值: {result['obj_val']:.6f}")
        >>> print(f"求解时间: {result['solve_time']:.4f} 秒")
        >>> x_opt = result['x']  # 最优解
        """
        if not self._setup_called:
            raise RuntimeError("请先调用 setup() 方法设置问题")
            
        # 调用底层Cython扩展的solve方法
        result = self._solver.solve()
        
        return result
        
    def update_P(self, P_new: Union[np.ndarray, sp.spmatrix, 'cp.ndarray']):
        """
        更新二次目标矩阵 P
        
        支持热启动：在多次求解相似问题时，只更新P而不重新设置整个问题。
        
        Parameters
        ----------
        P_new : array_like, sparse matrix, or CuPy array
            新的P矩阵（必须与原始P有相同的稀疏模式）
            - 如果gpu_mode=True且P_new是CuPy数组，使用GPU直接传输
            - 否则使用CPU传输
            
        Raises
        ------
        RuntimeError
            如果setup()尚未调用
        ValueError
            如果P_new的维度或稀疏模式与原始P不匹配
            
        Examples
        --------
        >>> # 初始求解
        >>> solver = ClarabelSolver(P, q, A, b, cone_dims)
        >>> result1 = solver.solve()
        >>> 
        >>> # 更新P后重新求解（热启动）
        >>> P_new = 2 * P  # 新的目标函数矩阵
        >>> solver.update_P(P_new)
        >>> result2 = solver.solve()
        """
        if not self._setup_called:
            raise RuntimeError("无法在调用 setup() 之前更新 P")
            
        self._solver.update_P(P_new)
        
    def update_q(self, q_new: Union[np.ndarray, 'cp.ndarray']):
        """
        更新线性目标向量 q
        
        Parameters
        ----------
        q_new : array_like or CuPy array
            新的q向量
            - 如果gpu_mode=True且q_new是CuPy数组，使用GPU直接传输
            - 否则使用CPU传输
            
        Raises
        ------
        RuntimeError
            如果setup()尚未调用
        ValueError
            如果q_new的维度与原始q不匹配
        """
        if not self._setup_called:
            raise RuntimeError("无法在调用 setup() 之前更新 q")
            
        self._solver.update_q(q_new)
        
    def update_A(self, A_new: Union[np.ndarray, sp.spmatrix, 'cp.ndarray']):
        """
        更新约束矩阵 A
        
        Parameters
        ----------
        A_new : array_like, sparse matrix, or CuPy array
            新的A矩阵（必须与原始A有相同的稀疏模式）
            - 如果gpu_mode=True且A_new是CuPy数组，使用GPU直接传输
            - 否则使用CPU传输
            
        Raises
        ------
        RuntimeError
            如果setup()尚未调用
        ValueError
            如果A_new的维度或稀疏模式与原始A不匹配
        """
        if not self._setup_called:
            raise RuntimeError("无法在调用 setup() 之前更新 A")
            
        self._solver.update_A(A_new)
        
    def update_b(self, b_new: Union[np.ndarray, 'cp.ndarray']):
        """
        更新约束向量 b
        
        Parameters
        ----------
        b_new : array_like or CuPy array
            新的b向量
            - 如果gpu_mode=True且b_new是CuPy数组，使用GPU直接传输
            - 否则使用CPU传输
            
        Raises
        ------
        RuntimeError
            如果setup()尚未调用
        ValueError
            如果b_new的维度与原始b不匹配
        """
        if not self._setup_called:
            raise RuntimeError("无法在调用 setup() 之前更新 b")
            
        self._solver.update_b(b_new)
        
    def update_data(self, 
                    P: Optional[Union[np.ndarray, sp.spmatrix, 'cp.ndarray']] = None,
                    q: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
                    A: Optional[Union[np.ndarray, sp.spmatrix, 'cp.ndarray']] = None,
                    b: Optional[Union[np.ndarray, 'cp.ndarray']] = None):
        """
        一次更新多个问题数据
        
        这是一个便捷方法，可以同时更新多个数据项。
        
        Parameters
        ----------
        P, q, A, b : array_like, sparse matrix, or CuPy array, optional
            新的数据值（只有非None的值会被更新）
            
        Examples
        --------
        >>> # 同时更新q和b
        >>> solver.update_data(q=q_new, b=b_new)
        >>> result = solver.solve()
        """
        if P is not None:
            self.update_P(P)
        if q is not None:
            self.update_q(q)
        if A is not None:
            self.update_A(A)
        if b is not None:
            self.update_b(b)
    
    def update_P_gpu(self, P_gpu: 'cp.ndarray'):
        """
        使用GPU数据直接更新P矩阵（零拷贝）
        
        此方法仅在gpu_mode=True时有效，使用GPU到GPU的直接传输，
        避免CPU中转，性能最优。
        
        Parameters
        ----------
        P_gpu : CuPy array or CuPy sparse matrix
            GPU上的新P矩阵数据
            
        Raises
        ------
        RuntimeError
            如果GPU模式未启用或setup()尚未调用
        """
        if not self._setup_called:
            raise RuntimeError("无法在调用 setup() 之前更新 P")
        if not self._gpu_mode:
            raise RuntimeError("GPU直接更新仅在gpu_mode=True时可用")
            
        self._solver.update_P_gpu(P_gpu)
        
    def update_q_gpu(self, q_gpu: 'cp.ndarray'):
        """
        使用GPU数据直接更新q向量（零拷贝）
        
        Parameters
        ----------
        q_gpu : CuPy array
            GPU上的新q向量
        """
        if not self._setup_called:
            raise RuntimeError("无法在调用 setup() 之前更新 q")
        if not self._gpu_mode:
            raise RuntimeError("GPU直接更新仅在gpu_mode=True时可用")
            
        self._solver.update_q_gpu(q_gpu)
        
    def update_A_gpu(self, A_gpu: 'cp.ndarray'):
        """
        使用GPU数据直接更新A矩阵（零拷贝）
        
        Parameters
        ----------
        A_gpu : CuPy array or CuPy sparse matrix
            GPU上的新A矩阵数据
        """
        if not self._setup_called:
            raise RuntimeError("无法在调用 setup() 之前更新 A")
        if not self._gpu_mode:
            raise RuntimeError("GPU直接更新仅在gpu_mode=True时可用")
            
        self._solver.update_A_gpu(A_gpu)
        
    def update_b_gpu(self, b_gpu: 'cp.ndarray'):
        """
        使用GPU数据直接更新b向量（零拷贝）
        
        Parameters
        ----------
        b_gpu : CuPy array
            GPU上的新b向量
        """
        if not self._setup_called:
            raise RuntimeError("无法在调用 setup() 之前更新 b")
        if not self._gpu_mode:
            raise RuntimeError("GPU直接更新仅在gpu_mode=True时可用")
            
        self._solver.update_b_gpu(b_gpu)
    
    def get_memory_info(self) -> Optional[Dict]:
        """
        获取显存使用信息（仅在使用RMM时）
        
        Returns
        -------
        dict or None
            如果使用RMM，返回显存统计信息字典：
            - 'rmm_enabled' : bool, RMM是否启用
            - 'cupy_used_bytes' : int, CuPy使用的字节数
            - 'cupy_total_bytes' : int, CuPy分配的总字节数
            - 'cupy_used_mb' : float, CuPy使用的MB数
            - 'cupy_total_mb' : float, CuPy分配的总MB数
            如果未使用RMM，返回None
            
        Examples
        --------
        >>> solver = ClarabelSolver(P, q, A, b, cone_dims, gpu_mode=True)
        >>> result = solver.solve()
        >>> mem_info = solver.get_memory_info()
        >>> if mem_info:
        >>>     print(f"显存使用: {mem_info['cupy_used_mb']:.2f} MB")
        """
        if hasattr(self._solver, 'get_memory_info'):
            return self._solver.get_memory_info()
        return None
    
    def reset_memory_pool(self):
        """
        重置RMM内存池（释放未使用的显存）
        
        此操作会释放内存池中未使用的内存块，但不会影响正在使用的分配。
        适用于需要释放显存给其他程序使用的场景。
        
        Examples
        --------
        >>> solver = ClarabelSolver(P, q, A, b, cone_dims, gpu_mode=True)
        >>> result = solver.solve()
        >>> solver.reset_memory_pool()  # 释放未使用的显存
        """
        if hasattr(self._solver, 'reset_memory_pool'):
            self._solver.reset_memory_pool()
            
    def _validate_inputs(self):
        """验证问题输入参数"""
        P = self._problem_data['P']
        q = self._problem_data['q']
        A = self._problem_data['A']
        b = self._problem_data['b']
        cone_dims = self._problem_data['cone_dims']
        
        # 检查必需参数
        if q is None:
            raise ValueError("线性目标向量 q 必须提供")
            
        n = len(q)
        
        # 检查P的维度
        if P is not None:
            P_shape = P.shape if hasattr(P, 'shape') else (len(P), len(P[0]))
            if P_shape != (n, n):
                raise ValueError(f"P 必须是 {n}x{n} 的方阵，实际为 {P_shape}")
                
        # 检查A的维度
        if A is None:
            raise ValueError("约束矩阵 A 必须提供")
            
        A_shape = A.shape if hasattr(A, 'shape') else (len(A), len(A[0]))
        m, n_A = A_shape
        
        if n_A != n:
            raise ValueError(f"A 必须有 {n} 列，实际为 {n_A} 列")
            
        # 检查b的维度
        if b is None:
            raise ValueError("约束向量 b 必须提供")
            
        if len(b) != m:
            raise ValueError(f"b 的长度必须为 {m}，实际为 {len(b)}")
            
        # 检查锥约束维度
        if cone_dims is None:
            raise ValueError("锥约束维度 cone_dims 必须提供")
            
        # 计算总锥维度
        total_cone_dim = 0
        if 'z' in cone_dims:
            total_cone_dim += cone_dims['z']
        if 'l' in cone_dims:
            total_cone_dim += cone_dims['l']
        if 'q' in cone_dims:
            if isinstance(cone_dims['q'], list):
                total_cone_dim += sum(cone_dims['q'])
            else:
                total_cone_dim += cone_dims['q']
            
        if total_cone_dim != m:
            raise ValueError(
                f"锥约束总维度 {total_cone_dim} 与约束数量 {m} 不匹配。"
                f"cone_dims = {cone_dims}"
            )


def create_cone_dims(n_eq: int = 0, n_ineq: int = 0, soc_dims: Optional[List[int]] = None) -> Dict:
    """
    辅助函数：创建锥约束维度字典
    
    Parameters
    ----------
    n_eq : int, default=0
        等式约束数量（零锥）
    n_ineq : int, default=0
        不等式约束数量（非负锥）
    soc_dims : list of int, optional
        二阶锥维度列表
        
    Returns
    -------
    dict
        锥约束维度字典
        
    Examples
    --------
    >>> # 10个不等式约束
    >>> cone_dims = create_cone_dims(n_ineq=10)
    >>> # {'l': 10}
    >>> 
    >>> # 5个等式约束 + 10个不等式约束
    >>> cone_dims = create_cone_dims(n_eq=5, n_ineq=10)
    >>> # {'z': 5, 'l': 10}
    >>> 
    >>> # 5个等式约束 + 2个二阶锥（维度分别为3和4）
    >>> cone_dims = create_cone_dims(n_eq=5, soc_dims=[3, 4])
    >>> # {'z': 5, 'q': [3, 4]}
    """
    cone_dims = {}
    
    if n_eq > 0:
        cone_dims['z'] = n_eq
        
    if n_ineq > 0:
        cone_dims['l'] = n_ineq
        
    if soc_dims:
        cone_dims['q'] = soc_dims
        
    return cone_dims


# 为了向后兼容，提供别名
ClarabelGPU = ClarabelSolver


__all__ = ['ClarabelSolver', 'ClarabelGPU', 'create_cone_dims', 'HAS_CUPY']
