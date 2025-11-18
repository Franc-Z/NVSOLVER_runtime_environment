"""
High-level Python interface for Clarabel GPU solver
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

# Import the solver implementation
try:
    # Try relative import first (when used as a package)
    from . import clarabel_gpu_enhanced as clarabel_gpu
except ImportError:
    try:
        # Try direct import
        import clarabel_gpu_enhanced as clarabel_gpu
    except ImportError:
        try:
            # Try original name
            import clarabel_gpu
        except ImportError:
            # Fall back to ctypes implementation if Cython extension is not available
            import warnings
            warnings.warn(
                "无法导入编译的 clarabel_gpu 扩展，使用 ctypes 接口作为后备方案。\n"
                "要获得最佳性能，请编译 Cython 扩展。"
            )
            import clarabel_gpu_ctypes as clarabel_gpu

class ClarabelGPU:
    """
    High-level interface for Clarabel GPU solver
    
    This solver supports quadratic programming (QP) and second-order cone 
    programming (SOCP) problems of the form:
    
    minimize    0.5 * x' * P * x + q' * x
    subject to  A * x + s = b
                s in K
                
    where K is a product of zero, nonnegative, and second-order cones.
    """
    
    def __init__(self, 
                 P: Optional[Union[np.ndarray, sp.spmatrix]] = None,
                 q: np.ndarray = None,
                 A: Union[np.ndarray, sp.spmatrix] = None,
                 b: np.ndarray = None,
                 cone_dims: Dict[str, Union[int, List[int]]] = None,
                 **settings):
        """
        Initialize the Clarabel GPU solver
        
        Parameters
        ----------
        P : array_like or sparse matrix, optional
            Quadratic objective matrix (must be symmetric positive semidefinite)
        q : array_like
            Linear objective vector
        A : array_like or sparse matrix
            Linear constraint matrix
        b : array_like
            Linear constraint vector
        cone_dims : dict
            Dictionary specifying cone dimensions:
            - 'z': int, dimension of zero cone (equality constraints)
            - 'l': int, dimension of nonnegative cone (inequality constraints)
            - 'q': list of int, dimensions of second-order cones
        **settings : keyword arguments
            Solver settings (see SolverSettings for options)
        """
        # Note: clarabel_gpu should be the real Cython extension (.so file), not a mock (.py file)
        # Check if we're using the real compiled extension
        module_file = getattr(clarabel_gpu, '__file__', '')
        if module_file.endswith('.py'):
            raise ImportError(
                f"错误：正在使用 Python 模拟实现 ({module_file})。\n"
                "请编译真正的 Clarabel GPU 扩展：\n"
                "  cd /clarabel/python\n"
                "  python setup.py build_ext --inplace"
            )
        
        # Use the real PyGPUSolver from the compiled extension
        self._solver = clarabel_gpu.PyGPUSolver()
        self._problem_data = {
            'P': P,
            'q': q,
            'A': A,
            'b': b,
            'cone_dims': cone_dims
        }
        self._settings = settings
        self._setup_called = False
        
    def setup(self):
        """Setup the optimization problem"""
        if self._setup_called:
            warnings.warn("Setup already called. Ignoring repeated call.")
            return
            
        # Validate inputs
        self._validate_inputs()
        
        # Setup the solver
        self._solver.setup(
            self._problem_data['P'],
            self._problem_data['q'],
            self._problem_data['A'],
            self._problem_data['b'],
            self._problem_data['cone_dims'],
            **self._settings
        )
        
        self._setup_called = True
        
    def solve(self) -> Dict:
        """
        Solve the optimization problem
        
        Returns
        -------
        dict
            Solution dictionary containing:
            - 'status': solver status string
            - 'x': primal solution
            - 'y': dual solution (Lagrange multipliers)
            - 's': slack variables
            - 'obj_val': objective value
            - 'solve_time': solution time in seconds
            - 'iterations': number of iterations
        """
        if not self._setup_called:
            self.setup()
            
        result = self._solver.solve()
        
        # Map 'z' to 'y' for consistency with CVXPy
        result['y'] = result.pop('z', None)
        
        return result
        
    def update_P(self, P_new: Union[np.ndarray, sp.spmatrix, 'cp.ndarray']):
        """
        Update the quadratic objective matrix P
        
        Parameters
        ----------
        P_new : array_like, sparse matrix, or CuPy array
            New P matrix (must have same sparsity pattern as original)
        """
        if not self._setup_called:
            raise RuntimeError("Cannot update P before calling setup()")
            
        self._solver.update_P(P_new)
        
    def update_q(self, q_new: Union[np.ndarray, 'cp.ndarray']):
        """
        Update the linear objective vector q
        
        Parameters
        ----------
        q_new : array_like or CuPy array
            New q vector
        """
        if not self._setup_called:
            raise RuntimeError("Cannot update q before calling setup()")
            
        self._solver.update_q(q_new)
        
    def update_A(self, A_new: Union[np.ndarray, sp.spmatrix, 'cp.ndarray']):
        """
        Update the constraint matrix A
        
        Parameters
        ----------
        A_new : array_like, sparse matrix, or CuPy array
            New A matrix (must have same sparsity pattern as original)
        """
        if not self._setup_called:
            raise RuntimeError("Cannot update A before calling setup()")
            
        self._solver.update_A(A_new)
        
    def update_b(self, b_new: Union[np.ndarray, 'cp.ndarray']):
        """
        Update the constraint vector b
        
        Parameters
        ----------
        b_new : array_like or CuPy array
            New b vector
        """
        if not self._setup_called:
            raise RuntimeError("Cannot update b before calling setup()")
            
        self._solver.update_b(b_new)
        
    def update_data(self, 
                    P: Optional[Union[np.ndarray, sp.spmatrix, 'cp.ndarray']] = None,
                    q: Optional[Union[np.ndarray, 'cp.ndarray']] = None,
                    A: Optional[Union[np.ndarray, sp.spmatrix, 'cp.ndarray']] = None,
                    b: Optional[Union[np.ndarray, 'cp.ndarray']] = None):
        """
        Update multiple problem data components at once
        
        Parameters
        ----------
        P, q, A, b : array_like, sparse matrix, or CuPy array, optional
            New data values (only non-None values are updated)
        """
        if P is not None:
            self.update_P(P)
        if q is not None:
            self.update_q(q)
        if A is not None:
            self.update_A(A)
        if b is not None:
            self.update_b(b)
            
    def _validate_inputs(self):
        """Validate problem inputs"""
        P = self._problem_data['P']
        q = self._problem_data['q']
        A = self._problem_data['A']
        b = self._problem_data['b']
        cone_dims = self._problem_data['cone_dims']
        
        # Check dimensions
        if q is None:
            raise ValueError("Linear objective vector q must be provided")
            
        n = len(q)
        
        if P is not None:
            P_shape = P.shape if hasattr(P, 'shape') else (len(P), len(P[0]))
            if P_shape != (n, n):
                raise ValueError(f"P must be square matrix of size {n}x{n}, got {P_shape}")
                
        if A is None:
            raise ValueError("Constraint matrix A must be provided")
            
        A_shape = A.shape if hasattr(A, 'shape') else (len(A), len(A[0]))
        m, n_A = A_shape
        
        if n_A != n:
            raise ValueError(f"A must have {n} columns, got {n_A}")
            
        if b is None:
            raise ValueError("Constraint vector b must be provided")
            
        if len(b) != m:
            raise ValueError(f"b must have length {m}, got {len(b)}")
            
        # Check cone dimensions
        if cone_dims is None:
            raise ValueError("Cone dimensions must be provided")
            
        total_cone_dim = 0
        if 'z' in cone_dims:
            total_cone_dim += cone_dims['z']
        if 'l' in cone_dims:
            total_cone_dim += cone_dims['l']
        if 'q' in cone_dims:
            total_cone_dim += sum(cone_dims['q'])
            
        if total_cone_dim != m:
            raise ValueError(f"Total cone dimension {total_cone_dim} != number of constraints {m}")


def create_cone_dims(n_eq: int = 0, n_ineq: int = 0, soc_dims: List[int] = None) -> Dict:
    """
    Helper function to create cone dimension dictionary
    
    Parameters
    ----------
    n_eq : int
        Number of equality constraints
    n_ineq : int
        Number of inequality constraints
    soc_dims : list of int, optional
        Dimensions of second-order cones
        
    Returns
    -------
    dict
        Cone dimensions dictionary
    """
    cone_dims = {}
    
    if n_eq > 0:
        cone_dims['z'] = n_eq
        
    if n_ineq > 0:
        cone_dims['l'] = n_ineq
        
    if soc_dims:
        cone_dims['q'] = soc_dims
        
    return cone_dims
