# ClarabelGPU User Guide --- A QP/Conic Solver written in CUDA C++ 

> **Important**: The convex optimization solver produced by this project (ClarabelGPU) **can be used as a fully standalone, high-performance convex optimizer** without any dependency on the differentiable functionality (DiffSolver). Built on a GPU-accelerated interior-point method, it achieves 2.5x–8x speedup over MOSEK and 2x–15x speedup over CPU-based Clarabel across workloads such as portfolio optimization, optimal power flow, and model predictive control. It is especially well-suited for parametric optimization scenarios that require high-frequency repeated solves (e.g., daily or intraday portfolio rebalancing).

---

## Table of Contents

1. [Computational Principles and Supported Problem Types](#1-computational-principles-and-supported-problem-types)
2. [C++ Solver Configuration and Usage](#2-c-solver-configuration-and-usage)
3. [Python Bindings and API Reference](#3-python-bindings-and-api-reference)
4. [CBF File Format Support](#4-cbf-file-format-support)
5. [CVXPY Integration and Usage](#5-cvxpy-integration-and-usage)
6. [Constraint Tightness Analysis](#6-constraint-tightness-analysis)
7. [Performance Optimization Tips](#7-performance-optimization-tips)
8. [Frequently Asked Questions](#8-frequently-asked-questions)

---

## 1. Computational Principles and Supported Problem Types

### 1.1 Standard Problem Form

ClarabelGPU solves convex optimization problems of the following form:

$$
\begin{aligned}
\text{minimize} \quad & \frac{1}{2} x^T P x + q^T x \\
\text{subject to} \quad & Ax + s = b, \quad s \in \mathcal{K}
\end{aligned}
$$

where:
- $x \in \mathbb{R}^n$ is the decision variable
- $P \in \mathbb{S}_+^n$ is a positive semidefinite cost matrix (nonzero for QP, zero for LP)
- $q \in \mathbb{R}^n$ is the linear cost vector
- $A \in \mathbb{R}^{m \times n}$ is the constraint matrix
- $b \in \mathbb{R}^m$ is the constraint right-hand side vector
- $s \in \mathbb{R}^m$ is the slack variable
- $\mathcal{K}$ is a Cartesian product of cone constraints

### 1.2 Supported Cone Types

| Cone Type | Mathematical Definition | Dimension | Typical Applications |
|-----------|------------------------|-----------|---------------------|
| **Zero Cone** (ZeroConeT) | $\{0\}^n$ | arbitrary | Equality constraints |
| **Nonnegative Cone** (NonnegativeConeT) | $\mathbb{R}_+^n = \{x \mid x_i \ge 0\}$ | arbitrary | Inequality constraints |
| **Second-Order Cone** (SecondOrderConeT) | $\{(t,x) \mid \|x\|_2 \le t\}$ | arbitrary | SOCP, risk constraints |
| **Exponential Cone** (ExponentialConeT) | $\{(x,y,z) \mid y e^{x/y} \le z, y > 0\}$ | 3 | Entropy maximization, log constraints |
| **Power Cone** (PowerConeT) | $\{(x,y,z) \mid x^\alpha y^{1-\alpha} \ge |z|\}$ | 3 | Geometric programming |
| **PSD Triangle Cone** (PSDTriangleConeT) | $\{X \in \mathbb{S}^n \mid X \succeq 0\}$ | $n(n+1)/2$ | SDP (mixed dims; $n \le 32$ batched, $n > 32$ per-cone) |

### 1.3 Supported Problem Types

By combining the above cones, ClarabelGPU can solve:

- **Linear Programming (LP)**: Zero cone + Nonnegative cone
- **Quadratic Programming (QP)**: Zero cone + Nonnegative cone + Quadratic objective
- **Second-Order Cone Programming (SOCP)**: Second-order cone constraints
- **Exponential Cone Programming**: Exponential cone constraints
- **Power Cone Programming**: Power cone constraints
- **Semidefinite Programming (SDP)**: PSD cone constraints (mixed dimensions supported; $n \le 32$ uses batched cuSOLVER, $n > 32$ falls back to per-cone cuSOLVER)
- **Mixed Conic Programming**: Any combination of the above

### 1.4 Solution Algorithm

ClarabelGPU is based on the **Primal-Dual Interior-Point Method**. The core workflow is:

1. **Data Equilibration**: Ruiz equilibration to improve problem conditioning
2. **KKT System Solve**: Sparse $LDL^T$ factorization on GPU via the NVIDIA cuDSS library
3. **Cone Operation Parallelization**: Mixed Parallel Computing strategy — different cone families are processed on independent CUDA streams
4. **Iterative Refinement**: Optional iterative refinement for improved numerical stability
5. **Termination Check**: Based on primal/dual residuals, duality gap, and complementary slackness

Key performance characteristics:
- Symbolic analysis is performed only once; subsequent parameter updates require only numeric refactorization
- Both forward and backward solvers use `CuDSSContext` for unified cuDSS lifecycle management — the forward solver uses minimal configuration (no pivoting, no cuDSS-internal IR) for maximum speed, with persistent RHS/SOL descriptors (`solve_persistent`) to avoid per-solve descriptor allocation
- RMM memory pool auto-initialization with adaptive GPU sizing (256-byte aligned); cuDSS internal allocations are also routed through the pool via `cudssDeviceMemHandler`
- A `tracking_resource_adaptor` layer on the pool provides live GPU memory usage statistics (accessible via `get_memory_info()`)

### 1.5 Reference

1. Chen Y., Tse D., Nobel P., Goulart P., Boyd S. "CuClarabel: GPU Acceleration for a Conic Optimization Solver." arXiv:2412.19027, 2024.
2. [clarabel online document](https://clarabel.org/stable/)
---

## 2. C++ Solver Configuration and Usage

### 2.1 Basic Usage

```cpp
#include <raft/core/handle.hpp>
#include <clarabel/core/solver.hpp>
#include <clarabel/core/settings.hpp>
#include <clarabel/utils/host_sparse_matrix.hpp>
#include <clarabel/cones/cone_specs.hpp>
#include <clarabel/core/data_updater.hpp>

using namespace clarabel;

// 1. Create a RAFT handle (manages CUDA context)
raft::handle_t handle;

// 2. Construct problem data (CSR format)
HostCsrMatrix<double, int32_t> P(n, n, P_values, P_col_indices, P_row_offsets);
HostCsrMatrix<double, int32_t> A(m, n, A_values, A_col_indices, A_row_offsets);
std::vector<double> q = {...};
std::vector<double> b = {...};

// 3. Define cone constraints
std::vector<cones::ConeSpec> cones;
cones.push_back(cones::ZeroCone(n_eq));           // Equality constraints
cones.push_back(cones::NonnegativeCone(n_ineq));  // Inequality constraints
cones.push_back(cones::SecondOrderCone(soc_dim));  // SOC constraints

// 4. Configure solver settings
SolverSettings settings;
settings.verbose = true;
settings.max_iter = 200;
settings.tol_feas = 1e-8;
settings.tol_gap_abs = 1e-8;
settings.tol_gap_rel = 1e-8;

// 5. Create the solver and solve
Solver<double> solver(&handle, P, q, A, b, cones, settings);
SolverStatus status = solver.solve();

// 6. Retrieve the solution
if (status == SolverStatus::SOLVED) {
    const auto& sol = solver.solution();
    // sol.x()           — primal variables (GPU device_uvector)
    // sol.z()           — dual variables
    // sol.s()           — slack variables
    // sol.obj_val()     — primal objective value
    // sol.obj_val_dual() — dual objective value
    // sol.iterations()  — iteration count
    // sol.solve_time()  — solve time (seconds)
}
```

### 2.2 Parameter Updates and Re-solves

```cpp
// Create a data updater
auto& problem_data = const_cast<ProblemData<double>&>(solver.data());
DataUpdater<double> updater(handle, problem_data);

// Update q vector (full replacement)
std::vector<double> q_new = {...};
updater.update_q(q_new);

// Update b vector (sparse update: modify only selected elements)
std::vector<std::pair<int, double>> b_updates = {{5000, 0.95}, {5001, 1.05}};
updater.update_b(b_updates);

// Update P matrix values (sparsity pattern must remain unchanged)
std::vector<double> P_new_values = {...};
updater.update_P(P_new_values);

// Re-solve (reuses symbolic analysis)
SolverStatus status = solver.solve();
```

### 2.3 Solver Parameter Reference

#### Main Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `verbose` | `false` | Print solve progress details |
| `max_iter` | `200` | Maximum number of iterations |
| `time_limit` | `Inf` | Maximum solve time (seconds) |
| `max_step_fraction` | `0.99` | Maximum step-size fraction |

#### Full Accuracy Tolerances

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tol_gap_abs` | `1e-8` | Absolute duality gap tolerance |
| `tol_gap_rel` | `1e-8` | Relative duality gap tolerance |
| `tol_feas` | `1e-8` | Feasibility tolerance |
| `tol_infeas_abs` | `1e-8` | Absolute infeasibility tolerance |
| `tol_infeas_rel` | `1e-8` | Relative infeasibility tolerance |
| `tol_ktratio` | `1e-6` | kappa/tau ratio tolerance |

#### Reduced Accuracy Tolerances

The corresponding `reduced_tol_*` parameters are used when full accuracy cannot be achieved.

#### Regularization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sr_enable` | `true` | Enable static regularization |
| `sr_constant` | `1e-8` | Static regularization constant $\delta_s$ |
| `sr_proportional` | `eps^2` | Proportional regularization coefficient |
| `dr_enable` | `true` | Enable dynamic regularization |
| `dr_eps` | `1e-13` | Dynamic regularization threshold |
| `dr_delta` | `2e-7` | Dynamic regularization increment |

#### Iterative Refinement Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ir_enable` | `true` | Enable iterative refinement |
| `ir_reltol` | `1e-12` | Iterative refinement relative tolerance |
| `ir_abstol` | `1e-12` | Iterative refinement absolute tolerance |
| `ir_max_iter` | `10` | Maximum iterative refinement steps |
| `ir_stop_ratio` | `5.0` | Iterative refinement stopping ratio |

#### Data Equilibration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `equilibrate_enable` | `true` | Enable Ruiz equilibration |
| `equilibrate_max_iter` | `10` | Maximum equilibration iterations |
| `equilibrate_min_scaling` | `1e-4` | Minimum scaling factor |
| `equilibrate_max_scaling` | `1e+4` | Maximum scaling factor |

### 2.4 Solver Status Enum

| Status | Meaning |
|--------|---------|
| `SOLVED` | Successfully solved to full accuracy |
| `ALMOST_SOLVED` | Solved to reduced accuracy |
| `PRIMAL_INFEASIBLE` | Primal problem is infeasible |
| `DUAL_INFEASIBLE` | Dual problem is infeasible |
| `MAX_ITERATIONS` | Maximum iteration limit reached |
| `MAX_TIME` | Time limit reached |
| `NUMERICAL_ERROR` | Numerical error encountered |
| `INSUFFICIENT_PROGRESS` | Insufficient iteration progress |

### 2.5 RMM Memory Pool

**Automatic pool initialization**: Starting from the current version, the solver automatically detects whether a high-performance RMM memory resource is active. If not (i.e., the default bare `cudaMalloc` allocator), it upgrades to a `pool_memory_resource<cuda_memory_resource>` with a fixed 256 MB initial pool (grows on demand up to 90% of free GPU memory). This means **no manual RMM setup is required** for basic C++ usage.

If you prefer explicit control, you can set up the pool before creating the solver:

```cpp
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

// Adaptive: query available GPU memory
size_t free_bytes = 0, total_bytes = 0;
cudaMemGetInfo(&free_bytes, &total_bytes);

auto upstream = std::make_shared<rmm::mr::cuda_memory_resource>();
auto pool = std::make_unique<rmm::mr::pool_memory_resource<
    rmm::mr::cuda_memory_resource>>(
    upstream.get(),
    256 << 20,            // initial: 256 MB
    free_bytes * 9 / 10   // maximum: 90% of free GPU memory
);
rmm::mr::set_current_device_resource(pool.get());

// Now create the solver — it will detect the pool and skip auto-init.
Solver<double> solver(&handle, P, q, A, b, cones, settings);
```

**Why pool matters**: The solver's interior-point iterations involve repeated allocation/deallocation of cuDSS work buffers and Thrust temporaries. Without pooling, each allocation triggers a synchronous `cudaMalloc`. The pool provides O(1) sub-allocation and also allows cuDSS internal allocations to share the same memory pool (via `cudssDeviceMemHandler`), avoiding two allocators competing for GPU memory.

**Important**: If you set up RMM yourself, ensure the pool object outlives the solver — RMM resources must persist until all `rmm::device_uvector` objects allocated through them are destroyed.

---

## 3. Python Bindings and API Reference

### 3.1 ClarabelGPU Class (Standalone Forward Solver)

`ClarabelGPU` is the Cython-wrapped Python interface defined in `clarabel_gpu_enhanced.pyx`, compiled into `clarabel_gpu.so`.

#### Basic Usage

```python
from clarabel_gpu import ClarabelGPU, ZeroConeT, NonnegativeConeT, SecondOrderConeT

# Create the solver
solver = ClarabelGPU()

# Set up the problem (P, q, A, b can be numpy or scipy.sparse)
solver.setup(
    P=P_csr,       # scipy.sparse CSR matrix or None
    q=q_array,     # numpy 1-D array
    A=A_csr,       # scipy.sparse CSR matrix
    b=b_array,     # numpy 1-D array
    cone_dims=cones,  # Cone constraints (three formats, see below)
    gpu_mode=True,    # Enable GPU mode
    verbose=True,
    max_iter=200,
    tol_feas=1e-8,
    tol_gap_abs=1e-8,
    tol_gap_rel=1e-8,
)

# Solve
result = solver.solve()
# result dict contains:
#   'status'       — solver status string
#   'obj_val'      — primal objective value
#   'obj_val_dual' — dual objective value
#   'solve_time'   — solve time (seconds)
#   'iterations'   — iteration count
#   'r_prim'       — primal residual
#   'r_dual'       — dual residual
#   'x'            — primal variables (numpy/cupy array)
#   'z'            — dual variables
#   's'            — slack variables
```

#### Three Cone Specification Formats

```python
# Format 1: Official Clarabel object list (recommended)
cones = [
    ZeroConeT(5),            # 5 equality constraints
    NonnegativeConeT(10),    # 10 inequality constraints
    SecondOrderConeT(51),    # SOC of dimension 51
]

# Format 2: Tuple list
cones = [
    ('z', 5),
    ('l', 10),
    ('q', 51),
    ('ep',),       # Exponential cone (always 3-D)
    ('p', 0.3),    # Power cone (alpha=0.3)
]

# Format 3: SCS/CVXPY dict format
cones = {
    'z': 5,           # Zero cone
    'l': 10,          # Nonnegative cone
    'q': [51, 51],    # SOC dimension list
    'ep': 3,          # 3 exponential cones
    'p': [0.3, 0.7],  # Power cone alpha list
}
```

#### Parameter Updates (CPU)

After the initial `setup()` + `solve()`, you can update problem data and re-solve
**without** rebuilding the solver.  The symbolic analysis from the first solve is
cached; subsequent solves only perform numeric refactorization, yielding 2–10x
speedup over a cold start.

| Method | Input | What is Updated |
|--------|-------|-----------------|
| `update_P(P_new)` | scipy sparse matrix **or** 1-D numpy array of values | Nonzero values of the $P$ matrix. Sparsity pattern must remain unchanged. |
| `update_q(q_new)` | numpy 1-D array of length $n$ | Linear cost vector $q$. |
| `update_A(A_new)` | scipy sparse matrix **or** 1-D numpy array of values | Nonzero values of the $A$ matrix. Sparsity pattern must remain unchanged. |
| `update_b(b_new)` | numpy 1-D array of length $m$ | Constraint RHS vector $b$. |

**Constraints**:
- The problem dimensions ($n$, $m$) and sparsity patterns of $P$ and $A$ **must not change** between updates. Only numerical values may differ.
- When passing a sparse matrix, the method extracts `.data` (the nonzero values) internally. When passing a 1-D array, it must contain exactly the same number of elements as the original matrix's `nnz`.

```python
# Typical parametric solve loop
solver.setup(P, q, A, b, cones, verbose=True, **settings)
result = solver.solve()          # Cold start (includes symbolic analysis)

for q_new, b_new in data_stream:
    solver.update_q(q_new)       # Update linear cost
    solver.update_b(b_new)       # Update constraint RHS
    result = solver.solve()      # Warm re-solve (numeric refactorization only)

# P and A can also be updated when their values change
solver.update_P(P_new)           # Update objective matrix values
solver.update_A(A_new)           # Update constraint matrix values
result = solver.solve()
```

#### GPU Direct Transfer Updates (Zero-Copy)

When `gpu_mode=True`, you can update data using CuPy arrays directly on the GPU,
avoiding CPU↔GPU round trips entirely.

| Method | Input | What is Updated |
|--------|-------|-----------------|
| `update_P(P_new)` | NumPy/SciPy or CuPy array | $P$ matrix nonzero values (auto-detects GPU/CPU) |
| `update_q(q_new)` | NumPy or CuPy array of length $n$ | Linear cost vector $q$ (GPU direct if CuPy) |
| `update_A(A_new)` | NumPy/SciPy or CuPy array | $A$ matrix nonzero values (always syncs KKT) |
| `update_b(b_new)` | NumPy or CuPy array of length $m$ | Constraint RHS vector $b$ (GPU direct if CuPy, handles cone permutation) |

```python
import cupy as cp

q_gpu = cp.asarray(q_new)
b_gpu = cp.asarray(b_new)

solver.update_q(q_gpu)    # auto-detects CuPy → GPU direct transfer
solver.update_b(b_gpu)
result = solver.solve()
```

> **Auto-detection**: The standard `update_q()` / `update_b()` / `update_P()` / `update_A()` methods also accept CuPy arrays when `gpu_mode=True` — they automatically dispatch to the GPU path.

#### Runtime Controls

```python
solver.set_verbose(True)    # Toggle verbose output at runtime
solver.get_verbose()        # Query current setting

# GPU memory diagnostics
info = solver.get_memory_info()   # Returns dict with GPU memory usage
# info['cupy_used_mb']  — CuPy-visible allocation (if RMM enabled)
# info['rmm_enabled']   — Whether RMM pool is active

solver.reset_memory_pool()  # Release unused pool memory back to the OS
```

> **Note**: The solver auto-initializes an RMM pool if none is configured.
> `get_memory_info()` reports CuPy-visible allocations; for full-stack tracking
> (including cuDSS internal buffers), use `rmm.statistics` on the Python side
> or DiffSolver's `get_memory_stats()['rmm_current_mb']`.

---

## 4. CBF File Format Support

ClarabelGPU includes an integrated parser for the **MOSEK Conic Benchmark Format (CBF)**, a standard file format for conic optimization problems.  CBF files can be loaded and solved directly from both C++ and Python without any external dependencies.

> **Reference**: [MOSEK CBF Format Specification](https://docs.mosek.com/latest/capi/cbf-format.html)

### 4.1 Supported CBF Features

| Feature | Status | Notes |
|---------|--------|-------|
| **All 17 keywords** (VER … DCOORD) | ✓ | Full spec compliance |
| **All 12 cone types** (F, L=, L+, L−, Q, QR, EXP, EXP\*, GMEANABS, GMEANABS\*, @k:POW, @k:POW\*) | ✓ | Parsed and recognized |
| **PSD variables / constraints** (PSDVAR, PSDCON, OBJFCOORD, FCOORD, HCOORD, DCOORD) | ✓ | svec column-major upper-triangular expansion |
| **Integer variables** (INT) | ⚠ | Parsed with warning; integrality ignored (continuous relaxation) |
| **Information group ordering** | ✓ | File format → Problem structure → Problem data |
| **Windows line endings** (\r\n) | ✓ | Carriage returns stripped |

#### CBF → Clarabel Conversion Details

The loader automatically converts CBF's conic form into Clarabel's standard form $Ax + s = b,\; s \in \mathcal{K}$:

| CBF Cone | Clarabel Cone | Conversion |
|----------|--------------|------------|
| `F` (free) | — | No constraint rows added |
| `L=` (zero) | `ZeroConeT` | Direct mapping |
| `L+` (nonneg) | `NonnegativeConeT` | Direct mapping |
| `L−` (nonpos) | `NonnegativeConeT` | Sign flip: $s = -x \ge 0$ |
| `Q` (SOC) | `SecondOrderConeT` | Direct mapping |
| `QR` (rotated SOC) | `SecondOrderConeT` | $1/\sqrt{2}$ linear transform: $(w_1, w_2) = \frac{1}{\sqrt{2}}(p \pm q)$ |
| `EXP` | `ExponentialConeT` | Row reorder `[2,1,0]`: CBF $(t,s,r)$ → Clarabel $(r,s,t)$ |
| `@k:POW` (3-dim) | `PowerConeT(α)` | $\alpha = \alpha_1 / (\alpha_1 + \alpha_2)$ |
| `PSDVAR` / `PSDCON` | `PSDTriangleConeT` | svec expansion with $\sqrt{2}$ off-diagonal scaling |

### 4.2 C++ Usage

#### One-step load and solve

```cpp
#include "clarabel/io/cbf_loader.hpp"
#include "clarabel/core/solver.hpp"

raft::handle_t handle;

// Load CBF file → Clarabel standard form
auto prob = clarabel::load_cbf<double>("problem.cbf");

// Create solver and solve
SolverSettings settings;
settings.verbose = true;
Solver<double> solver(&handle, prob.P, prob.q, prob.A, prob.b, prob.cones, settings);
solver.solve();

// Correct objective for MAX sense and OBJBCOORD constant
double obj = solver.solution().obj_val();
if (prob.was_maximize) obj = -obj;
obj += prob.obj_constant;
```

#### Command-line solver

```bash
# Build
make cbf_solver

# Solve one or more CBF files
./cbf_solver problem.cbf
./cbf_solver -q --tol 1e-6 a.cbf b.cbf c.cbf
./cbf_solver --help
```

Options: `-q` (quiet), `-t` (tolerance), `-i` (max iterations), `-p` (GPU pool MB), `-x` (print x components).

### 4.3 Python Usage

#### Method 1: `solve_cbf()` — Load + solve in one call (fastest)

All processing stays in C++ — no scipy/numpy round-trip overhead.

```python
from clarabel_gpu import ClarabelGPU

solver = ClarabelGPU()
result = solver.solve_cbf("problem.cbf", verbose=True, tol_feas=1e-8)

print(result['status'])       # 'solved'
print(result['obj_val'])      # objective (corrected for MAX and OBJBCOORD)
print(result['iterations'])   # interior-point iterations
print(result['x'][:5])        # first 5 solution components
```

**Return dict keys**: `status`, `obj_val`, `obj_val_dual`, `solve_time`, `iterations`, `r_prim`, `r_dual`, `x`, `z`, `s`, `n`, `m`, `obj_constant`, `was_maximize`.

#### Method 2: `load_cbf()` — Load as Python objects, inspect/modify, then solve

```python
from clarabel_gpu import ClarabelGPU, load_cbf

# Load CBF → Python objects
data = load_cbf("problem.cbf")

# Inspect
print(f"n={data['n']}, m={data['m']}")
print(f"cones: {data['cones']}")
# [ZeroConeT(2), NonnegativeConeT(10), SecondOrderConeT(3), ...]

print(f"A shape: {data['A'].shape}, nnz: {data['A'].nnz}")
print(f"q: {data['q']}")

# Optionally modify
data['q'] *= 1.5  # scale objective

# Solve
solver = ClarabelGPU()
solver.setup(data['P'], data['q'], data['A'], data['b'], data['cones'],
             verbose=True, tol_feas=1e-8)
result = solver.solve()
```

**Return dict keys**: `P` (scipy CSR), `q` (numpy), `A` (scipy CSR), `b` (numpy), `cones` (list of cone objects), `n`, `m`, `obj_constant`, `was_maximize`.

#### Method 3: Command-line from Python

```python
import sys
from clarabel_gpu import ClarabelGPU

solver = ClarabelGPU()
for path in sys.argv[1:]:
    result = solver.solve_cbf(path, verbose=True)
    print(f"{path}: {result['status']}, obj={result['obj_val']:.8e}")
```

### 4.4 Example: Solving Benchmark CBF Files

```python
from clarabel_gpu import ClarabelGPU

solver = ClarabelGPU()

# EXP cones + integer relaxation
r1 = solver.solve_cbf("exp_ising.cbf", verbose=False)
print(f"exp_ising:  {r1['status']}, obj={r1['obj_val']:.6f}, iter={r1['iterations']}")

# PSD constraint (21×21 SDP)
r2 = solver.solve_cbf("sdp_cardls.cbf", verbose=False)
print(f"sdp_cardls: {r2['status']}, obj={r2['obj_val']:.6f}, iter={r2['iterations']}")

# Rotated SOC (QR → SOC conversion)
r3 = solver.solve_cbf("sssd_strong_15_4.cbf", verbose=False)
print(f"sssd:       {r3['status']}, obj={r3['obj_val']:.6f}, iter={r3['iterations']}")
```

### 4.5 CBF API Reference

#### `solver.solve_cbf(path, **settings) → dict`

Load a CBF file and solve on GPU in one step.  Settings keywords are the same as `setup()` (`verbose`, `max_iter`, `tol_feas`, `tol_gap_abs`, etc.).  The returned objective value is automatically corrected for MAX direction and OBJBCOORD constant.

#### `load_cbf(path) → dict`   *(module-level function)*

Parse a CBF file and return all problem data as Python objects.  `P` and `A` are `scipy.sparse.csr_matrix`, `q` and `b` are `numpy.ndarray`, and `cones` is a list of Clarabel cone objects (`ZeroConeT`, `NonnegativeConeT`, `SecondOrderConeT`, `ExponentialConeT`, `PowerConeT`, `PSDTriangleConeT`).  The data can be inspected, modified, and passed to `solver.setup()`.

#### `clarabel::load_cbf<T>(path) → CbfProblem<T>`   *(C++ function)*

C++ template function returning a `CbfProblem<T>` struct with fields: `P` (`HostCsrMatrix`), `q`, `A`, `b`, `cones` (`vector<ConeSpec>`), `n`, `m`, `obj_constant`, `was_maximize`.

---

## 5. CVXPY Integration and Usage

### 5.1 Integration Architecture

CVXPY integration is implemented via the `ClarabelGPU` class in `clarabelgpu_conif.py`. This class inherits from `ConicSolver`, canonicalizes the CVXPY problem into conic form, and dispatches it to the `ClarabelGPU` backend.

**Core mechanisms**:

- `ClarabelGPU` registers as a custom solver backend for CVXPY, it owns all the functionality as `CuClarabel`
- Supports **DPP (Disciplined Parameterized Programming)**: reuses the compiled solver instance across parameter updates via CVXPY's standard `solver_cache` mechanism
- Status mapping: the backend's lowercase status strings (e.g., `solved`, `almost_solved`) are mapped to CVXPY constants (`OPTIMAL`, `OPTIMAL_INACCURATE`) in the `invert()` method
- Supported constraint types: `SOC`, `ExpCone`, `PowCone3D`, `PSD` (CVXPY automatically reduces `PowConeND` to `PowCone3D` before dispatching to the solver)

### 5.2 Basic Usage

```python
import cvxpy as cp
import numpy as np

# Define variables and parameters
x = cp.Variable(n, nonneg=True)
mu = cp.Parameter(n)  # DPP parameter
mu.value = expected_returns

# Build the problem
objective = cp.Minimize(-mu @ x + gamma * cp.quad_form(x, Sigma))
constraints = [cp.sum(x) == 1, x <= 0.1]
problem = cp.Problem(objective, constraints)

# Solve with ClarabelGPU
problem.solve(
    solver='CLARABELGPU',
    verbose=True,
    max_iter=200,
    tol_feas=1e-8,
    tol_gap_abs=1e-8,
    tol_gap_rel=1e-8,
)

print(f"Status: {problem.status}")
print(f"Objective: {problem.value}")
print(f"x = {x.value}")
```

### 5.3 DPP Parameterized Repeated Solves

```python
from clarabel_gpu import UpdateHint

# Initial solve (includes compilation and initialization)
problem.solve(solver='CLARABELGPU', verbose=True, **settings)

# Update parameters and re-solve (DPP acceleration, typically 5–50x speedup)
for new_data in data_stream:
    mu.value = new_data
    problem.solve(
        solver='CLARABELGPU',
        warm_start=True,  # Reuse cached solver instance (CVXPY standard)
        update_hints={UpdateHint.q_values_changed},  # Skip np.array_equal comparisons
        verbose=False,
        **settings
    )
    # problem.value contains the new optimal value
```

> **`update_hints`**: By default the CVXPY integration layer compares `A.indptr`, `A.indices`, `A.data`, etc. via `np.array_equal` to decide what changed. For large matrices this comparison itself can be costly. When you know exactly which data changed, pass a set of `UpdateHint` enum values to skip these comparisons entirely. See §7.1 for the full list of hints and their semantics.

### 5.4 CVXPY Solver Parameters

Solver settings can be passed as keyword arguments to `problem.solve()`. Two naming styles are supported:

| Official Name (CVXPY style) | Backend Short Name | Description |
|-----------------------------|-------------------|-------------|
| `static_regularization_enable` | `sr_enable` | Static regularization |
| `dynamic_regularization_enable` | `dr_enable` | Dynamic regularization |
| `iterative_refinement_enable` | `ir_enable` | Iterative refinement |
| `equilibrate_enable` | — | Data equilibration |
| `chordal_decomposition_enable` | — | Chordal decomposition (SDP) |

DPP warm-start behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warm_start` | `True` | CVXPY standard parameter. When `True`, the solver instance is cached in CVXPY's per-problem `solver_cache` and reused on subsequent solves if the problem structure (dimensions and cone types) matches. All problem data ($P$, $q$, $A$, $b$) is automatically updated in-place via the native `update_P/q/A/b` methods. When `False`, a fresh solver is always created. |
| `update_hints` | `None` | Optional `set` of `UpdateHint` enum values (from `clarabel_gpu`) specifying which data changed. When provided (and not containing `UpdateHint.automatic`), skips all `np.array_equal` comparisons in the warm-start cache path and directly performs the indicated updates. See §7.1 for the full enum definition and usage examples. |

> **Note on native update methods**: When warm-starting through CVXPY, the conif layer calls the native `update_P()`, `update_q()`, `update_A()`, and `update_b()` methods (documented in §3.1) to update all data in-place. This avoids tearing down and rebuilding the solver instance, and the symbolic analysis from the first solve is fully reused.  For direct control over which data to update, use the native `ClarabelGPU` interface (§3.1) instead of CVXPY.

### 5.5 SOCP Example (CVXPY)

```python
import cvxpy as cp

x = cp.Variable(n, nonneg=True)
t = cp.Variable()  # Risk auxiliary variable

constraints = [
    cp.sum(x) == 1,
    x <= 0.1,
    cp.norm(L @ x, 2) <= t,     # SOC risk constraint
    t <= max_risk,               # Risk upper bound
]

objective = cp.Minimize(-mu @ x + gamma * t + tc * cp.sum(z))
problem = cp.Problem(objective, constraints)
problem.solve(solver='CLARABELGPU', **settings)
```

### 5.6 Extracting Data from CVXPY for the Native Interface

For maximum performance, you can extract the standard-form data from CVXPY and call `ClarabelGPU` directly:

```python
# Extract CVXPY standard-form data
data = problem.get_problem_data('CLARABELGPU')
P, q, A, b = data[0]['P'], data[0]['c'], data[0]['A'], data[0]['b']
dims = data[0]['dims']

# Build cones
cone_dims = {'z': dims.zero, 'l': dims.nonneg}
if dims.soc:
    cone_dims['q'] = list(dims.soc)

# Use the native interface
from clarabel_gpu import ClarabelGPU
solver = ClarabelGPU()
solver.setup(P, q, A, b, cone_dims, gpu_mode=True, **settings)
result = solver.solve()
```

---

## 6. Constraint Tightness Analysis

After solving, determining whether each constraint is "tight" (active) or "loose" (inactive) is critical for optimization analysis. The methods differ by API level.

### 6.1 Core Principles

In the standard form $Ax + s = b, \; s \in \mathcal{K}$:
- The **slack variable $s$** is the **sole definitive criterion** for binding/slack determination. It is the mathematical definition itself, independent of KKT or duality theory.
- For scalar inequality ($s_i \ge 0$): $s_i = 0 \Leftrightarrow$ tight; $s_i > 0 \Leftrightarrow$ loose.
- For SOC ($s = (s_0, \bar{s}) \in \mathcal{K}_{soc}$): margin $m = s_0 - \|\bar{s}\|_2$; $m = 0 \Leftrightarrow$ tight (on cone boundary); $m > 0 \Leftrightarrow$ loose (cone interior).
- The **dual variable $z$** (shadow price) is **NOT needed** for binding determination. It only provides:
  - **Sensitivity analysis**: $\lambda_i$ is the marginal change in optimal objective per unit change in $b_i$.
  - **Degenerate constraint identification**: among tight constraints ($s_i = 0$), $\lambda_i > 0$ means the constraint actively restricts the optimum, while $\lambda_i = 0$ indicates a degenerate tight constraint.
  - The relationship is one-directional: $\lambda_i > 0 \Rightarrow s_i = 0$ (certain), but $s_i = 0 \not\Rightarrow \lambda_i > 0$ (may be degenerate).

Classification rules (based on slack variable — definitive):

| Constraint Type | Tight Condition | Loose Condition |
|-----------------|----------------|-----------------|
| Inequality (Nonnegative cone) | $s_i \le \text{tol}$ | $s_i \ge 10 \cdot \text{tol}$ |
| SOC constraint | margin $m = s_0 - \|\bar{s}\|_2 \le \text{tol}$ | margin $m \ge 10 \cdot \text{tol}$ |
| Equality (Zero cone) | Always tight ($s = 0$) | N/A |

#### SOC Constraint Tightness in Detail

**Lorentz cone and slack structure.**
A second-order cone (Lorentz cone) constraint has the standard form $(t, u) \in \mathcal{K}_{soc}$, meaning $t \ge \|u\|_2$. In the IPM conic framework $Ax + s = b,\; s \in \mathcal{K}$, the slack for an SOC constraint is a **vector** $s = (s_0, \bar{s}) \in \mathbb{R} \times \mathbb{R}^{d-1}$ that must itself lie in the Lorentz cone: $s_0 \ge \|\bar{s}\|_2$.

The tightness indicator is the **margin** (cone boundary distance):

$$
m = s_0 - \|\bar{s}\|_2
$$

| Condition | Meaning |
|-----------|---------|
| $m > 0$ | Strictly feasible — point is in the **interior** of the cone (loose) |
| $m \approx 0$ | On the **boundary** of the cone (tight / binding) |
| $m = 0$ and $s = 0$ | At the **apex** of the cone (most extreme tight) |
| $m < 0$ | Constraint **violated** (infeasible) |

This is fundamentally different from scalar inequalities where slack is a single number compared to zero. For SOC, you must check whether the entire slack vector lies on the cone boundary, which reduces to the scalar comparison $m \approx 0$.

**`cp.SOC(t, X)` in CVXPY — construction and args.**
In CVXPY, an SOC constraint is created via `cp.SOC(t, X)`, representing $\|X\|_2 \le t$:

| Field | Meaning | Mathematical role |
|-------|---------|-------------------|
| `c.args[0]` | `t` — the upper bound (scalar or vector) | Lorentz cone component $s_0$ |
| `c.args[1]` | `X` — the expression whose norm is bounded | Lorentz cone "body" $\bar{s}$ |

Together they form the Lorentz cone vector $(t, X) \in \mathcal{K}_{soc}$.

Note: `cp.norm(u, 2) <= t` in CVXPY typically creates a `NonPos` constraint (not `SOC`), in which case the scalar slack $-(\|u\|_2 - t) = t - \|u\|_2$ already equals the correct margin. The explicit `SOC` type applies when using `cp.SOC(t, X)` directly.

**Case 1: Scalar `t` (single SOC constraint).**
This is the most common case, e.g., `cp.SOC(max_risk, risk_vector)`:

```python
# c is a cp.constraints.second_order.SOC object
t_val = float(np.asarray(c.args[0].value).ravel()[0])   # scalar upper bound
X_val = np.asarray(c.args[1].value, dtype=float).ravel() # vector body

margin = t_val - np.linalg.norm(X_val)

tol = 1e-6
if margin <= tol:
    print(f"SOC constraint: TIGHT (on cone boundary, margin = {margin:.2e})")
elif margin >= 10 * tol:
    print(f"SOC constraint: LOOSE (cone interior, margin = {margin:.2e})")
else:
    print(f"SOC constraint: BORDERLINE (margin = {margin:.2e})")
```

**Case 2: Vector `t` (batch of SOC constraints).**
When `t` has shape `(n,)` and `X` has shape `(d, n)`, this represents `n` independent SOC constraints $\|X_{:,i}\|_2 \le t_i$:

```python
t_vals = np.asarray(c.args[0].value, dtype=float).ravel()  # shape (n,)
X_vals = np.asarray(c.args[1].value, dtype=float)           # shape (d, n)

margins = t_vals - np.linalg.norm(X_vals, axis=0)           # shape (n,)

for i, m in enumerate(margins):
    status = "TIGHT" if m <= tol else ("LOOSE" if m >= 10 * tol else "BORDERLINE")
    print(f"  SOC[{i}]: {status} (margin = {m:.2e})")
```

**Comparison: scalar inequality vs. SOC constraint slack.**

| Constraint type | Slack type | Tight condition |
|-----------------|-----------|-----------------|
| Scalar inequality $g(x) \le 0$ | Scalar $s \ge 0$ | $s = 0$ |
| SOC $(t, u) \in \mathcal{K}_{soc}$ | Vector $s \in \mathcal{K}_{soc}$ | $s_0 = \|\bar{s}\|_2$, i.e., $m = 0$ |
| SDP $X \succeq 0$ | Matrix $S \succeq 0$ | $\lambda_{\min}(S) = 0$, i.e., $S$ is singular |

### 6.2 Tightness Analysis in C++

```cpp
// Retrieve the solution
const auto& sol = solver.solution();
// sol.s() returns a GPU device_uvector<double>

// Copy slack variables to host
std::vector<double> s_host(m);
cudaMemcpy(s_host.data(), sol.s().data(), m * sizeof(double), cudaMemcpyDeviceToHost);

// Copy dual variables to host
std::vector<double> z_host(m);
cudaMemcpy(z_host.data(), sol.z().data(), m * sizeof(double), cudaMemcpyDeviceToHost);

double tol = 1e-6;
int offset = 0;

// 1. Zero cone (equality constraints) — always tight
// s[0..n_eq-1] should be near 0
offset += n_eq;

// 2. Nonnegative cone (inequality constraints)
for (int i = offset; i < offset + n_ineq; i++) {
    if (s_host[i] <= tol) {
        // Tight (active) constraint
    } else {
        // Loose (inactive) constraint
    }
}
offset += n_ineq;

// 3. SOC constraint: (t, x) in SOC, s = (s_t, s_x)
// Tight condition: s_t ≈ ||s_x||_2
for (auto soc_dim : soc_dims) {
    double s_t = s_host[offset];
    double s_x_norm = 0;
    for (int j = 1; j < soc_dim; j++) {
        s_x_norm += s_host[offset + j] * s_host[offset + j];
    }
    s_x_norm = std::sqrt(s_x_norm);
    double slack = s_t - s_x_norm;  // SOC slack
    if (slack <= tol) {
        // SOC constraint is tight
    }
    offset += soc_dim;
}
```

### 6.3 Tightness Analysis in ClarabelGPU Python API

```python
result = solver.solve()
s = result['s']  # Slack variables (numpy or cupy array)
z = result['z']  # Dual variables

tol = 1e-6
offset = 0

# 1. Equality constraints (Zero cone)
offset += n_eq

# 2. Inequality constraints (Nonnegative cone)
s_ineq = s[offset:offset + n_ineq]
tight_mask = s_ineq <= tol
print(f"Tight constraints: {tight_mask.sum()} / {n_ineq}")
offset += n_ineq

# 3. SOC constraints
for soc_dim in soc_dims:
    s_soc = s[offset:offset + soc_dim]
    s_t = s_soc[0]
    s_x_norm = np.linalg.norm(s_soc[1:])
    slack = s_t - s_x_norm
    if slack <= tol:
        print(f"SOC constraint (dim {soc_dim}): TIGHT")
    else:
        print(f"SOC constraint (dim {soc_dim}): LOOSE, slack = {slack:.2e}")
    offset += soc_dim

# Note: dual variable z is for sensitivity analysis only (shadow price),
# NOT needed for binding/slack determination — slack s is definitive.
# z_i > 0 implies s_i = 0 (certain), but s_i = 0 does NOT imply z_i > 0.
# Use z for: sensitivity analysis (marginal objective change per unit b_i change)
#            and degenerate constraint identification (tight but z_i = 0).
```

### 6.4 Tightness Analysis in CVXPY

CVXPY provides high-level abstractions. **Binding/slack determination uses only the slack variable** — `dual_value` is not needed for this purpose (see §6.1).

```python
# Method 1: Direct slack-based analysis
for i, c in enumerate(problem.constraints):
    if isinstance(c, cp.constraints.nonpos.NonPos):
        # expr <= 0, slack = -expr (definitive criterion)
        slack = -np.asarray(c.args[0].value).ravel()
        if np.min(slack) <= tol:
            print(f"Constraint {i}: TIGHT (slack_min = {np.min(slack):.2e})")
        else:
            print(f"Constraint {i}: LOOSE (slack_min = {np.min(slack):.2e})")

    elif isinstance(c, cp.constraints.zero.Zero):
        print(f"Constraint {i}: EQUALITY (violation = {c.violation():.2e})")

    elif isinstance(c, cp.constraints.second_order.SOC):
        # SOC: margin m = t - ||u||_2 (NOT violation)
        t_val = float(np.asarray(c.args[0].value).ravel()[0])
        x_val = np.asarray(c.args[1].value, dtype=float).ravel()
        margin = t_val - np.linalg.norm(x_val)
        if margin <= tol:
            print(f"Constraint {i}: SOC TIGHT (margin = {margin:.2e})")
        else:
            print(f"Constraint {i}: SOC LOOSE (margin = {margin:.2e})")

# Method 2: Using the analyze_constraint_tightness function from the examples
# (see portfolio_optimization_qp_ClarabelGPU.py for the implementation)
from portfolio_optimization_qp_ClarabelGPU import analyze_constraint_tightness

results, summary = analyze_constraint_tightness(
    problem,
    constraint_names=["weight_ub", "budget_eq", "risk_soc"],
    tol=1e-6
)
# summary['tight']      — list of tight constraint indices
# summary['loose']      — list of loose constraint indices
# summary['borderline'] — list of borderline constraint indices
```

The `analyze_constraint_tightness()` function computes:
1. **Slack / margin** (definitive criterion): scalar slack for inequalities, cone margin $m = t - \|u\|_2$ for SOC
2. **Feasibility residual** (`violation`): verification that constraint is satisfied
3. **Dual variable norm**: for **sensitivity analysis only** (shadow price), NOT for binding determination

Classification rules (based on slack variable — definitive):

```
Scalar inequality:  slack_min <= tol          -> TIGHT
                    tol < slack_min < 10*tol  -> BORDERLINE
                    slack_min >= 10*tol       -> LOOSE

SOC constraint:     margin <= tol             -> TIGHT  (on cone boundary)
                    margin >= 10*tol          -> LOOSE  (cone interior)
```

---

## 7. Performance Optimization Tips

### 7.1 Sequential Solve Strategies (Three-Case Detection)

When solving a sequence of related problems (e.g., daily portfolio rebalancing), the overhead of each solve depends on **what changed** between problems. The solver and CVXPY integration layer (`clarabelgpu_conif.py`) automatically detect three cases:

| Case | What Changed | Solver Action | cuDSS Phases | Typical Speedup |
|------|-------------|---------------|-------------|-----------------|
| **Case 1** | A/P sparsity pattern | `rebuild()` — C++ native method; internally preserves `etree_cache_` | ANALYSIS (with cached etree) + FACTORIZATION + SOLVE | ~1.0–1.2x vs cold |
| **Case 2** | A/P values only (same pattern) | `update_A()` / `update_P()` — scatter new values into KKT | REFACTORIZATION + SOLVE (ANALYSIS skipped entirely) | **~1.8–2.0x** vs cold |
| **Case 3** | Only q and/or b | `update_q()` / `update_b()` — update RHS only | REFACTORIZATION + SOLVE | **~2.0–3.6x** vs cold |

**CVXPY warm-start** (`warm_start=True`): the `_try_cached_solver()` method in `clarabelgpu_conif.py` performs the detection automatically by comparing `A.indptr`, `A.indices`, and `A.data` against cached values:

```python
# Automatic three-case detection (internal to CVXPY integration)
problem.solve(solver='CLARABELGPU', warm_start=True, **settings)
# Case 1: if A structure changed → rebuild() (etree cache survives)
# Case 2: if A values changed   → update_A()  (no ANALYSIS)
# Case 3: if A unchanged        → skip update (lightest path)
```

**Native interface** — explicit control:

```python
from clarabel_gpu import ClarabelGPU

solver = ClarabelGPU()
solver.setup(P, q, A, b, cones, **settings)   # Cold start
result = solver.solve()

# Case 3: only q/b changed
solver.update_q(q_new)
solver.update_b(b_new)
result = solver.solve()                        # 2–3.6x faster

# Case 2: A values changed, same sparsity pattern
solver.update_A(A_new)                         # scatter into KKT
solver.update_q(q_new)
result = solver.solve()                        # ~1.8x faster

# Case 1: A sparsity pattern changed (requires rebuild)
solver.rebuild(P, q, A_new_structure, b, cones, **settings)
result = solver.solve()                        # etree cache reused
```

**Important constraints**:
- `update_P/A()` requires the sparsity pattern (indptr/indices) to be identical to the original `setup()`. Only numerical values may differ.
- `equilibrate_enable` must be `False` when using `update_A/P()` across different problems, because equilibration scaling factors from the first problem would be incorrectly applied to new data.
- The elimination-tree cache is managed internally by the C++ `Solver` class (`etree_cache_` member). It persists across `rebuild()` calls automatically — no external cache management is needed. It is only effective when the KKT matrix **dimension** is unchanged (same n, m, cones).

#### Caller-Supplied Hints: `UpdateHint` (CVXPY only)

By default, the CVXPY integration layer runs `np.array_equal` comparisons on `A.indptr`, `A.indices`, `A.data`, `P.indptr`, `P.indices`, and `P.data` to decide which case applies. For large matrices (e.g., 5000-asset portfolio with 20000+ constraint rows), these comparisons alone can take measurable time.

If you know exactly what changed between solves — which is common in DPP loops where only specific parameters are updated — you can pass `update_hints` to bypass all comparisons:

```python
from clarabel_gpu import UpdateHint

# Only mu changed → only q vector changed in conic form
problem.solve(
    solver='CLARABELGPU',
    warm_start=True,
    update_hints={UpdateHint.q_values_changed},
    **settings
)

# Factor loadings F and mu both changed → A values + q changed
problem.solve(
    solver='CLARABELGPU',
    warm_start=True,
    update_hints={UpdateHint.A_values_changed, UpdateHint.q_values_changed},
    **settings
)
```

**`UpdateHint` enum members** (defined in `clarabel_gpu`):

| Member | Meaning | Solver Action |
|--------|---------|---------------|
| `automatic` | Fall back to comparison-based auto-detection | (original three-case logic) |
| `P_pattern_changed` | P sparsity pattern changed (dimensions unchanged) | `rebuild()` with etree cache |
| `P_values_changed` | P numerical values changed (same pattern) | `update_P()` |
| `A_pattern_changed` | A sparsity pattern changed (dimensions unchanged) | `rebuild()` with etree cache |
| `A_values_changed` | A numerical values changed (same pattern) | `update_A()` |
| `q_values_changed` | Linear cost vector q changed | `update_q()` |
| `b_values_changed` | Constraint RHS vector b changed | `update_b()` |

**Rules**:
- Pass `None` or `{UpdateHint.automatic}` to use the default comparison-based detection.
- Pass a specific set (e.g., `{UpdateHint.q_values_changed}`) to skip all comparisons and only execute the indicated updates.
- Pass an empty set `set()` when nothing changed (returns cached solver as-is).
- `P_pattern_changed` / `A_pattern_changed` refer to sparsity pattern changes while matrix **dimensions** remain the same. If dimensions change, the structural key mismatch forces a cold start before hints are consulted.

### 7.2 Parametric Repeated Solves (DPP)

- Use `DataUpdater` (C++) or `update_q/b` (Python) to update parameters
- After the first solve, symbolic analysis results are cached; subsequent solves only require numeric refactorization
- Typical speedup: 2x–10x over the initial solve
- For CVXPY DPP, declare frequently-changing inputs as `cp.Parameter` and use `warm_start=True`

### 7.3 GPU Mode and Memory Management

- With `gpu_mode=True`, solution vectors are returned as CuPy arrays (data remains on GPU)
- Pass CuPy arrays to `update_q()` / `update_b()` for GPU-to-GPU direct transfer, avoiding CPU-GPU round trips
- The solver automatically initializes an RMM pool if none is configured. For advanced control or PyTorch/CuPy interop, see §2.5

### 7.4 Data Equilibration

- `equilibrate_enable=True` (default) can significantly improve problem conditioning
- Note: after equilibration, internal problem data is scaled, but the solution is automatically unscaled before being returned

### 7.5 Iterative Refinement

- `ir_enable=True` (default) improves numerical stability
- For well-conditioned problems (e.g., QP), reducing `ir_max_iter` can speed up solves
- For complex problems (SOCP/SDP/exponential cones), keep the defaults

### 7.6 Problem Size Guidelines

| Problem Type | Typical Solvable Scale | Scale Where GPU Acceleration is Significant |
|-------------|----------------------|---------------------------------------------|
| LP/QP | Tens of thousands of variables | $n \ge 5{,}000$ |
| SOCP | Hundreds of thousands of constraints | Number of SOCs $\ge 2{,}000$ |
| SDP | Thousands of PSD cones (mixed dims OK) | Number of PSD cones $\ge 100$ |
| Exponential Cone | Tens of thousands of cones | $n \ge 2{,}000$ |

---

## 8. Frequently Asked Questions

### Q1: When should I use CVXPY + ClarabelGPU vs. native ClarabelGPU?

- **CVXPY + ClarabelGPU**: Best for rapid prototyping, leveraging DPP parameterization, and when dual variable mapping back to original constraints is needed
- **Native ClarabelGPU**: Best for production environments, latency-sensitive high-frequency scenarios, and maximum performance

### Q2: Is the `ALMOST_SOLVED` status acceptable?

Yes. `ALMOST_SOLVED` means the reduced accuracy standards (`reduced_tol_*`) are satisfied, which is typically sufficient for practical applications. It maps to CVXPY's `optimal_inaccurate`.

### Q3: How do I handle numerical instability?

1. Enable `equilibrate_enable=True`
2. Increase `sr_constant` (e.g., `1e-7`)
3. Increase `ir_max_iter` (e.g., `3`)
4. Verify that the problem is well-posed (P is positive semidefinite, constraints are consistent)

### Q4: Should the P matrix be upper-triangular or full?

The C++ backend accepts the **full** CSR-format P matrix. There is no requirement to extract the upper-triangular part before passing it to the solver.

### Q5: What is the required ordering of cone constraints?

Cone constraints must match the row ordering of the A matrix and b vector. Recommended order:
`ZeroCone -> NonnegativeCone -> SecondOrderCone -> PSDTriangleCone -> ExponentialCone -> PowerCone`

The CVXPY integration layer handles ordering automatically.

### Q6: Can solver instances be shared across threads?

No. Both `Solver` and `ClarabelGPU` are **not thread-safe**. In multi-threaded scenarios, create a separate instance per thread.

### Q7: I get `cusparseSpMV(): dimension mismatch, matA.rows != vecY.size`. What happened?

This error means the internal constraint matrix A has a different number of rows than the solver expects. Common causes and solutions:

**Cause 1: Calling `update_A()` after the constraint structure changed.**

`update_A()` only updates the **numerical values** of the existing CSR matrix — the sparsity pattern (row count, column indices, indptr) is fixed at `setup()` time. If your new A has different dimensions or nnz, use `rebuild()` instead:

```python
# ✗ Wrong: A dimensions changed, but using update_A
solver.update_A(A_new)  # → dimension mismatch error

# ✓ Correct: use rebuild() when structure changes
solver.rebuild(P, q, A_new, b, cones, **settings)
```

**Cause 2: Chordal decomposition augmented the problem internally.**

When `chordal_decomposition_enable=True` and the problem has PSD cones, the solver may decompose large PSD cones into smaller ones, **adding extra rows** to A internally. The internal A dimensions no longer match the user-supplied A. In this case, do not call `update_A()` — either:

- Disable chordal decomposition: `chordal_decomposition_enable=False`
- Or use `rebuild()` which correctly re-applies the decomposition

**Summary: when to use `update_*()` vs `rebuild()`**

| Scenario | Method | Notes |
|----------|--------|-------|
| Only q or b values change | `update_q()` / `update_b()` | Fastest: no KKT refactorization needed |
| P or A values change (same pattern) | `update_P()` / `update_A()` | KKT values updated, refactorization on next solve |
| A sparsity pattern changes (same dims) | `rebuild()` | Reuses elimination-tree cache (~50-80% ANALYSIS savings) |
| Problem dimensions change | New `setup()` | Full rebuild from scratch |
