# Building the cuFOLIO Runtime Environment (SM_75/80/86/89/90/120)

> Supported OS: Ubuntu 22.04–24.04, RockyLinux 8–10, CentOS 8-10 etc.

---

## 1. Pull the Base Container and Enter

```bash
docker pull nvcr.io/nvidia/base/ubuntu:jammy-20251013
docker run -it --gpus 1 -v /path/to/prebuilt:/clarabel nvcr.io/nvidia/base/ubuntu:jammy-20251013 /bin/bash

apt update && apt install -y wget
```

---

## 2. Install cuDSS (taking Ubuntu 22.04 as example)

cuDSS (CUDA Sparse Direct Solver) must be installed separately (see [cuDSS Downloads](https://developer.nvidia.com/cudss-downloads)):

```bash
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-repo-ubuntu2204-0.7.1_0.7.1-1_amd64.deb
dpkg -i cudss-local-repo-ubuntu2204-0.7.1_0.7.1-1_amd64.deb
cp /var/cudss-local-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cudss
```

---

## 3. Install Miniconda (Python 3.12)

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
```

---

## 4. Create and Activate the `cufolio` Environment

The shared library is built with CUDA 12.8:

```bash
conda create -n cufolio \
  -c rapidsai -c conda-forge -c nvidia \
  numpy scipy cupy rmm=25.10 \
  cvxpy=1.7.5 openmp fmt \
  cuda-version=12.8 python=3.12
```

---

## 5. Install ClarabelGPU / DiffSolver

Use the integration script to automatically copy all files and apply CVXPy patches:

```bash
cd /clarabel
./python/integrate_cuFOLIO.sh --yes
```

The script performs the following operations:

| Operation | File | Destination |
|-----------|------|-------------|
| Overwrite | `defines.py` | `$SITE_PACKAGES/cvxpy/reductions/solvers/defines.py` |
| Overwrite | `settings.py` | `$SITE_PACKAGES/cvxpy/settings.py` |
| Copy | `clarabelgpu_conif.py` | `$SITE_PACKAGES/cvxpy/reductions/solvers/conic_solvers/` |
| Copy | `clarabel_gpu.abi3.so` | `$SITE_PACKAGES/` |

`$SITE_PACKAGES/` is typically `/root/miniconda3/envs/cufolio/lib/python3.12/site-packages/`.

> **Note**: `clarabel_gpu.abi3.so` uses the Python Stable ABI, compatible with Python 3.10–3.13.

---

## 6. Configure Runtime Environment Variables

```bash
cat >> ~/.bashrc <<'EOF'
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
EOF

source ~/.bashrc
```

---

## 7. Verify Installation

```bash
conda activate cufolio
```

```python
# Verify ClarabelGPU (forward solver)
import cvxpy as cp
solvers = cp.installed_solvers()
print(solvers)                      # Should contain 'CLARABELGPU'
assert 'CLARABELGPU' in solvers

```

---

## 8. Save the Image (Run on Host Machine)

```bash
docker commit <container-id-or-name> cufolio:py3.12
```

---

## 9. CUDA Forward-Compatible Driver (Optional)

If the GPU driver version is lower than the minimum required by the CUDA Toolkit, install the forward-compatible driver:

| CUDA Toolkit | Minimum Native Driver | Forward-Compat Driver Floor |
|--------------|-----------------------|-----------------------------|
| 12.8         | 570.xx                | 535.xx                      |

```bash
# Ubuntu
sudo apt install cuda-compat-12-8
# CentOS/RHEL/Rocky
sudo yum install cuda-compat-12-8

export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
```
