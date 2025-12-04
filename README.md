
# 1. 构建 cufolio 镜像步骤

> 操作系统：rockylinux:10-ubi 
> 基础镜像：`rockylinux/rockylinux:10-ubi`


---

## 2. 安装 Miniconda（Python 3.13）

```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py313_25.9.1-1-Linux-x86_64.sh
sh ./Miniconda3-py313_25.9.1-1-Linux-x86_64.sh -b  # 默认安装到 /root/miniconda3
source /root/miniconda3/etc/profile.d/conda.sh
```

---

## 3. 创建并激活 `cufolio` 环境

```bash
conda create -n cufolio \
  -c rapidsai -c conda-forge -c nvidia \
  numpy scipy cython cupy pylibraft rmm fmt=11 libcudss-dev libcnpy libzlib nvmath-python zlib\
  cuda-cccl cvxpy=1.7.2 cuda-version=12.8 cxx-compiler cmake=4.1.2 make python=3.13

conda init bash
conda activate cufolio
```

---

## 4. 替换/补充 CVXPY 文件

操作    目标路径    
用自定义 `defines.py` 覆盖    `/root/miniconda3/envs/cufolio/lib/python3.13/site-packages/cvxpy/reductions/solvers/defines.py`    
用自定义 `settings.py` 覆盖    `/root/miniconda3/envs/cufolio/lib/python3.13/site-packages/cvxpy/settings.py`    
拷贝 `nvsolver_conif.py` 到    `/root/miniconda3/envs/cufolio/lib/python3.13/site-packages/cvxpy/reductions/solvers/conic_solvers/nvsolver_conif.py`    
拷贝 `clarabel_gpu.cpython-313-x86_64-linux-gnu.so` 与 `clarabel_solver.py` 到    `/root/miniconda3/envs/cufolio/lib/python3.13/site-packages/`    

---

## 5. 配置运行时环境变量

```bash
cat >> ~/.bashrc <<'EOF'
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=/root/miniconda3/envs/cufolio/lib:/root/miniconda3/envs/cufolio/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
EOF

source ~/.bashrc
```

---

## 6. 保存镜像（在宿主机【容器外】执行）

```bash
# 将容器 xxxx 保存为镜像
docker commit <container-id-or-name> cufolio:py3.13
```

## 8. 如果驱动版本不满足CUDA 12.8的最低要求，怎办？

CUDA Toolkit    最低驱动（native）    可用 forward-compat 驱动的下限    
12.8            570.xx                535.xx    

```bash
sudo apt install cuda-compat-12-8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH
```


