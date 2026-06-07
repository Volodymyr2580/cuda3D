# AGENT_LOG.md

本日志用于记录 CUDA3D 项目中每一步环境配置、源码修改、构建、测试和基准对比。后续所有测试和优化都应追加记录，不覆盖历史内容。

## 2026-06-02 - 建立可运行环境与第一版 smoke baseline

### 操作目标

在服务器 `/data/shengwz/swz/cuda3D` 中建立基本可运行环境，使当前 CUDA3D 程序能够编译并完成最小 1GPU/3GPU smoke test。

### 环境检查

服务器信息：

- 登录用户：`shengwz`
- 项目目录：`/data/shengwz/swz/cuda3D`
- 系统：Ubuntu 22.04.5 LTS
- GPU：4 张 NVIDIA GeForce RTX 4090，每张约 24GB 显存
- 内存：约 503GiB
- `/data` 磁盘：约 29T，总可用约 15T

工具链：

- CUDA driver 显示支持 CUDA 12.4
- 使用 CUDA 编译器：`/usr/local/cuda-12.2/bin/nvcc`
- Intel MPI：`/opt/intel/oneapi/mpi/latest`
- 原计划去除 MKL 运行依赖，当前服务器构建只链接 MPI

### 修改文件

已对当前可运行版做过以下最小修补：

- `include/inc3D/common.h`
  - 移除 MKL 相关 include。
- `include/inc3D/cu_common.h`
  - 移除 MKL 相关 include。
- `include/inc3D/alloc.h`
  - 移除 `<mkl.h>` include。
  - 增加本地兼容类型 `MKL_Complex8` 和 `MKL_Complex16`。
- `src/makefile.server`
  - 新增服务器专用构建文件。
  - 使用 `/usr/local/cuda-12.2/bin/nvcc`。
  - 使用 `/opt/intel/oneapi/mpi/latest/bin/mpicc`。
  - 只链接 MPI，不链接 MKL。
- `src/main.cu`
  - 将 `char order[1]` 改为 `char order[8]`，避免 `scanf("%s", order)` 写越界。
  - 增加 `mrecu=1` 初始化，避免未初始化变量参与内存分配。
  - 将固定打印前 10 炮改为 `MIN(10, ns)`，避免小样例炮数少于 10 时越界。
- `tools/create_smoke_case.py`
  - 新增 smoke case 生成脚本。

说明：以上修改没有改动 CUDA kernel、有限差分公式、PML 公式、震源注入、检波器抽取、速度模型读取和炮检点读取的计算路径。

### 执行命令

服务器环境检查示例：

```bash
hostname
uname -a
nvidia-smi
free -h
df -h /data/shengwz /tmp
```

编译命令：

```bash
cd /data/shengwz/swz/cuda3D/src
make -f makefile.server test
```

生成 smoke case：

```bash
cd /data/shengwz/swz/cuda3D
python3 tools/create_smoke_case.py
```

1GPU smoke test：

```bash
source /opt/intel/oneapi/setvars.sh
cd /data/shengwz/swz/cuda3D/bench_smoke
CUDA_VISIBLE_DEVICES=0 timeout 120s /opt/intel/oneapi/mpi/latest/bin/mpirun -np 1 ../bin/cuda_3D_FM < input_smoke.in
```

3GPU smoke test：

```bash
source /opt/intel/oneapi/setvars.sh
cd /data/shengwz/swz/cuda3D/bench_smoke
CUDA_VISIBLE_DEVICES=0,1,2 timeout 120s /opt/intel/oneapi/mpi/latest/bin/mpirun -np 3 ../bin/cuda_3D_FM < input_smoke_3gpu.in
```

### 测试结果

编译结果：

- `make -f makefile.server test` 成功。
- 生成可执行文件：`/data/shengwz/swz/cuda3D/bin/cuda_3D_FM`

1GPU smoke test：

- 成功完成。
- 日志包含 `ALL DONE`。
- 程序识别 `avail=1 gpus`。
- 3 炮均完成。

3GPU smoke test：

- 成功完成。
- 日志包含 `ALL DONE`。
- 程序识别 `avail=3 gpus`。
- `TOTAL SHOT=3, each core process 1 shots`。
- 说明 MPI 分炮和多 GPU 绑定链路已跑通。

### 输出与哈希摘要

当前 smoke case 文件：

```text
bench_smoke/input_smoke.in
bench_smoke/input_smoke_3gpu.in
bench_smoke/nav_smoke_3shots_25rec.nav
bench_smoke/vel_smoke_ny48_nx48_nz48.dir
```

已记录过的关键 SHA256：

```text
f31311fbba8716ef4e2c085cb61d061cdcd5ca96428437a0d7f4c46c2f4780fc  bin/cuda_3D_FM
16cceedbb9a9a140ba453106dd79719edf5913c9a7a0d06b519f96b296b31b9e  bench_smoke/input_smoke.in
5c4bf8545e5d58bb3436f68565905a8516ed2e336c9c120728902a209df67970  bench_smoke/input_smoke_3gpu.in
bf93615c57f0be04266a0509d347a51c266944d474380164a878b360243d063a  bench_smoke/nav_smoke_3shots_25rec.nav
abd9941201c6be48ee5bddf2fe369c9a03c55d1a76055c715169bb0b9ae21960  bench_smoke/vel_smoke_ny48_nx48_nz48.dir
```

3GPU smoke 输出文件：

```text
d_obs_salt_gpu_cpu_checked_ricker1_8hz_3d_ny_384_nx_384_nz95_nbell_1_bscl_0.9_moffy_9.5625_moffx_9.5625_h_obs_nt_1501_dt_2ms_shot_0.dir 5100 bytes
d_obs_salt_gpu_cpu_checked_ricker1_8hz_3d_ny_384_nx_384_nz95_nbell_1_bscl_0.9_moffy_9.5625_moffx_9.5625_h_obs_nt_1501_dt_2ms_shot_1.dir 5100 bytes
d_obs_salt_gpu_cpu_checked_ricker1_8hz_3d_ny_384_nx_384_nz95_nbell_1_bscl_0.9_moffy_9.5625_moffx_9.5625_h_obs_nt_1501_dt_2ms_shot_2.dir 5100 bytes
```

尺寸符合 `25 receivers * 51 time steps * 4 bytes = 5100 bytes`。

### 风险与下一步

风险：

- 当前可运行版不是严格证明过的未修改原始版。
- `orig_code` 目录与当前可运行源码关键文件哈希一致，不能作为真正原始源码 baseline。
- smoke case 太小，`mod time` 显示为 `0.000000s`，只能验证链路，不能用于性能结论。
- 服务器 GPU 曾有其他 Python 和 MPI 任务占用，正式性能测试前需要检查 GPU 空闲状态。

下一步：

- 建立 `benchmarks/` 目录结构。
- 冻结 `current_runnable` baseline。
- 生成 `correctness` 和 `perf_3gpu` 测试样例。
- 编写输出对比脚本，默认相对 L2 误差门槛为 `1e-5`。
- 后续每次测试或源码修改后继续追加本日志。

## 2026-06-02 - 新增项目协作说明与日志制度

### 操作目标

将 CUDA 基准策略和日志维护要求写入项目根目录文档，保证后续优化过程可追踪、可复现。

### 修改文件

- `AGENTS.md`
  - 新增项目目标、基准原则、数值正确性要求、测试层级、服务器测试环境、工作日志要求和安全约束。
- `AGENT_LOG.md`
  - 新增本日志文件。
  - 补记服务器环境检查、去 MKL 构建、`makefile.server`、smoke case、1GPU/3GPU 跑通结果。

### 执行命令

本次主要通过本地文件编辑完成，并计划同步到服务器项目目录：

```powershell
C:\Windows\System32\OpenSSH\scp.exe E:\cuda3D\AGENTS.md shengwz@162.105.91.239:/data/shengwz/swz/cuda3D/AGENTS.md
C:\Windows\System32\OpenSSH\scp.exe E:\cuda3D\AGENT_LOG.md shengwz@162.105.91.239:/data/shengwz/swz/cuda3D/AGENT_LOG.md
```

### 测试结果

待同步后检查：

- `AGENTS.md` 存在。
- `AGENT_LOG.md` 存在。
- `AGENTS.md` 包含相对 L2 误差 `1e-5`。
- `AGENT_LOG.md` 包含 1GPU 和 3GPU smoke test 记录。

### 风险与下一步

- 后续所有修改和测试都必须追加日志。
- 下一步应建立正式 `benchmarks/` 目录和 baseline 冻结脚本。

## 2026-06-02 - 建立正式 benchmark 工具链并冻结 current_runnable baseline

### 操作目标

建立可重复运行的 CUDA3D benchmark 工具链，生成 `correctness` 和 `perf_3gpu` 样例，冻结当前可运行版 baseline，并验证输出比较脚本能正确工作。

### 修改文件

- `tools/create_benchmark_cases.py`
  - 新增 correctness / perf_3gpu 样例生成脚本。
  - 生成确定性的速度模型、nav 文件、input 文件和 case manifest。
- `tools/run_benchmark.py`
  - 新增统一运行脚本。
  - 自动记录运行日志、环境信息、输入/二进制哈希、输出文件和输出哈希。
  - 支持写入 `benchmarks/baselines/current_runnable/` 或 `benchmarks/runs/`。
- `tools/compare_outputs.py`
  - 新增输出比较脚本。
  - 默认相对 L2 容差 `1e-5`，绝对容差 `1e-7`。
  - 输出 `comparison.json` 和 `comparison.md`。

### 执行命令

同步脚本到服务器：

```powershell
C:\Windows\System32\OpenSSH\scp.exe E:\cuda3D\tools\create_benchmark_cases.py shengwz@162.105.91.239:/data/shengwz/swz/cuda3D/tools/create_benchmark_cases.py
C:\Windows\System32\OpenSSH\scp.exe E:\cuda3D\tools\run_benchmark.py shengwz@162.105.91.239:/data/shengwz/swz/cuda3D/tools/run_benchmark.py
C:\Windows\System32\OpenSSH\scp.exe E:\cuda3D\tools\compare_outputs.py shengwz@162.105.91.239:/data/shengwz/swz/cuda3D/tools/compare_outputs.py
```

服务器语法检查与样例生成：

```bash
cd /data/shengwz/swz/cuda3D
python3 -m py_compile tools/create_benchmark_cases.py tools/run_benchmark.py tools/compare_outputs.py
python3 tools/create_benchmark_cases.py --case all
```

冻结 current baseline：

```bash
cd /data/shengwz/swz/cuda3D
python3 tools/run_benchmark.py --case smoke_1gpu --tag baseline --baseline
python3 tools/run_benchmark.py --case smoke_3gpu --tag baseline --baseline
python3 tools/run_benchmark.py --case correctness --tag baseline --baseline
python3 tools/run_benchmark.py --case perf_3gpu --tag baseline_contended --baseline
```

重复 correctness 并比较：

```bash
cd /data/shengwz/swz/cuda3D
python3 tools/run_benchmark.py --case correctness --tag repeat_compare
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs \
  --candidate benchmarks/runs/correctness_repeat_compare_20260601_165128/outputs \
  --out benchmarks/reports/correctness_repeat_compare_20260601_165128
```

### 测试结果

样例生成：

```text
benchmarks/cases/correctness/input_correctness.in 258 bytes
benchmarks/cases/correctness/nav_correctness_6shots_49rec.nav 7056 bytes
benchmarks/cases/correctness/vel_correctness_ny96_nx96_nz64.dir 2359296 bytes
benchmarks/cases/perf_3gpu/input_perf_3gpu.in 255 bytes
benchmarks/cases/perf_3gpu/nav_perf_3gpu_9shots_81rec.nav 17496 bytes
benchmarks/cases/perf_3gpu/vel_perf_3gpu_ny128_nx128_nz95.dir 6225920 bytes
```

baseline 运行目录：

```text
benchmarks/baselines/current_runnable/smoke_1gpu_baseline_20260601_165045
benchmarks/baselines/current_runnable/smoke_3gpu_baseline_20260601_165049
benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054
benchmarks/baselines/current_runnable/perf_3gpu_baseline_contended_20260601_165108
```

baseline 结果：

- `smoke_1gpu`：退出码 0，输出 3 个 `.dir` 文件。
- `smoke_3gpu`：退出码 0，输出 3 个 `.dir` 文件。
- `correctness`：退出码 0，输出 6 个 `.dir` 文件。
- `perf_3gpu`：退出码 0，输出 9 个 `.dir` 文件。

比较工具验证：

- baseline：`benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs`
- candidate：`benchmarks/runs/correctness_repeat_compare_20260601_165128/outputs`
- 报告：`benchmarks/reports/correctness_repeat_compare_20260601_165128/comparison.md`
- 结果：通过。
- 6 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。

### 风险与下一步

风险：

- 服务器 GPU 当前有其他任务占用，`perf_3gpu_baseline_contended_20260601_165108` 只可作为争用环境下的临时性能记录。
- 当前 `perf_3gpu` 样例运行约 4 秒，太轻，后续若用于真实性能优化，需要增加规模或使用 profiler 指标。
- 本地 `python -m py_compile` 因 Windows `tools/__pycache__` 写入权限失败；服务器上 `py_compile` 已成功。

下一步：

- 读取 baseline run logs，提取每炮耗时和总体耗时。
- 用更明确的计时或 profiler 找出主要热点。
- 首个优化方向优先考虑 `fd_3d_f` 内每炮重复 `cudaMalloc/cudaMemcpy/cudaFree` 开销，以及时间步循环中的全网格 CUDA kernel。

## 2026-06-02 01:05:57 +08:00

### 操作目标

- 建立可重复 CUDA 基准测试工具链。
- 扩大 `perf_3gpu` 到更接近真实性能测试的规模。
- 尝试 profiler 定位热点。
- 完成第一轮低风险 CUDA 优化，并与 current runnable baseline 做数值和性能对比。

### 修改文件

- `tools/create_benchmark_cases.py`
  - 新增 `correctness`、`perf_3gpu`、`profile_1gpu` 测试样例生成。
  - `perf_3gpu` 调整为 `384x384x95`、`nt=1501`、9 炮、每炮 441 个检波点、3 GPU。
- `tools/run_benchmark.py`
  - 新增统一 benchmark runner，自动复制输入、运行程序、收集输出、计算 SHA256、保存 `manifest.json` 与 `run.log`。
- `tools/compare_outputs.py`
  - 新增 float32 `.dir` 输出比较工具，检查文件大小、NaN/Inf、相对 L2、最大绝对误差、RMS。
- `src/single_solver.cu`
  - 在实际调用的 `cuda_fd3d_p_pml_ns` kernel 中，把 `vzz/vxx/vyy` 的中间结果改为线程本地寄存器变量 `vzz_loc/vxx_loc/vyy_loc`。
  - 保持数值公式和 kernel 接口不变，避免第一轮优化扩大改动范围。

### 执行命令

服务器样例生成与语法检查：

```bash
cd /data/shengwz/swz/cuda3D
python3 -m py_compile tools/create_benchmark_cases.py tools/run_benchmark.py tools/compare_outputs.py
python3 tools/create_benchmark_cases.py --case correctness
python3 tools/create_benchmark_cases.py --case perf_3gpu
python3 tools/create_benchmark_cases.py --case profile_1gpu
```

冻结与追加 baseline：

```bash
cd /data/shengwz/swz/cuda3D
python3 tools/run_benchmark.py --case smoke_1gpu --tag baseline --baseline
python3 tools/run_benchmark.py --case smoke_3gpu --tag baseline --baseline
python3 tools/run_benchmark.py --case correctness --tag baseline --baseline
python3 tools/run_benchmark.py --case perf_3gpu --tag baseline_heavy_contended --baseline
```

重复正确性验证：

```bash
cd /data/shengwz/swz/cuda3D
python3 tools/run_benchmark.py --case correctness --tag repeat_compare
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs \
  --candidate benchmarks/runs/correctness_repeat_compare_20260601_165128/outputs \
  --out benchmarks/reports/correctness_repeat_compare_20260601_165128
```

Profiler 尝试：

```bash
cd /data/shengwz/swz/cuda3D
nvprof ./bin/cuda_3D_FM benchmarks/cases/profile_1gpu/input_profile_1gpu.in
/usr/local/cuda-12.2/bin/ncu --target-processes all ./bin/cuda_3D_FM benchmarks/cases/profile_1gpu/input_profile_1gpu.in
```

第一轮优化后构建与测试：

```bash
cd /data/shengwz/swz/cuda3D/src
source /opt/intel/oneapi/setvars.sh --force
make -f makefile.server test

cd /data/shengwz/swz/cuda3D
python3 tools/run_benchmark.py --case correctness --tag opt_p_local
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs \
  --candidate benchmarks/runs/correctness_opt_p_local_20260601_170306/outputs \
  --out benchmarks/reports/correctness_opt_p_local_latest

python3 tools/run_benchmark.py --case perf_3gpu --tag opt_p_local
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs \
  --candidate benchmarks/runs/perf_3gpu_opt_p_local_20260601_170330/outputs \
  --out benchmarks/reports/perf_3gpu_opt_p_local_latest
```

### 测试结果

基准目录：

```text
benchmarks/baselines/current_runnable/smoke_1gpu_baseline_20260601_165045
benchmarks/baselines/current_runnable/smoke_3gpu_baseline_20260601_165049
benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054
benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602
```

重复 correctness 对比：

- 报告：`benchmarks/reports/correctness_repeat_compare_20260601_165128/comparison.md`
- 结果：通过。
- 6 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。

Profiler 结果：

- `nvprof` 失败：RTX 4090 属于 Compute Capability 8.0+ 后的 GPU，`nvprof` 已不支持。
- Nsight Compute `ncu` 失败：服务器限制了 GPU performance counter 权限，日志报 `ERR_NVGPUCTRPERM`。
- 因此本轮先采用源码审计、CUDA event 计时和端到端 benchmark 做优化判断。

第一轮优化正确性：

- candidate：`benchmarks/runs/correctness_opt_p_local_20260601_170306`
- report：`benchmarks/reports/correctness_opt_p_local_latest/comparison.md`
- 结果：通过。
- 6 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。

第一轮优化性能：

- baseline：`benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602`
- candidate：`benchmarks/runs/perf_3gpu_opt_p_local_20260601_170330`
- report：`benchmarks/reports/perf_3gpu_opt_p_local_latest/comparison.md`
- 结果：通过。
- 9 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。

性能摘要：

```text
baseline WP computing time = 3.491814 s
candidate WP computing time = 2.831085 s
speedup = 1.23x
time reduction = 18.9%

baseline mod times = 1.118448 s, 1.212862 s, 1.116182 s
candidate mod times = 0.893013 s, 0.967806 s, 0.896852 s
```

### 输出、哈希与误差摘要

```text
src/single_solver.cu SHA256 = 5056afb6d89065fe03a6d7728ec4bf47d175f80980d9916b05eff3423f79d86e
bin/cuda_3D_FM SHA256 = 3eedecc11c6e7ad0fa1e1845cc0819e8bc528915258c294b6d512e2a24c21edd
tools/create_benchmark_cases.py SHA256 = d9cfa8d83f5ecee9e111f2e3ffc4337baaaaf736ed437e419bcf2a2d62eb7f08
tools/run_benchmark.py SHA256 = f7ced59e4882ad483a7bb208648afac0af4e604cd0f6c9ec7112d514b3eb9c63
tools/compare_outputs.py SHA256 = e810dbfd95b83ae45fa07998f6901d9072695f5269602b5284aaae88b524d075
```

### 风险与下一步

风险：

- `perf_3gpu` baseline 与 candidate 都是在服务器 GPU 有其他任务争用时运行的，绝对时间有噪声；不过每炮 kernel 时间均有同向下降，可以作为初步有效信号。
- `ncu` 目前不能读取硬件性能计数器，除非管理员开启 performance counter 权限；短期内用 wall time、CUDA event time 与数值对比推进。

下一步：

- 清理 `cuda_fd3d_p_pml_ns` 已经不再需要的 `vzz/vxx/vyy` 参数。
- 在 `fd_3d_f` 中去掉 `d_vzz/d_vxx/d_vyy` 的分配、清零和释放，减少显存占用和每炮初始化开销。
- 每次改动后继续跑 `correctness` 和 `perf_3gpu`，要求相对 L2 `<= 1e-5` 且无 NaN/Inf。

## 2026-06-02 01:35:02 +08:00

### 操作目标

- 按用户要求建立 `0.5x` 速度阈值存档机制。
- 继续优化 CUDA 运行效率，目标先突破 `1.5x`。
- 对每个候选版本继续执行 correctness/perf 对比，确保数值结果不变。

### 修改文件

- `AGENTS.md`
  - 新增速度阈值存档规则。
  - 规定以 `perf_3gpu_baseline_heavy_contended_20260601_165602` 的 `WP computing time = 3.491814 s` 作为 `1.0x`。
  - 规定突破 `1.5x`、`2.0x`、`2.5x`、`3.0x` 等门槛时归档到 `archives/speedups/`。
- `tools/archive_speedup.py`
  - 新增阈值版本归档工具，只新增目录和文件，不删除、不覆盖已有存档。
  - 归档内容包括关键源码、头文件、benchmark 工具、对比报告、run manifest 与日志。
- `src/rem_fd.cu`
  - 删除 `d_vyy/d_vxx/d_vzz` 的声明、`cudaMalloc`、`cudaMemset`、kernel 参数传递和 `cudaFree`。
- `src/single_solver.cu`
  - 删除 `cuda_fd3d_p_pml_ns` 中已经无用的 `vyy/vxx/vzz` 参数。
  - 删除实际执行的 `_ns` 速度/压力 kernel 中不需要的 `__syncthreads()`。
- `include/inc3D/single_solver.h`
  - 同步删除 `cuda_fd3d_p_pml_ns` 的 `vyy/vxx/vzz` 参数。
- `include/inc3D/cu_common.h`
  - `BlockSize1/2/3` 从 `32/32/1` 调整为 `64/4/2`。
- `tools/sweep_blocksize.py`
  - 新增 block-size 扫描工具，用于编译并测试不同 CUDA block 形状。

### 执行命令

同步文件到服务器：

```powershell
C:\Windows\System32\OpenSSH\scp.exe ...
```

构建：

```bash
cd /data/shengwz/swz/cuda3D/src
source /opt/intel/oneapi/setvars.sh --force
make -f makefile.server test
make -B -f makefile.server test
```

第二轮中间版本测试：

```bash
cd /data/shengwz/swz/cuda3D
python3 tools/run_benchmark.py --case correctness --tag opt_p_no_scratch
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs \
  --candidate benchmarks/runs/correctness_opt_p_no_scratch_20260601_171304/outputs \
  --out benchmarks/reports/correctness_opt_p_no_scratch_latest
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_p_no_scratch
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs \
  --candidate benchmarks/runs/perf_3gpu_opt_p_no_scratch_20260601_171331/outputs \
  --out benchmarks/reports/perf_3gpu_opt_p_no_scratch_latest
```

第三轮中间版本测试：

```bash
cd /data/shengwz/swz/cuda3D
python3 tools/run_benchmark.py --case correctness --tag opt_p_no_scratch_no_sync
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs \
  --candidate benchmarks/runs/correctness_opt_p_no_scratch_no_sync_20260601_171531/outputs \
  --out benchmarks/reports/correctness_opt_p_no_scratch_no_sync_latest
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_p_no_scratch_no_sync
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs \
  --candidate benchmarks/runs/perf_3gpu_opt_p_no_scratch_no_sync_20260601_171555/outputs \
  --out benchmarks/reports/perf_3gpu_opt_p_no_scratch_no_sync_latest
```

block-size 扫描：

```bash
cd /data/shengwz/swz/cuda3D
python3 tools/sweep_blocksize.py --case perf_3gpu --baseline-wp 3.491814
python3 tools/sweep_blocksize.py --case perf_3gpu --baseline-wp 3.491814 \
  --variants 128x2x1 128x1x1 96x4x1 96x2x1 80x4x1 64x2x1 64x2x2 48x4x1
python3 tools/sweep_blocksize.py --case perf_3gpu --baseline-wp 3.491814 \
  --variants 64x1x4 64x2x3 64x3x2 64x4x2 48x2x2 48x2x3 96x1x2 128x1x2 32x2x4
python3 tools/sweep_blocksize.py --case perf_3gpu --baseline-wp 3.491814 \
  --variants 64x5x2 64x6x2 64x8x2 64x4x3 64x3x3 128x2x2 96x2x2 80x2x2 32x4x4
```

最终 `1.5x` 候选验证：

```bash
cd /data/shengwz/swz/cuda3D
python3 tools/run_benchmark.py --case correctness --tag opt_1p5_block_64x4x2
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs \
  --candidate benchmarks/runs/correctness_opt_1p5_block_64x4x2_20260601_173331/outputs \
  --out benchmarks/reports/correctness_opt_1p5_block_64x4x2_latest
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_1p5_block_64x4x2
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs \
  --candidate benchmarks/runs/perf_3gpu_opt_1p5_block_64x4x2_20260601_173358/outputs \
  --out benchmarks/reports/perf_3gpu_opt_1p5_block_64x4x2_latest
```

### 测试结果

中间版本 `opt_p_no_scratch`：

- correctness：通过，6 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。
- perf：通过，9 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。
- `WP computing time = 2.797758 s`。
- speedup = `3.491814 / 2.797758 = 1.25x`，未达到 `1.5x` 存档线。

中间版本 `opt_p_no_scratch_no_sync`：

- correctness：通过，6 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。
- perf：通过，9 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。
- `WP computing time = 2.753577 s`。
- speedup = `3.491814 / 2.753577 = 1.27x`，未达到 `1.5x` 存档线。

block-size 扫描摘要：

```text
32x32x1 default after previous kernel optimization: 2.753577 s, 1.27x
64x4x1: 2.580169 s, 1.35x
64x2x2: 2.222405 s, 1.57x
64x4x2: 2.208669 s, 1.58x
128x2x2: 2.222031 s, 1.57x
```

最终 `1.5x` 候选 `opt_1p5_block_64x4x2`：

- correctness：通过，6 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。
- perf：通过，9 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。
- `mod time = 0.720422 s, 0.782596 s, 0.719227 s`。
- `Gradient TIME all = 3.710990 s`。
- `WP computing time = 2.215294 s`。
- speedup = `3.491814 / 2.215294 = 1.576x`。
- 达到并超过 `1.5x` 存档线。

### 输出、哈希与误差摘要

最终 `1.5x` 候选：

```text
candidate correctness run = benchmarks/runs/correctness_opt_1p5_block_64x4x2_20260601_173331
candidate perf run = benchmarks/runs/perf_3gpu_opt_1p5_block_64x4x2_20260601_173358
correctness report = benchmarks/reports/correctness_opt_1p5_block_64x4x2_latest/comparison.md
perf report = benchmarks/reports/perf_3gpu_opt_1p5_block_64x4x2_latest/comparison.md
bin/cuda_3D_FM SHA256 = 101eb55f577521bf842ea1c5aaef9e735ae249a319065f20371bb159f804b186
```

### 风险与下一步

风险：

- 服务器仍有 GPU 负载波动，`Gradient TIME all` 受 MPI 等待和资源争用影响明显；本轮正式比较继续以 `WP computing time` 为主要性能指标。
- block-size 是和当前数据形状相关的调优结果，后续若 `nz/nx/ny` 或 PML 厚度显著变化，应重新扫描。

下一步：

- 创建 `1.5x` 版本存档。
- 继续寻找下一阶段 `2.0x` 的优化点，优先考虑 `_ns` kernel 的内存访问、PML 分支分离、读写数组数量和常量/只读数据限定。

## 2026-06-02 01:37:33 +08:00

### 操作目标

- 创建并验证第一个正式速度阈值存档：`1.5x`。

### 修改文件

- 未修改源码。
- `archives/speedups/1.5x_20260601_173716_opt_1p5_block_64x4x2/` 为新增远程存档目录。

### 执行命令

第一次归档命令由于远程 shell 引号导致 `--notes` 被拆分，未创建存档；随后使用无空格备注重跑成功：

```bash
cd /data/shengwz/swz/cuda3D
python3 tools/archive_speedup.py \
  --threshold 1.5 \
  --speedup 1.576 \
  --baseline-time 3.491814 \
  --candidate-time 2.215294 \
  --tag opt_1p5_block_64x4x2 \
  --correctness-report benchmarks/reports/correctness_opt_1p5_block_64x4x2_latest \
  --perf-report benchmarks/reports/perf_3gpu_opt_1p5_block_64x4x2_latest \
  --candidate-run benchmarks/runs/perf_3gpu_opt_1p5_block_64x4x2_20260601_173358 \
  --baseline-run benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602 \
  --notes pressure_locals_no_scratch_no_sync_block_64x4x2
```

### 测试结果

存档创建成功：

```text
archive_dir = /data/shengwz/swz/cuda3D/archives/speedups/1.5x_20260601_173716_opt_1p5_block_64x4x2
manifest = /data/shengwz/swz/cuda3D/archives/speedups/1.5x_20260601_173716_opt_1p5_block_64x4x2/archive_manifest.json
```

存档内容检查：

- `archive_manifest.json` 存在。
- 源码快照存在：`AGENTS.md`、`AGENT_LOG.md`、`src/main.cu`、`src/rem_fd.cu`、`src/single_solver.cu`、`src/makefile.server`、`include/inc3D/*.h`。
- 工具脚本快照存在：`create_smoke_case.py`、`create_benchmark_cases.py`、`run_benchmark.py`、`compare_outputs.py`、`archive_speedup.py`、`sweep_blocksize.py`。
- correctness/perf 对比报告已复制。
- baseline/candidate 的 `manifest.json` 与 `run.log` 已复制。
- `nvidia-smi.txt` 单独文件不存在，但 GPU 信息已经在各 run 的 `manifest.json` 内记录。

### 输出、哈希与误差摘要

```text
threshold = 1.5x
baseline WP computing time = 3.491814 s
candidate WP computing time = 2.215294 s
speedup = 1.576x
candidate binary SHA256 = 101eb55f577521bf842ea1c5aaef9e735ae249a319065f20371bb159f804b186
archive source/src/single_solver.cu SHA256 = ddc3f33162b2a75a270277aaea06f8990346923c121a6584a67f424122248fc9
archive source/include/inc3D/cu_common.h SHA256 = fd1334b888cf25f3b2fbf863f50b2b916806e6dd15362a200451cd7c55cff611
```

### 风险与下一步

风险：

- 当前项目不是 git 仓库，本轮使用目录快照做版本存档。
- 若后续需要回退，可从 `archives/speedups/1.5x_20260601_173716_opt_1p5_block_64x4x2/source/` 复制对应源码文件恢复。

下一步：

- 继续向 `2.0x` 阈值推进。
- 优先研究能否减少 `cuda_fd3d_v_pml_ns` 与 `cuda_fd3d_p_pml_ns` 的全局内存流量，以及是否能把 PML 边界区域与内部区域拆开，减少主域线程的分支判断。

## 2026-06-02 01:43:35 +08:00

### 操作目标

- 尝试 `const`/`__restrict__` 指针限定优化。
- 如果性能退化，则撤回该尝试，确保当前工作树回到已存档的 `1.5x` 源码状态。

### 修改文件

- 临时修改后已撤回：
  - `src/single_solver.cu`
  - `include/inc3D/single_solver.h`
- 当前保留状态与 `1.5x` 存档源码一致：
  - `src/single_solver.cu`
  - `src/rem_fd.cu`
  - `include/inc3D/single_solver.h`
  - `include/inc3D/cu_common.h`

### 执行命令

临时优化构建和验证：

```bash
cd /data/shengwz/swz/cuda3D/src
source /opt/intel/oneapi/setvars.sh --force
make -B -f makefile.server test

cd /data/shengwz/swz/cuda3D
python3 tools/run_benchmark.py --case correctness --tag opt_restrict_const
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs \
  --candidate benchmarks/runs/correctness_opt_restrict_const_20260601_173943/outputs \
  --out benchmarks/reports/correctness_opt_restrict_const_latest
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_restrict_const
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs \
  --candidate benchmarks/runs/perf_3gpu_opt_restrict_const_20260601_174011/outputs \
  --out benchmarks/reports/perf_3gpu_opt_restrict_const_latest
```

撤回后恢复构建和验证：

```bash
cd /data/shengwz/swz/cuda3D/src
source /opt/intel/oneapi/setvars.sh --force
make -B -f makefile.server test

cd /data/shengwz/swz/cuda3D
python3 tools/run_benchmark.py --case perf_3gpu --tag restore_1p5_after_restrict_revert
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs \
  --candidate benchmarks/runs/perf_3gpu_restore_1p5_after_restrict_revert_20260601_174218/outputs \
  --out benchmarks/reports/perf_3gpu_restore_1p5_after_restrict_revert_latest
sha256sum src/single_solver.cu src/rem_fd.cu include/inc3D/single_solver.h include/inc3D/cu_common.h
```

### 测试结果

`opt_restrict_const`：

- correctness：通过，6 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。
- perf：通过，9 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。
- `mod time = 0.735389 s, 0.800412 s, 0.735823 s`。
- `WP computing time = 3.643586 s`，主指标退化。
- 判定：不采用，已撤回。

恢复后的 `restore_1p5_after_restrict_revert`：

- perf 输出对比通过，9 个输出文件全部 `rel_l2 = 0`、`max_abs = 0`、`rms = 0`。
- `mod time = 0.723223 s, 0.782299 s, 0.720468 s`。
- `WP computing time = 3.926151 s`，该轮服务器 GPU 1/2 在运行前已高占用，MPI 等待噪声明显；不作为性能结论。
- 源码哈希已确认与 `1.5x` 存档一致。

### 输出、哈希与误差摘要

当前远程源码哈希：

```text
src/single_solver.cu SHA256 = ddc3f33162b2a75a270277aaea06f8990346923c121a6584a67f424122248fc9
src/rem_fd.cu SHA256 = ec3712449627fd1b91a71d8dd44cd267623f740471d03885f279cc502978c237
include/inc3D/single_solver.h SHA256 = 2579906056d6687a8f60f17540c4f124d19335d19d743497359aa79d9618eb3e
include/inc3D/cu_common.h SHA256 = fd1334b888cf25f3b2fbf863f50b2b916806e6dd15362a200451cd7c55cff611
```

这些哈希与 `archives/speedups/1.5x_20260601_173716_opt_1p5_block_64x4x2/archive_manifest.json` 中记录的源码哈希一致。

### 风险与下一步

风险：

- 当前 perf 恢复验证时 GPU 1/2 已有高负载，`WP computing time` 不适合判断源码速度。
- `const`/`__restrict__` 的退化可能来自编译器寄存器/调度变化，也可能被 GPU 争用放大；由于已出现主指标退化，先不采用。

下一步：

- 在 GPU 负载较低时重跑一次 `perf_3gpu_opt_1p5_block_64x4x2` 或使用更稳定的 GPU 选择策略。
- 继续推进 `2.0x` 前，应优先改进 benchmark runner，让它记录每个 MPI rank 的耗时，避免只看 rank 0 的 `mod time`。

## 2026-06-02 02:41:11 +08:00 - 接续 2.0x 优化：CorePmlMargin 扫描、Profiler、sm_89 编译与本地 shared p_pml 原型

### 操作目标

- 接续当前约 `1.74x` 的有效优化版本，继续向 `2.0x` 阈值推进。
- 验证 `CorePmlMargin` 是否能进一步缩小边界保守区域。
- 用 `nsys` 定位主耗时 kernel。
- 测试 RTX 4090 原生 `sm_89` 编译参数。
- 本地实现 shared-memory `p_pml` 原型，准备下一轮远程 correctness/perf 验证。

### 修改文件

- 已同步并验证到远程：
  - `include/inc3D/cu_common.h`
    - 测试过 `CorePmlMargin=3`、`2`，最终恢复为 `4`。
  - `src/makefile.server`
    - 增加 `NV_ARCH ?= sm_89`
    - `NVFLAGS = -O3 -g -arch=$(NV_ARCH)`
- 本地已修改但尚未同步/验证：
  - `include/inc3D/single_solver.h`
    - 新增 `cuda_fd3d_p_pml_shared_ns(...)` 声明。
  - `src/single_solver.cu`
    - 新增 shared-memory 版本 `cuda_fd3d_p_pml_shared_ns(...)`。
  - `src/rem_fd.cu`
    - 本地将压力 PML kernel 调用切到 `cuda_fd3d_p_pml_shared_ns(...)`。

### 执行命令

核心验证命令摘要：

```bash
cd /data/shengwz/swz/cuda3D
python3 tools/run_benchmark.py --case correctness --tag opt_core_margin3
python3 tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs --candidate benchmarks/runs/correctness_opt_core_margin3_20260601_181521/outputs --out benchmarks/reports/correctness_opt_core_margin3_latest
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_core_margin3
python3 tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs --candidate benchmarks/runs/perf_3gpu_opt_core_margin3_20260601_181553/outputs --out benchmarks/reports/perf_3gpu_opt_core_margin3_latest

python3 tools/run_benchmark.py --case correctness --tag opt_core_margin2
python3 tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs --candidate benchmarks/runs/correctness_opt_core_margin2_20260601_181749/outputs --out benchmarks/reports/correctness_opt_core_margin2_latest

python3 tools/run_benchmark.py --case perf_3gpu --tag opt_margin4_restored
python3 tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs --candidate benchmarks/runs/perf_3gpu_opt_margin4_restored_20260601_182507/outputs --out benchmarks/reports/perf_3gpu_opt_margin4_restored_latest

/usr/local/cuda-12.2/bin/nsys profile --stats=true --force-overwrite=true -o /data/shengwz/swz/cuda3D/benchmarks/profiles/profile_1gpu_margin4_nsys /opt/intel/oneapi/mpi/latest/bin/mpirun -np 1 /data/shengwz/swz/cuda3D/bin/cuda_3D_FM < input_profile_1gpu.in

python3 tools/run_benchmark.py --case correctness --tag opt_restrict
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_restrict_retry

python3 tools/run_benchmark.py --case correctness --tag opt_sm89_norestrict
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_sm89_norestrict
```

### 测试结果

`CorePmlMargin=3`：

- correctness：通过。
- correctness 最大相对 L2 约 `4.296933e-06`，低于 `1e-5`。
- perf：通过。
- `WP computing time = 2.071675 s`，相对冻结基准 `3.491814 s` 约 `1.686x`，比当前最佳 `~2.004 s` 慢。
- 判定：正确但不采用为加速方向。

`CorePmlMargin=2`：

- correctness：失败。
- 相对 L2 约 `4.555289e-05` 到 `4.711041e-05`，超过 `1e-5`。
- 判定：不采用，已恢复 `CorePmlMargin=4`。

恢复 `CorePmlMargin=4`：

- perf 输出比较通过。
- `WP computing time = 2.006040 s`，相对冻结基准约 `1.740x`。

`nsys profile_1gpu`：

- `cuda_fd3d_p_pml_ns`：`36.8%` GPU kernel time。
- `cuda_fd3d_v_pml_ns`：`34.6%` GPU kernel time。
- `cuda_fd3d_p_core_ns`：`28.0%` GPU kernel time。
- `lint3d_inject_bell_extract_gpu_zz`：`0.7%` GPU kernel time。
- 结论：注入/检波已不是瓶颈，后续应主要优化有限差分主循环。

`ncu`：

- 尝试使用 Nsight Compute 采样 `cuda_fd3d_p_pml_ns`。
- 失败原因：服务器未开放 NVIDIA GPU Performance Counters，报错 `ERR_NVGPUCTRPERM`。
- 判定：后续主要依赖 `nsys` 和 A/B benchmark。

`const/__restrict__`：

- correctness：通过。
- perf：`WP computing time = 2.046238 s`。
- 判定：比恢复版慢，已撤回本地和远程源码中的 `const/__restrict__` 参数修改。

`sm_89` 编译：

- correctness：通过。
- perf：`WP computing time = 2.005830 s`。
- 判定：与恢复版 `2.006040 s` 基本一致，没有可量化加速；保留 `makefile.server` 的 `NV_ARCH ?= sm_89` 可配置改动，因为目标服务器为 RTX 4090。

### 输出、哈希与误差摘要

- 当前已验证最快有效版本仍约 `WP = 2.004071 s` 到 `2.006040 s` 区间，约 `1.74x`。
- 尚未突破 `2.0x`；`2.0x` 阈值对应 `WP <= 1.745907 s`。
- `CorePmlMargin=2` 的错误幅度已明确超过阈值，不再作为可接受优化。

### 风险与下一步

风险：

- 当前本地源码包含尚未远程验证的 `cuda_fd3d_p_pml_shared_ns` 原型；远程服务器尚未同步这三处 shared kernel 相关修改。
- 远程同步在本地平台审批处被拦截：`You've hit your usage limit... try again at 6:10 AM`。
- 服务器 GPU 2/3 持续有外部高负载，`perf_3gpu` 使用 `CUDA_VISIBLE_DEVICES=0,1,2` 时仍可能受到 GPU 2 干扰。

下一步：

- 审批额度恢复后，先同步并编译本地 shared `p_pml` 原型。
- 运行 `correctness --tag opt_p_pml_shared`，若通过再运行 `perf_3gpu --tag opt_p_pml_shared`。
- 如果 shared `p_pml` 通过但速度不佳，立即回退 `rem_fd.cu` 的调用到 `cuda_fd3d_p_pml_ns`，保留原型供后续拆解。
- 若 shared `p_pml` 明显提速，再继续对 `cuda_fd3d_v_pml_ns` 做类似 shared-memory 处理。

---

## 2026-06-02 10:30 CST - 收束试错并重建 CUDA 优化纪律

### 操作目标

- 响应用户反馈，停止频繁请求确认，改成更 autonomous 的 CUDA 优化工作流。
- 先收束上一轮留下的实验状态，再基于 `nsys`/benchmark 数据推进。
- 保持 correctness 门槛：相对 L2 `<= 1e-5`，禁止 NaN/Inf。

### 修改文件

- `src/rem_fd.cu`
  - 保留 `cudaFuncSetCacheConfig(..., cudaFuncCachePreferL1)`，让 `cuda_fd3d_v_pml_ns`、`cuda_fd3d_p_core_ns`、`cuda_fd3d_p_pml_ns` 偏向 L1 cache。
  - 尝试 CUDA Graph 将每步 `V -> P core -> P PML` 三个主 kernel 合成 graph launch；验证后撤回，因为 graph capture/instantiate 位于每个 shot 内部，`WP computing time` 会计入一次性开销。
  - 撤回 CUDA Graph 后恢复普通 kernel launch 与 `d_p0/d_p1` 指针交换。
- `include/inc3D/cu_common.h`
  - 尝试 `64x4x1` block 配置；验证后撤回。
  - 当前恢复为 `128x2x1`：`VBlockSize*`、`PBlockSize*`、`PCoreBlockSize*`、`PmlBlockSize*` 均为 `128,2,1`。
- `src/single_solver.cu`
  - 尝试单 launch compact PML 映射；正确但慢，已撤回。
  - 确认没有 `region0_n`、`PmlCompactBlockSize`、`dimg_pml`、CUDA Graph 残留。

### 执行命令摘要

```bash
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_l1cache_g0 --gpus 0
/usr/local/cuda-12.2/bin/nsys profile --stats=true --force-overwrite=true -o benchmarks/profiles/profile_1gpu_l1cache_g0_nsys/report ...
python3 tools/run_benchmark.py --case correctness --tag opt_step_graph_l1
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_step_graph_l1_g0 --gpus 0
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_step_graph_l1_gpus031 --gpus 0,3,1
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_step_graph_l1_default
python3 tools/sweep_blocksize.py --case profile_1gpu --gpus 0 --variants 128x2x1 128x1x2 64x4x1 64x2x2 32x8x1 32x4x2 256x1x1 16x16x1
python3 tools/run_benchmark.py --case correctness --tag opt_l1_bs64x4x1
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_l1_bs64x4x1_g0 --gpus 0
```

### 测试结果

`L1 cache` 实验：

- correctness：通过。
- `profile_1gpu` on GPU0：`WP computing time = 0.280381 s`，复测 clean 状态 `0.283379 s`。
- `nsys` kernel 分布：
  - `cuda_fd3d_v_pml_ns`：`96.808 ms`，约 `36.5%`。
  - `cuda_fd3d_p_pml_ns`：`96.381 ms`，约 `36.4%`。
  - `cuda_fd3d_p_core_ns`：`69.892 ms`，约 `26.4%`。
  - 注入/检波约 `0.7%`。
- 判定：低风险，保留。

`CUDA Graph` 实验：

- correctness：通过。
- `profile_1gpu` on GPU0：`mod time = 0.264888 s`，但 `WP computing time = 0.432740 s`。
- `perf_3gpu --gpus 0,3,1`：输出比较通过，`WP computing time = 2.253569 s`，约 `1.55x`，非默认卡组且有负载，不作为里程碑。
- 默认 `perf_3gpu`：`WP computing time = 3.651085 s`，受 GPU1/2 外部负载污染。
- 判定：由于 graph capture/instantiate 在每个 shot 内部发生，`WP` 计入一次性开销；当前代码结构下不采用，已撤回。

`single launch compact PML` 实验：

- correctness：通过。
- `profile_1gpu` on GPU0：`WP computing time = 0.373494 s`，慢于 clean `128x2x1 + L1`。
- 判定：映射/除法开销和访存形状抵消了减少无效线程的收益；已撤回。

`64x4x1` block 配置：

- sweep 临时结果：`64x4x1 WP = 0.282299 s`，略优于 sweep 中 `128x2x1 WP = 0.290441 s`。
- 正式写入头文件并重编译后：
  - correctness：通过。
  - `profile_1gpu` on GPU0：`WP computing time = 0.294813 s`。
- 判定：正式复测慢于 `128x2x1 + L1`，已撤回。

### 当前有效候选

- 源码/服务器二进制已恢复为：
  - `NVFLAGS = -O3 -arch=$(NV_ARCH)`。
  - `NV_ARCH ?= sm_89`。
  - `V/P/PCore/Pml block = 128x2x1`。
  - 三个主 kernel 设置 `cudaFuncCachePreferL1`。
- 当前 profile 参考：
  - `benchmarks/runs/profile_1gpu_opt_l1_clean_g0_20260602_102525`
  - `WP computing time = 0.283379 s`。
- 当前 correctness 参考：
  - `benchmarks/reports/correctness_opt_l1cache_latest`
  - rel L2 约 `3e-7` 到 `4e-7`。

### 风险与下一步

- 默认三卡 `0,1,2` 当前经常受外部任务污染，不能把污染结果作为里程碑依据。
- 尚未突破 `2.0x`，因此没有新建 `2.0x` archive。
- 后续应聚焦 kernel 本体，而不是继续做 launch/参数小试错：
  - 首要热点仍是 `v_pml` 与 `p_pml`，二者合计约 `73%` GPU kernel time。
  - 下一步优先考虑对 `v_pml/p_pml` 做访存模式和分支结构级重写，或构造更稳定的长步数单卡开发 case 来降低一次性开销噪声。

## 2026-06-02 10:50 CST - 自主工作流校正与 p_core/p_pml shared-memory 实验

### 操作目标

- 根据用户反馈，停止频繁确认，改为自主闭环：profile -> 单点修改 -> 编译 -> correctness -> profile/perf -> 记录日志 -> 达阈值归档。
- 对上一轮 `p_core` z 向 shared-memory 优化做安全审查，避免将未定义 shared-memory 访问当作性能收益。
- 尝试更克制的 `p_pml` z 向 shared-memory 缓存，只缓存 `vz`，验证是否能降低 `p_pml` 热点开销。

### 修改文件

- `src/single_solver.cu`
  - 保留 `p_core` z 向 shared-memory 思路，但新增 `CoreStencilRadius = 7`，将 `z_tile` halo 从全局 `radius=4` 扩到该 kernel 实际访问的 `+-7`。
  - 尝试在 `cuda_fd3d_p_pml_ns` 中加入 `vz_tile` shared-memory 缓存；验证正确但显著变慢后已撤回。
  - 撤回位置错误的 `vz_tile` 声明，确认 `v_pml` 没有残留 shared-memory 实验代码。

### 执行命令摘要

```bash
scp src/single_solver.cu shengwz@162.105.91.239:/data/shengwz/swz/cuda3D/src/single_solver.cu
cd /data/shengwz/swz/cuda3D/src
source /opt/intel/oneapi/setvars.sh --force >/tmp/cuda3d_setvars.log 2>&1
make -B -f makefile.server test
python3 tools/run_benchmark.py --case correctness --tag opt_pcore_halo7
python3 tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs --candidate benchmarks/runs/correctness_opt_pcore_halo7_20260602_104332/outputs --out benchmarks/reports/correctness_opt_pcore_halo7_20260602_104332
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_pcore_halo7_g0 --gpus 0
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_pcore_halo7_g0_r2 --gpus 0
python3 tools/run_benchmark.py --case correctness --tag opt_ppml_zshared
python3 tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs --candidate benchmarks/runs/correctness_opt_ppml_zshared_20260602_104644/outputs --out benchmarks/reports/correctness_opt_ppml_zshared_20260602_104644
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_ppml_zshared_g0 --gpus 0
```

### 测试结果

`p_core halo7` 安全修复：

- correctness：通过。
- rel L2 范围：约 `2.98e-7` 到 `4.10e-7`，低于 `1e-5` 门槛。
- `profile_1gpu opt_pcore_halo7_g0`：`WP computing time = 0.431598 s`，判定为噪声样本。
- `profile_1gpu opt_pcore_halo7_g0_r2`：`WP computing time = 0.276802 s`。
- 判定：安全修复保留，单卡 profile 至少不劣于 L1 clean 参考。

`p_pml zshared` 实验：

- correctness：通过。
- rel L2 范围：约 `2.98e-7` 到 `4.10e-7`。
- `profile_1gpu opt_ppml_zshared_g0`：`WP computing time = 0.721128 s`。
- 判定：正确但显著慢，已撤回。推测原因是 PML 分支区域内引入 `__syncthreads()` 和 shared-memory halo 装载后，开销大于 z 向访存复用收益。

### 输出、哈希或误差摘要

- correctness report：
  - `benchmarks/reports/correctness_opt_pcore_halo7_20260602_104332/comparison.md`
  - `benchmarks/reports/correctness_opt_ppml_zshared_20260602_104644/comparison.md`
- profile runs：
  - `benchmarks/runs/profile_1gpu_opt_pcore_halo7_g0_20260602_104358`
  - `benchmarks/runs/profile_1gpu_opt_pcore_halo7_g0_r2_20260602_104421`
  - `benchmarks/runs/profile_1gpu_opt_ppml_zshared_g0_20260602_104709`

### 风险与下一步

- `p_core` 原先的 shared-memory halo 宽度不足，虽然 correctness 样例通过，但存在潜在未定义访问风险；已通过 `CoreStencilRadius=7` 修正。
- `p_pml` 单纯 z 向 shared-memory 缓存不是有效方向，后续不再沿“给 PML kernel 加同步 shared tile”的路径盲试。
- 下一步应转向更高收益的结构性优化：
  - 用 `nsys/ncu` 精确拆解 `v_pml`/`p_pml` 的访存效率、occupancy、寄存器压力。
  - 优先考虑减少 PML kernel 的全域无效线程与分支，而不是在全域 kernel 内加同步。
  - 若要拆 PML 区域，必须避免除法/模运算映射和过多 launch，倾向按 6 个面或按连续内存 slab 设计专用 kernel。

## 2026-06-02 11:00 CST - 三卡窗口检查、pressure 融合实验与正式 perf 复测

### 操作目标

- 响应用户提示，尝试使用 `0,2,3` 三张卡运行 `perf_3gpu`。
- 基于 `nsys` 发现的 launch/API 开销，尝试将 `p_core` 与 `p_pml` pressure update 融合为单个 kernel，验证减少 launch 是否带来收益。
- 在实验失败后恢复稳定主线，避免后续测试混入失败代码。

### 修改文件

- `src/single_solver.cu`
  - 临时在 `cuda_fd3d_p_pml_ns` 中加入 core 分支，使其同时处理 core 与 PML pressure update。
  - 验证正确但慢后，已撤回融合分支，恢复为 PML kernel 只处理非 core 区域。
- `src/rem_fd.cu`
  - 临时移除单独的 `cuda_fd3d_p_core_ns` launch。
  - 验证慢后，已恢复 `V -> P core -> P pml -> inject/extract` 原调度。

### 执行命令摘要

```bash
python3 tools/run_benchmark.py --case correctness --tag opt_fused_p
python3 tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs --candidate benchmarks/runs/correctness_opt_fused_p_20260602_105310/outputs --out benchmarks/reports/correctness_opt_fused_p_20260602_105310
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_fused_p_g0 --gpus 0
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_fused_p_g2 --gpus 2
python3 tools/run_benchmark.py --case perf_3gpu --tag stable_pcore_halo7_gpus023 --gpus 0,2,3
python3 tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs --candidate benchmarks/runs/perf_3gpu_stable_pcore_halo7_gpus023_20260602_105654/outputs --out benchmarks/reports/perf_3gpu_stable_pcore_halo7_gpus023_20260602_105654
```

### 测试结果

`pressure fusion` 实验：

- correctness：通过，rel L2 约 `2.98e-7` 到 `4.10e-7`。
- `profile_1gpu opt_fused_p_g0`：`WP computing time = 0.566275 s`。
- `profile_1gpu opt_fused_p_g2`：`WP computing time = 0.363296 s`。
- 判定：正确但慢，已撤回。简单融合造成大分支/访存路径混杂，kernel 效率下降，节省的 launch 无法抵消。

`perf_3gpu stable_pcore_halo7_gpus023`：

- `CUDA_VISIBLE_DEVICES=0,2,3`，输出文件数 `9`。
- `WP computing time = 2.492231 s`。
- 相对 baseline `3.491814 s` speedup = `1.401x`。
- perf 输出比较：通过，rel L2 约 `2.29e-6` 到 `2.53e-6`。
- 判定：低于 `1.5x` 阈值，且 GPU0 同时存在另一个项目的 `train_rl_fwi.py` 进程，不作为正式里程碑。

### 环境观察

- `ncu` 2023.2 可启动，但服务器普通用户无权访问 GPU performance counters，报 `ERR_NVGPUCTRPERM`；后续只能依赖 `nsys`、kernel timing 和输出对比，除非管理员放开 counter 权限。
- 用户提示 `0,2,3` 可用后，实际检查到 GPU0 上有：
  - PID `336581`
  - `/data/shengwz/swz/RL-seismic-inversion/train_rl_fwi.py`
  - 持续约 `70%+` SM 占用
- 该进程属于另一个项目目录，未擅自终止。

### 风险与下一步

- 当前没有足够干净的三卡窗口；正式 3GPU 结论需要避开 GPU0/GPU1 外部任务。
- 下一步继续用空卡 GPU2/GPU3 做单卡 profile 和小步优化。
- 结构上不再做“大分支融合”，改为研究专用 PML 面/edge kernel 或更稳定的 CUDA Graph 生命周期方案。

## 2026-06-02 11:12 CST - fast math 主线与新版 1.5x 存档

### 操作目标

- 测试 `--use_fast_math` 是否能在不破坏 correctness 的前提下提升当前候选性能。
- 抓住 GPU1/2/3 空闲窗口，运行更干净的 `perf_3gpu`。
- 对超过 `1.5x` 阈值的新最佳版本做新增存档，不覆盖旧存档。

### 修改文件

- `src/makefile.server`
  - `NVFLAGS` 从 `-O3 -arch=$(NV_ARCH)` 改为 `-O3 -arch=$(NV_ARCH) --use_fast_math`。
- `include/inc3D/single_solver.h`、`src/single_solver.cu`
  - 临时尝试给热点 `_ns` kernel 增加 `__restrict__` 指针限定。
  - correctness 通过但 profile 无收益，已撤回。

### 执行命令摘要

```bash
make -B -f makefile.server 'NVFLAGS=-O3 -arch=sm_89 --use_fast_math' test
python3 tools/run_benchmark.py --case correctness --tag opt_fastmath_temp
python3 tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs --candidate benchmarks/runs/correctness_opt_fastmath_temp_20260602_110348/outputs --out benchmarks/reports/correctness_opt_fastmath_temp_20260602_110348
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_fastmath_temp_g2 --gpus 2
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_fastmath_temp_g3 --gpus 3
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_fastmath_gpus023 --gpus 0,2,3
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_fastmath_gpus123 --gpus 1,2,3
python3 tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs --candidate benchmarks/runs/perf_3gpu_opt_fastmath_gpus123_20260602_110535/outputs --out benchmarks/reports/perf_3gpu_opt_fastmath_gpus123_20260602_110535
```

### 测试结果

`__restrict__` 实验：

- correctness：通过。
- `profile_1gpu opt_restrict_hot_ns_g2`：`WP computing time = 0.350464 s`。
- `profile_1gpu opt_restrict_hot_ns_g3`：`WP computing time = 0.401139 s`。
- 判定：无收益，已撤回。

`--use_fast_math` 实验：

- correctness：通过，rel L2 约 `2.98e-7` 到 `4.10e-7`。
- `profile_1gpu opt_fastmath_temp_g2`：`WP computing time = 0.278264 s`。
- `profile_1gpu opt_fastmath_temp_g3`：`WP computing time = 0.310789 s`。
- 判定：数值安全，保留为当前主线编译选项。

`perf_3gpu opt_fastmath_gpus023`：

- `CUDA_VISIBLE_DEVICES=0,2,3`。
- `WP computing time = 2.521664 s`。
- GPU0 同时存在外部 python 训练负载，不作为里程碑。

`perf_3gpu opt_fastmath_gpus123`：

- `CUDA_VISIBLE_DEVICES=1,2,3`。
- `WP computing time = 2.074612 s`。
- speedup = `3.491814 / 2.074612 = 1.683x`。
- perf 输出比较：通过，rel L2 约 `2.31e-6` 到 `2.45e-6`。
- 判定：超过 `1.5x`，未达到 `2.0x`。

### 存档

- 新增存档：
  - `archives/speedups/1.5x_20260602_110535_fastmath_pcore_halo7`
- 存档内容：
  - `source_snapshot/`：`single_solver.cu`、`rem_fd.cu`、`makefile.server`、`single_solver.h`、`cu_common.h`。
  - `benchmark_tools/`：`run_benchmark.py`、`compare_outputs.py`。
  - `runs/`：fast math correctness 与 `perf_3gpu opt_fastmath_gpus123` run。
  - `reports/`：对应 correctness/perf comparison。
  - `summary.md`、`manifest.json`。
- 存档哈希文件数：`30`。

### 风险与下一步

- `--use_fast_math` 当前样例数值通过，但它是编译级近似数学选项；后续新增更大或更敏感样例时必须复查误差。
- 当前最好版本为 `1.683x`，尚未达到 `2.0x`。
- 下一步继续围绕 `v_pml` 与 `p_pml` 热点，优先尝试不会引入大分支融合和全域同步的优化。

## 2026-06-02 11:25 CST - block sweep、PML slab 与 pressure streams 试验

### 操作目标

- 在 fast math 主线下重新扫描 CUDA block size，避免复用旧编译条件下的结论。
- 尝试将 `p_pml` 从全域 launch 改为 6 个不重叠 slab launch，减少 core 空线程。
- 尝试用 CUDA stream 并发执行 `p_core` 和 `p_pml`，验证 pressure 阶段能否 overlap。

### 修改文件

- `tools/sweep_blocksize.py`
  - 新增 `--extra-nvflags` 参数，使 sweep 可以显式带上 `--use_fast_math`。
- `include/inc3D/single_solver.h`、`src/single_solver.cu`
  - 临时扩展 `cuda_fd3d_p_pml_ns`，支持 region offset/size 参数。
  - 6-slab 验证后变慢，已撤回。
- `src/rem_fd.cu`
  - 临时将 `p_pml` 改为 6 个 slab launch。
  - 临时增加 `stream_v/stream_core/stream_pml`，让 `p_core` 与 `p_pml` 在 `V` 后并发。
  - 两个实验均验证变慢，已撤回。

### 执行命令摘要

```bash
python3 tools/sweep_blocksize.py --case profile_1gpu --gpus 2 --extra-nvflags=--use_fast_math --variants 128x2x1 64x4x1 32x8x1 256x1x1 128x1x2 64x2x2 32x4x2 16x16x1
python3 tools/run_benchmark.py --case correctness --tag opt_ppml_6slab
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_ppml_6slab_g2 --gpus 2
nsys profile --stats=true ... profile_1gpu_ppml_6slab_nsys_20260602_1115
python3 tools/run_benchmark.py --case correctness --tag opt_pressure_streams
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_pressure_streams_g2 --gpus 2
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_pressure_streams_gpus123 --gpus 1,2,3
```

### 测试结果

Fast math block sweep：

- Report：`benchmarks/reports/blocksize_sweep_20260602_110816/summary.md`
- 最佳：`128x2x1`，`profile_1gpu WP = 0.280638 s`。
- 结论：当前 block size 保持 `128x2x1`。

`p_pml 6-slab`：

- correctness：通过。
- `profile_1gpu opt_ppml_6slab_g2`：`WP computing time = 0.285626 s`。
- `nsys` kernel 分解：
  - `cuda_fd3d_p_pml_ns` 从主线约 `95.3 ms / 501 calls` 变为 `104.9 ms / 3006 calls`。
- 判定：减少空线程不抵消额外 launch 和小 slab 低效率，已撤回。

`pressure streams`：

- correctness：通过。
- `profile_1gpu opt_pressure_streams_g2`：
  - `mod time = 0.260504 s`
  - `WP computing time = 0.349019 s`
- `perf_3gpu opt_pressure_streams_gpus123`：
  - `WP computing time = 3.050908 s`
  - perf 输出比较通过。
- 判定：GPU event timing 显示 overlap 发生，但正式 WP 被每步 event/wait/API 开销拖慢，已撤回。

### 风险与下一步

- 多 launch、多 event 的结构性改动在当前 benchmark 下很容易损失 WP，即使 GPU kernel 时间看起来更低。
- 下一步优先寻找“少引入 API 开销”的优化：
  - 单 kernel 内减少真实算术或访存。
  - 改善当前 full-grid kernel 内的分支/访存顺序。
  - 谨慎考虑 graph，但必须避免 per-shot graph instantiate 进入 WP。

## 2026-06-02 11:35 CST - PML block 独立化与 CorePmlMargin 边界测试

### 操作目标

- 测试 `p_pml` 使用独立 PML block size 是否能让纯 core block 整块跳过。
- 测试 `CorePmlMargin` 从 `4` 降到 `3` 或 `2` 是否能减少 PML 工作量并保持数值正确。

### 修改文件

- `include/inc3D/cu_common.h`
  - 临时将 `PmlBlockSize` 从 `128x2x1` 改为 `16x16x1`。
  - 验证变慢后已恢复为 `128x2x1`。
- `src/rem_fd.cu`
  - 临时增加独立 `dimg_pml/dimb_pml`。
  - 验证变慢后已撤回。
- `src/single_solver.cu`
  - 临时给 `cuda_fd3d_p_pml_ns` 增加 block-level core skip。
  - 验证变慢后已撤回。

### 执行命令摘要

```bash
python3 tools/run_benchmark.py --case correctness --tag opt_pml_block16_skip
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_pml_block16_skip_g2 --gpus 2
make -B -f makefile.server 'NVFLAGS=-O3 -arch=sm_89 --use_fast_math -DCorePmlMargin=2' test
python3 tools/run_benchmark.py --case correctness --tag opt_margin2_temp
make -B -f makefile.server 'NVFLAGS=-O3 -arch=sm_89 --use_fast_math -DCorePmlMargin=3' test
python3 tools/run_benchmark.py --case correctness --tag opt_margin3_temp
python3 tools/run_benchmark.py --case profile_1gpu --tag opt_margin3_temp_g2 --gpus 2
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_margin3_gpus123 --gpus 1,2,3
```

### 测试结果

`PmlBlockSize=16x16x1 + block skip`：

- correctness：通过。
- `profile_1gpu opt_pml_block16_skip_g2`：`WP computing time = 0.644116 s`。
- 判定：严重退化，已撤回。原因是破坏了 z-fast 方向的访存合并，收益远小于损失。

`CorePmlMargin=2`：

- correctness：失败。
- rel L2 约 `4.56e-5` 到 `4.71e-5`，超过 `1e-5` 门槛。
- 判定：不可用。

`CorePmlMargin=3`：

- correctness：通过。
- correctness rel L2 约 `3.84e-6` 到 `4.30e-6`。
- `profile_1gpu opt_margin3_temp_g2`：`WP computing time = 0.290565 s`。
- `perf_3gpu opt_margin3_gpus123`：`WP computing time = 2.163083 s`，慢于当前最佳 `2.074612 s`。
- 判定：正确但慢，已撤回。

### 当前主线状态

- `src/makefile.server`：`NVFLAGS = -O3 -arch=$(NV_ARCH) --use_fast_math`。
- `CorePmlMargin = 4`。
- `V/P/PCore/Pml block = 128x2x1`。
- 当前最佳正式三卡结果：
  - `benchmarks/runs/perf_3gpu_opt_fastmath_gpus123_20260602_110535`
  - `WP computing time = 2.074612 s`
  - speedup = `1.683x`
  - 已归档：`archives/speedups/1.5x_20260602_110535_fastmath_pcore_halo7`

### 风险与下一步

- `2.0x` 阈值要求 `WP <= 1.745907 s`，当前还差约 `0.329 s`。
- 已验证失败方向：
  - 简单 pressure fusion。
  - PML 6-slab launch。
  - pressure streams。
  - PML block 改成 `16x16x1`。
  - `CorePmlMargin <= 2`。
- 下一步应考虑更深层的单 kernel 算法改写，特别是减少 `v_pml`/`p_pml` 内部实际访存，而不是再增加 launch 或 event。

## 2026-06-03 00:25 CST - 干净 0/1/2 三卡主线复测

### 操作目标

- 在用户释放 CUDA 0/1/2 后，复测当前 fast math 主线的默认三卡 `perf_3gpu`。
- 确认当前主线在干净默认卡组上的正式 WP 和数值误差。

### 执行命令摘要

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi pmon -c 1
python3 tools/run_benchmark.py --case perf_3gpu --tag mainline_fastmath_clean012 --gpus 0,1,2
python3 tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs --candidate benchmarks/runs/perf_3gpu_mainline_fastmath_clean012_20260602_162358/outputs --out benchmarks/reports/perf_3gpu_mainline_fastmath_clean012_20260602_162358
```

### 测试结果

- GPU 状态：CUDA 0/1/2 空闲，CUDA 3 有外部 python 负载。
- Run：`benchmarks/runs/perf_3gpu_mainline_fastmath_clean012_20260602_162358`
- Report：`benchmarks/reports/perf_3gpu_mainline_fastmath_clean012_20260602_162358`
- `WP computing time = 2.143985 s`
- speedup = `3.491814 / 2.143985 = 1.628x`
- perf 输出比较：通过，rel L2 约 `2.31e-6` 到 `2.45e-6`。

### 风险与下一步

- 这轮默认卡组干净，但速度略慢于此前 `1,2,3` 的最佳 `2.074612 s`；判断为正常卡组/系统噪声差异，不更新存档。
- 当前仍未达到 `2.0x` 阈值，后续继续聚焦单 kernel 内部访存与算术优化，避免增加额外 launch/event。

## 2026-06-03 03:22 CST - 远端被外部 agent 改乱后的恢复

### 操作目标

- 按用户要求尝试用本地项目覆盖远端 `/data/shengwz/swz/cuda3D`。
- 验证覆盖后发现本地副本也含有外部 agent 插入的 FP16 实验代码，不能作为可信恢复源。
- 立即从已归档且通过 correctness/perf 的 `1.5x_20260602_110535_fastmath_pcore_halo7` 源码快照恢复远端核心源码，并反向同步回本地。

### 修改/覆盖文件

- 初始本地覆盖包包含：
  - `src/`
  - `include/`
  - `tools/`
  - `AGENTS.md`
  - `AGENT_LOG.md`
  - `README.txt`
  - `input.in`
- 发现问题后，从归档恢复：
  - `src/rem_fd.cu`
  - `src/single_solver.cu`
  - `src/makefile.server`
  - `include/inc3D/single_solver.h`

### 执行命令摘要

```bash
python3 -m zipfile -e __incoming_source_overlay.zip .
make -B -f makefile.server test
python3 tools/run_benchmark.py --case correctness --tag after_local_overlay --gpus 0
python3 tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs --candidate benchmarks/runs/correctness_after_local_overlay_20260603_031959/outputs --out benchmarks/reports/correctness_after_local_overlay_20260603_031959
cp archives/speedups/1.5x_20260602_110535_fastmath_pcore_halo7/source_snapshot/src/rem_fd.cu src/rem_fd.cu
cp archives/speedups/1.5x_20260602_110535_fastmath_pcore_halo7/source_snapshot/src/single_solver.cu src/single_solver.cu
cp archives/speedups/1.5x_20260602_110535_fastmath_pcore_halo7/source_snapshot/src/makefile.server src/makefile.server
cp archives/speedups/1.5x_20260602_110535_fastmath_pcore_halo7/source_snapshot/include/inc3D/single_solver.h include/inc3D/single_solver.h
python3 tools/run_benchmark.py --case correctness --tag restored_archive_after_bad_overlay --gpus 0
```

### 测试结果

- 本地覆盖后的 correctness 失败：
  - Run：`benchmarks/runs/correctness_after_local_overlay_20260603_031959`
  - Report：`benchmarks/reports/correctness_after_local_overlay_20260603_031959`
  - rel L2 约 `1.0`，部分输出全零。
  - 直接原因：本地副本中残留 Hermes/外部 agent 插入的 FP16 wavefield 实验代码和 `__half` 原型。
- 从 1.5x 归档恢复后 correctness 通过：
  - Run：`benchmarks/runs/correctness_restored_archive_after_bad_overlay_20260603_032156`
  - Report：`benchmarks/reports/correctness_restored_archive_after_bad_overlay_20260603_032156`
  - rel L2 约 `2.98e-7` 到 `4.10e-7`。
- 恢复后的源码哈希：
  - `src/rem_fd.cu`: `9d9e6a2bfa08b3e5865f6ad3bf423d9915307f7edaac931782f3d1778c65e45c`
  - `src/single_solver.cu`: `46ac1d9820af6b6258aa9e639cdcf0c9f4333c29979460738c717920d2158925`
  - `src/makefile.server`: `2789763712a9d10a32036249c4e66e1e7f6e3ba39aec20035b8f41117c74c392`
  - `include/inc3D/single_solver.h`: `063a685b26fa17501e80646fe3499dd67c4ff9373e7942b8a441ac5301dd1bf6`

### 风险与下一步

- 本地和远端均已恢复到可信 1.5x 归档底座；后续不能再使用恢复前的本地 FP16 副本作为上传源。
- 当前四张 GPU 一度均被外部 `xspecfem3D` 占用，因此本轮只做 correctness，不做正式 `perf_3gpu` 阈值测试。
- 下一步继续从可信底座出发做结构性 CUDA 优化；任何 FP16/混合精度实验必须单独分支、先过 correctness，不允许直接覆盖主线。
## 2026-06-03 11:37 +08:00 - 恢复可信 1.5x 源码后发现并存档 2.0x 调度层加速

### 操作目标
- 用户要求先用本地可信项目覆盖/恢复云端被其他 agent 改乱的代码，然后继续推进提速。
- 验证刚恢复后的源码状态，撤回无收益的 persistent GPU workspace 实验。
- 从更结构性的角度测试多 MPI rank 复用 3 张 RTX 4090，让 9 炮并发度高于默认 `np=3`。

### 修改文件
- `src/rem_fd.cu`
  - 远端从 `archives/speedups/1.5x_20260602_110535_fastmath_pcore_halo7/source_snapshot/src/rem_fd.cu` 恢复。
  - 本地通过 `scp` 从远端恢复后的 `src/rem_fd.cu` 拉回，避免本地残留 Hermes FP16/坏 overlay 或 persistent workspace 代码。
- `AGENT_LOG.md`
  - 追加本条日志。

### 执行命令摘要

```bash
cd /data/shengwz/swz/cuda3D
cp archives/speedups/1.5x_20260602_110535_fastmath_pcore_halo7/source_snapshot/src/rem_fd.cu src/rem_fd.cu
cd src
source /opt/intel/oneapi/setvars.sh --force >/tmp/cuda3d_setvars.log 2>&1
make -B -f makefile.server test >/tmp/cuda3d_build_revert_persistent.log 2>&1

python3 tools/run_benchmark.py --case perf_3gpu --tag persistent_workspace_clean_gpus012_recheck --gpus 0,1,2
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs \
  --candidate benchmarks/runs/perf_3gpu_persistent_workspace_clean_gpus012_recheck_20260603_032811/outputs \
  --out benchmarks/reports/perf_3gpu_persistent_workspace_clean_gpus012_recheck_20260603_032811

python3 tools/run_benchmark.py --case profile_1gpu --tag restored_1p5_baseline_profile_g0 --gpus 0
CUDA_VISIBLE_DEVICES=0 ncu --target-processes all --set speedOfLight ...
CUDA_VISIBLE_DEVICES=0 nvprof --print-gpu-trace ...

python3 tools/run_benchmark.py --case perf_3gpu --tag rank9_3gpu_concurrent_shots --np 9 --gpus 0,1,2
python3 tools/run_benchmark.py --case perf_3gpu --tag rank3_restored_recheck_after_rank_sweep --np 3 --gpus 0,1,2
python3 tools/run_benchmark.py --case perf_3gpu --tag rank6_3gpu_concurrent_shots_serial --np 6 --gpus 0,1,2
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/outputs \
  --candidate benchmarks/runs/perf_3gpu_rank6_3gpu_concurrent_shots_serial_20260603_033547/outputs \
  --out benchmarks/reports/perf_3gpu_rank6_3gpu_concurrent_shots_serial_20260603_033547

python3 tools/archive_speedup.py \
  --threshold 2.0 \
  --speedup 2.254 \
  --baseline-time 3.491814 \
  --candidate-time 1.549196 \
  --tag rank6_3gpu_concurrent_shots \
  --correctness-report benchmarks/reports/correctness_restored_archive_after_bad_overlay_20260603_032156 \
  --perf-report benchmarks/reports/perf_3gpu_rank6_3gpu_concurrent_shots_serial_20260603_033547 \
  --candidate-run benchmarks/runs/perf_3gpu_rank6_3gpu_concurrent_shots_serial_20260603_033547 \
  --baseline-run benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602
```

### 测试结果
- persistent GPU workspace clean recheck：
  - 输出对比通过。
  - `WP computing time = 2.589584 s`，相对冻结 baseline speedup `1.348x`。
  - 判定：慢于当前可信 1.5x 存档 `2.074612 s`，已撤回。
- `profile_1gpu restored_1p5_baseline_profile_g0`：
  - `WP computing time = 0.424883 s`，该轮受系统/外部噪声影响偏慢。
  - `ncu` 失败：Nsight Compute 2021.3.1 section/rule 路径不可用。
  - `nvprof` 失败：不支持 RTX 4090 / Compute Capability 8.0+。
  - 继续采用既有 `nsys` 结论：`v_pml` 与 `p_pml` 合计约 73% GPU kernel time。
- `perf_3gpu --np 9 --gpus 0,1,2`：
  - 输出对比通过。
  - `WP computing time = 2.194983 s`，在外部 `xspecfem3D` 进程存在时仍能接近 1.5x 存档性能。
  - 判定：正确，但不如 `np=6`。
- `perf_3gpu --np 3 --gpus 0,1,2`：
  - `WP computing time = 3.284489 s`，运行时 GPU 上有外部 `xspecfem3D` 进程，性能不作为结论。
- `perf_3gpu --np 6 --gpus 0,1,2`：
  - 输出对比通过，9 个 `.dir` 文件全部通过 `rel_l2 <= 1e-5`。
  - `WP computing time = 1.549196 s`。
  - speedup = `3.491814 / 1.549196 = 2.254x`。
  - 已突破 `2.0x` 阈值。

### 输出、哈希或误差摘要
- 恢复后关键源码哈希：
  - `src/rem_fd.cu`: `9d9e6a2bfa08b3e5865f6ad3bf423d9915307f7edaac931782f3d1778c65e45c`
  - `src/single_solver.cu`: `46ac1d9820af6b6258aa9e639cdcf0c9f4333c29979460738c717920d2158925`
  - `include/inc3D/single_solver.h`: `063a685b26fa17501e80646fe3499dd67c4ff9373e7942b8a441ac5301dd1bf6`
  - `src/makefile.server`: `2789763712a9d10a32036249c4e66e1e7f6e3ba39aec20035b8f41117c74c392`
- `np=6` perf report：
  - `benchmarks/reports/perf_3gpu_rank6_3gpu_concurrent_shots_serial_20260603_033547/comparison.md`
  - 每炮 `Rel L2` 约 `2.306270e-06` 到 `2.454066e-06`，全部通过。
- 2.0x 存档：
  - `archives/speedups/2.0x_20260603_033651_rank6_3gpu_concurrent_shots`
  - manifest: `archives/speedups/2.0x_20260603_033651_rank6_3gpu_concurrent_shots/archive_manifest.json`

### 风险与下一步
- 本轮 `np=6` 是调度层优化，不是 CUDA kernel 源码优化；它利用 `cudaSetDevice(mytid % gpus_p_node)` 让每张 4090 同时承载 2 个 MPI rank，从而把 9 炮从 3 轮压到 2 轮。
- 运行前 `nvidia-smi` 记录有轻微外部 GPU context/利用率，因此建议在完全空卡时复测 `np=6`，若稳定仍低于 `1.745907 s`，则确认 2.0x 里程碑非常稳。
- 下一步应把 `run_benchmark.py` 或新增调度脚本扩展为可记录 `np/gpus/ranks_per_gpu` 的正式 benchmark 配置，并继续寻找 kernel 级优化，优先围绕 `v_pml` 与 `p_pml` 的真实访存。

## 2026-06-03 11:48 +08:00 - 纠正跨 MPI 配置的计时口径，2.0x 调度存档改为 provisional

### 操作目标
- 审计 `np=6` 调度实验的计时口径。
- 避免把 root rank 的局部 `WP computing time` 误当成整作业性能。
- 更新长期协作规则，要求调度层优化必须同时比较 `Gradient TIME all` 与 wall-clock。

### 修改文件
- `AGENTS.md`
  - 新增计时口径补充：只有 MPI rank 数、GPU 数、shot 分配方式一致时，才允许直接用 `WP computing time` 判断阈值。
  - 调度层优化必须同时报告 `WP computing time`、`Gradient TIME all`、`Elapsed (wall clock) time`。
- `tools/run_benchmark.py`
  - 新增 `perf_3gpu_rank6` case，默认 `np=6`、`gpus=0,1,2`，用于后续复现实验。
- `archives/speedups/2.0x_20260603_033651_rank6_3gpu_concurrent_shots/VALIDATION_NOTE.md`
  - 追加说明：该存档是 provisional scheduling experiment，不是已确认 2.0x CUDA 性能里程碑。
- `AGENT_LOG.md`
  - 追加本条日志。

### 执行命令摘要

```bash
python3 -m py_compile tools/run_benchmark.py
python3 tools/run_benchmark.py --case perf_3gpu_rank6 --tag formal_rank6_case_recheck --gpus 0,1,2
python3 tools/run_benchmark.py --case perf_3gpu --tag rank12_3gpu_concurrent_shots --np 12 --gpus 0,1,2
python3 tools/run_benchmark.py --case perf_3gpu_rank6 --tag formal_rank6_case_solo_recheck --gpus 0,1,2

grep -E 'Gradient TIME all|Elapsed \(wall clock\)|WP computing time' \
  benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602/run.log \
  archives/speedups/1.5x_20260602_110535_fastmath_pcore_halo7/runs/perf_3gpu_opt_fastmath_gpus123_20260602_110535/run.log \
  benchmarks/runs/perf_3gpu_rank6_3gpu_concurrent_shots_serial_20260603_033547/run.log \
  benchmarks/runs/perf_3gpu_rank6_formal_rank6_case_solo_recheck_20260603_033949/run.log
```

### 测试结果
- 服务器端 `python3 -m py_compile tools/run_benchmark.py` 通过。
- 本地 `python -m py_compile E:\cuda3D\tools\run_benchmark.py` 失败：
  - 原因：Windows 拒绝写入 `tools/__pycache__/*.pyc` 临时缓存文件。
  - 影响：不影响服务器运行，服务器 py_compile 已通过。
- `perf_3gpu_rank6 formal_rank6_case_recheck`：
  - 输出对比通过。
  - 与 `np=12` 并行运行，互相抢 GPU，性能不作为正式结论。
  - `WP computing time = 3.205986 s`。
- `perf_3gpu rank12_3gpu_concurrent_shots`：
  - 输出对比通过。
  - 与 `rank6` 并行运行，性能不作为正式结论。
  - `WP computing time = 3.537638 s`。
- `perf_3gpu_rank6 formal_rank6_case_solo_recheck`：
  - 输出对比通过。
  - 运行前仍有外部 `xspecfem3D` 进程。
  - `WP computing time = 2.435341 s`。
  - `Gradient TIME all = 2.984629 s`。
  - elapsed wall clock `0:07.63`。

### 计时口径审计
- 冻结 baseline：
  - `Gradient TIME all = 4.135657 s`
  - `WP computing time = 3.491814 s`
  - elapsed wall clock `0:08.39`
- 已确认 1.5x 存档：
  - `Gradient TIME all = 2.530737 s`
  - `WP computing time = 2.074612 s`
  - elapsed wall clock `0:06.85`
- `np=6` 最好一轮：
  - `Gradient TIME all = 2.559642 s`
  - `WP computing time = 1.549196 s`
  - elapsed wall clock `0:07.68`
- `np=6` solo 复测：
  - `Gradient TIME all = 2.984629 s`
  - `WP computing time = 2.435341 s`
  - elapsed wall clock `0:07.63`

### 结论
- `np=6` 调度方案数值正确，但 whole-job 指标没有确认 2.0x。
- 之前按 `WP computing time` 创建的 `2.0x_20260603_033651_rank6_3gpu_concurrent_shots` 存档保留，但标为 provisional，不作为正式 2.0x 性能里程碑。
- 后续真正冲 2.0x/2.5x 必须回到 CUDA kernel 级优化，或用 whole-job 指标确认调度收益。

## 2026-06-03 11:58 +08:00 - block-level PML core skip 实验，正确但不采用

### 操作目标
- 在 `cuda_fd3d_v_pml_ns` 与 `cuda_fd3d_p_pml_ns` 中尝试 block-level core skip。
- 目标是让完全位于非 PML 核心区的 CUDA block 整块提前返回，减少每线程边界判断开销。

### 修改文件
- `src/single_solver.cu`
  - 临时新增 `block1/block2/block3` 整块核心区判断。
  - 实验完成后已从 1.5x 可信存档恢复。
- `AGENT_LOG.md`
  - 追加本条日志。

### 执行命令摘要

```bash
scp src/single_solver.cu shengwz@162.105.91.239:/data/shengwz/swz/cuda3D/src/single_solver.cu
cd /data/shengwz/swz/cuda3D/src
source /opt/intel/oneapi/setvars.sh --force >/tmp/cuda3d_setvars.log 2>&1
make -B -f makefile.server test >/tmp/cuda3d_build_block_core_skip.log 2>&1

python3 tools/run_benchmark.py --case correctness --tag opt_block_core_skip
python3 tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/correctness_baseline_20260601_165054/outputs \
  --candidate benchmarks/runs/correctness_opt_block_core_skip_20260603_034611/outputs \
  --out benchmarks/reports/correctness_opt_block_core_skip_20260603_034611

python3 tools/run_benchmark.py --case profile_1gpu --tag opt_block_core_skip_g0 --gpus 0
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_block_core_skip_gpus012 --gpus 0,1,2
python3 tools/run_benchmark.py --case perf_3gpu --tag opt_block_core_skip_gpus012_solo --gpus 0,1,2

cp archives/speedups/1.5x_20260602_110535_fastmath_pcore_halo7/source_snapshot/src/single_solver.cu src/single_solver.cu
make -B -f makefile.server test >/tmp/cuda3d_build_revert_block_skip.log 2>&1
```

### 测试结果
- correctness：
  - 通过。
  - `Rel L2` 约 `2.980189e-07` 到 `4.098143e-07`。
- `profile_1gpu opt_block_core_skip_g0`：
  - `WP computing time = 0.547291 s`。
  - 与 `perf_3gpu` 并行运行，受 GPU0 互相干扰，只作筛查。
- `perf_3gpu opt_block_core_skip_gpus012`：
  - 输出对比通过。
  - `WP computing time = 2.132702 s`，与 profile 并行运行，作筛查。
- `perf_3gpu opt_block_core_skip_gpus012_solo`：
  - 输出对比通过。
  - `Gradient TIME all = 2.580772 s`
  - `WP computing time = 2.144295 s`
  - elapsed wall clock `0:06.74`

### 结论
- block-level core skip 正确但没有超过当前 1.5x 可信存档：
  - 当前最佳 `WP computing time = 2.074612 s`
  - 本实验 solo `WP computing time = 2.144295 s`
- 判定：不采用，已撤回 `src/single_solver.cu`。
- 恢复后本地/远端 `src/single_solver.cu` hash 均为：
  - `46ac1d9820af6b6258aa9e639cdcf0c9f4333c29979460738c717920d2158925`

### 风险与下一步
- 本次发现一次同步竞态：远端回退与本地 `scp` 拉取并行执行时，本地可能抢先拉到旧实验文件。
- 后续涉及恢复源码时必须串行执行：先远端恢复并确认 hash，再拉回本地。
- 下一步继续寻找能真实减少 `v_pml`/`p_pml` 访存的 kernel 级改写；简单提前返回不是足够大的收益点。
## 2026-06-04 CST - PML/profile 驱动优化与 p_core core-box 保留

- 操作目标：
  - 继续冲击 `2.0x` 速度阈值。
  - 使用 profiler 定位当前主要瓶颈，优先尝试结构性 CUDA 优化。
  - 保留 correctness 通过且 perf 有收益的改动，撤回退化实验。
- 修改文件：
  - `src/rem_fd.cu`
    - 保留：PML 系数 device 指针初始化为 `NULL`，不再为已搬到 constant memory 的 12 个 PML 系数数组做 `cudaMalloc/cudaMemcpy`。
    - 保留：`p_core` grid 改为只覆盖 core box，尺寸为：
      - `core_nz = nbz - 2 * (nbd + CorePmlMargin)`
      - `core_nx = nbx - 2 * (nbd + CorePmlMargin)`
      - `core_ny = nby - 2 * (nbd + CorePmlMargin)`
    - 撤回：PML compact shell 1D 映射 launch。
    - 撤回：`p_pml` 六 slab launch。
    - 撤回：active kernel `__restrict__` 标注。
    - 撤回：`cuda_fd3d_p_pml_shared_ns` active call。
  - `src/single_solver.cu`
    - 保留：PML 常量内存系数读取。
    - 保留：`cuda_fd3d_p_core_ns` 内部增加 core offset，使 kernel 只处理 core box launch 对应区域。
    - 保留：`cuda_fd3d_p_pml_shared_ns` 的 PML 系数读取改为 constant memory；该 kernel 当前不作为 active path。
    - 撤回：PML compact shell 设备端映射函数。
    - 撤回：active kernel `__restrict__` 标注。
  - `include/inc3D/single_solver.h`
    - 同步 active kernel 声明，最终不保留 `__restrict__` 和 `p_pml` slab 签名。
  - 远程 `src/single_solver.h`
    - 同步 `include/inc3D/single_solver.h`，避免远程 `src` 同名头文件遮蔽 include 目录声明。
- 执行命令摘要：
  - 多次同步：
    - `scp E:\cuda3D\src\single_solver.cu shengwz@162.105.91.239:/data/shengwz/swz/cuda3D/src/single_solver.cu`
    - `scp E:\cuda3D\src\rem_fd.cu shengwz@162.105.91.239:/data/shengwz/swz/cuda3D/src/rem_fd.cu`
    - `scp E:\cuda3D\include\inc3D\single_solver.h .../include/inc3D/single_solver.h`
    - `scp E:\cuda3D\include\inc3D\single_solver.h .../src/single_solver.h`
  - 多次编译：
    - `cd /data/shengwz/swz/cuda3D/src && source /opt/intel/oneapi/setvars.sh --force && make -B -f makefile.server test`
  - profiler：
    - `ncu` 因 `ERR_NVGPUCTRPERM` 无权限访问 GPU performance counters，改用 `nsys`。
    - `nsys profile -t cuda ...`
    - `nsys stats --report cuda_gpu_kern_sum --format csv ...`
  - benchmark：
    - `python3 tools/run_benchmark.py --case correctness --tag opt_pml_const_no_coeff_alloc`
    - `python3 tools/run_benchmark.py --case perf_3gpu --tag opt_pml_const_no_coeff_alloc_gpus012 --gpus 0,1,2`
    - `python3 tools/run_benchmark.py --case smoke_1gpu --tag opt_pml_shell_compact`
    - `python3 tools/run_benchmark.py --case correctness --tag opt_pml_shell_compact`
    - `python3 tools/run_benchmark.py --case perf_3gpu --tag opt_pml_shell_compact_gpus012 --gpus 0,1,2`
    - `python3 tools/run_benchmark.py --case correctness --tag opt_pml_shared_const`
    - `python3 tools/run_benchmark.py --case perf_3gpu --tag opt_pml_shared_const_gpus012 --gpus 0,1,2`
    - `python3 tools/run_benchmark.py --case correctness --tag opt_restrict_active_kernels`
    - `python3 tools/run_benchmark.py --case perf_3gpu --tag opt_restrict_active_kernels_gpus012 --gpus 0,1,2`
    - `python3 tools/run_benchmark.py --case correctness --tag opt_dlcm_ca`
    - `python3 tools/run_benchmark.py --case perf_3gpu --tag opt_dlcm_ca_gpus012 --gpus 0,1,2`
    - `python3 tools/run_benchmark.py --case correctness --tag opt_dlcm_cg`
    - `python3 tools/run_benchmark.py --case perf_3gpu --tag opt_dlcm_cg_gpus012 --gpus 0,1,2`
    - `python3 tools/run_benchmark.py --case correctness --tag opt_pcore_box`
    - `python3 tools/run_benchmark.py --case perf_3gpu --tag opt_pcore_box_gpus012 --gpus 0,1,2`
    - `python3 tools/run_benchmark.py --case correctness --tag opt_ppml_slabs`
    - `python3 tools/run_benchmark.py --case perf_3gpu --tag opt_ppml_slabs_gpus012 --gpus 0,1,2`
    - `python3 tools/run_benchmark.py --case perf_3gpu --tag opt_pcore_box_confirm_gpus012 --gpus 0,1,2`
- 测试结果：
  - `nsys` 初始 profile，profile_1gpu：
    - `cuda_fd3d_v_pml_ns`: 36.9%，约 192.5 us/step。
    - `cuda_fd3d_p_pml_ns`: 36.5%，约 190.3 us/step。
    - `cuda_fd3d_p_core_ns`: 25.9%，约 135.4 us/step。
  - `opt_pml_const_no_coeff_alloc_gpus012_20260603_173331`
    - correctness/perf 输出均通过。
    - `WP computing time = 1.960844s`
    - `Gradient TIME all = 2.394609s`
    - elapsed `0:06.61`
  - `opt_pml_shell_compact_gpus012_20260603_174239`
    - correctness/perf 输出通过，但性能退化。
    - `WP computing time = 2.060386s`
    - 结论：撤回。线性壳层映射的整数除法/取模开销超过收益。
  - `opt_pml_shared_const_gpus012_20260603_174736`
    - correctness/perf 输出通过，但性能明显退化。
    - `WP computing time = 2.922701s`
    - 结论：撤回 active call。shared tile 额外加载和同步成本过高。
  - `opt_restrict_active_kernels_gpus012_20260603_175206`
    - correctness/perf 输出通过，但性能退化。
    - `WP computing time = 2.101572s`
    - 结论：撤回 active kernel `__restrict__` 标注。
  - `opt_dlcm_ca_gpus012_20260603_175535`
    - correctness/perf 输出通过，但慢于当前最好。
    - `WP computing time = 2.033722s`
    - 结论：不固化 `-Xptxas -dlcm=ca`。
  - `opt_dlcm_cg_gpus012_20260603_175723`
    - correctness/perf 输出通过，但明显退化。
    - `WP computing time = 2.564389s`
    - 结论：不固化 `-Xptxas -dlcm=cg`。
  - `opt_pcore_box_gpus012_20260603_180157`
    - correctness/perf 输出均通过。
    - `WP computing time = 1.920610s`
    - `Gradient TIME all = 2.400399s`
    - elapsed `0:06.51`
    - speedup vs baseline WP `3.491814 / 1.920610 = 1.818x`
  - `opt_ppml_slabs_gpus012_20260603_180636`
    - correctness/perf 输出通过，但性能退化。
    - `WP computing time = 2.059746s`
    - 结论：撤回。多 kernel launch 成本超过少跑空线程收益。
  - 最终确认 run：`perf_3gpu_opt_pcore_box_confirm_gpus012_20260603_180923`
    - correctness/perf 输出对比通过。
    - `WP computing time = 1.921693s`
    - `Gradient TIME all = 2.355070s`
    - elapsed `0:06.49`
    - speedup vs baseline WP `3.491814 / 1.921693 = 1.817x`
- 输出/误差摘要：
  - `correctness_opt_pcore_box_20260603_180111` 通过。
  - `perf_3gpu_opt_pcore_box_confirm_gpus012_20260603_180923` 通过。
  - perf 对比最大 rel L2 约 `2.45e-6`，低于 `1e-5` 门槛。
  - 所有保留版本未发现 NaN/Inf、缺失文件或额外输出文件。
- 风险与下一步：
  - 当前仍未达到正式 `2.0x` 阈值；不创建新的 `2.0x` 存档。
  - 当前最佳是 `p_core core-box + PML constant memory + no coeff alloc`，约 `1.817x`。
  - `p_core` 已从约 `135.4 us/step` 降到约 `121.1 us/step`。
  - 下一步应继续针对 `v_pml_ns` 和 `p_pml_ns`，但避免会显著增加 kernel launch 或整数映射开销的方案。
  - 可探索方向：
    - 单 kernel 内轻量 block-level 区域剪枝的更低开销版本。
    - PML kernel 中按 component 拆分或减少不必要分支，但必须控制 launch 数。
    - 更细地分析 `v_pml/p_pml` 访存模式，考虑只对 z/x/y 中特定方向做局部共享缓存，而不是三分量全 shared。
## 2026-06-06 CST - 迁移到 RTX 5090 新服务器并建立基础环境

- 操作目标：
  - 将本地项目迁移到新的稳定测试服务器。
  - 在 `/work/wenzhe/cuda3D` 建立可编译、可运行的 CUDA 13 / RTX 5090 环境。
  - 创建项目专用 conda 环境，避免修改系统全局 Python。
- 修改文件：
  - 新增 `tools/remote_exec.py`
    - 使用 `paramiko` 通过 SSH 执行远程命令。
    - 密码只通过环境变量传入，不写入脚本。
  - 新增 `tools/remote_upload.py`
    - 使用 SFTP 上传项目目录。
    - 跳过 `.git` 和 `__pycache__`。
  - 新增 `env_5090.sh`
    - 设置 `CUDA_HOME=/usr/local/cuda-13.0`。
    - 设置 `MPI_HOME=/opt/intel/oneapi/mpi/latest`。
    - source oneAPI `setvars.sh`。
    - activate conda 环境 `cuda3d`。
  - 新增 `src/makefile.rtx5090`
    - 使用 CUDA 13.0。
    - 默认 `NV_ARCH=sm_120`。
    - 使用 Intel MPI。
  - 修改 `tools/run_benchmark.py`
    - `tool_versions` 改为读取 `CUDA_HOME` 和 `MPI_HOME`。
    - `mpirun` 路径改为使用 `MPI_HOME`。
  - 修改 `AGENTS.md`
    - 增加 RTX 5090 新服务器环境、编译命令、smoke 初始结果和注意事项。
- 远程服务器信息：
  - 连接：`ssh -p 25804 -X zz@162.105.95.56`
  - 主机名：`cotopaxi`
  - 系统：Ubuntu 24.04.4 LTS
  - GPU：1 张 NVIDIA GeForce RTX 5090，32607 MiB
  - Driver：595.71.05
  - CUDA：`/usr/local/cuda-13.0`
  - CUDA 架构支持：包含 `compute_120`、`compute_121`
  - MPI：Intel MPI 2021.18
  - 工作目录：`/work/wenzhe/cuda3D`
- 执行命令摘要：
  - 远程检查：
    - `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader`
    - `/usr/local/cuda/bin/nvcc --version`
    - `source /opt/intel/oneapi/setvars.sh --force`
    - `mpirun --version`
  - 权限修复：
    - `/work/wenzhe` 初始为 `root:root`，`zz` 无法写入。
    - 使用 `sudo -S chown zz:descfly /work/wenzhe` 只修改该明确目录的所有者。
    - 写入测试通过。
  - 上传：
    - 通过 `tools/remote_upload.py` 上传本地项目到 `/work/wenzhe/cuda3D`。
    - 上传 71 个文件，约 1.46GB。
  - Conda：
    - 安装 Miniforge 到 `/work/wenzhe/miniforge3`。
    - 创建环境：`mamba create -y -n cuda3d python=3.11 numpy`
  - 编译：
    - `cd /work/wenzhe/cuda3D`
    - `source ./env_5090.sh`
    - `cd src`
    - `make -B -f makefile.rtx5090 test`
- 测试结果：
  - 编译通过：
    - `nvcc -O3 -arch=sm_120 --use_fast_math`
    - 输出：`FINISHED COMPILING RTX5090`
  - smoke case 生成：
    - `python tools/create_smoke_case.py`
  - smoke 运行：
    - `python tools/run_benchmark.py --case smoke_1gpu --tag rtx5090_initial`
    - run：`benchmarks/runs/smoke_1gpu_rtx5090_initial_20260606_000133`
    - returncode：0
    - outputs：3
    - `ALL DONE`
    - `WP computing time = 0.002072s`
    - `Gradient TIME all = 0.003854s`
    - elapsed `0:02.38`
    - binary SHA256：`7c9c761fd9da24c24ba78d135f2e63ab996f41a7dc4ad66704a870556b3ad640`
- 风险与下一步：
  - 新平台只有 1 张 RTX 5090，旧的 `perf_3gpu` 速度阈值不能直接沿用。
  - 需要为 RTX 5090 平台重新定义并冻结 `correctness` 和 `perf_1gpu` baseline。
  - `run_benchmark.py` 已修正 CUDA/MPI 版本采集，但需在下一轮 smoke/correctness/perf 中确认 manifest 不再引用 CUDA 12.2。
  - 密码未写入项目文件；后续建议配置 SSH key，减少密码自动化。

## 2026-06-06 CST - RTX 5090 run_benchmark 环境采集复测

- 操作目标：
  - 验证 `tools/run_benchmark.py` 使用 `CUDA_HOME/MPI_HOME` 后能正确记录新服务器工具版本。
- 修改文件：
  - 无新增源码修改；复测前已同步 `tools/run_benchmark.py`、`env_5090.sh`、`AGENTS.md` 和 `AGENT_LOG.md`。
- 执行命令：
  - `cd /work/wenzhe/cuda3D`
  - `source ./env_5090.sh`
  - `python tools/run_benchmark.py --case smoke_1gpu --tag rtx5090_envcheck`
- 测试结果：
  - run：`benchmarks/runs/smoke_1gpu_rtx5090_envcheck_20260606_000541`
  - returncode：0
  - outputs：3
  - `ALL DONE`
  - `WP computing time = 0.002216s`
  - `Gradient TIME all = 0.002968s`
  - elapsed `0:02.35`
  - manifest 已正确记录：
    - CUDA 13.0，`V13.0.48`
    - Intel MPI 2021.18
    - GCC 13.3.0
- 风险与下一步：
  - RTX 5090 基础环境已可编译、可运行 smoke。
  - 下一步应创建并冻结 RTX 5090 平台专用 `correctness` 与 `perf_1gpu` baseline。

## 2026-06-06 CST - 新增 Kimi-K2.6 CUDA 工程交接 soul.md

- 操作目标：
  - 为后续 Claude Code + Kimi-K2.6 接手 CUDA 核心重写准备长期交接文档。
  - 明确 Kimi 在本项目中的工程角色、正确性纪律、baseline 纪律、安全规则、已知优化历史和下一步技术方向。
- 修改文件：
  - `soul.md`
  - `AGENT_LOG.md`
- 执行命令：
  - `Get-Content -Encoding UTF8 AGENT_LOG.md -Tail 80`
  - `Get-Content -Encoding UTF8 soul.md`
  - 后续同步到远程：`python3 tools/remote_put.py ... soul.md ... AGENT_LOG.md ...`
- 测试结果：
  - 文档变更，不涉及源码编译或 benchmark 运行。
  - `soul.md` 已记录 RTX 5090 当前环境、smoke 初始结果、旧 4090 优化历史、失败实验清单和新平台 baseline 要求。
- 输出、哈希或误差摘要：
  - 未生成数值输出。
  - 未记录任何服务器密码到项目文件、脚本、日志或文档。
- 风险与下一步：
  - 后续 Kimi 接手后应先冻结 RTX 5090 `correctness` 和 `perf_1gpu` baseline，再开始 CUDA time-stepping 核心重写。

## 2026-06-06 CST - 审计 Kimi 远程记录并新增 CUDA 重写执行指令

- 操作目标：
  - 查看 `/work/wenzhe/cuda3D` 中 Kimi/Claude 留下的工作记录、benchmark run 和 baseline。
  - 将下一阶段 CUDA 底层结构化重写意见整理为项目文档。
  - 明确 Kimi 后续只作为高级实现工程师执行代码编写、测试和反馈报告，不单独决定总体架构路线。
- 修改文件：
  - 新增 `KIMI_CUDA_REWRITE_DIRECTIVE.md`
  - 修改 `AGENT_LOG.md`
- 执行命令：
  - 远程查看项目记录文件：`find . -maxdepth 3 -type f | grep -Ei "kimi|report|log|note|\\.md$"`
  - 远程读取 `CLAUDE.md`、`progress.md` 和 `AGENT_LOG.md` 尾部。
  - 远程列出 `benchmarks/runs`、`benchmarks/reports`、`benchmarks/baselines/current_runnable`。
  - 远程抽取关键 run 的 `WP computing time`、`Gradient TIME all` 和 `ALL DONE`。
- 测试结果：
  - 文档和审计操作，不涉及源码编译或新的 benchmark 运行。
  - 远程已发现 `CLAUDE.md`、`progress.md`、`AGENT_LOG.md`。
  - RTX 5090 `perf_1gpu` baseline 已存在：`WP computing time = 0.545397s`，`Gradient TIME all = 0.576524s`。
  - 已审计的 Kimi 实验包括 CUDA Graph、block size、stream overlap、device memory pool、`v_pml+p_pml` 简单融合、`p_core` LDG、`p_core` 2D shared tile、`v_pml/p_pml` 寄存器整理、`v_pml` y-tile。
- 输出、哈希或误差摘要：
  - CUDA Graph：`WP=0.546443s`，无加速。
  - block size `32x8x2`：`WP=0.577354s`，退化。
  - stream overlap：`WP=0.546808s`，无加速。
  - memory pool：`WP=0.545344s`，单炮无实际收益。
  - `p_core` LDG：`WP=0.546795s`，无加速。
  - `p_core` 2D tile：`WP=0.898886s`，严重退化。
  - `vp_regopt`：`WP=0.546916s`，无加速。
  - `v_pml_ytile`：一次 `WP=0.665555s` 退化，复测 `WP=0.548109s` 接近 baseline 但无实质收益。
- 风险与下一步：
  - Kimi 后续不得继续随机微优化。
  - 新文档要求 Kimi 先写 `docs/cuda_core_dependency_map.md`，再设计 `core_opt` 与 `pml_opt` 两条最小 prototype。
  - 每次实验后必须新增 `feedback/kimi_report_*.md` 并追加 `AGENT_LOG.md`。

## 2026-06-06 CST - 审阅 Kimi core_y2 反馈报告

- 操作目标：
  - 阅读远程 `feedback/kimi_report_20260606_172800_core_y2.md`。
  - 检查对应 `AGENT_LOG.md` 尾部、run 列表、`docs/cuda_core_dependency_map.md`、`docs/core_pml_prototype_design.md` 和当前 `p_core` kernel/launch 代码。
  - 对 Kimi 的失败归因和下一步方向提出审阅意见。
- 修改文件：
  - 新增 `KIMI_REPORT_REVIEW_20260606_CORE_Y2.md`
  - 修改 `AGENT_LOG.md`
- 执行命令：
  - 远程读取 `feedback/kimi_report_20260606_172800_core_y2.md`。
  - 远程读取 `AGENT_LOG.md` 尾部。
  - 远程列出 `feedback/` 和 `benchmarks/runs/` 最新条目。
  - 远程读取 `docs/cuda_core_dependency_map.md`、`docs/core_pml_prototype_design.md`。
  - 远程查看 `src/rem_fd.cu` 中 `dimg_p` 设置和 `src/single_solver.cu` 中 `cuda_fd3d_p_core_ns` 映射。
- 测试结果：
  - 只读审阅和文档新增，不涉及源码编译或新 benchmark。
  - `core_y2` correctness 失败，报告记录 rel L2 约 `1.05e-01`，已 revert。
  - baseline verify 已存在：`correctness_baseline_verify_20260606_172523`。
- 输出、哈希或误差摘要：
  - 审阅结论：`revert` 决策正确，但 Kimi 的失败归因不够干净。
  - 主要问题：`p_core_ns` 只占约 20.9% GPU kernel time，不应作为当前主战场；one-thread-two-y 与现有 z-shared tile 模型不匹配；minimal halved-grid 版本漏 odd y 是预期结果，不能证明 halved grid 有未知缺陷。
- 风险与下一步：
  - 建议停止 `p_core` 微优化。
  - 下一步要求 Kimi 先写 PML layout design 和 one-step PML debug plan，再实现 `pml_opt` prototype。

## 2026-06-06 18:05 CST — PML Layout Design Report

- **Status**: design-only, no code change
- **File**: `feedback/kimi_report_20260606_180500_pml_layout_design.md`
- **Key Finding**: Pure face-major compact buffer for PML is **infeasible** under 8th-order stencil. Each PML point needs 1D neighbor lines that cross core boundaries (4-point coupling depth). Any compact buffer would need halo covering full-domain span, negating compactness.
- **Answers to review questions**:
  1. `p_pml_ns` reads 24 floats/point (vz×8 + vx×8 + vy×8), all written by `v_pml_ns`.
  2. Strict face buffer: **no** (halo problem). Directional slice buffer: possible but storage reduction minimal.
  3. 6-face disjoint partition avoids duplicate/miss, but stencil halo needs cross-face data access.
  4. 6-face kernel: no extra integer mapping. Compact 1D: rejected per review.
  5. 6-face kernel: +5 launches/step. 3-group: +2 launches/step.
  6. Kernel split alone does **not** reduce global load/store volume (557 MB/step unchanged). Data transpose cost exceeds L2 miss savings.
  7. Proposed `debug_1step` case + PML shell dump + per-point comparison script.
- **Recommended next step**: Build one-step PML debug harness first, then experiment 3-group kernel (Z/X/Y face groups).

## 2026-06-06 CST - Codex PML tile-list prototype and sweep

- 操作目标：
  - 根据 GPT 5.5 Pro 的建议，直接在 `/work/wenzhe/cuda3D` 上实现 PML tile-list 单 launch prototype。
  - 保持现有数学公式、full-domain `vx/vy/vz` 数组和 CPML memory layout 不变，只减少完全位于 inactive core 内的 PML CTA 调度。
  - 测试 `32x4x2` 以及多个 tile block shape，判断该结构方向是否有实质收益。
- 修改文件：
  - `include/inc3D/cu_common.h`
    - 新增 `PmlTileBlockSize1/2/3`，默认 `32x4x2`，允许通过 `NVFLAGS -D` 覆盖。
  - `include/inc3D/single_solver.h`
    - 新增 `PmlTile` 结构体。
    - 新增 `cuda_fd3d_v_pml_tile_ns`、`cuda_fd3d_p_pml_tile_ns` 声明。
  - `src/single_solver.cu`
    - 新增 V/P PML tile-list kernel。
    - 公式与现有 `cuda_fd3d_v_pml_ns`、`cuda_fd3d_p_pml_ns` 保持一致，只改变 `gtid1/2/3` 来源。
  - `src/rem_fd.cu`
    - 新增 host 端 tile-list 构造。
    - 新增编译宏：
      - `CUDA3D_PML_TILE_LIST`：同时启用 V 和 P tile-list。
      - `CUDA3D_PML_TILE_LIST_V`：只启用 V tile-list。
      - `CUDA3D_PML_TILE_LIST_P`：只启用 P tile-list。
    - 默认构建仍走原始 active path。
  - `feedback/codex_report_20260606_223000_pml_tile_list.md`
- 执行命令摘要：
  - 默认路径编译：
    - `cd /work/wenzhe/cuda3D && source ./env_5090.sh && cd src && make -B -f makefile.rtx5090 test`
  - tile-list 编译：
    - `make -B -f makefile.rtx5090 NVFLAGS="-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_TILE_LIST -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2" test`
  - 正确性测试：
    - `python tools/run_benchmark.py --case smoke_1gpu --tag pml_tile_list_32x4x2`
    - `python tools/run_benchmark.py --case correctness --tag pml_tile_list_32x4x2`
    - `python tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/correctness_rtx5090_baseline_20260606_002850/outputs --candidate ... --out benchmarks/reports/correctness_pml_tile_list_32x4x2_latest`
  - final 正确性与性能：
    - `python tools/run_benchmark.py --case correctness --tag pml_tile_32x4x2_final`
    - `python tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/correctness_rtx5090_baseline_20260606_002850/outputs --candidate benchmarks/runs/correctness_pml_tile_32x4x2_final_20260606_222838/outputs --out benchmarks/reports/correctness_pml_tile_32x4x2_final`
    - `python tools/run_benchmark.py --case perf_1gpu --tag pml_tile_32x4x2_final`
    - `python tools/compare_outputs.py --baseline benchmarks/baselines/current_runnable/perf_1gpu_rtx5090_baseline_20260606_002902/outputs --candidate benchmarks/runs/perf_1gpu_pml_tile_32x4x2_final_20260606_222841/outputs --out benchmarks/reports/perf_1gpu_pml_tile_32x4x2_final`
  - sweep variants：
    - V+P tile-list：`32x8x1`、`16x8x2`、`16x4x4`、`32x4x2`、`32x4x1`、`32x2x2`、`32x2x4`、`24x4x2`
    - P-only：`32x4x2`
    - V-only：`32x4x2`
- 测试结果：
  - 默认路径编译通过。
  - `CUDA3D_PML_TILE_LIST` 编译通过。
  - `smoke_1gpu_pml_tile_list_32x4x2_20260606_221929`：
    - `ALL DONE`
    - `WP computing time = 0.002246s`
    - `Gradient TIME all = 0.003046s`
  - `correctness_pml_tile_list_32x4x2_20260606_221932`：
    - 输出 6 个文件。
    - 与 baseline 对比通过，所有文件 `rel L2 = 0`。
  - final correctness：
    - run：`benchmarks/runs/correctness_pml_tile_32x4x2_final_20260606_222838`
    - report：`benchmarks/reports/correctness_pml_tile_32x4x2_final`
    - 输出 6 个文件，与 baseline 对比全部 `rel L2 = 0`。
  - final perf：
    - run：`benchmarks/runs/perf_1gpu_pml_tile_32x4x2_final_20260606_222841`
    - report：`benchmarks/reports/perf_1gpu_pml_tile_32x4x2_final`
    - `v_tiles = 23100`
    - `p_tiles = 22188`
    - `WP computing time = 0.539543s`
    - `Gradient TIME all = 0.569638s`
    - 与 perf baseline 输出对比通过，`rel L2 = 0`。
- 输出、哈希或误差摘要：
  - RTX 5090 perf baseline：`WP = 0.545397s`，`Gradient = 0.576524s`。
  - final candidate：`WP = 0.539543s`，`Gradient = 0.569638s`。
  - speedup by WP：`0.545397 / 0.539543 = 1.0109x`。
  - speedup by Gradient：`0.576524 / 0.569638 = 1.0121x`。
  - sweep 最好 run：`perf_1gpu_pml_tile_32x4x2_20260606_222244`
    - `WP = 0.538781s`
    - `Gradient = 0.568863s`
    - speedup by WP：`1.0123x`。
  - sweep 结果：
    - `32x8x1`：`WP = 0.542812s`
    - `16x8x2`：`WP = 0.578214s`
    - `16x4x4`：`WP = 0.567895s`
    - `32x4x1`：`WP = 0.545382s`
    - `32x2x2`：`WP = 0.552509s`
    - `32x2x4`：`WP = 0.546216s`
    - `24x4x2`：`WP = 0.558097s`
    - P-only `32x4x2`：`WP = 0.547992s`
    - V-only `32x4x2`：`WP = 0.540711s`
- 决策：
  - 保留源码中的 tile-list experimental path，但不设为默认 active path。
  - 当前远程二进制已按最佳 `32x4x2` tile-list 宏重新编译，便于后续继续测试。
  - 不创建 speedup archive；当前收益约 `1.01x`，远低于 `1.5x` 里程碑。
- 风险与下一步：
  - tile-list 只减少 CTA 调度，并没有减少主瓶颈 `p1 -> vx/vy/vz -> p0` 的 global memory round trip，因此收益上限很低。
  - P-only tile-list 退化，说明少发 CTA 本身不足以带来显著收益。
  - V-only tile-list 小幅有效，说明 `v_pml` 调度仍有少量可挖空间，但不是 3x/5x 路径。
  - 下一步应建立 one-step PML debug harness，然后尝试方向性 PML 数据流改写，目标是减少或局部消除 `vx/vy/vz` 的 global write/read 往返。

## 2026-06-06 23:01 +08:00 - PML debug dump 与 velocity recompute 实验矩阵

- 操作目标：
  - 按结构化 CUDA 核心改写路线，验证 PML 区 `p1 -> velocity -> p0` 数据流是否可以减少全局内存往返。
  - 建立 one-step PML debug dump 工具，方便确认 kernel 内部状态差异。
  - 测试 `RECOMPUTE_X/Y/Z/XYZ` 与 tile-list 组合，筛选可继续推进的方向。
- 修改文件：
  - `include/inc3D/single_solver.h`
    - 扩展 `cuda_fd3d_p_pml_ns` 与 `cuda_fd3d_p_pml_tile_ns` 参数，传入 `d_memory_dz/dx/dy`。
  - `src/single_solver.cu`
    - 新增 `CUDA3D_PML_RECOMPUTE_Z`、`CUDA3D_PML_RECOMPUTE_X`、`CUDA3D_PML_RECOMPUTE_Y` 实验路径。
    - 新增 `recompute_vz_from_p1_mem`、`recompute_vx_from_p1_mem`、`recompute_vy_from_p1_mem`。
    - 对 `v_pml` 与 `v_pml_tile` 加宏保护，启用重算方向时跳过对应 `vx/vy/vz` global store。
    - 对 `p_pml` 与 `p_pml_tile` 加宏保护，启用重算方向时直接从 `p1 + velocity memory` 重算对应 velocity。
  - `src/rem_fd.cu`
    - 新增 `CUDA3D_PML_DEBUG_DUMP` dump helper。
    - `p_pml` launch 传入 `d_memory_dz/dx/dy`。
  - `tools/run_benchmark.py`
    - 支持将 `CUDA3D_PML_DUMP_DIR`、`CUDA3D_PML_DUMP_STEP` 传入 MPI 环境。
    - 确认包含 `perf_1gpu` 与 `perf_1gpu_6shots` case。
  - `tools/compare_debug_dumps.py`
    - 新增 PML dump 对比工具。
  - `feedback/codex_report_20260606_230100_pml_recompute_z_tile.md`
    - 记录本轮实验矩阵、结果与下一步建议。
- 执行命令摘要：
  - 默认路径编译：
    - `cd /work/wenzhe/cuda3D && source ./env_5090.sh && cd src && make -B -f makefile.rtx5090 test`
  - debug dump 编译与手动 dump：
    - `make -B -f makefile.rtx5090 NVFLAGS="-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_DEBUG_DUMP" test`
    - 使用 MPI `-genv CUDA3D_PML_DUMP_DIR ... -genv CUDA3D_PML_DUMP_STEP ...` 运行 smoke。
  - recompute 变体编译：
    - `-DCUDA3D_PML_RECOMPUTE_X`
    - `-DCUDA3D_PML_RECOMPUTE_Y`
    - `-DCUDA3D_PML_RECOMPUTE_Z`
    - `-DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_RECOMPUTE_X -DCUDA3D_PML_RECOMPUTE_Y`
    - `-DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST_V -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2`
    - `-DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2`
  - 测试命令：
    - `python3 tools/run_benchmark.py --case smoke_1gpu --tag <variant>`
    - `python3 tools/run_benchmark.py --case correctness --tag <variant>`
    - `python3 tools/run_benchmark.py --case perf_1gpu --tag <variant>`
    - `python3 tools/run_benchmark.py --case perf_1gpu_6shots --tag <variant>`
    - `python3 tools/compare_outputs.py --baseline ... --candidate ... --out ...`
  - 环境小结：
    - 远端裸 shell 中 `python` 不存在，改用 `python3` 后正常。
- 测试结果：
  - 默认路径编译通过。
  - debug dump：
    - tile-list vs baseline `it=0`：所有 dump 数组通过。
    - recompute-z vs baseline `it=1`：`p0/p1/vx/vy` 和 memory 数组通过；`vz` dump 不一致为预期，因为该路径刻意不再写回 `vz`。
  - frozen RTX 5090 perf baseline：
    - `WP = 0.545397s`
    - `Gradient = 0.576524s`
  - 变体矩阵：
    - `RECOMPUTE_X`
      - correctness：Pass，所有输出 `rel L2 = 0`。
      - perf output compare：Pass。
      - `WP = 0.627311s`，`Gradient = 0.657990s`，慢于 baseline。
    - `RECOMPUTE_Y`
      - correctness：Pass，所有输出 `rel L2 = 0`。
      - perf output compare：Pass。
      - `WP = 0.652586s`，`Gradient = 0.683030s`，慢于 baseline。
    - `RECOMPUTE_Z`
      - correctness：Pass，所有输出 `rel L2 = 0`。
      - perf output compare：Pass。
      - `WP = 0.518128s`，`Gradient = 0.547818s`。
    - `RECOMPUTE_XYZ`
      - correctness：Pass，所有输出 `rel L2 = 0`。
      - perf output compare：Pass。
      - `WP = 0.673621s`，`Gradient = 0.703244s`，明显退化。
    - `RECOMPUTE_Z + TILE_LIST_V`
      - correctness：Pass，所有输出 `rel L2 = 0`。
      - perf output compare：Pass。
      - `WP = 0.507620s`，`Gradient = 0.538299s`。
    - `RECOMPUTE_Z + TILE_LIST(V+P)`
      - correctness：Pass，所有输出 `rel L2 = 0`。
      - perf output compare：Pass。
      - `WP = 0.507413s`，`Gradient = 0.538789s`。
  - 同场复测：
    - default `perf_1gpu`：
      - run：`benchmarks/runs/perf_1gpu_default_retest_after_recompute_20260606_225948`
      - `WP = 0.547345s`
      - `Gradient = 0.578054s`
      - output compare vs frozen baseline：Pass。
    - best repeat `perf_1gpu`：
      - run：`benchmarks/runs/perf_1gpu_recompute_z_tile_all_repeat_20260606_230004`
      - `WP = 0.506966s`
      - `Gradient = 0.537213s`
      - output compare vs frozen baseline：Pass。
    - default `perf_1gpu_6shots`：
      - run：`benchmarks/runs/perf_1gpu_6shots_default_retest_after_recompute_20260606_225952`
      - `WP = 2.705801s`
      - `Gradient = 2.853960s`
    - best `perf_1gpu_6shots`：
      - run：`benchmarks/runs/perf_1gpu_6shots_recompute_z_tile_all_20260606_230008`
      - `WP = 2.506537s`
      - `Gradient = 2.629992s`
      - output compare vs default 6-shot：Pass。
- 输出、哈希或误差摘要：
  - 当前远端最佳二进制和关键文件 SHA256：
    - `bin/cuda_3D_FM`：`c67cecd2fe675267c7d32d8ece70300ee74b30ee0c4e77bfe77b7f0a98b7f271`
    - `include/inc3D/single_solver.h`：`b6aa858d68b123f2744ca0b65d68d151089dc6c9432013da0f071d725f051379`
    - `src/single_solver.cu`：`db7cde40938315fe0891d442f84903cc2d198b4653ca39073b23b0fc4ce7c66b`
    - `src/rem_fd.cu`：`7bad299265a7781dde3be874f45cb52db7b30229560462458447dfcfb5d13c3a`
    - `tools/run_benchmark.py`：`90a02d4d2b2cc77ed60bbf75718b86f9a70cf28d1aeb34f46e3ceee6eecfcd5f`
    - `tools/compare_debug_dumps.py`：`238a65ce10458aa90136a20ef80f2d4935681cbf3cea38a776a1be77e681ba65`
  - 1-shot best speedup by frozen WP：`0.545397 / 0.506966 = 1.0758x`。
  - 1-shot best speedup by frozen Gradient：`0.576524 / 0.537213 = 1.0732x`。
  - 6-shot same-window speedup by WP：`2.705801 / 2.506537 = 1.0795x`。
  - 6-shot same-window speedup by Gradient：`2.853960 / 2.629992 = 1.0852x`。
  - 所有 correctness/perf 输出对比均通过；记录到对应 `benchmarks/reports/*/comparison.md`。
- 决策：
  - 当前最佳候选为：
    - `-DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2`
  - 当前远端二进制已按该最佳候选重新编译。
  - 不创建 `archives/speedups` 阈值存档；当前约 `1.08x`，未达到 `1.5x`。
  - 保留 `RECOMPUTE_X/Y` 源码宏路径作为可复现实验证据，但不建议作为优化主线继续推进。
- 风险与下一步：
  - `RECOMPUTE_X/Y` 数值正确但性能退化，说明全方向 velocity 重算不是 2x/3x 路线。
  - 后续应保留 z 方向重算，转向更大结构：
    - 方向专用 PML kernel，减少分支和无效 work。
    - kernel fusion，重点研究 core 与 z-PML 压力更新能否合并。
    - temporal blocking，但这是更大重写，需要重新设计时间推进和 halo/source injection 时序。

## 2026-06-06 23:40 +08:00 - z-face PML pressure 专用化受控实验

- 操作目标：
  - 按 Pro 新路线执行：先 profile 当前 best，再实现 `CUDA3D_PML_ZFACE_P_SPECIALIZE`。
  - 只专门化 z-PML face 中 `x/y` 位于 core 区域的 pressure update。
  - `x/y` face、edge、corner 和 residual 区域继续走 generic PML tile-list。
  - 用 debug dump、correctness、`perf_1gpu`、`perf_1gpu_6shots` 判断是否继续做 `ZFACE_V_SPECIALIZE`。
- 修改文件：
  - `include/inc3D/cu_common.h`
    - 新增 `PmlZFaceBlockSize1/2/3` 默认宏。
  - `include/inc3D/single_solver.h`
    - 新增 `cuda_fd3d_p_pml_zface_ns` 声明。
  - `src/single_solver.cu`
    - 新增 `pml_zface_p_special_point`。
    - 新增 `cuda_fd3d_p_pml_zface_ns`。
    - generic `p_pml`/`p_pml_tile` 在宏开启时跳过 zface-special 点，避免重复更新。
    - 后续将 skip 放到 core early-return 后，并加 tile-level gate，减少无关线程开销。
  - `src/rem_fd.cu`
    - 新增 zface pressure tile-list 构建、launch 和释放。
  - `tools/remote_exec.py`
    - 追加清理 stdin 命令开头 UTF-8 BOM 的健壮性修补，避免 PowerShell 管道偶发把远端命令变成 `﻿cd`。
  - `feedback/codex_report_20260606_234000_zface_p_specialize.md`
    - 记录 profile、实现、debug dump、sweep 和决策。
- 执行命令摘要：
  - 当前 best nsys profile：
    - `nsys profile -t cuda,nvtx -f true -o benchmarks/profiles/pml_recompute_z_tile_20260606/best_nsys ...`
    - `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv --output ...`
  - 当前 best ncu profile：
    - `ncu --target-processes all --set full --kernel-name 'regex:.*pml.*' --launch-count 20 ...`
  - zface 编译宏：
    - `-DCUDA3D_PML_RECOMPUTE_Z`
    - `-DCUDA3D_PML_TILE_LIST`
    - `-DCUDA3D_PML_ZFACE_P_SPECIALIZE`
    - `-DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2`
    - `-DPmlZFaceBlockSize1/2/3=<variant>`
  - debug dump：
    - baseline：`RECOMPUTE_Z + TILE_LIST + CUDA3D_PML_DEBUG_DUMP`
    - candidate：`RECOMPUTE_Z + TILE_LIST + ZFACE_P_SPECIALIZE + CUDA3D_PML_DEBUG_DUMP`
    - dump step：`it=0/1/2`
  - sweep 测试：
    - `python3 tools/run_benchmark.py --case correctness --tag <tag>`
    - `python3 tools/run_benchmark.py --case perf_1gpu --tag <tag>`
    - `python3 tools/run_benchmark.py --case perf_1gpu_6shots --tag <tag>`
    - `python3 tools/compare_outputs.py --baseline ... --candidate ... --out ...`
- 测试结果：
  - nsys profile 通过：
    - `cuda_fd3d_p_pml_tile_ns`：46.7% GPU kernel time。
    - `cuda_fd3d_v_pml_tile_ns`：30.0% GPU kernel time。
    - `cuda_fd3d_p_core_ns`：22.3% GPU kernel time。
    - `cudaLaunchKernel`：2004 calls，总计约 80.37 ms。
  - ncu profile 未能获取硬件指标：
    - 失败原因：`ERR_NVGPUCTRPERM`，当前用户无 NVIDIA GPU performance counter 权限。
  - debug dump：
    - `benchmarks/reports/debug_dump_zface_p_8x8x4_vs_best_it0`：Pass。
    - `benchmarks/reports/debug_dump_zface_p_8x8x4_vs_best_it1`：Pass。
    - `benchmarks/reports/debug_dump_zface_p_8x8x4_vs_best_it2`：Pass。
  - first correctness：
    - `zface_p_8x8x4` correctness pass。
    - 输出相对 L2 约 `3e-7` 到 `4.6e-7`，低于 `1e-5` 门槛。
  - sweep：
    - `8x8x4`
      - correctness/perf1/6shot compare：Pass。
      - `perf_1gpu WP = 0.585157s`，`Gradient = 0.614705s`。
      - `perf_1gpu_6shots WP = 2.865097s`，`Gradient = 2.994406s`。
      - vs current best 6shot WP：`0.875x`，退化。
    - `16x8x2`
      - correctness/perf1/6shot compare：Pass。
      - `perf_1gpu WP = 0.555189s`，`Gradient = 0.584692s`。
      - `perf_1gpu_6shots WP = 2.723829s`，`Gradient = 2.854656s`。
      - vs current best 6shot WP：`0.920x`，退化。
    - `16x4x4`
      - correctness/perf1/6shot compare：Pass。
      - `perf_1gpu WP = 0.554979s`，`Gradient = 0.585131s`。
      - `perf_1gpu_6shots WP = 2.694538s`，`Gradient = 2.824677s`。
      - vs current best 6shot WP：`0.930x`，退化。
    - `8x16x2`
      - correctness/perf1/6shot compare：Pass。
      - `perf_1gpu WP = 0.586863s`，`Gradient = 0.617514s`。
      - `perf_1gpu_6shots WP = 2.844403s`，`Gradient = 2.984499s`。
      - vs current best 6shot WP：`0.881x`，退化。
    - `16x4x4 gated`
      - correctness/perf1/6shot compare：Pass。
      - `perf_1gpu WP = 0.553700s`，`Gradient = 0.583606s`。
      - `perf_1gpu_6shots WP = 2.712018s`，`Gradient = 2.838111s`。
      - 仍慢于 current best。
  - 恢复 current best 后 quick perf：
    - run：`benchmarks/runs/perf_1gpu_restore_best_after_zface_20260606_233741`
    - `WP = 0.506107s`
    - `Gradient = 0.535563s`
    - output compare vs frozen perf baseline：Pass。
- 输出、哈希或误差摘要：
  - 当前 best 6shot baseline：
    - run：`benchmarks/runs/perf_1gpu_6shots_recompute_z_tile_all_20260606_230008`
    - `WP = 2.506537s`
    - `Gradient = 2.629992s`
  - 远端当前二进制已恢复为 current best 宏构建：
    - `bin/cuda_3D_FM`：`5d2b7c5e4e0fdfd1590bb6e736c21ba11b0d85ba92dc564d4c6a0227ff305a71`
    - `feedback/codex_report_20260606_234000_zface_p_specialize.md`：`d5465c69e71723e53123aafab04e0028d5e9ae8904b43b0fb22238018165449f`
    - `include/inc3D/cu_common.h`：`b184a1b52697982f1e18caa73db8b7b0127f81241bfd1d3458b40b8dc2293180`
    - `include/inc3D/single_solver.h`：`34b50fad827e6895be922274fc88c8a140cecf9ad3ed1d898e2fed79d99f6fe2`
    - `src/single_solver.cu`：`cfa076ea6a863e7e1f3f920d8c4409950d10ea13841101456c298fa409f77bc7`
    - `src/rem_fd.cu`：`1bbbfb909b4c0086c14eb1a252b10c2e191358117594d990a773fdb9c5213a4e`
- 决策：
  - 按 Pro 规则属于“情况 C：`ZFACE_P_SPECIALIZE` 变慢”。
  - 不继续做 `CUDA3D_PML_ZFACE_V_SPECIALIZE`。
  - 不创建 speedup archive；没有达到 `1.5x`，并且相对 current best 退化。
  - 保留源码中的 zface 实验宏作为可复现实验证据，但远端二进制恢复为 current best。
- 风险与下一步：
  - zface pressure 拆 kernel 数值正确，但 kernel launch、tile 切分和 generic residual 成本超过收益。
  - 后续不应继续拆 PML faces。
  - 建议转向：
    - 重新设计 PML generic kernel 的 tile shape/occupancy。
    - 或进入 core stencil/time blocking/fusion 的大结构实验。

## 2026-06-07 00:17 CST - Implement ZMEM_IN_P scaffold and compile gates

- ?????? Pro ???????? `CUDA3D_PML_ZMEM_IN_P`??????/current-best ???????
- ?????`include/inc3D/single_solver.h`?`src/single_solver.cu`?`src/rem_fd.cu`?
- ??????? `mem_dz_next_v` ????? `recompute_vz_after_update_from_old_mem`?`ZMEM_IN_P` ??? `v_pml` ???? z velocity/mem update?host ??? `d_memory_dz_next`??? `mem_dz`/`mem_dz_next` swap??? debug-only `CUDA3D_PML_ZMEM_DEBUG_FILL` NaN ?????
- ????????? 3 ???? `/work/wenzhe/cuda3D`??? `make -B -f makefile.rtx5090 test`??? current-best ??? `RECOMPUTE_Z + TILE_LIST + PmlTileBlockSize=32x4x2`?
- ????????????current-best ?????
- ?????`benchmarks/build_logs/default_after_zmem_patch_20260607_001658.log` rc=0?`benchmarks/build_logs/current_best_after_zmem_patch_20260607_001658.log` rc=0?
- ??????????? candidate ???????? `CUDA3D_PML_ZMEM_IN_P` candidate???? debug dump step 0/1/2?correctness?perf_1gpu?perf_1gpu_6shots?

## 2026-06-07 00:27 CST - Validate CUDA3D_PML_ZMEM_IN_P and restore current best

- ??????? `CUDA3D_PML_ZMEM_IN_P` ??????? debug dump?correctness?performance?profiling fallback????? current best?
- ?????`include/inc3D/single_solver.h`?`src/single_solver.cu`?`src/rem_fd.cu`????? `feedback/codex_report_20260607_002700_zmem_in_p.md`?
- ??????? debug baseline/candidate??? smoke debug dump step 0/1/2??? release correctness?perf_1gpu?perf_1gpu_6shots??? perf_1gpu_6shots repeat??? profiling ????? ptxas/cuobjdump fallback????? current best ? smoke?
- ?????debug dump step 0/1/2 ?? pass?candidate `CUDA3D_PML_ZMEM_DEBUG_FILL` ????correctness/perf ?????? pass???? `rel_l2 = 0`?final smoke `ALL DONE`?
- ??/?????`zmem_in_p_correctness_vs_best_fixed_20260607_002024`?`zmem_in_p_perf1_vs_best_fixed_20260607_002024`?`zmem_in_p_perf6_vs_best_fixed_20260607_002024`?`zmem_in_p_perf6_repeat_vs_best_20260607_002416` ? pass?
- ?????first perf_1gpu WP `0.508251s -> 0.481991s`?speedup `1.0545x`?first perf_1gpu_6shots WP `2.506582s -> 2.421955s`?speedup `1.0349x`?repeat perf_1gpu_6shots WP `2.499682s -> 2.420177s`?speedup `1.0329x`?
- profiling ???`RmProfilingAdminOnly: 1`????????? NCU counters?fallback ??? `benchmarks/profiles/current_best_ptxas_resource_20260607_002231.log` ? `benchmarks/profiles/current_best_cuobjdump_resource_20260607_002231.txt`?
- ???`ZMEM_IN_P` ???? 3%+ ????????????????????????????????????? current best?sha256 `0b921158eb9b05ffde7f1688b2e641a6371acc8ad485ac7984092f5022166565`?
- ???????wall-clock ?????????????? kernel/WP ????????????? PML fused shared-halo ? core temporal blocking prototype?
## 2026-06-07 00:35 CST - Start overnight autonomous optimization loop

- Goal: run staged CUDA3D optimization loop in isolated overnight_20260607 workspace.
- Workspace: /work/wenzhe/cuda3D/overnight_20260607.
- Script: overnight_20260607/scripts/build_and_test_variant.sh.
- Initial records: git_status_initial.txt, bin_initial.sha256, start_time.txt.
- Safety: no deletion, no global environment changes, all experiments macro-gated.

## 2026-06-07 00:45 CST - Stage 0 references completed

- Objective: build overnight harness and run current_best_reference / zmem_reference.
- Files changed: added overnight_20260607/scripts/build_and_test_variant.sh.
- Commands: ran build_and_test_variant.sh for current best and zmem references with correctness, perf_1gpu, perf_1gpu_6shots, perf_1gpu_6shots repeat.
- Results: both variants build and run successfully; zmem output compare vs current best passed for correctness/perf1/perf6/repeat with rel_l2=0.
- Timings: current_best perf6 WP=2.507059s Gradient=2.632964s; current_best perf6_repeat WP=2.508503s Gradient=2.632298s.
- Timings: zmem perf6 WP=2.393577s Gradient=2.514862s; zmem perf6_repeat WP=2.390644s Gradient=2.514458s.
- Decision: zmem_reference reproduced >3% 6-shot speedup and becomes overnight best_so_far.
- Next: implement CUDA3D_PML_ZMEM_V_TILE_PRUNE.

## 2026-06-07 00:50 CST - Stage 1 ZMEM_V_TILE_PRUNE completed

- Objective: implement and test CUDA3D_PML_ZMEM_V_TILE_PRUNE.
- Files changed: src/rem_fd.cu.
- Commands: uploaded rem_fd.cu; ran zmem_debug_reference; ran zmem_v_prune_debug_32x4x2 debug dump step 0/1/2; ran zmem_v_prune_32x4x2 correctness, perf_1gpu, perf_1gpu_6shots, repeat.
- Correctness: debug dump step 0/1/2 pass vs zmem_debug_reference; correctness/perf compares vs zmem_reference pass with rel_l2=0.
- Tile stats: perf1 original=23100 kept=23100 pruned=0; perf6 original=18800 kept=18800 pruned=0; prune_ratio=0.
- Performance vs zmem_reference: perf6 WP speedup=0.98895; perf6 repeat WP speedup=0.98738; no benefit.
- Decision: keep macro-gated code as evidence, but do not carry prune as best_so_far. Continue Stage 2 from zmem_reference.

## 2026-06-07 00:56 CST - Stage 2 TILE_MASK_FASTPATH completed

- Objective: implement tile mask metadata and p_pml_tile single-kernel fastpath.
- Files changed: include/inc3D/single_solver.h, src/rem_fd.cu, src/single_solver.cu.
- Commands: compiled default/current_best/zmem+maskfast; ran zmem_prune_maskfast debug dump step 0/1/2; ran correctness, perf_1gpu, perf_1gpu_6shots, repeat with output compares vs zmem_reference.
- Correctness: debug and all output compares passed with rel_l2=0.
- Performance vs zmem_reference: perf6 WP speedup=0.98389; perf6 repeat WP speedup=0.98122; slower.
- Decision: keep macro-gated code as evidence, but do not enable TILE_MASK_FASTPATH in best_so_far. Continue Stage 3 from zmem_reference only.


## 2026-06-07 01:04:04  - Overnight Stage3 top2 repeat/debug

- ???? Stage3 block sweep ????? perf6_repeat ? debug dump ???
- ?????????????/?? `overnight_20260607/reports/stage3_top2_repeat_summary.json`?
- ?????`build_and_test_variant.sh` ? `zmem_shape_32x2x4_repeat`?`zmem_shape_32x4x2_repeat`??? debug top ?????
- ??????? repeat ??????? zmem_reference perf6_repeat ??? rel_l2=0?debug it0/it1/it2 ????
- ?????
  - zmem_shape_32x2x4_repeat: WP=2.416933s, Gradient=2.528195s, WP speedup vs zmem_repeat=0.989123, Gradient speedup=0.994566.
  - zmem_shape_32x4x2_repeat: WP=2.418862s, Gradient=2.526810s, WP speedup vs zmem_repeat=0.988334, Gradient speedup=0.995112.
- ???Stage3 ????? zmem_reference ??????? Stage4 ??/?? sweep?
- ????????????????????????? correctness ????????

## 2026-06-07 01:11:43  - Overnight Stage4 p_core block sweep

- ???? zmem_reference ????? `cuda_fd3d_p_core_ns` ? `PBlockSize1/2/3`???????????
- ?????`include/inc3D/cu_common.h` ? BlockSize/VBlockSize/PBlockSize/PCoreBlockSize/PmlBlockSize ??? `#ifndef`????????????????
- ?????? 12 ? p_core block ???? `build_and_test_variant.sh --correctness --perf-1gpu --perf-1gpu-6shots`??? zmem_reference ???
- ???????
  - stage4_pcore_128x1x2: status=ok, compare_ok=True, perf6 WP=2.388011, Gradient=2.511048, WP speedup vs zmem=1.0023308100339572.
  - stage4_pcore_32x2x4: status=ok, compare_ok=True, perf6 WP=2.412678, Gradient=2.541207, WP speedup vs zmem=0.9920830711765101.
  - stage4_pcore_32x4x2: status=ok, compare_ok=True, perf6 WP=2.415801, Gradient=2.543195, WP speedup vs zmem=0.9908005667685377.
  - stage4_pcore_128x2x1: status=ok, compare_ok=True, perf6 WP=2.417007, Gradient=2.527869, WP speedup vs zmem=0.9903061927416843.
  - stage4_pcore_64x2x2: status=ok, compare_ok=True, perf6 WP=2.417128, Gradient=2.543447, WP speedup vs zmem=0.9902566185986014.
  - stage4_pcore_64x4x1: status=ok, compare_ok=True, perf6 WP=2.454251, Gradient=2.573546, WP speedup vs zmem=0.9752779972382613.
- ?????????????????? zmem_reference 2%????????????? `-maxrregcount` ???? kernel ???

## 2026-06-07 01:15:28  - Overnight Stage4b combo/reg sweep

- ????? Stage3/Stage4 ?? PML/core block???? `-maxrregcount` ???????????????
- ????????????????? `include/inc3D/cu_common.h` ???????
- ?????3 ? PML/core ?? + 5 ? register cap ?????? correctness/perf1/perf6 ?? zmem_reference ???
- ???????
  - stage4b_pml32x2x4_pcore128x1x2: status=ok, compare_ok=True, perf6 WP=2.415699, Gradient=2.532146, WP speedup vs zmem=0.9908424021370212.
  - stage4b_pml32x8x1_pcore128x1x2: status=ok, compare_ok=True, perf6 WP=2.458359, Gradient=2.565222, WP speedup vs zmem=0.9736482751298732.
  - stage4b_pml16x4x4_pcore128x1x2: status=ok, compare_ok=True, perf6 WP=2.501256, Gradient=2.616666, WP speedup vs zmem=0.9569500283057791.
  - stage4b_reg64_pml32x4x2_pcore128x1x2: status=ok, compare_ok=True, perf6 WP=2.513425, Gradient=2.63247, WP speedup vs zmem=0.9523168584700161.
  - stage4b_reg128_pml32x4x2_pcore128x1x2: status=ok, compare_ok=True, perf6 WP=2.562998, Gradient=2.685349, WP speedup vs zmem=0.9338973342936671.
  - stage4b_reg96_pml32x4x2_pcore128x1x2: status=ok, compare_ok=True, perf6 WP=2.566937, Gradient=2.687183, WP speedup vs zmem=0.9324642560374486.
- ??????????? zmem_reference 2%???????? zmem_reference ????????????? sweep ???
- ???????register cap ??????????????? repeat ???

## 2026-06-07 01:16:17  - Overnight Stage4 best p_core repeat

- ????? Stage4 ?????? `PBlockSize=128x1x2`?
- ???????
- ?????`build_and_test_variant.sh --perf-1gpu-6shots-repeat`?baseline ? zmem_reference perf6_repeat?
- ?????status=ok, compare_ok=True, WP=2.386368s, Gradient=2.507736s, WP speedup vs zmem_repeat=1.001792, Gradient speedup=1.002681.
- ???repeat ??? 2% ?????????????
- ????????? final rule ?????????????

## 2026-06-07 01:19:11  - Overnight Stage5 finalization

- ???? overnight plan ???????????????????
- ???????/?? `overnight_20260607/reports/final_report.md`?`final_summary.json`?`best_variant.env`?`best_binary.sha256`?`failed_variants.md`?`next_steps.md`?
- ???????? final zmem binary??? final correctness compare ? smoke?
- ?????final correctness status=ok, compare_ok=True; smoke ALL DONE=True, WP=0.002515, Gradient=0.003288.
- ??/?????ad993cb6a5cd2301e2bbf077186ec350f3565eab41861ebbc8e5f6fd0f3c4d1e  bin/cuda_3D_FM
- ?????????? zmem_reference?Stage4 `128x1x2` repeat ?? 1.0018x???? >=2% ??????
- ????????????????????? profiler-guided region-specialized PML ???? temporal/blocking ???

## 2026-06-07 08:48:00 +0800 - Git synchronization for local and RTX 5090 server

- 操作目标：初始化本地 `E:\cuda3D` 与服务器 `/work/wenzhe/cuda3D` 的 Git 同步，并推送到 `https://github.com/Volodymyr2580/cuda3D.git`。
- 修改文件：
  - 新增 `.gitignore`，忽略大模型数据、二进制、目标文件、benchmark runs/profiles/logs、overnight 生成物和备份文件。
  - 新增 `.gitattributes`，固定源码、脚本、Markdown、配置文件使用 LF，避免 Windows/服务器换行差异。
  - 新增 `README.md`，记录项目用途和当前 RTX 5090 validated build flags。
  - 更新 `AGENT_LOG.md`，记录本次 Git 同步。
- 执行命令摘要：
  - 本地：`git init`、`git add .`、`git commit`、`git remote add origin`、`git -c http.proxy= -c https.proxy= push -u origin main`。
  - 本地：通过 `git archive` 和 `git bundle` 生成同步包，上传到服务器 `overnight_20260607/builds/`。
  - 服务器：展开 tracked 文件、`git init`、从 bundle 建立 `origin/main`、配置 GitHub origin，并执行 `git pull --ff-only`。
- 测试/验证结果：
  - GitHub `main` 已推送成功。
  - 本地最新提交：`3b9f586 chore(git): ignore generated overnight reports`，追加日志后会再生成一个日志提交。
  - 服务器最新同步时 `git log` 显示 `main` 与 `origin/main` 对齐，`git status --short` 干净。
  - 服务器 `git ls-remote --heads origin main` 成功返回 GitHub main ref。
- 输出/哈希摘要：
  - 初始提交：`7486ea0 chore(init): first commit`。
  - Git 规则提交：`e6a8ae9 chore(git): ignore backup files`。
  - 生成报告忽略提交：`3b9f586 chore(git): ignore generated overnight reports`。
- 风险与下一步：
  - 根目录 1.4GB `.nav` 与 56MB `.dir` 未纳入 Git，仍保留在本地/服务器文件系统。
  - perf benchmark 所需大体积 `.dir` 数据也未纳入 Git；clone 后若要跑 perf，需要服务器原始数据或重新生成/复制。
  - 以后新增重要 benchmark 报告时，如果被 `.gitignore` 忽略，需要显式 `git add -f <path>`。

## 2026-06-07 09:05:00 +0800 - Profiler gate and architecture freeze

- 操作目标：落实 Pro 架构反馈，把 `ZMEM_IN_P` 固化为 RTX 5090 稳定基线，并检查是否具备 Nsight Compute 硬件 counter 权限。
- 修改文件：
  - 更新 `AGENTS.md`，加入当前稳定基线、do-not-invest 列表、profiler gate 和两个允许的 prototype 门槛。
  - 新增 `docs/profiler_inventory.md`、`docs/final_arch_report.md`、`docs/pml_fusion_result.md`、`docs/core_zpencil_result.md`。
  - 更新 `overnight_20260607/reports/best_variant.env`、`best_binary.sha256`、`final_report.md`、`final_summary.json` 中的当前二进制 hash。
- 执行命令摘要：
  - 服务器执行 `cat /proc/driver/nvidia/params | grep -E "RmProfilingAdminOnly|Profiling"`。
  - 服务器执行 `ncu --target-processes all ... --kernel-name regex:".*(p_pml_tile|v_pml_tile|p_core).*"` 对 `profile_1gpu` 尝试采集。
  - 服务器用正式 zmem flags 加 `-Xptxas -v` 生成 `ptxas` verbose log，并用 `cuobjdump --dump-resource-usage` 生成静态资源表。
  - 服务器最后用正式 zmem flags 重新编译 `bin/cuda_3D_FM`，恢复当前 best binary 口径。
- 测试结果：
  - Nsight Compute 已安装：`/usr/local/cuda-13.0/bin/ncu`，版本 `2025.3.0.0`。
  - GPU counter 权限不可用：`RmProfilingAdminOnly: 1`。
  - NCU 尝试连接到进程并且应用 `ALL DONE`，但 NCU 返回 `ERR_NVGPUCTRPERM`，return code `1`。
  - 因 profiler gate 未通过，未启动 `CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE` 或 `CUDA3D_CORE_ZPENCIL_SHARED`。
- 输出/哈希摘要：
  - 当前正式 zmem binary SHA256：`db55f6505d3bf2460a07028056a3b00da8bf0b884ffb69a250b2d1bd023ab488  bin/cuda_3D_FM`。
  - fallback 静态资源：`p_pml_tile` 44 regs/0 spill，`v_pml_tile` 38 regs/0 spill，`p_core` 48 regs/0 spill/约 2KB shared。
- 风险与下一步：
  - 当前只有静态资源信息，没有 memory throughput、occupancy、stall reason，不能据此启动大结构重写。
  - 需要管理员把 NVIDIA profiling counter 权限打开，目标是 `RmProfilingAdminOnly: 0`。
  - 权限打开后，先重跑 `docs/profiler_inventory.md` 中的 NCU 命令，再决定是否进入 PML z-slab fusion 或 p_core z-pencil。

## 2026-06-07 09:18:00 +0800 - NVIDIA profiler permission configuration written

- 操作目标：使用管理员权限为 RTX 5090 服务器开放 NVIDIA GPU performance counters，以便 Nsight Compute 采集硬件级 profiler 数据。
- 修改文件：
  - 服务器系统文件：`/etc/modprobe.d/nvidia-profiler.conf`。
  - 服务器 initramfs：执行 `update-initramfs -u` 更新启动镜像。
  - 项目文件：仅追加本日志条目。
- 执行命令摘要：
  - 先执行 `sudo -S true` 验证 sudo 密码可用。
  - 写入 `/etc/modprobe.d/nvidia-profiler.conf`：
    `options nvidia NVreg_RestrictProfilingToAdminUsers=0`
  - 执行 `update-initramfs -u`。
  - 检查 `/proc/driver/nvidia/params` 中的 runtime 参数。
- 测试结果：
  - sudo 验证通过。
  - 配置文件写入成功。
  - `update-initramfs` 成功生成 `/boot/initrd.img-6.17.0-29-generic`。
  - 当前 runtime 参数仍为 `RmProfilingAdminOnly: 1`，说明需要重启或重新加载 NVIDIA 驱动模块后生效。
- 输出/哈希摘要：
  - 配置文件内容：`options nvidia NVreg_RestrictProfilingToAdminUsers=0`。
- 风险与下一步：
  - 尚未重启服务器，当前 Nsight Compute 仍会因 `ERR_NVGPUCTRPERM` 失败。
  - 下一步需要用户确认是否允许重启服务器；重启后重新检查 `RmProfilingAdminOnly` 是否变为 `0`，然后重跑 NCU。

## 2026-06-07 09:30:00 +0800 - Server reboot and successful NCU profiler inventory

- 操作目标：重启 RTX 5090 服务器使 NVIDIA profiler counter 配置生效，并采集 `zmem_reference` 的 NCU profiler inventory。
- 修改文件：
  - 更新 `docs/profiler_inventory.md`、`docs/final_arch_report.md`、`docs/pml_fusion_result.md`、`docs/core_zpencil_result.md`。
  - 追加本 `AGENT_LOG.md` 条目。
  - 服务器生成 raw profiler artifacts 于 `benchmarks/profiles/`，这些文件按 `.gitignore` 不进入 Git。
- 执行命令摘要：
  - 使用 sudo 发起 `/sbin/reboot`。
  - 轮询 SSH，服务器恢复后检查 `cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly`。
  - 运行 NCU 主采集：`SpeedOfLight`、`MemoryWorkloadAnalysis`、`SchedulerStats`、`Occupancy`、`LaunchStats`。
  - 补跑 NCU `WarpStateStats`，提取 warp stall breakdown。
- 测试结果：
  - 服务器重启成功，SSH 恢复。
  - `RmProfilingAdminOnly` 从 `1` 变为 `0`。
  - NCU main run return code `0`，应用日志包含 `ALL DONE`。
  - NCU warp-state run return code `0`，应用日志包含 `ALL DONE`。
- 输出/哈希/指标摘要：
  - main report：`benchmarks/profiles/zmem_ncu_main_20260607.ncu-rep`。
  - warp report：`benchmarks/profiles/zmem_ncu_warpstates_20260607.ncu-rep`。
  - `p_pml_tile`: time avg `189.366`，compute/mem throughput `56.15%`，DRAM throughput `40.43%`，warps active `72.82%`，top stalls long_scoreboard `8.70`、wait `1.97`、short_scoreboard `1.91`。
  - `v_pml_tile`: time avg `71.299`，compute/mem throughput `55.40%`，DRAM throughput `45.74%`，warps active `82.35%`，top stalls long_scoreboard `15.63`、wait `1.56`、not_selected `1.49`。
  - `p_core`: time avg `93.555`，compute/mem throughput `96.94%`，DRAM throughput `42.44%`，warps active `66.33%`，top stalls long_scoreboard `8.64`、barrier `3.49`、not_selected `1.67`。
- 风险与下一步：
  - profiler gate 已通过，下一步可进入 `CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE`。
  - NCU 证据支持数据流/依赖延迟优化，不支持继续做 block-size 或 register-cap 随机 sweep。
  - `p_core` z-pencil 有依据但暂缓，优先做 PML z-slab；若 PML prototype 低于 5% repeat speedup，再转向 p_core 或重新评估。

## 2026-06-07 11:00:05 +0800 - p_core source-level profile and z-pencil gate decision

- 操作目标：根据 Pro 反馈停止 PML fused z-slab 方向，在 `exp/core-zpencil-shared` 分支上对 `p_core` 做 source-level NCU，判断是否实现 `CUDA3D_CORE_ZPENCIL_SHARED`。
- 修改文件：
  - 新增 `docs/pcore_source_profile.md`，记录 `p_core` 行级 NCU 热点与判断。
  - 更新 `docs/core_zpencil_result.md`，把状态从 deferred 改为 stopped before implementation。
  - 新增 `docs/temporal_blocking_feasibility.md`，记录失败后的数据流/temporal blocking 可行性分析。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地切换并确认分支：`git checkout -B exp/core-zpencil-shared`。
  - 服务器在 `/work/wenzhe/cuda3D` 使用 zmem baseline flags 加 `-lineinfo` 编译。
  - 服务器执行 `ncu --target-processes all --section SpeedOfLight --section MemoryWorkloadAnalysis --section SchedulerStats --section Occupancy --section SourceCounters --section WarpStateStats --kernel-name regex:".*p_core.*" --launch-skip 10 --launch-count 20 ...`。
  - 导出 `benchmarks/profiles/pcore_zmem_lineinfo_20260607_source_cuda_sass.csv`。
  - 服务器重新用不带 `-lineinfo` 的 zmem baseline flags 编译，恢复正式 binary。
  - 服务器运行 `benchmarks/cases/profile_1gpu` 做 post-restore sanity check。
- 测试结果：
  - NCU return code `0`，profile run 日志包含 `ALL DONE`。
  - `p_core` 20-launch source profile：Duration avg `93.630 us`，DRAM throughput avg `42.40%`，L2 throughput avg `96.83%`，achieved occupancy avg `66.51%`。
  - source-level counters 显示 z-neighbor stencil 行 `1110/1114/1118/1122/1126/1130/1134` 均为 `Shared(2)` load，没有 global L1 tag 或 L2 global sector 计数。
  - 主要 long-scoreboard 热点：line 1086 `z_tile[...] = p1[base]`，long SB `25483`；line 1116 y-neighbor global load，long SB `17627`；line 1111 x-neighbor global load，long SB `8713`。
  - barrier stall 主要归因到 line 1101 的 core-boundary return 附近，结合 line 1097 `__syncthreads()`，说明现有 shared tile 已经带来同步成本。
  - post-restore sanity check：`Gradient TIME all=0.189999s`，`WP computing time=0.161748s`，日志包含 `ALL DONE`。
- 输出/哈希/误差摘要：
  - lineinfo profiling binary SHA256：`019e615560c090aa849ac84a9075cc5baff017cfe801eb85af917f1f9ea896ce`。
  - restored zmem binary SHA256：`cfc502cf8a27038e54a1bdf1c3665b057a3b306046bd547a1baf70a204e17293`。
  - 关键 remote artifacts：`benchmarks/profiles/pcore_zmem_lineinfo_20260607.ncu-rep`、`.csv`、`_raw.csv`、`_source_cuda_sass.csv`、`benchmarks/runs/profile_1gpu_after_pcore_profile_20260607/run.log`。
  - 本轮未生成 candidate 输出，因此无 correctness 误差对比。
- 风险与下一步：
  - 本地 Windows Python 缺少 `paramiko`，直接运行 `tools/remote_exec.py` 会报 `ModuleNotFoundError: No module named 'paramiko'`；本轮改用 WSL Python helper，未修改全局 Python 环境。
  - 曾按旧文档尝试 `benchmarks/cases/smoke_1gpu`，远程当前不存在该 case；改用现有最小 `profile_1gpu` 做 sanity check。
  - 结论：不实现 `CUDA3D_CORE_ZPENCIL_SHARED`，因为当前 baseline 已经包含该 z-pencil shared-memory 路径。继续做同类宏只会重复现有代码，缺少达到 `p_core >=10%` 的可信路径。
  - 下一步应转向更大尺度数据流重构，例如 core-interior temporal blocking feasibility，而不是继续 p_core block-size sweep 或重复 shared-memory variant。

## 2026-06-07 11:45:37 +0800 - Core 2-step interior prototype scaffolding

- 操作目标：根据最新架构反馈，冻结 `ZMEM_IN_P` 稳定基线，停止 PML z-slab 与 p_core z-pencil 路线，启动 `CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE` 的 dependency map、debug harness 和最小安全 case。
- 修改文件：
  - 更新 `AGENTS.md`，把允许 prototype 顺序改为 core strict-interior two-step temporal/dataflow。
  - 新增 `docs/architecture_decision_20260607.md`、`docs/architecture_decision_log.md`。
  - 新增 `docs/core_2step_dependency_map.md`、`docs/core_2step_interior_design.md`、`docs/core_2step_interior_result.md`。
  - 更新 `include/inc3D/cu_common.h`，新增 `CUDA3D_CORE_STENCIL_RADIUS=7`。
  - 更新 `src/single_solver.cu`，让 `CoreStencilRadius` 由 `CUDA3D_CORE_STENCIL_RADIUS` 派生。
  - 更新 `src/rem_fd.cu`，新增 `CUDA3D_CORE_2STEP_DEBUG_DUMP` / `CUDA3D_CORE_2STEP_INTERIOR_COMPARE` 下的 strict core interior p0/p1 dump harness。
  - 新增 `tools/create_core_2step_case.py`、`tools/compare_core_interior_dumps.py`。
  - 新增 `benchmarks/cases/core_2step_interior_1gpu/` 的 input/nav/manifest；大体积 `.dir` 由生成脚本重建，不进 Git。
- 执行命令摘要：
  - 本地打 tag：`stable-zmem-rtx5090-20260607`，并推送到 GitHub。
  - 新建并推送分支：`exp/core-2step-interior-prototype`。
  - 本地执行 `python -m py_compile tools/create_core_2step_case.py tools/compare_core_interior_dumps.py`。
  - 本地与服务器执行 `python3 tools/create_core_2step_case.py` 生成最小 case。
  - 服务器默认 zmem build：`make -B -f makefile.rtx5090 test`，使用稳定 baseline flags。
  - 服务器 debug build：额外加入 `-DCUDA3D_CORE_2STEP_DEBUG_DUMP`。
  - 服务器运行 `benchmarks/cases/core_2step_interior_1gpu/input_core_2step_interior_1gpu.in`。
  - 服务器执行 `tools/compare_core_interior_dumps.py` 对 debug dump 目录做 self-compare。
  - 服务器最后重新编译不带 debug 宏的 zmem binary，并跑最小 case sanity check。
- 测试结果：
  - Python 工具语法检查通过。
  - 默认 zmem build 通过，初次 scaffolding binary SHA256：`496e09b9021ea03c1461b242cee400b90d5782970b3dafaccc86a8601c9a0d45`。
  - 初始 `xpad=0.05` 的 debug run 失败，错误为 strict interior 在 x/y 裁剪后为空：`n=(96,25,25)`，region x/y 上界为 `-1`。
  - 修复 case generator：默认 `xpad=0.5`，避免 acquisition crop 过小。
  - debug build 通过，debug binary SHA256：`afa215446262be563c2a753a79c04f115afd050aff4bfe0bf4ce7e7b19d8d244`。
  - 修复后的 debug run 通过：`Gradient TIME all=0.004658s`，`WP computing time=0.003526s`，日志包含 `ALL DONE`。
  - debug dump 生成 18 个文件：6 个 timestep，每步 `p0_core.bin`、`p1_core.bin`、`core_meta.txt`。
  - meta 样例：`source_in_region=0`，`receivers_in_region=0`，region `z=[26,70)`、`x=[26,35)`、`y=[26,35)`，count `3564`。
  - self-compare 通过：`benchmarks/reports/core2step_debug_self_compare_xpad_20260607/comparison.md`，12 个 `.bin` 全部 pass。
  - 恢复不带 debug 宏的 zmem binary 后，SHA256：`e2e48089353443fbbf3088ef7e1131bec9023e4721f25e9ff4a1f3e90a8a045a`。
  - post-restore 最小 case sanity check 通过：`Gradient TIME all=0.002856s`，`WP computing time=0.001342s`，日志包含 `ALL DONE`。
- 输出/哈希/误差摘要：
  - stable tag：`stable-zmem-rtx5090-20260607`。
  - 分支：`exp/core-2step-interior-prototype`。
  - remote run artifacts：`benchmarks/runs/core_2step_debug_dump_xpad_20260607/`、`benchmarks/reports/core2step_debug_self_compare_xpad_20260607/`、`benchmarks/build_logs/core2step_default_build_20260607.log`、`core2step_debug_build_20260607.log`、`core2step_restore_default_20260607.log`。
  - 本阶段未实现 `p(t+2)` prediction kernel，因此没有 candidate-vs-baseline 数值误差；当前验证目标是 dump harness 正确生成并可比较。
- 风险与下一步：
  - acquisition-based subdomain crop 会改变实际 `nby/nbx/nbz`，不能只按输入全域判断 strict interior；后续必须以 dump meta 中的实际 region 为准。
  - debug dump 通过并不代表 temporal blocking 已经正确，只说明第一阶段的 dependency map、safe case 和 per-step dump/compare 工具可用。
  - 下一步可以实现 debug-only `p(t+2)` strict-interior prediction，但仍不得改变主计算输出或 source/receiver 时序。

## 2026-06-07 11:52:20 +0800 - Ignore generated acquisition text artifacts

- 操作目标：保持服务器 `exp/core-2step-interior-prototype` 工作区在运行最小 case 后可读，避免程序自动生成的 acquisition 文本文件显示为未跟踪文件。
- 修改文件：
  - 更新 `.gitignore`，忽略 `benchmarks/cases/**/nrec_shot_new.txt`、`s_cor_new.txt`、`r_cor_new.txt`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 服务器运行最小 case 后，`git status --short` 显示 `benchmarks/cases/core_2step_interior_1gpu/nrec_shot_new.txt`、`s_cor_new.txt`、`r_cor_new.txt` 为未跟踪文件。
- 测试结果：
  - 未删除这些服务器文件，仅新增 ignore 规则。
- 输出/哈希/误差摘要：
  - 无数值输出变化。
- 风险与下一步：
  - 这些文件是 acquisition 调试/派生文本，不属于 CUDA 计算结果；已存在的 tracked case 中文件不受 ignore 规则影响。

## 2026-06-07 15:32:57 +0800 - Validate core 2-step debug-only p2 predictor

- 操作目标：在不改变主计算路径的前提下，为 `CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE` 增加 debug-only `p(t+2)` strict-interior 预测，并验证 `p2(it)` 是否与下一步 baseline `p0(it+1)` 完全一致。
- 修改文件：
  - 更新 `include/inc3D/single_solver.h`，在 `CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE` 下声明 `cuda_fd3d_p_core_2step_predict_ns`。
  - 更新 `src/single_solver.cu`，新增受 prototype 宏保护的 strict-interior `p2` 预测 kernel。
  - 更新 `src/rem_fd.cu`，在 debug dump 阶段分配辅助 `d_p2_core_debug`，预测并 dump `p2_core.bin`；source/receiver 落入 region 时停止。
  - 更新 `tools/compare_core_interior_dumps.py`，新增 `--mode p2-shift`，比较 candidate `p2(it)` 与 baseline `p0(it+1)`。
  - 更新 `docs/core_2step_interior_design.md`、`docs/core_2step_interior_result.md`，记录 debug-only predictor 的设计和验收结果。
- 执行命令摘要：
  - 本地执行 `python -m py_compile tools/compare_core_interior_dumps.py` 与 `git diff --check`。
  - 通过 SSH 将当前源码 diff 临时应用到 `/work/wenzhe/cuda3D`，未写入密码到项目文件。
  - 服务器 debug build flags：稳定 zmem flags 加 `-DCUDA3D_CORE_2STEP_DEBUG_DUMP -DCUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE`。
  - 服务器运行 `benchmarks/cases/core_2step_interior_1gpu/input_core_2step_interior_1gpu.in`，dump 到 `benchmarks/runs/core_2step_p2_debug_20260607_152450/dumps`。
  - 服务器执行 `python3 tools/compare_core_interior_dumps.py --mode p2-shift`，报告写入 `benchmarks/reports/core2step_p2_shift_compare_20260607_152450/`。
  - 服务器重新编译 non-debug zmem build，并运行同一最小 case 做 post-restore sanity check。
- 测试结果：
  - debug prototype 编译通过；debug binary SHA256：`b3ba2be23e7f07aa4b7593ba154649bf1b4c616b87c880548ef38dc29bc90f36`。
  - debug run 通过：`Gradient TIME all=0.005677s`，`WP computing time=0.004523s`，elapsed `0:02.49`，日志包含 `ALL DONE`。
  - dump 文件数符合预期：总计 `23`，其中 `p2_core.bin` 为 `5` 个。
  - `p2-shift` 比较通过：5 个 timestep 全部 `pass=True`，`rel_l2=0.0`，`max_abs=0.0`。
  - post-restore non-debug sanity check 通过：`Gradient TIME all=0.002968s`，`WP computing time=0.001357s`，elapsed `0:02.48`，日志包含 `ALL DONE`。
- 输出/哈希/误差摘要：
  - shifted compare report：`benchmarks/reports/core2step_p2_shift_compare_20260607_152450/comparison.md`。
  - post-restore binary SHA256：`b0996e463e6a89c8adfaf5daf84c6441d3bf6289356ec2f3637110791858289b`。
  - 观察到同 flags 连续 non-debug rebuild 的 binary SHA256 不完全稳定，因此本阶段记录 hash 作为追溯信息，但验收以 build/run/correctness report 为主。
  - 远端命令经验总结：PowerShell 直接把复杂多行命令作为 argv 传给 WSL/SSH 时容易出现引号或执行范围问题；后续远端多行命令优先使用 `remote_exec.py ... -` 的 stdin-command 模式。
- 风险与下一步：
  - 当前 `p2` 只是辅助 buffer 中的 debug-only 预测，还没有减少任何正式计算量，因此不声明性能提升。
  - 下一步可以进入 commit mode：在 single GPU、source/receiver 不落入 strict region 的条件下，提交已验证的 `p(t+2)` interior，并在下一 timestep 跳过同一区域的 baseline core 计算。
  - commit mode 必须保留 PML、guard region、source injection、receiver extraction 和 pointer swap 的 baseline 时序。

## 2026-06-07 16:01:48 +0800 - Validate core 2-step commit correctness prototype

- 操作目标：实现 `CUDA3D_CORE_2STEP_COMMIT_INTERIOR` correctness prototype，在下一 timestep 跳过 strict-interior baseline core 计算并提交上一轮预测的 `p(t+2)`，验证其数值正确性和性能方向。
- 修改文件：
  - 更新 `include/inc3D/single_solver.h`，声明 commit-mode skip/copy kernels。
  - 更新 `src/single_solver.cu`，新增 `cuda_fd3d_p_core_ns_skip_region` 与 `cuda_core2step_copy_region`。
  - 更新 `src/rem_fd.cu`，抽出 `Core2StepRegion` helper，接入 commit region 初始化、skip-region core launch、predicted-region copy 和下一步 prediction。
  - 更新 `docs/core_2step_interior_design.md`、`docs/core_2step_interior_result.md` 与 `AGENTS.md`，记录 commit prototype 结论和后续 fused two-step 方向。
- 执行命令摘要：
  - 本地执行 `git diff --check`。
  - 服务器临时应用源码 diff 后编译：稳定 zmem flags 加 `-DCUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE -DCUDA3D_CORE_2STEP_COMMIT_INTERIOR -DCUDA3D_DEBUG_CHECKS`。
  - 服务器运行 `core_2step_interior_1gpu` baseline/candidate 输出对比。
  - 服务器运行 baseline-debug 与 commit-debug strict-interior dump same-name 对比。
  - 服务器运行 `correctness` 6-shot case；默认 region 因裁剪后子域过小按设计停止，随后用 `CUDA3D_CORE_2STEP_REGION=26:54,12:15,12:15` 重跑。
  - 服务器最后重新编译 non-debug zmem build。
- 测试结果：
  - commit prototype 编译通过；candidate binary SHA256：`5beb9c6c5698a4131689e82b698a9dbb4e45f726a58d30da1a55618c13ced974`。
  - minimal output correctness 通过：`benchmarks/reports/core2step_commit_correctness_20260607_154200/comparison.md`，1 个输出文件 `rel_l2=0.0`，`max_abs=0.0`。
  - minimal timing：baseline `WP=0.001391s`，candidate `WP=0.001453s`，candidate 略慢。
  - strict-interior dump compare 通过：`benchmarks/reports/core2step_commit_dump_compare_20260607_154800/comparison.md`，23 个 dump 文件，17 个 `.bin` 比较均 `rel_l2=0.0`，`max_abs=0.0`。
  - full correctness 默认 region 失败且未作为通过结果：`ERROR invalid CUDA3D_CORE_2STEP commit region z=[26,54) x=[26,1) y=[26,1), n=(80,27,27)`。
  - full correctness 显式安全 region 通过：`benchmarks/reports/core2step_commit_correctness_full_region_20260607_155200/comparison.md`，6 个输出文件全部 `rel_l2=0.0`，`max_abs=0.0`。
  - full correctness timing：baseline `WP=0.012992s`，candidate `WP=0.015493s`，candidate 变慢。
  - post-restore non-debug zmem binary SHA256：`099bd05e192cd6017b2a851f09c923c91599c0ecb68683a949af029ede2be1f5`。
- 输出/哈希/误差摘要：
  - 关键 run：`benchmarks/runs/core_2step_commit_correctness_20260607_154200`、`core_2step_commit_dump_compare_20260607_154800`、`core_2step_commit_correctness_full_20260607_155200`。
  - 关键 reports：`benchmarks/reports/core2step_commit_correctness_20260607_154200/`、`core2step_commit_dump_compare_20260607_154800/`、`core2step_commit_correctness_full_region_20260607_155200/`。
  - 所有有效 correctness/dump comparison 均为零误差；失败的默认 full correctness run 是 region 安全门触发，不是数值误差。
- 风险与下一步：
  - standalone predict+copy commit prototype 不会带来速度提升，因为 `p(t+2)` 仍由额外 kernel 单独计算；它只证明调度正确性。
  - 默认 strict region 会被 acquisition crop 破坏，后续 fused kernel 必须支持 per-shot region fallback、tile list 或显式 region。
  - 真正下一步应实现 fused two-step core kernel：在同一 kernel 内计算 `p(t+1)` 与 strict-interior `p(t+2)`，复用 shared/global loads，并避免为已提交区域加载整块 shared tile。

## 2026-06-07 16:07:53 +0800 - Write Pro feedback report for core 2-step work

- 操作目标：根据用户请求，整理一份面向 Pro 的反馈报告，说明 core two-step interior 当前验证结果、性能结论和下一步请求。
- 修改文件：
  - 新增 `feedback/codex_report_20260607_160753_core2step_commit.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 检查 `feedback/` 目录已有报告命名风格。
  - 新增 Markdown 报告。
- 测试结果：
  - 未修改 CUDA 源码或构建文件。
  - 未运行新的编译或数值测试。
- 输出/哈希/误差摘要：
  - 报告总结了 `248ebba` 与 `616280f` 两个 commit 的验证结果。
- 风险与下一步：
  - 报告明确当前 commit prototype 不是 speedup 路线，下一步应要求 Pro 设计 fused two-step core temporal blocking kernel。

## 2026-06-07 16:42:56 +0800 - Validate debug-only fused core two-step predictor

- 操作目标：按照 Pro 反馈停止 standalone predict+copy 性能路线，启动 `CUDA3D_CORE_2STEP_FUSED_INTERIOR`，先完成 design doc、meaningful case、debug-only fused p2 predictor，并在 RTX 5090 上验证 `p2(it)` 与 baseline 下一步 `p0(it+1)` 一致。
- 修改文件：
  - 新增 `docs/core_2step_fused_design.md`，记录 fused two-step 设计约束、tile/inner/outer/halo、cross-CTA 依赖、commit/skip 与 benchmark gate。
  - 新增 `docs/core_2step_fused_result.md`，记录本阶段 meaningful case、编译 flags、p2-shift 和输出对比结果。
  - 更新 `AGENTS.md`，把 `CUDA3D_CORE_2STEP_FUSED_INTERIOR` 标记为当前 active architecture path，并记录 debug 阶段通过但尚无 speedup 结论。
  - 新增 `tools/create_core_2step_meaningful_case.py` 与 `benchmarks/cases/core_2step_meaningful_1gpu/` case 文件。
  - 更新 `include/inc3D/single_solver.h`，声明 `cuda_fd3d_p_core_2step_fused_predict_ns`。
  - 更新 `src/single_solver.cu`，新增 correctness-only fused p2 predictor；它在 baseline `p_core` 前使用 `p(t-1), p(t)` 局部重算第一步依赖，不读取其他 CTA 未同步的 `p(t+1)`。
  - 更新 `src/rem_fd.cu`，接入 fused region 初始化、source/receiver exclusion、debug dump 辅助 buffer、`CUDA3D_CORE_2STEP_FUSED_DEBUG` dump-step gate。
- 执行命令摘要：
  - 本地执行 `python -m py_compile tools/create_core_2step_meaningful_case.py tools/compare_core_interior_dumps.py`。
  - 本地执行 `git diff --check`。
  - 本地执行 `python tools/create_core_2step_meaningful_case.py` 生成 meaningful case。
  - 通过 WSL Python 的 `remote_put.py` / `remote_upload.py` 上传明确文件到 `/work/wenzhe/cuda3D`；Windows Python 因缺少 `paramiko` 不能运行这些脚本，未改全局 Python 环境。
  - 服务器执行 `python3 tools/create_core_2step_meaningful_case.py` 复核 case manifest。
  - 服务器 debug build flags：稳定 zmem flags 加 `-DCUDA3D_CORE_2STEP_FUSED_INTERIOR -DCUDA3D_CORE_2STEP_FUSED_DEBUG -DCUDA3D_CORE_2STEP_DEBUG_DUMP -DCUDA3D_DEBUG_CHECKS`。
  - 服务器运行 meaningful case 两次：`CUDA3D_CORE_2STEP_DUMP_STEP=0` dump fused `p2`，`CUDA3D_CORE_2STEP_DUMP_STEP=1` dump baseline `p0`，手动配对后运行 `tools/compare_core_interior_dumps.py --mode p2-shift --rel-tol 1e-6`。
  - 服务器重新编译 non-debug zmem build，运行同一 meaningful case，并用 `tools/compare_outputs.py` 对比 fused-debug 输出与 zmem baseline 输出。
- 测试结果：
  - meaningful case 生成通过：`core_points=2033152`，`fused_eligible_points=922560`，`eligible_ratio=0.453758`，`source_in_fused_region=no`，`receivers_in_fused_region=0`，fused region `30:90,30:154,30:154`。
  - debug build 通过；debug binary SHA256：`593e58ccc415e60f9a5700c40280ed8bd4c2a77c945f67588668c582c9f5e42b`。
  - meaningful debug run 通过，日志包含 `ALL DONE`；dump it=0：`Gradient TIME all=0.085542s`，`WP computing time=0.078205s`，elapsed `0:02.66`；dump it=1：`Gradient TIME all=0.085960s`，`WP computing time=0.078484s`，elapsed `0:02.16`。
  - p2-shift 比较通过：`benchmarks/reports/core2step_fused_p2_shift_meaningful_20260607_163942/comparison.md`，`count=922560`，`rel_l2=0.0`，`max_abs=0.0`，`max_rel=0.0`。
  - 恢复 zmem build 通过；restored zmem binary SHA256：`86617a8a4bb549e916c0681d7833b85b8516ceb8293104b9f1b2cd734a6f77ba`。
  - meaningful 输出对比通过：`benchmarks/reports/core2step_fused_output_meaningful_20260607_163942/comparison.md`，1 个 `.dir` 文件 `rel_l2=0.0`，`max_abs=0.0`。
  - restored zmem meaningful baseline timing：`Gradient TIME all=0.073288s`，`WP computing time=0.064357s`，elapsed `0:02.57`。
- 输出/哈希/误差摘要：
  - run：`benchmarks/runs/core_2step_fused_debug_meaningful_20260607_163942`。
  - p2-shift report：`benchmarks/reports/core2step_fused_p2_shift_meaningful_20260607_163942/`。
  - output report：`benchmarks/reports/core2step_fused_output_meaningful_20260607_163942/`。
  - debug-only fused predictor 没有修改主输出，输出相对 zmem baseline 为零误差。
  - 本地工具经验总结：`rg.exe` 在当前 PowerShell 中被系统拒绝执行时，立即切换 `Get-ChildItem | Select-String`；PowerShell `Select-Object -Index 560..760` 会把 range 当字符串，后续用 `-Skip/-First` 更稳。
  - 远端命令经验总结：`source ./env_5090.sh` 这类环境脚本不要配 `set -u`，oneAPI/CUDA 环境脚本可能引用未定义变量；后续远端 bash 默认用 `set -eo pipefail`。
- 风险与下一步：
  - 当前 fused predictor 只是 correctness probe，不是性能路径；debug timing 变慢是预期结果，不能作为 speedup 结论。
  - 下一步应实现真正 `CUDA3D_CORE_2STEP_FUSED_COMMIT`：同一 kernel 计算 `p(t+1)` 与 strict-inner `p(t+2)`，下一步对已提交 inner tile 做 block-level early skip，避免 shared-memory fill。
  - commit 版本必须先在 meaningful case 上通过 correctness；若 repeat speedup `<5%`，应停止 fused two-step route；若 meaningful `>=5%` 但 `perf_1gpu_6shots repeat <2%`，则保持 disabled 并报告适用性受 cropped domains 限制。

## 2026-06-07 16:49:30 +0800 - Write Pro feedback for fused debug stage and stage-4 race

- 操作目标：根据用户要求整理面向 Pro 的阶段反馈，汇报 `CUDA3D_CORE_2STEP_FUSED_INTERIOR` stage 1-3 验收结果，并指出 stage 4 不能直接把 debug helper 塞进 in-place `p_core` 的竞态风险。
- 修改文件：
  - 新增 `feedback/codex_report_20260607_164600_core2step_fused_debug.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 审查 `src/single_solver.cu` 中 `cuda_fd3d_p_core_ns` 与 `cuda_fd3d_p_core_ns_skip_region` 的 shared z-tile 加载位置和 skip 判断位置。
  - 编写反馈报告，记录 `p0` 同时作为 old field 输入与 new field 输出时，在单 kernel fused commit 中会产生无 grid sync 的 old/new 竞态。
- 测试结果：
  - 未新增编译或运行测试。
  - 反馈报告明确当前 stage 1-3 已验证通过，但 stage 4 需要先确定 shared-staged CTA tile、old-p0 read-only buffer 或其他安全设计。
- 输出/哈希/误差摘要：
  - 报告引用 commit `9115859`、meaningful case stats、p2-shift 零误差结果、输出零误差结果和 restored zmem binary hash。
- 风险与下一步：
  - 如果直接在 `p_core` 内读取 global old `p0` 计算 `p2`，同时写 global `p0(t+1)`，会有跨 CTA 读写竞态，不能作为正确实现。
  - 下一步应先拿到或制定明确 tile/shared-memory budget，再实现 `CUDA3D_CORE_2STEP_FUSED_COMMIT`。
