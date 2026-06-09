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

## 2026-06-08 00:34:22 +0800 - Start WAVESTEP Engine V2 and validate CPML velocity memory double buffer

- 操作目标：
  - 按用户提供的新 Pro 路线启动 `CUDA3D_WAVESTEP_ENGINE_V2`。
  - 从 `main` 创建新分支 `exp/wavestep-engine-v2-pml-vp-fusion`。
  - 完成 Phase 0 设计盘点文档。
  - 实现 Phase 1：全方向 CPML velocity memory double-buffer，不做 PML fusion。
  - 在 RTX 5090 服务器上完成 debug fill、debug dump、correctness、perf6 repeat gate。
- 修改文件：
  - 新增 `docs/wavestep_engine_v2_design.md`。
  - 新增 `docs/architecture_decision_log.md`。
  - 更新 `AGENTS.md`，记录 `CUDA3D_WAVESTEP_ENGINE_V2` 当前主线、Phase 1 结果和下一步 fused VP gate。
  - 更新 `include/inc3D/single_solver.h`。
  - 更新 `src/single_solver.cu`。
  - 更新 `src/rem_fd.cu`。
  - 新增本地同步报告目录：`reports/wavestep_engine_v2_phase1_cpml_vmem_20260608_003000/`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地：
    - `git checkout main`
    - `git -c http.proxy= -c https.proxy= pull --ff-only origin main`
    - `git checkout -b exp/wavestep-engine-v2-pml-vp-fusion`
  - 服务器：
    - `git checkout -B exp/wavestep-engine-v2-pml-vp-fusion origin/main`
    - 上传 `docs/wavestep_engine_v2_design.md`、`single_solver.h`、`single_solver.cu`、`rem_fd.cu`。
  - 默认 zmem 编译：
    - `-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2`
  - Phase 1 debug 编译：
    - zmem flags + `-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL -DCUDA3D_CPML_VMEM_DISABLE_MPI -DCUDA3D_CPML_VMEM_DEBUG_FILL -DCUDA3D_DEBUG_CHECKS`
  - Debug dump 编译：
    - zmem dump flags + `-DCUDA3D_PML_DEBUG_DUMP`
    - Phase1 dump flags + `-DCUDA3D_PML_DEBUG_DUMP`
  - Phase 1 release 编译：
    - zmem flags + `-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL -DCUDA3D_CPML_VMEM_DISABLE_MPI`
  - 运行：
    - zmem `smoke_1gpu`、`correctness`、`perf_1gpu_6shots`、repeat。
    - Phase1 debug `smoke_1gpu`、`correctness`。
    - zmem/Phase1 debug dump step `0/1/2`。
    - Phase1 release `smoke_1gpu`、`correctness`、`perf_1gpu`、`perf_1gpu_6shots`、repeat。
    - post A/B：重建 zmem 后再次运行 `perf_1gpu_6shots`、repeat。
  - 对比：
    - `tools/compare_outputs.py` 比较 smoke/correctness/perf6/perf6_repeat。
    - `tools/compare_debug_dumps.py` 比较 step `0/1/2` dump。
- 测试结果：
  - 默认 zmem 编译通过。
  - Phase1 debug-fill 编译通过，`smoke_1gpu` 和 `correctness` 通过；未发现 next CPML velocity memory 未写回。
  - Debug dump step `0/1/2` 全部通过。
  - Phase1 release `smoke_1gpu`、`correctness`、`perf_1gpu`、`perf_1gpu_6shots`、repeat 全部 return code `0`。
  - release 输出对比全部通过，rel L2 满足 `<=1e-5`。
- 输出/哈希/误差摘要：
  - Phase 1 report：`reports/wavestep_engine_v2_phase1_cpml_vmem_20260608_003000/phase1_report.md`
  - Phase 1 summary：`reports/wavestep_engine_v2_phase1_cpml_vmem_20260608_003000/phase1_ab_summary.json`
  - zmem pre perf6：
    - `benchmarks/runs/perf_1gpu_6shots_wavestep_v2_zmem_ref_20260608_002557`：WP `2.444596s`，Gradient `2.558980s`
    - `benchmarks/runs/perf_1gpu_6shots_wavestep_v2_zmem_ref_repeat_20260608_002602`：WP `2.455479s`，Gradient `2.565440s`
  - Phase1 release perf6：
    - `benchmarks/runs/perf_1gpu_6shots_wavestep_v2_cpml_vmem_release_20260608_002937`：WP `2.364870s`，Gradient `2.481050s`
    - `benchmarks/runs/perf_1gpu_6shots_wavestep_v2_cpml_vmem_release_repeat_20260608_002943`：WP `2.366573s`，Gradient `2.487687s`
  - zmem post perf6：
    - `benchmarks/runs/perf_1gpu_6shots_wavestep_v2_zmem_post_ab_20260608_003136`：WP `2.431655s`，Gradient `2.545308s`
    - `benchmarks/runs/perf_1gpu_6shots_wavestep_v2_zmem_post_ab_repeat_20260608_003142`：WP `2.439698s`，Gradient `2.552431s`
  - A/B 汇总：
    - Phase1 mean WP：`2.3657215s`
    - zmem all mean WP：`2.442857s`
    - speedup vs all zmem WP：`1.032605x`
    - speedup vs all zmem Gradient：`1.028648x`
    - gate：`continue`
  - Phase1 release binary SHA256：`0749563fd944e6a275de4ffb6ef63822ec710551b7da520acdbb41b90d2eaee8`
  - 服务器最终恢复 zmem binary SHA256：`ad99db7cb09ed2b223607ef06df05589b388d411bfb5bd711c598db45ee0a195`
- 风险与下一步：
  - Phase1 是 ownership scaffold，默认关闭；不要把它误认为最终 fused VP。
  - Phase1 在本次 A/B 中有约 `3.26%` WP 正收益，但仍需后续更多 repeat 确认稳定性。
  - 下一步允许进入 `CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY`。
  - fused VP 必须真正消除 fused region 内 `vx/vy/vz` global write/read round trip，禁止 pressure-only split。
  - pure z-face gate：meaningful case repeat speedup `>=10%`，`perf_1gpu_6shots repeat >=5%`，否则停止 PML fused VP。

## 2026-06-08 01:08:00 +0800 - Test and stop direct PML fused VP z-face

- 操作目标：
  - 继续 `CUDA3D_WAVESTEP_ENGINE_V2` Phase 2。
  - 实现并测试 `CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY`。
  - 验证是否能通过 correctness 和 `perf_1gpu_6shots repeat >=5%` gate。
- 修改文件：
  - 更新 `include/inc3D/single_solver.h`。
  - 更新 `src/single_solver.cu`。
  - 更新 `src/rem_fd.cu`。
  - 新增 `reports/wavestep_engine_v2_phase2_fused_zface_20260608_010000/phase2_fused_zface_report.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 更新 `docs/wavestep_engine_v2_design.md`。
  - 更新 `AGENTS.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地 `git diff --check`。
  - 上传 `single_solver.cu`、`rem_fd.cu`、`single_solver.h` 到 `/work/wenzhe/cuda3D`。
  - 编译默认 zmem flags，确认宏关闭路径可编译。
  - 编译 fused debug/release flags：
    - zmem flags
    - `-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`
    - `-DCUDA3D_CPML_VMEM_DISABLE_MPI`
    - `-DCUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY`
  - 运行 fused separate-kernel 版：
    - `smoke_1gpu`
    - `correctness`
    - `perf_1gpu_6shots`
    - `perf_1gpu_6shots` repeat
  - 运行 fused inline 版：
    - `smoke_1gpu`
    - `correctness`
    - `perf_1gpu_6shots`
    - `perf_1gpu_6shots` repeat
  - 重建 zmem 并同机复跑 `perf_1gpu_6shots`、repeat，最终服务器 binary 恢复为 zmem release。
  - 使用 `tools/compare_outputs.py` 对比 smoke/correctness/perf6 repeat 输出。
- 测试结果：
  - 默认 zmem 编译通过。
  - fused separate-kernel 版编译通过；smoke/correctness/perf6 repeat 输出对比均通过。
  - fused inline 版编译通过；smoke/correctness/perf6 repeat 输出对比均通过。
  - 两个 fused 版本都未达到性能 gate，均慢于同机 zmem。
- 输出/哈希/误差摘要：
  - separate fused binary SHA256：`63531df023ebab0bf8104eafe4b420658f6eb1cfd0801b6cbe3d5d4dd4adad4a`
  - inline fused binary SHA256：`c88b2acf88025f7796288603250d3f63749a2af8b548449af9b1373507e1cff9`
  - final restored zmem binary SHA256：`c768270c431b3922f803fc787b1eaaffdc8967b072dfc5f74f30c0a94bf459e5`
  - same-session zmem perf6:
    - `20260608_010619`：WP `2.432802s`，Gradient `2.545055s`
    - repeat `20260608_010625`：WP `2.436119s`，Gradient `2.550727s`
    - mean WP `2.434461s`
  - separate fused perf6:
    - `20260608_005956`：WP `2.656186s`，Gradient `2.771636s`
    - repeat `20260608_010002`：WP `2.663968s`，Gradient `2.776977s`
    - mean WP `2.660077s`，speed ratio vs zmem `0.915184x`
  - inline fused perf6:
    - `20260608_010449`：WP `2.691287s`，Gradient `2.812600s`
    - repeat `20260608_010455`：WP `2.693871s`，Gradient `2.817426s`
    - mean WP `2.692579s`，speed ratio vs zmem `0.904137x`
  - perf6 repeat 最大 rel L2：`6.358816e-07`，满足 `<=1e-5`。
- 风险与下一步：
  - 直接用 p1 x/y second derivatives 替代 `vx/vy` global round trip 的 z-face fusion 已停止。
  - 后续不要重复 separate zface kernel 或 inline p_pml direct recompute。
  - 只有新设计使用 CTA-local shared-memory velocity intermediates，或有新的 NCU 证据说明总 memory stall 下降，才允许重开 PML z-face fusion。

## 2026-06-08 02:12:39 +08:00 - WAVESTEP V2 shared VP night sprint

- 操作目标：
  - 按 Pro 路线继续推进 `/work/wenzhe/cuda3D` 上的 CUDA 性能优化。
  - 固化 zmem baseline，执行 NCU forensic、CPML double-buffer 复测、z-face shared VP 预算与 prototype 验收。
  - 所有候选必须对比 `zmem_reference`，并在失败后恢复服务器 binary。
- 修改文件：
  - 新增 `tools/pml_zface_shared_tile_budget.py`。
  - 新增 `tools/ncu_csv_summary.py`。
  - 更新 `include/inc3D/cu_common.h`。
  - 更新 `include/inc3D/single_solver.h`。
  - 更新 `src/single_solver.cu`。
  - 更新 `src/rem_fd.cu`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 新增 `docs/wavestep_v2/pml_zface_shared_tile_budget.md`。
  - 新增 `docs/wavestep_v2/pml_zface_shared_vp_design.md`。
  - 新增 `docs/wavestep_v2/cpml_vmem_double_buffer_all_result.md`。
  - 新增 `docs/wavestep_v2/ncu_forensics_summary.md`。
  - 新增 `docs/wavestep_v2/phase2_fused_zface_forensics.md`。
  - 新增 `docs/wavestep_v2/shared_vp_debug_result.md`。
  - 新增 `docs/wavestep_v2/post_shared_vp_fallback_plan.md`。
  - 新增 `reports/wavestep_v2_night_20260608/final_report.md`。
  - 新增 `reports/wavestep_v2_night_20260608/final_summary.json`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地 `git checkout -B exp/wavestep-v2-shared-vp-night`。
  - 远端 `git checkout -B exp/wavestep-v2-shared-vp-night`。
  - 远端构建 zmem baseline：
    - `-O3 -arch=sm_120 --use_fast_math`
    - `-DCUDA3D_PML_RECOMPUTE_Z`
    - `-DCUDA3D_PML_TILE_LIST`
    - `-DCUDA3D_PML_ZMEM_IN_P`
    - `-DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2`
  - 远端运行：
    - `smoke_1gpu`
    - `correctness`
    - `perf_1gpu_6shots` x2
  - 远端构建并运行 `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` release A/B。
  - 远端使用 Nsight Compute 2025.3.0 对 zmem、CPML double-buffer、direct inline fused zface 执行短 profile。
  - 远端构建并运行 `CUDA3D_PML_ZFACE_SHARED_VP_DEBUG`：
    - S2 p-only：`12x16x12`，smem `81120`。
    - S4 p-only：`12x12x12`，smem `70304`。
    - S4 staged-V：`12x12x12`，smem `92192`。
  - 使用 `tools/compare_outputs.py` 对 smoke/correctness/perf 输出进行 rel L2 对比。
  - 远端最终重建 zmem 并运行 final smoke。
- 测试结果：
  - zmem baseline 复跑通过。
  - CPML double-buffer correctness 通过，保留为 scaffold。
  - NCU forensic 完成，direct inline fused zface 的 `p_pml_tile` sampled duration 从 `188.856us` 增至 `248.200us`。
  - S2 p-only shared VP correctness 通过，但性能失败。
  - S4 p-only shared VP correctness 通过，但性能失败。
  - S4 staged-V shared VP correctness 通过，但性能失败。
  - 服务器最终 binary 已恢复为 zmem，final smoke 通过。
- 输出/哈希/误差摘要：
  - zmem same-session mean：WP `2.448577s`，Gradient `2.560774s`。
  - CPML double-buffer mean：WP `2.369180s`，speed `1.033512x`；Gradient `2.486222s`，speed `1.029986x`；rel L2 `0`。
  - S2 p-only shared VP mean：WP `3.007605s`，speed `0.814129x`；Gradient `3.169875s`，speed `0.807847x`；rel L2 `0`。
  - S4 p-only shared VP：WP `3.039426s`，speed `0.805605x`；Gradient `3.188930s`，speed `0.803020x`；rel L2 `0`。
  - S4 staged-V shared VP mean：WP `3.090552s`，speed `0.792278x`；Gradient `3.236344s`，speed `0.791255x`；rel L2 `0`。
  - S2 binary SHA256：`143a3a19fa7e57ddadb0c1cb80b10397c7e2b4b6263df2723c5f621f7ac7b324`。
  - S4 binary SHA256：`e7cdd11d3d0de5654d836679dbef5242b0adc6232deaa2b28a0b0f570c960ef4`。
  - S4 staged-V binary SHA256：`288103e236d3c4bba160073f372a2b9bc61cf6486fcf8c439d8b094dd3e2202b`。
  - final restored zmem SHA256：`0e54c4938ea60bdb606fa67e450a5fc992b71ffe5823d2238a1414e0f30e9d6d`。
- 风险与下一步：
  - `CUDA3D_PML_ZFACE_SHARED_VP_DEBUG` 当前形态已停止；禁止继续重复 S2/S4 p-only 或 S4 staged-V。
  - 新增 shared VP 代码全部默认关闭，仅保留为 traceable failed prototype。
  - 后续建议保留 CPML double-buffer scaffold，转向 PML compact-state audit 或 global-region temporal pipeline 的 byte-budget/profiler 设计。

## 2026-06-08 09:50:00 +08:00 - Day sprint Phase 0 zmem baseline rebuild

- 操作目标：
  - 根据 2026-06-08 白天 sprint 要求，建立同机、同日、同 flags 的 zmem 对照。
  - 避免后续 CPML double-buffer A/B 测试复用错误 binary。
- 修改文件：
  - 追加本 `AGENT_LOG.md` 条目。
  - 远端新增报告目录：`reports/day_20260608`、`docs/day_20260608`、`benchmarks/profiles/day_20260608`、`benchmarks/build_logs/day_20260608`。
- 执行命令摘要：
  - 本地创建工作分支：`git checkout -B exp/day-20260608-cpml-compact-temporal`。
  - 远端记录 sprint 前状态：`reports/day_20260608/git_status_before_day_sprint_20260608.txt`。
  - 远端构建 zmem：
    - `-O3 -arch=sm_120 --use_fast_math`
    - `-DCUDA3D_PML_RECOMPUTE_Z`
    - `-DCUDA3D_PML_TILE_LIST`
    - `-DCUDA3D_PML_ZMEM_IN_P`
    - `-DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2`
  - 远端运行：
    - `python3 tools/run_benchmark.py --case smoke_1gpu --tag day_zmem_phase0_smoke`
    - `python3 tools/run_benchmark.py --case correctness --tag day_zmem_phase0_correctness`
- 测试结果：
  - zmem 编译通过。
  - `smoke_1gpu` 通过，输出 3 个 `.dir`。
  - `correctness` 通过，输出 6 个 `.dir`。
- 输出/哈希/误差摘要：
  - zmem binary SHA256：`23760de8a255cccb133d7c54657ee8cb77407031f4ae382ea681fbf8fbb8d754`。
  - smoke run：`benchmarks/runs/smoke_1gpu_day_zmem_phase0_smoke_20260608_094936`。
  - correctness run：`benchmarks/runs/correctness_day_zmem_phase0_correctness_20260608_094940`。
- 风险与下一步：
  - 远端 worktree 保留夜间实验改动与测试工件，今天不执行清理或 reset。
  - 继续 Phase 1：每个 zmem/CPML perf6 run 前显式重建对应 binary，完成三轮 A/B。

## 2026-06-08 09:58:00 +08:00 - Day sprint Phase 1 CPML double-buffer revalidation

- 操作目标：
  - 复验 `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` 在 RTX 5090 平台上是否有可重复小收益。
  - 使用同机同日三轮 zmem/CPML A/B，且每次候选运行前重新编译对应 binary，避免混用可执行文件。
- 修改文件：
  - 追加本 `AGENT_LOG.md` 条目。
  - 远端新增构建日志：`benchmarks/build_logs/day_20260608/*_perf6_*_build.log`。
  - 远端新增汇总：`reports/day_20260608/cpml_dbuf_perf6_ab_summary.json`、`reports/day_20260608/cpml_dbuf_perf6_ab_paths.tsv`。
  - 远端新增 correctness/perf 对比目录：`reports/day_20260608/cpml_dbuf_phase1_correctness_vs_zmem`、`reports/day_20260608/cpml_dbuf_perf6_{a,b,c}_vs_zmem`。
- 执行命令摘要：
  - 构建 CPML double-buffer：
    - `-O3 -arch=sm_120 --use_fast_math`
    - `-DCUDA3D_PML_RECOMPUTE_Z`
    - `-DCUDA3D_PML_TILE_LIST`
    - `-DCUDA3D_PML_ZMEM_IN_P`
    - `-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`
    - `-DCUDA3D_CPML_VMEM_DISABLE_MPI`
    - `-DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2`
  - 远端运行 CPML smoke/correctness/perf_1gpu。
  - 远端运行三轮 `perf_1gpu_6shots` A/B：`day_zmem_perf6_{a,b,c}` 与 `day_cpml_dbuf_perf6_{a,b,c}`。
  - 使用 `tools/compare_outputs.py` 对 correctness 与每轮 perf6 输出做 rel L2 对比。
- 测试结果：
  - CPML smoke/correctness/perf_1gpu 均通过。
  - CPML correctness vs zmem 通过。
  - 三轮 perf6 输出对比均通过。
  - Phase 1 gate 通过：平均 WP speedup `1.0323x`，平均 Gradient speedup `1.0284x`，所有单轮均高于 `1.015x`。
- 输出/哈希/误差摘要：
  - CPML phase1 binary SHA256：`4654be0284377da9fcb046836fd4379f2bcb57a5a5de9dfdb9b68e92f4d4dfc6`。
  - Round a：zmem WP `2.456481s`，CPML WP `2.379958s`，WP speedup `1.032153x`；Gradient speedup `1.028239x`。
  - Round b：zmem WP `2.422000s`，CPML WP `2.353140s`，WP speedup `1.029263x`；Gradient speedup `1.027842x`。
  - Round c：zmem WP `2.426137s`，CPML WP `2.342810s`，WP speedup `1.035567x`；Gradient speedup `1.029032x`。
  - all-mean：zmem WP `2.434873s`，CPML WP `2.358636s`，WP speedup `1.032329x`。
  - all-mean：zmem Gradient `2.550490s`，CPML Gradient `2.480129s`，Gradient speedup `1.028370x`。
- 风险与下一步：
  - CPML double-buffer 是稳定小收益，可保留为后续 ownership scaffold。
  - 继续 Phase 2：做 PML compact-state byte budget 与 NCU 短 profile；没有 `>=5%` 理论/实测意义收益则不进入 compact prototype。

## 2026-06-08 10:10:00 +08:00 - Day sprint Phase 2 compact-state audit and gate

- 操作目标：
  - 根据白天 sprint 要求，判断 PML compact-state 是否值得进入 CUDA prototype。
  - 用静态 byte budget 和 NCU 短 profile 避免继续做低收益 micro-prototype。
- 修改文件：
  - 新增 `tools/pml_state_traffic_audit.py`。
  - 新增 `docs/day_20260608/cpml_vmem_dbuf_revalidation.md`。
  - 新增 `docs/day_20260608/pml_compact_state_audit.md`。
  - 新增 `docs/day_20260608/pml_compact_state_audit_static.md`。
  - 新增 `docs/day_20260608/pml_state_ncu_summary.md`。
  - 新增 `docs/day_20260608/phase2_compact_state_gate_decision.md`。
  - 新增 `reports/day_20260608/cpml_vmem_dbuf_summary.json`。
  - 新增 `reports/day_20260608/cpml_dbuf_perf6_ab_summary.json`。
  - 新增 `reports/day_20260608/pml_compact_state_audit.json`。
  - 新增 `reports/day_20260608/pml_compact_state_audit_static.json`。
  - 新增 `reports/day_20260608/pml_state_ncu_summary.json`。
  - 新增 `reports/day_20260608/phase2_compact_state_gate_summary.json`。
  - 新增 `benchmarks/profiles/day_20260608/zmem_pml_state_ncu.csv`。
  - 新增 `benchmarks/profiles/day_20260608/cpml_dbuf_pml_state_ncu.csv`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地运行：`python tools/pml_state_traffic_audit.py --case benchmarks/cases/perf_1gpu_6shots --variant cpml_dbuf`。
  - 上传 `tools/pml_state_traffic_audit.py` 到 `/work/wenzhe/cuda3D/tools/`。
  - 远端运行同一 audit，生成 `docs/day_20260608/pml_compact_state_audit*.md/json`。
  - 远端分别重建 zmem 与 CPML double-buffer，并使用 Nsight Compute 2025.3.0 profile：
    - `--section SpeedOfLight`
    - `--section MemoryWorkloadAnalysis`
    - `--section SourceCounters`
    - `--section WarpStateStats`
    - `--launch-skip 10`
    - `--launch-count 12`
    - kernel filter：`regex:.*cuda_fd3d_[vp]_pml_tile.*`
  - 远端运行 `tools/ncu_csv_summary.py` 生成 NCU 汇总。
  - 远端最终重建 zmem 并运行 final smoke。
- 测试结果：
  - `tools/pml_state_traffic_audit.py` 本地 `py_compile` 通过。
  - zmem NCU CSV 生成，423 行。
  - CPML NCU CSV 生成，423 行。
  - compact-state gate 失败，停止 compact-state CUDA prototype。
  - 远端最终 zmem restore smoke 通过。
- 输出/哈希/误差摘要：
  - CPML state footprint：`72.391 MiB`。
  - six padded wavefield/cw2 array floor：`503.039 MiB`。
  - safe z-face compact share of `memory_dz`：`84.93%`。
  - residual z edge/corner elements：`602112`。
  - mandatory CPML state update traffic floor：`96.521 MiB/step`。
  - zmem `memory_dz` old reads from recompute path：`111.762 MiB/step`。
  - pressure PML vx/vy load estimate：`458.344 MiB/step`。
  - estimated compact-state WP speedup ceiling：`1.005x`。
  - NCU `cuda_fd3d_p_pml_tile_ns` duration：zmem `189.840us`，CPML `190.293us`。
  - NCU `cuda_fd3d_v_pml_tile_ns` duration：zmem `71.493us`，CPML `66.000us`。
  - restored zmem binary SHA256：`4083a1f39428e2bbb0f204e330c409dd55c135ffd088441c24d331288ed1ad7e`。
  - restored zmem smoke run：`benchmarks/runs/smoke_1gpu_day_zmem_restore_after_phase2_smoke_20260608_100828`。
- 风险与下一步：
  - 不实现 `CUDA3D_PML_COMPACT_STATE_DEBUG_MIRROR` 或 `CUDA3D_PML_COMPACT_ZFACE_STATE`，除非新 profiler 证明 CPML state layout 是主瓶颈且 byte model 预测 `>=5%` WP speedup。
  - 下一步进入 global-region temporal pipeline 设计/原型阶段；继续禁止重复 z-face fusion、shared zface VP、RECOMPUTE_X/Y/XYZ、block/register sweep 等已失败路线。

## 2026-06-08 10:18:00 +08:00 - Day sprint Phase 4 temporal pipeline design gate

- 操作目标：
  - 在 compact-state gate 失败后，按要求进入更大粒度的 global-region temporal pipeline 方向。
  - 先补主核 profile 与 byte/synchronization 设计门，不直接写可能跨 CTA 读半更新数据的不安全 two-step kernel。
- 修改文件：
  - 新增 `docs/day_20260608/zmem_core_pml_sol_ncu_summary.md`。
  - 新增 `docs/day_20260608/global_temporal_pipeline_phase4_design.md`。
  - 新增 `reports/day_20260608/zmem_core_pml_sol_ncu_summary.json`。
  - 新增 `reports/day_20260608/phase4_global_temporal_pipeline_design_summary.json`。
  - 新增 `benchmarks/profiles/day_20260608/zmem_core_pml_sol_ncu.csv`。
  - 更新 `AGENTS.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端在已恢复的 zmem binary 上运行 Nsight Compute 短 profile：
    - `--section SpeedOfLight`
    - `--launch-skip 10`
    - `--launch-count 30`
    - kernel filter：`regex:.*cuda_fd3d_(p_core|v_pml_tile|p_pml_tile).*`
  - 使用 `tools/ncu_csv_summary.py` 生成 summary。
- 测试结果：
  - NCU CSV 生成并成功汇总。
  - Phase 4 设计门已打开，但没有写 CUDA prototype。
- 输出/哈希/误差摘要：
  - `cuda_fd3d_p_pml_tile_ns` duration：`189.562us`，约占 sampled main kernels `53.43%`。
  - `cuda_fd3d_p_core_ns` duration：`93.670us`，约占 `26.40%`，Memory SOL `96.810%`。
  - `cuda_fd3d_v_pml_tile_ns` duration：`71.610us`，约占 `20.18%`。
  - sampled main-kernel total：`354.842us`。
  - sampled main kernels 若要达到 `>=5%` speedup，需要节省约 `16.897us`。
  - 如果只优化 `p_core`，需要至少 `18.04%` 的 `p_core` reduction。
  - K=2 deep-core temporal geometry 约覆盖原 pressure core `77.7%`；K=3 约 `58.1%`。
- 风险与下一步：
  - 普通 CUDA kernel 没有 grid-wide barrier；不能实现会跨 CTA 读取半更新 `p(t+1)` 的 fused two-step kernel。
  - 下一步应先写 K=2 deep-core temporal byte/synchronization model，再决定是否进入 `CUDA3D_WAVESTEP_ENGINE_V2_TEMPORAL_PIPELINE` prototype。

## 2026-06-08 10:45:00 +08:00 - Day sprint Phase 4.1 temporal byte/sync model

- 操作目标：
  - 按 Pro 指示继续推进 Phase 4：先完成 K=2 deep-core temporal pipeline 的 byte/synchronization model。
  - 判断是否可以进入直接 CUDA prototype。
- 修改文件：
  - 新增 `tools/temporal_pipeline_model.py`。
  - 新增 `docs/day_20260608/temporal_pipeline_model.md`。
  - 新增 `docs/day_20260608/phase4_1_temporal_model_gate_decision.md`。
  - 新增 `reports/day_20260608/temporal_pipeline_model.json`。
  - 新增 `reports/day_20260608/phase4_1_temporal_model_gate_summary.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地运行：`python -m py_compile tools/temporal_pipeline_model.py`。
  - 本地运行：`python tools/temporal_pipeline_model.py --case benchmarks/cases/perf_1gpu_6shots --json-out reports/day_20260608/temporal_pipeline_model.json --md-out docs/day_20260608/temporal_pipeline_model.md`。
  - 上传 `tools/temporal_pipeline_model.py` 到 `/work/wenzhe/cuda3D/tools/`。
  - 远端运行同一模型，生成正式远端路径报告。
  - 拉回远端 `docs/day_20260608/temporal_pipeline_model.md` 与 `reports/day_20260608/temporal_pipeline_model.json`。
- 测试结果：
  - 模型工具本地 `py_compile` 通过。
  - 远端模型运行通过。
  - Gate 结果：`stop_cuda_prototype`。
- 输出/哈希/误差摘要：
  - pressure core：`87 x 376 x 376`，`12299712` points。
  - K=2 deep core：`73 x 362 x 362`，coverage `77.78%`。
  - current p_core bytes/output estimate：`128.438`。
  - current p_core bytes/core step estimate：`1506.562 MiB`。
  - ideal K=2 saved bytes/pair：`1062.265 MiB`。
  - ideal K=2 p_core pair reduction upper bound：`35.25%`。
  - ideal K=2 sampled-main speedup upper bound：`1.103x`。
  - cooperative grid blocks required：`70688`。
  - conservative resident block capacity assumption：`1360`。
  - cooperative over-capacity factor：`51.98x`。
  - CTA-local candidate local pair bytes / baseline：`11.29x` 到 `21.30x`，计入 p_mid halo duplication 后全部失败。
- 风险与下一步：
  - 不写 direct K=2 fused CUDA kernel；safe global-middle 设计无 meaningful speedup，cooperative grid-sync 不可行，CTA-local p_mid reuse 计入 halo duplication 后字节模型失败，且仍属于禁止的 two-step 家族。
  - 下一步只能进入 `Phase 4.2 source-aware swept/wavefront temporal design`，先解决 p_mid ownership、source injection、intermediate receiver extraction、shell/PML reconciliation 和 half-updated value 依赖证明。

## 2026-06-08 11:20:00 +08:00 - Day sprint Phase 4.2 source-aware temporal gate

- 操作目标：
  - 继续 Phase 4.2，检查实际 source/receiver 布局是否阻止 K=2 deep-core temporal pipeline。
  - 复现 shot-local y/x 子域裁剪，避免用全域几何高估 temporal 覆盖率。
- 修改文件：
  - 新增 `tools/source_aware_temporal_model.py`。
  - 新增 `docs/day_20260608/source_aware_temporal_model.md`。
  - 新增 `docs/day_20260608/phase4_2_source_aware_temporal_gate_decision.md`。
  - 新增 `reports/day_20260608/source_aware_temporal_model.json`。
  - 新增 `reports/day_20260608/phase4_2_source_aware_temporal_gate_summary.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地运行：`python -m py_compile tools/source_aware_temporal_model.py`。
  - 本地运行：`python tools/source_aware_temporal_model.py --case benchmarks/cases/perf_1gpu_6shots --json-out reports/day_20260608/source_aware_temporal_model.json --md-out docs/day_20260608/source_aware_temporal_model.md`。
  - 上传 `tools/source_aware_temporal_model.py` 到 `/work/wenzhe/cuda3D/tools/`。
  - 远端复跑同一模型并拉回远端路径版报告。
- 测试结果：
  - 模型工具本地与远端 `py_compile` 通过。
  - 远端 source-aware model 运行通过。
  - Gate 结果：`stop_swept_wavefront_cuda_prototype`。
- 输出/哈希/误差摘要：
  - shot-local aggregate K=2 deep-core share：`73.22%`。
  - source influence overlaps K=2 deep core：`0` shots。
  - receiver footprint overlaps K=2 deep core：`0` shots。
  - shot domains：
    - shot 0：`216 x 216`，K2 share `72.99%`。
    - shot 1：`216 x 241`，K2 share `73.56%`。
    - shot 2：`216 x 217`，K2 share `73.02%`。
    - shot 3：`217 x 216`，K2 share `73.02%`。
    - shot 4：`217 x 241`，K2 share `73.58%`。
    - shot 5：`217 x 217`，K2 share `73.04%`。
- 风险与下一步：
  - Source/receiver placement 不阻止 temporal blocking，但 `p(t+1)` ownership/synchronization 与 halo duplication 仍失败。
  - 当前停止 swept/wavefront temporal CUDA prototype。
  - 下一自主方向转向 dominant `cuda_fd3d_p_pml_tile_ns` 的 pressure PML dataflow 或 wave-step scheduling。

## 2026-06-08 11:05:10 +08:00 - Day sprint Phase 4.3 pressure PML dataflow gate

- 操作目标：
  - 在 temporal K=2 路线 gate 失败后，转向当前 sampled-main 最大热点 `cuda_fd3d_p_pml_tile_ns`。
  - 复现 pressure PML tile list 和 shot-local 子域，量化 shell 点、tile/thread 效率和 z 方向中间速度重算复用上限。
- 修改文件：
  - 新增 `tools/pml_pressure_dataflow_audit.py`。
  - 新增 `docs/day_20260608/pml_pressure_dataflow_audit.md`。
  - 新增 `docs/day_20260608/phase4_3_pressure_pml_dataflow_gate_decision.md`。
  - 新增 `reports/day_20260608/pml_pressure_dataflow_audit.json`。
  - 新增 `reports/day_20260608/pml_pressure_dataflow_audit.md`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地运行：`python -m py_compile tools/pml_pressure_dataflow_audit.py`。
  - 本地运行：`python tools/pml_pressure_dataflow_audit.py --json-out reports/day_20260608/pml_pressure_dataflow_audit.json --md-out docs/day_20260608/pml_pressure_dataflow_audit.md`。
  - 上传 `tools/pml_pressure_dataflow_audit.py` 到远端 `/work/wenzhe/cuda3D/tools/`。
  - 远端运行同一 audit：
    - `python3 -m py_compile tools/pml_pressure_dataflow_audit.py`
    - `python3 tools/pml_pressure_dataflow_audit.py --json-out reports/day_20260608/pml_pressure_dataflow_audit.json --md-out reports/day_20260608/pml_pressure_dataflow_audit.md`
- 测试结果：
  - 本地 `py_compile` 通过。
  - 本地 audit 运行通过。
  - 远端 `py_compile` 通过。
  - 远端 audit 运行通过。
  - Gate 结果：`open_p_pml_z_recompute_line_cache_prototype`。
- 输出/哈希/误差摘要：
  - pressure PML tiles：`113840 / 181232`。
  - active thread efficiency：`65.60%`。
  - valid-domain thread efficiency：`87.32%`。
  - returned-core threads in kept tiles：`6328998`。
  - shell active points：`4143640`，占 active points `21.67%`。
  - true-PML active points：`14975304`，占 active points `78.33%`。
  - 当前 `recompute_vz_after_update` calls：`152951552`。
  - shared z-line cache calls estimate：`29093740`。
  - estimated z recompute call reduction：`80.98%`。
  - current p1 loads inside z recompute：`4667.711 MiB/step aggregate-shots`。
  - shared-cache p1 load estimate：`887.870 MiB/step aggregate-shots`。
  - NCU-linked p_pml sampled-main share：`53.42%`。
  - modeled p_pml speedup：`1.573x`。
  - modeled sampled-main speedup：`1.242x`。
  - 远端复现：gate `open_p_pml_z_recompute_line_cache_prototype`，model speedup `1.2417261903808379`，shell sum `4143640 == 4143640`。
- 风险与下一步：
  - 允许进入 `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE` prototype，macro 默认关闭。
  - 必须保持 `memory_dz_next` ownership：只有 tile-owned central z range 可以写 next z CPML memory。
  - 不重开 `CUDA3D_PML_TILE_MASK_FASTPATH`、z-face specialize/fusion、`CUDA3D_PML_ZFACE_SHARED_VP_DEBUG` 或 `RECOMPUTE_X/Y/XYZ`。
  - 下一步实现 pressure PML z-line cache 原型，先跑 debug dump step 0/1/2 和 correctness，再跑 `perf_1gpu_6shots repeat`。

## 2026-06-08 11:40:41 +08:00 - Day sprint Phase 4.4 pressure PML z-recompute cache prototype

- 操作目标：
  - 实现 `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`，复用 pressure PML tile 内 z-line 的 `vz_after_update` 中间值。
  - 测试 standalone z-cache、与 Phase 1 CPML vmem scaffold 的组合效果，并验证 aggressive vx/vy cache 是否可行。
- 修改文件：
  - `src/single_solver.cu`
    - 在 `cuda_fd3d_p_pml_tile_ns` 中新增宏默认关闭的 shared z-line cache。
    - 保持 `memory_dz_next` ownership：仅 tile-owned active central z positions 写 next z CPML memory。
    - 未保留已失败的 shared `vx/vy` cache 代码。
  - 新增 `docs/day_20260608/pressure_pml_zrecomp_cache_prototype.md`。
  - 新增/拉回远端报告：
    - `reports/day_20260608/zrecomp_cache_v2_correctness_comparison.md`
    - `reports/day_20260608/zrecomp_cache_v2_perf6_repeat_summary.md`
    - `reports/day_20260608/zrecomp_cache_v2_perf6_repeat_summary.json`
    - `reports/day_20260608/zrecomp_cache_v3_failed_perf6_repeat_summary.md`
    - `reports/day_20260608/zrecomp_cache_cpml_combo_debug_step0_comparison.md`
    - `reports/day_20260608/zrecomp_cache_cpml_combo_debug_step1_comparison.md`
    - `reports/day_20260608/zrecomp_cache_cpml_combo_debug_step2_comparison.md`
    - `reports/day_20260608/zrecomp_cache_cpml_combo_correctness_comparison.md`
    - `reports/day_20260608/zrecomp_cache_cpml_combo_perf6_repeat_summary.md`
    - `reports/day_20260608/zrecomp_cache_cpml_combo_perf6_repeat_summary.json`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 为避免污染远端原 dirty worktree，在 `/work/wenzhe/cuda3D` 下 `git worktree add /work/wenzhe/cuda3D_codex_day_20260608 origin/exp/day-20260608-cpml-compact-temporal`。
  - 由于 clean worktree 缺少 perf 大输入，非破坏性新增：
    - symlink：`benchmarks/cases/perf_1gpu_6shots/vel_perf_1gpu_6shots_ny384_nx384_nz95.dir -> /work/wenzhe/cuda3D/benchmarks/cases/perf_1gpu_6shots/vel_perf_1gpu_6shots_ny384_nx384_nz95.dir`
    - output dir：`benchmarks/cases/perf_1gpu_6shots/d_obs`
  - 编译 flags：
    - baseline：zmem reference flags。
    - standalone z-cache：baseline + `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`。
    - combo：standalone + `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` + `CUDA3D_CPML_VMEM_DISABLE_MPI`。
  - 运行：
    - standalone z-cache debug dump step `0/1/2`。
    - standalone z-cache correctness。
    - standalone z-cache `perf_1gpu_6shots` repeat 3 轮 A/B。
    - aggressive shared `vx/vy` cache correctness/perf repeat，用于判定是否保留。
    - combo debug dump step `0/1/2`。
    - combo correctness。
    - combo `perf_1gpu_6shots` repeat 3 轮 A/B。
- 测试结果：
  - standalone z-cache：
    - debug dump step `0/1/2` 通过。
    - correctness 通过，6 个输出 rel L2 全部 `0`。
    - perf repeat 输出对比全部通过。
    - mean WP speedup：`1.044955x`。
    - mean Gradient speedup：`1.045506x`。
  - aggressive shared `vx/vy` cache：
    - correctness 通过，但性能灾难性退化。
    - mean WP speedup：`0.419906x`。
    - mean Gradient speedup：`0.426565x`。
    - 结论：已从代码移除，禁止继续。
  - combo candidate：
    - debug dump step `0/1/2` 通过。
    - correctness 通过，6 个输出 rel L2 全部 `0`。
    - perf repeat 3 轮输出对比全部通过。
    - mean WP speedup：`1.083390x`。
    - mean Gradient speedup：`1.080857x`。
- 输出/哈希/误差摘要：
  - combo round 1：WP `2.435633 -> 2.249627`，speedup `1.082683x`；Gradient `2.545943 -> 2.357701`，speedup `1.079841x`。
  - combo round 2：WP `2.413101 -> 2.227910`，speedup `1.083123x`；Gradient `2.533939 -> 2.346707`，speedup `1.079785x`。
  - combo round 3：WP `2.416663 -> 2.228645`，speedup `1.084364x`；Gradient `2.542785 -> 2.348029`，speedup `1.082944x`。
  - combo correctness：6 个 `.dir` 输出全部 rel L2 `0`，无 missing/extra。
- 风险与下一步：
  - combo 当前只验证 single GPU / single MPI rank；`CUDA3D_CPML_VMEM_DISABLE_MPI` 仍限制 MPI 场景。
  - 下一步应 profile combo candidate，确认剩余 `cuda_fd3d_p_pml_tile_ns` latency 与 occupancy/source counters。
  - 不继续 shared `vx/vy` cache、tile-mask fastpath、z-face specialize/fusion 或 `RECOMPUTE_X/Y/XYZ`。

## 2026-06-08 11:48:00 +08:00 - Combo candidate NCU short profile

- 操作目标：
  - 对已通过 gate 的 combo candidate 做 Nsight Compute 短 profile。
  - 判断收益来源和下一步剩余瓶颈。
- 修改文件：
  - 新增 `docs/day_20260608/zrecomp_cache_cpml_combo_ncu_summary.md`。
  - 新增 `reports/day_20260608/zrecomp_cache_cpml_combo_ncu_summary.json`。
  - 新增 `benchmarks/profiles/day_20260608/zmem_vs_combo_zmem_ncu.csv`。
  - 新增 `benchmarks/profiles/day_20260608/zmem_vs_combo_combo_ncu.csv`。
  - 更新 `docs/day_20260608/pressure_pml_zrecomp_cache_prototype.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端在 clean worktree `/work/wenzhe/cuda3D_codex_day_20260608` 重建 zmem baseline 与 combo candidate。
  - 使用 Nsight Compute：
    - `--section SpeedOfLight`
    - `--section MemoryWorkloadAnalysis`
    - `--section SchedulerStats`
    - `--section WarpStateStats`
    - `--section Occupancy`
    - `--launch-skip 10`
    - `--launch-count 30`
    - kernel filter：`regex:.*cuda_fd3d_(p_core|v_pml_tile|p_pml_tile).*`
  - 使用 `tools/ncu_csv_summary.py` 生成 markdown/json summary。
- 测试结果：
  - zmem 与 combo NCU CSV 均生成成功。
  - summary 生成成功。
- 输出/哈希/误差摘要：
  - `cuda_fd3d_p_core_ns` duration：zmem `76.061us`，combo `75.306us`，基本不变。
  - `cuda_fd3d_p_pml_tile_ns` duration：zmem `158.291us`，combo `142.902us`，kernel speedup `1.108x`。
  - `cuda_fd3d_v_pml_tile_ns` duration：zmem `58.320us`，combo `53.101us`，kernel speedup `1.098x`。
  - combo `p_pml_tile`：
    - eligible warps/scheduler：`0.798`。
    - No Eligible：`60.879%`。
    - achieved occupancy：`75.965%`。
    - block limit registers：`5`，block limit shared mem：`7`。
- 风险与下一步：
  - combo 收益来自 `p_pml` 与 `v_pml`，`p_core` 仍是 L2/memory-throughput limited。
  - combo 后 `p_pml` 剩余瓶颈更像 issue/latency overhead，不是简单 DRAM 带宽。
  - 下一步可尝试降低 z-cache fill 的 integer/division/control overhead；继续禁止 shared `vx/vy` cache。

## 2026-06-08 12:01:00 +08:00 - Direct-fill z-cache optimization

- 操作目标：
  - 根据 NCU 指向的 issue/latency overhead，降低 z-cache fill 的 integer division/modulo/control overhead。
  - 将 linear cache-fill loop 改为 direct fill：每个 thread 填 central entry，少数 thread 填 halo。
- 修改文件：
  - `src/single_solver.cu`
    - 新增 `fill_pml_pressure_vz_cache_entry` `__device__ __forceinline__` helper。
    - `cuda_fd3d_p_pml_tile_ns` 内 z-cache fill 改为 direct fill。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 更新 `docs/day_20260608/pressure_pml_zrecomp_cache_prototype.md`。
  - 新增/拉回远端报告：
    - `reports/day_20260608/zrecomp_cache_directfill_combo_debug_step0_comparison.md`
    - `reports/day_20260608/zrecomp_cache_directfill_combo_debug_step1_comparison.md`
    - `reports/day_20260608/zrecomp_cache_directfill_combo_debug_step2_comparison.md`
    - `reports/day_20260608/zrecomp_cache_directfill_combo_correctness_comparison.md`
    - `reports/day_20260608/zrecomp_cache_directfill_combo_perf6_repeat_summary.md`
    - `reports/day_20260608/zrecomp_cache_directfill_combo_perf6_repeat_summary.json`
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 上传 `src/single_solver.cu` 到 clean worktree `/work/wenzhe/cuda3D_codex_day_20260608`。
  - 编译 combo debug flags：
    - zmem reference flags
    - `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`
    - `CUDA3D_CPML_VMEM_DISABLE_MPI`
    - `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
    - `CUDA3D_PML_DEBUG_DUMP`
  - 运行 combo debug dump step `0/1/2` 并对比 baseline dump。
  - 编译 combo release flags。
  - 运行 correctness 并对比 baseline outputs。
  - 运行 `perf_1gpu_6shots` repeat 3 轮 A/B，每轮均比较 outputs。
- 测试结果：
  - direct-fill combo debug dump step `0/1/2` 通过。
  - direct-fill combo correctness 通过，6 个输出 rel L2 全部 `0`。
  - direct-fill combo perf repeat 3 轮输出对比全部通过。
  - mean WP speedup：`1.100929x`。
  - mean Gradient speedup：`1.097530x`。
- 输出/哈希/误差摘要：
  - round 1：WP `2.438928 -> 2.217328`，speedup `1.099940x`；Gradient `2.549396 -> 2.324237`，speedup `1.096874x`。
  - round 2：WP `2.417782 -> 2.194350`，speedup `1.101821x`；Gradient `2.535653 -> 2.311585`，speedup `1.096933x`。
  - round 3：WP `2.415093 -> 2.193495`，speedup `1.101025x`；Gradient `2.541987 -> 2.313455`，speedup `1.098784x`。
- 风险与下一步：
  - direct-fill 是当前 best combo candidate。
  - 仍只验证 single GPU / single MPI rank。
  - 下一步需要 profile direct-fill combo，确认 p_pml issue/latency 是否改善，以及新的 dominant bottleneck。

## 2026-06-08 12:09:00 +08:00 - Direct-fill combo NCU profile

- 操作目标：
  - 对已提交的 direct-fill combo 版本重新运行 Nsight Compute short profile。
  - 确认 direct-fill 相对 zmem 的 kernel-level 收益，并定位下一步瓶颈。
- 修改文件：
  - 新增远端测试 worktree：`/work/wenzhe/cuda3D_codex_day_20260608_68de1a7`，固定到 commit `68de1a7`。
  - 新增远端临时 profile 脚本：`reports/day_20260608/directfill_combo_ncu_20260608_120449/run_profile.sh`。
  - 拉回本地报告：
    - `reports/day_20260608/directfill_combo_ncu_20260608_120449_summary.md`
    - `reports/day_20260608/directfill_combo_ncu_20260608_120449_summary.json`
    - `reports/day_20260608/directfill_combo_ncu_20260608_120449_zmem_bin.sha256`
    - `reports/day_20260608/directfill_combo_ncu_20260608_120449_directfill_bin.sha256`
  - 更新 `docs/day_20260608/pressure_pml_zrecomp_cache_prototype.md`。
  - 更新 `AGENTS.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `git fetch origin exp/day-20260608-cpml-compact-temporal`
  - `git worktree add --detach /work/wenzhe/cuda3D_codex_day_20260608_68de1a7 68de1a7`
  - 为 perf case 建立只读 velocity symlink，并创建 `d_obs` 输出目录。
  - 编译 zmem flags 并运行 NCU：
    - `--section SpeedOfLight`
    - `--section MemoryWorkloadAnalysis`
    - `--section SchedulerStats`
    - `--section WarpStateStats`
    - `--section Occupancy`
    - `--launch-skip 10`
    - `--launch-count 30`
    - kernel filter：`regex:.*cuda_fd3d_(p_core|v_pml_tile|p_pml_tile).*`
  - 编译 direct-fill combo flags 并运行同口径 NCU。
  - 使用 `tools/ncu_csv_summary.py` 生成 markdown/json summary。
- 测试结果：
  - zmem NCU run 通过，WP computing time：`2.436104s`。
  - direct-fill NCU run 通过，WP computing time：`2.222122s`。
  - 同轮 NCU WP speedup 约 `1.0963x`。
  - summary 生成成功。
- 输出/哈希/误差摘要：
  - `cuda_fd3d_p_core_ns` duration：zmem `75.942us`，direct-fill `75.270us`，speedup `1.009x`。
  - `cuda_fd3d_p_pml_tile_ns` duration：zmem `158.438us`，direct-fill `134.099us`，speedup `1.181x`。
  - `cuda_fd3d_v_pml_tile_ns` duration：zmem `58.794us`，direct-fill `53.590us`，speedup `1.097x`。
  - direct-fill `p_pml_tile`：
    - No Eligible：`59.885%`。
    - eligible warps/scheduler：`0.820`。
    - achieved occupancy：`74.662%`。
- 风险与下一步：
  - direct-fill 的主要收益确认来自 `p_pml_tile`，`p_core` 仍基本不动。
  - direct-fill 后 `p_pml_tile` 仍有明显 issue/latency 问题。
  - 下一步允许实现并测试 `CUDA3D_PML_PRESSURE_ZCACHE_WARP_RANGE`：每条 32-thread z-line 只计算一次 active z range，通过 warp broadcast 供 central/halo z-cache fill 复用。
  - 不得改变 `mem_dz_next` ownership，不得重开 shared `vx/vy` cache。

## 2026-06-08 12:20:00 +08:00 - Reject z-cache warp-range candidate

- 操作目标：
  - 测试 `CUDA3D_PML_PRESSURE_ZCACHE_WARP_RANGE` 是否能降低 direct-fill z-cache 的 active-range branch/control overhead。
  - 候选设计：每条 32-thread z-line 由 `threadIdx.x == 0` 计算 active z range，再用 `__shfl_sync` 广播给 central/halo z-cache fill。
- 修改文件：
  - 临时修改 `src/single_solver.cu` 实现 warp-range 候选。
  - 测试后已将 `src/single_solver.cu` 恢复到 commit `68de1a7` 的 direct-fill best 形态；本地和远端均无 warp-range 源码残留。
  - 拉回报告：
    - `reports/day_20260608/zcache_warp_range_correctness_comparison.md`
    - `reports/day_20260608/zcache_warp_range_perf6_repeat_summary.md`
    - `reports/day_20260608/zcache_warp_range_perf6_repeat_summary.json`
  - 更新 `docs/day_20260608/pressure_pml_zrecomp_cache_prototype.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 更新 `AGENTS.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 上传临时 `src/single_solver.cu` 到 `/work/wenzhe/cuda3D_codex_day_20260608_68de1a7`。
  - 编译 flags：
    - zmem reference flags
    - `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`
    - `CUDA3D_CPML_VMEM_DISABLE_MPI`
    - `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
    - `CUDA3D_PML_PRESSURE_ZCACHE_WARP_RANGE`
  - 运行 `correctness`，并用 `tools/compare_outputs.py` 对比 zmem baseline outputs。
  - 运行 `perf_1gpu_6shots` direct-fill vs warp-range 3 轮 A/B，每轮输出对比。
  - 用 `apply_patch` 恢复本地源码到 direct-fill best，并重新上传到远端 worktree。
- 测试结果：
  - 编译通过。
  - correctness 通过，6 个输出 rel L2 全部 `0`。
  - `perf_1gpu_6shots` repeat 3 轮输出对比全部通过。
  - mean WP speedup vs direct-fill：`0.997223x`。
  - mean Gradient speedup vs direct-fill：`0.997502x`。
- 输出/哈希/误差摘要：
  - round 1：direct WP `2.196203s`，warp WP `2.207274s`，speedup `0.994984x`；Gradient `0.997417x`。
  - round 2：direct WP `2.182848s`，warp WP `2.187369s`，speedup `0.997933x`；Gradient `0.997221x`。
  - round 3：direct WP `2.180586s`，warp WP `2.183309s`，speedup `0.998753x`；Gradient `0.997869x`。
- 风险与下一步：
  - 决策：拒绝 `CUDA3D_PML_PRESSURE_ZCACHE_WARP_RANGE`，不进入主线。
  - 原因：shuffle/control overhead 未抵消减少 active-range 重复计算的收益，未达到 `>=2%` small-candidate gate。
  - 当前 best 仍是 direct-fill z-cache combo：mean WP speedup `1.100929x`，mean Gradient speedup `1.097530x`。
  - 下一步需要离开这个微控制逻辑点，转向更大粒度的 pressure PML 数据流或 source-level profiling。

## 2026-06-08 12:36:00 +08:00 - Direct-fill SourceCounters and local-mem accumulation rejection

- 操作目标：
  - 使用 Nsight Compute SourceCounters 定位 direct-fill `p_pml_tile` 的剩余源码级瓶颈。
  - 测试一个由 source profile 引出的低风险候选：将 CPML memory update 显式改写为 local `new_mem` 变量。
- 修改文件：
  - 临时修改 `src/single_solver.cu`：在 `cuda_fd3d_p_pml_tile_ns` 中把 `mem_dzz/mem_dxx/mem_dyy` 更新改成 `new_mem` 局部变量。
  - 测试后已将 `src/single_solver.cu` 恢复到 commit `68de1a7` 的 direct-fill best 形态；本地和远端均无该候选源码残留。
  - 新增报告：
    - `reports/day_20260608/directfill_p_pml_source_ncu.csv`
    - `reports/day_20260608/directfill_source_profile_20260608_122553_bin.sha256`
    - `reports/day_20260608/directfill_source_profile_summary.md`
    - `reports/day_20260608/pml_local_mem_accum_correctness_comparison.md`
    - `reports/day_20260608/pml_local_mem_accum_perf6_repeat_summary.md`
    - `reports/day_20260608/pml_local_mem_accum_perf6_repeat_summary.json`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 更新 `docs/day_20260608/pressure_pml_zrecomp_cache_prototype.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 编译 direct-fill combo + `-lineinfo`。
  - 运行 NCU：
    - `--section SourceCounters`
    - `--section SchedulerStats`
    - `--section WarpStateStats`
    - `--launch-skip 10`
    - `--launch-count 10`
    - kernel filter：`regex:.*cuda_fd3d_p_pml_tile.*`
  - 生成 `.ncu-rep` 并在远端导出 `source_page.txt`；该文件约 19MB，未提交，只提交精简 CSV/summary。
  - 上传 local-mem-accum 临时源码并编译。
  - 运行 correctness，对比 zmem baseline outputs。
  - 运行 `perf_1gpu_6shots` direct-fill vs local-mem-accum 3 轮 A/B，每轮输出对比。
  - 用 `apply_patch` 恢复本地源码到 direct-fill best，并重新上传到远端 worktree。
- 测试结果：
  - SourceCounters profile 成功。
  - local-mem-accum 编译通过。
  - correctness 通过，6 个输出 rel L2 全部 `0`。
  - `perf_1gpu_6shots` repeat 3 轮输出对比全部通过。
  - mean WP speedup vs direct-fill：`1.000647x`。
  - mean Gradient speedup vs direct-fill：`0.998957x`。
- 输出/哈希/误差摘要：
  - direct-fill SourceCounters：
    - No Eligible：约 `60%`。
    - eligible warps/scheduler：约 `0.81`。
    - L1TEX scoreboard stall：约 `14.4 cycles/warp`。
    - uncoalesced global accesses：约 `19% excessive sectors`。
    - avg active threads/warp：约 `19.84`。
  - source page top lines：
    - `mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);`
    - `mem_dxx/mem_dyy` CPML memory update。
    - `p0[outIndex]=2*__ldg(p1+outIndex)-p0[outIndex]...`
  - local-mem-accum repeat：
    - round 1：WP `2.206771 -> 2.202571`，speedup `1.001907x`；Gradient `0.999970x`。
    - round 2：WP `2.203253 -> 2.204237`，speedup `0.999554x`；Gradient `0.999600x`。
    - round 3：WP `2.207174 -> 2.206112`，speedup `1.000481x`；Gradient `0.997301x`。
- 风险与下一步：
  - 决策：拒绝 local `new_mem` accumulation，未达到 `>=2%` gate。
  - direct-fill best 仍是当前主线。
  - 下一步应考虑更大粒度结构：降低 pressure PML divergence / active-thread loss，或重新组织 CPML memory traffic；不要继续重复 z-cache fill、warp-range、plain `new_mem` 表达式优化。

## 2026-06-08 12:58:00 +08:00 - Reject p0 __ldg final pressure read

- 操作目标：
  - 根据 SourceCounters 中 final `p0[outIndex]` pressure update 热点，测试把旧 `p0` 读取改成 read-only-cache load 是否能降低 pressure-PML latency。
  - 候选名：`pml_p0_ldg`。
- 修改文件：
  - 临时修改 `src/single_solver.cu`：只在 `cuda_fd3d_p_pml_tile_ns` 的 final pressure update 中把旧 `p0[outIndex]` 读改成 `__ldg(p0+outIndex)`。
  - 测试后已恢复到 commit `68de1a7` 的 direct-fill best 形态；本地 `git diff -- src/single_solver.cu` 为空。
  - 新增报告：
    - `reports/day_20260608/pml_p0_ldg_correctness_comparison.md`
    - `reports/day_20260608/pml_p0_ldg_perf6_repeat_summary.md`
    - `reports/day_20260608/pml_p0_ldg_perf6_repeat_summary.json`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 更新 `docs/day_20260608/pressure_pml_zrecomp_cache_prototype.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 上传临时 `src/single_solver.cu` 到 `/work/wenzhe/cuda3D_codex_day_20260608_68de1a7`。
  - 编译 direct-fill combo + `pml_p0_ldg` 临时源码。
  - 运行 correctness，对比 zmem baseline outputs。
  - 运行 `perf_1gpu_6shots` direct-fill vs `pml_p0_ldg` 3 轮 A/B，每轮输出对比。
  - 恢复本地与远端源码到 direct-fill best。
- 测试结果：
  - 编译通过。
  - correctness 通过，6 个输出 rel L2 全部 `0`。
  - `perf_1gpu_6shots` repeat 3 轮输出对比全部通过。
  - mean WP speedup vs direct-fill：`1.000054x`。
  - mean Gradient speedup vs direct-fill：`1.000694x`。
- 输出/哈希/误差摘要：
  - round 1：direct WP `2.208778s`，candidate WP `2.208617s`，speedup `1.000073x`；Gradient speedup `1.001689x`。
  - round 2：direct WP `2.191094s`，candidate WP `2.190333s`，speedup `1.000347x`；Gradient speedup `0.999835x`。
  - round 3：direct WP `2.188848s`，candidate WP `2.189415s`，speedup `0.999741x`；Gradient speedup `1.000559x`。
- 风险与下一步：
  - 决策：拒绝 `pml_p0_ldg`，未达到 `>=2%` small-candidate gate。
  - 不再重复 final `p0` read-only load 方向，除非出现新的 profiler 证据。
  - 下一步应转向更大粒度的 pressure-PML divergence / CPML memory traffic 结构，而不是继续做表达式级小修。

## 2026-06-08 13:18:00 +08:00 - Reject z-safe direct shared p1 pressure-z candidate

- 操作目标：
  - 测试一个结构化 pressure-PML 候选：对 z 方向完全远离 z-PML 的 tile，跳过 `recompute_vz_after_update_from_old_mem` line cache，改用 shared `p1` z-line +/-7 halo 直接计算 z 二阶项。
  - 候选名：`zsafe_direct_shared`，临时宏：`CUDA3D_PML_PRESSURE_Z_SAFE_DIRECT_SHARED`。
- 修改文件：
  - 临时修改 `src/single_solver.cu`：
    - 增加 shared `p1` z-line fill helper。
    - 增加 direct z second-derivative helper。
    - 对 z-safe tile 使用 direct shared `p1` path。
  - 测试后已恢复到 direct-fill best；本地 `git diff -- src/single_solver.cu` 为空，远端已重新上传并重编译 direct-fill best。
  - 新增报告：
    - `reports/day_20260608/zsafe_direct_correctness_comparison.md`
    - `reports/day_20260608/zsafe_direct_correctness_comparison.json`
    - `reports/day_20260608/zsafe_direct_perf6_repeat_summary.md`
    - `reports/day_20260608/zsafe_direct_perf6_repeat_summary.json`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 更新 `docs/day_20260608/pressure_pml_zrecomp_cache_prototype.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 上传临时 `src/single_solver.cu` 到 `/work/wenzhe/cuda3D_codex_day_20260608_68de1a7`。
  - 编译 direct-fill flags + `CUDA3D_PML_PRESSURE_Z_SAFE_DIRECT_SHARED`。
  - 运行 correctness 并对比 zmem correctness baseline。
  - 运行 `perf_1gpu_6shots` direct-fill vs candidate 3 轮 A/B，每轮输出对比。
  - 恢复本地和远端源码到 direct-fill best，并在远端重编译 direct-fill best。
- 测试结果：
  - 编译通过。
  - correctness 通过，6 个输出 rel L2 最大约 `2.180533e-10`，无 NaN/Inf。
  - `perf_1gpu_6shots` repeat 3 轮输出对比全部通过。
  - mean WP speedup vs direct-fill：`0.966920x`。
  - mean Gradient speedup vs direct-fill：`0.965779x`。
- 输出/哈希/误差摘要：
  - round 1：direct WP `2.206688s`，candidate WP `2.285624s`，speedup `0.965464x`；Gradient speedup `0.965635x`。
  - round 2：direct WP `2.191635s`，candidate WP `2.264459s`，speedup `0.967840x`；Gradient speedup `0.966443x`。
  - round 3：direct WP `2.192925s`，candidate WP `2.266692s`，speedup `0.967456x`；Gradient speedup `0.965260x`。
- 风险与下一步：
  - 决策：拒绝 `zsafe_direct_shared`，性能退化约 `3.3%`，不进入主线。
  - 不再重复当前 32x4x2 tile 下的 z-safe shared `p1` direct second derivative。
  - 下一步如果继续 pressure-PML，需要寻找能降低 CPML memory traffic 或 active-thread divergence 的更大改法，而不是扩大 z 方向 shared halo。

## 2026-06-08 13:24:00 +08:00 - Reject ptxas dlcm cache-policy sweep

- 操作目标：
  - 根据 SourceCounters 中 L1TEX scoreboard stall 信号，测试 whole-build ptxas global-load cache policy 是否能改善 direct-fill pressure-PML。
  - 候选 flags：
    - `-Xptxas -dlcm=ca`
    - `-Xptxas -dlcm=cg`
- 修改文件：
  - 未修改源码。
  - 新增报告：
    - `reports/day_20260608/dlcm_ca_perf6_repeat_summary.md`
    - `reports/day_20260608/dlcm_ca_perf6_repeat_summary.json`
    - `reports/day_20260608/dlcm_cg_perf6_repeat_summary.md`
    - `reports/day_20260608/dlcm_cg_perf6_repeat_summary.json`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 更新 `docs/day_20260608/pressure_pml_zrecomp_cache_prototype.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 对 `-dlcm=ca`：每轮重新编译 direct-fill default 与 candidate，各跑 `perf_1gpu_6shots`，共 3 轮 A/B，并逐轮对比输出。
  - 对 `-dlcm=cg`：同样 3 轮 A/B，并逐轮对比输出。
  - 测试结束后远端重新编译 direct-fill best，避免 binary 停留在失败 flags。
- 测试结果：
  - `-dlcm=ca`：输出对比全部通过，mean WP speedup `0.999263x`，mean Gradient speedup `0.999576x`。
  - `-dlcm=cg`：输出对比全部通过，mean WP speedup `0.859344x`，mean Gradient speedup `0.864052x`。
- 输出/哈希/误差摘要：
  - `-dlcm=ca` round WP speedup：`0.999543x`、`0.999403x`、`0.998842x`。
  - `-dlcm=cg` round WP speedup：`0.862083x`、`0.855110x`、`0.860840x`。
- 风险与下一步：
  - 决策：拒绝 ptxas `dlcm` cache-policy sweep。
  - `cg` 明显破坏当前 direct-fill cache locality；`ca` 不超过噪声。
  - 后续 memory work 必须更具体到源码/dataflow 或 per-load 证据，不能继续 whole-binary cache-policy sweep。

## 2026-06-08 13:28:00 +08:00 - Reject p_core explicit readonly LDG

- 操作目标：
  - 测试 `cuda_fd3d_p_core_ns` 中只读 `p1/cw2` load 显式使用 `__ldg` 是否能改善 p_core memory path。
  - 候选宏：`CUDA3D_P_CORE_READONLY_LDG`。
- 修改文件：
  - 临时修改 `src/single_solver.cu`：在 `p_core` 内用宏包装 `p1/cw2` 读取。
  - 测试后已恢复到 direct-fill best；本地 `git diff -- src/single_solver.cu` 为空，远端已重新上传并重编译 direct-fill best。
  - 新增报告：
    - `reports/day_20260608/pcore_readonly_ldg_correctness_comparison.md`
    - `reports/day_20260608/pcore_readonly_ldg_correctness_comparison.json`
    - `reports/day_20260608/pcore_readonly_ldg_perf6_repeat_summary.md`
    - `reports/day_20260608/pcore_readonly_ldg_perf6_repeat_summary.json`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 更新 `docs/day_20260608/pressure_pml_zrecomp_cache_prototype.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 上传临时 `src/single_solver.cu` 到 `/work/wenzhe/cuda3D_codex_day_20260608_68de1a7`。
  - 编译 direct-fill flags + `CUDA3D_P_CORE_READONLY_LDG`。
  - 运行 correctness，对比 zmem baseline outputs。
  - 运行 `perf_1gpu_6shots` direct-fill vs candidate 3 轮 A/B，每轮输出对比。
  - 恢复本地和远端源码到 direct-fill best，并在远端重编译 direct-fill best。
- 测试结果：
  - 编译通过。
  - correctness 通过，6 个输出 rel L2 全部 `0`。
  - `perf_1gpu_6shots` repeat 3 轮输出对比全部通过。
  - mean WP speedup vs direct-fill：`0.999319x`。
  - mean Gradient speedup vs direct-fill：`0.999254x`。
- 输出/哈希/误差摘要：
  - round 1：WP speedup `0.999245x`，Gradient speedup `1.000178x`。
  - round 2：WP speedup `0.999866x`，Gradient speedup `0.998967x`。
  - round 3：WP speedup `0.998847x`，Gradient speedup `0.998618x`。
- 风险与下一步：
  - 决策：拒绝 `CUDA3D_P_CORE_READONLY_LDG`。
  - p_core 的 read-only load syntax 不是有效优化点。
  - 后续 p_core work 需要数据复用、temporal ownership 或更大结构变化，而不是 load 包装。

## 2026-06-08 13:49:00 +08:00 - Reject inject/extract BS512 small-kernel candidate

- 操作目标：
  - 根据 NCU 提示 `lint3d_inject_bell_extract_gpu_zz` grid 太小、kernel duration 约 `5.109us`，测试 inject/extract helper 的 block size 从 `1024` 改为 `512` 是否能降低小 kernel 调度开销。
  - 候选宏：`CUDA3D_INJECT_EXTRACT_BS512`。
- 修改文件：
  - 临时修改 `src/rem_fd.cu`：用宏把 `BS=1024` 切换为 `BS=512`。
  - 测试后已恢复本地与远端 `src/rem_fd.cu` 到 direct-fill best；本地 `git diff -- src/rem_fd.cu` 为空。
  - 远端 `/work/wenzhe/cuda3D_codex_day_20260608_68de1a7` 已重新编译 direct-fill best binary。
  - 新增报告：
    - `reports/day_20260608/inject_extract_ncu_summary.md`
    - `reports/day_20260608/inject_extract_ncu_summary.json`
    - `reports/day_20260608/inject_extract_bs512_correctness_comparison.md`
    - `reports/day_20260608/inject_extract_bs512_correctness_comparison.json`
    - `reports/day_20260608/inject_extract_bs512_perf6_repeat_summary.md`
    - `reports/day_20260608/inject_extract_bs512_perf6_repeat_summary.json`
    - `docs/day_20260608/inject_extract_launch_overhead.md`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 更新 `docs/day_20260608/pressure_pml_zrecomp_cache_prototype.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 对 `perf_1gpu_6shots` 运行 Nsight Compute，采样 `lint3d_inject_bell_extract_gpu_zz`。
  - 上传临时 `src/rem_fd.cu` 到 `/work/wenzhe/cuda3D_codex_day_20260608_68de1a7`。
  - 编译 direct-fill flags + `CUDA3D_INJECT_EXTRACT_BS512`。
  - 运行 correctness，对比 zmem correctness baseline outputs。
  - 运行 `perf_1gpu_6shots` direct-fill vs candidate 3 轮 A/B，每轮输出对比。
  - 恢复本地和远端源码到 direct-fill best，并在远端重编译 direct-fill best。
- 测试结果：
  - NCU 成功生成 inject/extract profile。
  - 编译通过。
  - correctness 通过，6 个输出 rel L2 全部 `0`。
  - `perf_1gpu_6shots` repeat 3 轮输出对比全部通过。
  - mean WP speedup vs direct-fill：`0.999684x`。
  - mean Gradient speedup vs direct-fill：`0.998963x`。
- 输出/哈希/误差摘要：
  - NCU：duration `5.109us`，SOL compute `0.040%`，SOL memory `6.699%`，SOL DRAM `1.414%`。
  - Nsight Compute rule：grid 太小，只有 `0.0` full waves。
  - round 1：direct WP `2.210482s`，candidate WP `2.207253s`，WP speedup `1.001463x`；Gradient speedup `0.998848x`。
  - round 2：direct WP `2.188303s`，candidate WP `2.191148s`，WP speedup `0.998702x`；Gradient speedup `0.998355x`。
  - round 3：direct WP `2.189624s`，candidate WP `2.192060s`，WP speedup `0.998889x`；Gradient speedup `0.999687x`。
- 风险与下一步：
  - 决策：拒绝 `CUDA3D_INJECT_EXTRACT_BS512`。
  - inject/extract 小 kernel 确实存在 launch/small-grid 信号，但 block-size-only 调整没有收益。
  - 后续若重开此方向，必须改成 CUDA Graph、launch aggregation 或 wave-step scheduling 设计，并以 `perf_1gpu_6shots repeat` 证明至少 `>=2%` speedup。

## 2026-06-08 14:15:00 +08:00 - V-PML SourceCounters gate and component split rejection

- 操作目标：
  - 在当前 direct-fill best 上补采 `cuda_fd3d_v_pml_tile_ns` 的 SourceCounters / SchedulerStats / WarpStateStats。
  - 判断是否值得进入 `v_pml` 的 vx/vy component-owner split CUDA prototype。
- 修改文件：
  - 新增报告：
    - `reports/day_20260608/directfill_v_pml_source_ncu.csv`
    - `reports/day_20260608/directfill_v_pml_source_summary.md`
    - `reports/day_20260608/directfill_v_pml_source_summary.json`
    - `docs/day_20260608/v_pml_source_profile_gate.md`
  - 更新 `tools/ncu_csv_summary.py`：
    - 增加 SourceCounters 指标解析与 markdown 输出：branch efficiency、branch instructions、avg divergent branches。
    - 增加 Scheduler/WarpState 的更多输出行：issued/active warps、warp cycles、active threads。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端 `/work/wenzhe/cuda3D_codex_day_20260608_68de1a7` 重新编译 direct-fill best + `-lineinfo`。
  - 运行 Nsight Compute：
    - `--section SourceCounters`
    - `--section SchedulerStats`
    - `--section WarpStateStats`
    - `--launch-skip 10`
    - `--launch-count 10`
    - `--kernel-name regex:.*cuda_fd3d_v_pml_tile.*`
    - case：`perf_1gpu_6shots`
  - 导出 NCU `details` CSV，并用 `tools/ncu_csv_summary.py` 生成 summary。
  - 远端导出 source page 与 `cuda,sass` source page；原始文本约 `3.36MB` / `8.65MB`，只保留在远端，不提交入 Git。
  - 本地运行 `python -m py_compile tools/ncu_csv_summary.py`。
  - 本地静态预算 vx/vy component-owner split 的 tile/thread overlap。
  - 远端最终重新编译 direct-fill best，移除 `-lineinfo` profile binary 影响。
- 测试结果：
  - NCU profile 成功。
  - `tools/ncu_csv_summary.py` py_compile 通过。
  - 远端 direct-fill best restore build 通过。
  - 未写 CUDA candidate；component split 在静态 gate 被拒绝。
  - 输出/哈希/误差摘要：
  - 远端 restore 后 binary SHA256：`bf719d04f0fa1136af3f1afac54a936ee0d052a18ffd9a9d07863aa7f9dfca28`。
  - `cuda_fd3d_v_pml_tile_ns` SourceCounters：
    - No Eligible：`44.891%`。
    - eligible warps/scheduler：`1.629`。
    - active warps/scheduler：`10.170`。
    - warp cycles/issued inst：`18.456`。
    - avg active threads/warp：`23.700`。
    - avg not-predicated threads/warp：`21.670`。
    - branch efficiency：`86.970%`。
    - branch instructions：`2,079,334`。
    - avg divergent branches：`143.480`。
    - L1TEX scoreboard stall：约 `11.8 cycles/warp`。
    - uncoalesced excessive sectors：约 `22%`。
  - vx/vy split static budget：
    - current combined vx/vy tiles：`41,100`。
    - vx-only tiles：`40,848`。
    - vy-only tiles：`40,762`。
    - split tile sum / combined：`1.985645x`。
    - split active work sum / combined active：`1.963726x`。
    - overlap tiles：`40,510`。
- 风险与下一步：
  - 决策：拒绝当前 `32x4x2` tile geometry 下的 `v_pml` vx/vy component-owner split，不进入 CUDA implementation。
  - `v_pml` 剩余信号是真实的 memory-latency/coalescing 问题，而不是简单 branch 或表达式级问题。
  - 下一步如果继续 `v_pml`，必须先提出改变 memory layout/coalescing 的设计；不能做 tile block shape sweep，也不能重开 `CUDA3D_PML_ZMEM_V_TILE_PRUNE`。

## 2026-06-08 14:45:00 +08:00 - Nsight Systems scheduling gate and CUDA Graph rejection

- 操作目标：
  - 对当前 direct-fill best 运行 Nsight Systems，判断单卡 `perf_1gpu_6shots` 是否存在足够 launch/scheduling gap 来支持 CUDA Graph 或 launch aggregation prototype。
  - 将调度层结论写入项目规则，避免后续重复投入不满足 gate 的方向。
- 修改文件：
  - 新增工具：`tools/nsys_cuda_summary.py`。
  - 新增 raw/summary artifacts：
    - `reports/day_20260608/directfill_scheduling_cuda_api_sum.csv`
    - `reports/day_20260608/directfill_scheduling_cuda_gpu_kern_sum.csv`
    - `reports/day_20260608/directfill_scheduling_nsys_run.log`
    - `reports/day_20260608/directfill_scheduling_nsys_bin.sha256`
    - `reports/day_20260608/directfill_scheduling_nsys_summary.md`
    - `reports/day_20260608/directfill_scheduling_nsys_summary.json`
  - 新增报告：`docs/day_20260608/scheduling_nsys_cuda_graph_gate.md`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端 `/work/wenzhe/cuda3D_codex_day_20260608_68de1a7` 检查 `nsys` 可用，版本 `2025.3.2.367-253236224375v0`。
  - 远端检查 RTX 5090 空闲：约 `481 MiB` 显存占用，`0%` util。
  - 远端运行：
    - `nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true --stats=false`
    - case：`benchmarks/cases/perf_1gpu_6shots`
    - binary：当前 direct-fill best。
  - 远端导出 Nsight Systems stats：
    - `cuda_api_sum`
    - `cuda_gpu_kern_sum`
    - `cuda_gpu_trace` 保留在远端 raw profile 目录，未提交大文件。
  - 本地运行：
    - `python -m py_compile tools/nsys_cuda_summary.py`
    - `python tools/nsys_cuda_summary.py --api-csv reports/day_20260608/directfill_scheduling_cuda_api_sum.csv --kernel-csv reports/day_20260608/directfill_scheduling_cuda_gpu_kern_sum.csv --run-log reports/day_20260608/directfill_scheduling_nsys_run.log --md-out reports/day_20260608/directfill_scheduling_nsys_summary.md --json-out reports/day_20260608/directfill_scheduling_nsys_summary.json`
  - 将 `tools/nsys_cuda_summary.py` 上传到远端测试 worktree：`/work/wenzhe/cuda3D_codex_day_20260608_68de1a7/tools/nsys_cuda_summary.py`。
  - 远端复查 CUDA 源码无实验 diff；仅辅助工具 `tools/nsys_cuda_summary.py` 为未跟踪文件。
- 测试结果：
  - Nsight Systems profile 成功。
  - 程序日志包含 `ALL DONE`。
  - `tools/nsys_cuda_summary.py` py_compile 通过。
- 输出/哈希/误差摘要：
  - 远端 nsys run id：`scheduling_nsys_20260608_142948`。
  - 远端 profile：`/work/wenzhe/cuda3D_codex_day_20260608_68de1a7/reports/day_20260608/scheduling_nsys_20260608_142948/directfill_perf6.nsys-rep`。
  - binary SHA256：`bf719d04f0fa1136af3f1afac54a936ee0d052a18ffd9a9d07863aa7f9dfca28`。
  - `Gradient TIME all = 2.349826s`。
  - `WP computing time = 2.238769s`。
  - GPU kernel total：`2.232398465s`。
  - `WP - GPU kernel total = 0.006370535s`，visible gap fraction `0.2846%`。
  - ideal speedup if gap vanished：`1.002854x`。
  - `cudaLaunchKernel` CPU API total：`1.845401s`，`36,024` calls，avg `51.227us`。
  - main kernel totals：
    - `cuda_fd3d_p_pml_tile_ns`：`1.251216s`，`9006` instances，avg `138.931us`。
    - `cuda_fd3d_p_core_ns`：`0.557985s`，`9006` instances，avg `61.957us`。
    - `cuda_fd3d_v_pml_tile_ns`：`0.390576s`，`9006` instances，avg `43.368us`。
    - `lint3d_inject_bell_extract_gpu_zz`：`0.032622s`，`9006` instances，avg `3.622us`。
- 风险与下一步：
  - 决策：拒绝当前 single-GPU / single-MPI-rank CUDA Graph 或 launch aggregation prototype。
  - 原因：CPU launch API 累计时间很高，但大部分与 GPU kernel 执行重叠；WP timer 与 GPU kernel total 只差约 `6.37ms`。
  - 只有当未来 Nsight Systems 或多 rank wall-clock profile 显示 `>2%` visible scheduling gap / GPU idle 时，才允许重开调度层 CUDA Graph 路线。
  - 下一步应转向更可能有收益的 pressure-PML divergence / CPML memory traffic / memory coalescing ownership 结构，而不是继续调度层小修。

## 2026-06-08 15:10:00 +08:00 - Pressure-PML active segment compaction model

- 操作目标：
  - 在 CUDA Graph 被拒绝后，转向 pressure-PML lane utilization / active segment ownership 方向。
  - 量化普通 active-line list、exact active-point list、length-16 half-warp packing 三种候选的理论上界。
- 修改文件：
  - 新增工具：`tools/pml_active_segment_compaction_model.py`。
  - 新增报告：`docs/day_20260608/pml_active_segment_compaction_model.md`。
  - 新增 JSON：`reports/day_20260608/pml_active_segment_compaction_model.json`。
  - 生成辅助 direct-fill dataflow JSON/Markdown：
    - `reports/day_20260608/pml_pressure_dataflow_directfill_for_line_model.json`
    - `reports/day_20260608/pml_pressure_dataflow_directfill_for_line_model.md`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地创建远端隔离 worktree：
    - `/work/wenzhe/cuda3D/.codex_worktrees/sprint_0648`
    - branch：`codex/remote-sprint-20260608-0648`
    - HEAD：`f5f4037`
  - 本地运行：
    - `python tools/pml_pressure_dataflow_audit.py --case benchmarks/cases/perf_1gpu_6shots --ncu-summary-json reports/day_20260608/directfill_combo_ncu_20260608_120449_summary.json --json-out reports/day_20260608/pml_pressure_dataflow_directfill_for_line_model.json --md-out reports/day_20260608/pml_pressure_dataflow_directfill_for_line_model.md`
    - `python -m py_compile tools/pml_active_segment_compaction_model.py`
    - `python tools/pml_active_segment_compaction_model.py --case benchmarks/cases/perf_1gpu_6shots --ncu-summary-json reports/day_20260608/directfill_combo_ncu_20260608_120449_summary.json --json-out reports/day_20260608/pml_active_segment_compaction_model.json --md-out docs/day_20260608/pml_active_segment_compaction_model.md`
- 测试结果：
  - 远端隔离 worktree 创建成功，远端主目录未 reset/未覆盖。
  - `tools/pml_active_segment_compaction_model.py` py_compile 通过。
  - 模型报告生成成功。
- 输出/哈希/误差摘要：
  - pressure-PML current launched lanes：`29,143,040`。
  - active lanes after core return：`19,118,944`。
  - current lane efficiency：`65.60%`。
  - active line slots：`893,204`。
  - active z-line length histogram：
    - length `16`：`542,100` line slots，`8,673,600` active lanes。
    - length `23`：`87,776` line slots，`2,018,848` active lanes。
    - length `32`：`263,328` line slots，`8,426,496` active lanes。
  - whole length-16 tiles：`67,392`，对应当前 lanes `17,252,352`，约占当前 pressure-PML launched lanes `59.20%`。
  - simple active-line list：
    - lane reduction：`1.92%`。
    - p_pml lane speedup ceiling：`1.020x`。
    - sampled-main ceiling：`1.011x`。
    - gate：reject。
  - exact active-point list：
    - lane reduction：`34.40%`。
    - p_pml lane speedup ceiling：`1.524x`。
    - sampled-main ceiling：`1.228x`。
    - descriptor traffic：约 `72.933 MiB/step aggregate-shots`。
    - gate：design only。
  - length-16 half-warp packing：
    - lane reduction：`31.69%`。
    - p_pml lane speedup ceiling：`1.464x`。
    - sampled-main ceiling：`1.207x`。
    - gate：design only。
- 风险与下一步：
  - 决策：拒绝普通 active-line list，不写该 CUDA prototype。
  - length-16 half-warp packing 有足够模型上界，下一步允许进入设计/原型 gate。
  - 该路线必须保持 direct-fill pressure z-cache 数值路径，不得变成已禁止的 z-face direct derivative、z-face fusion 或 shared-VP 变体。
  - 原型必须 macro-default-off，先跑 debug dump step `0/1/2`、correctness、`perf_1gpu_6shots repeat`；若 repeat 没有 `>=5%` WP speedup，应立即停止。

## 2026-06-08 15:55:00 +08:00 - Length-16 half-warp pressure-PML prototype accepted

- 操作目标：
  - 按 active segment compaction model 的 gate，实现并验收 `length-16 half-warp pressure-PML active segment packing`。
  - 保留 direct-fill pressure z-cache 数值路径，只改变 whole length-16 active-z pressure-PML tile 的 warp lane ownership。
  - 在远端隔离 worktree 内完成构建、debug dump、correctness、`perf_1gpu_6shots repeat`，不污染主远端目录。
- 修改文件：
  - `include/inc3D/single_solver.h`
  - `src/rem_fd.cu`
  - `src/single_solver.cu`
  - 新增报告：`docs/day_20260608/len16_halfwarp_pressure_pml_prototype.md`
  - 新增/取回报告摘要：
    - `reports/day_20260608/len16_halfwarp_perf6_compare_20260608_152909/comparison.md`
    - `reports/day_20260608/len16_halfwarp_correctness_compare_20260608_152526/comparison.md`
    - `reports/day_20260608/len16_halfwarp_correctness_compare_20260608_152526/comparison.json`
    - `reports/day_20260608/len16_halfwarp_perf6_repeat_20260608_152944/summary.md`
    - `reports/day_20260608/len16_halfwarp_perf6_repeat_20260608_152944/summary.json`
    - `reports/day_20260608/len16_halfwarp_debug_profile_20260608_153419/compare_step0/comparison.md`
    - `reports/day_20260608/len16_halfwarp_debug_profile_20260608_153419/compare_step1/comparison.md`
    - `reports/day_20260608/len16_halfwarp_debug_profile_20260608_153419/compare_step2/comparison.md`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地实现新宏默认关闭：`CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK`。
  - 远端隔离 worktree：`/work/wenzhe/cuda3D/.codex_worktrees/sprint_0648`。
  - 远端构建命令使用 makefile command-line override：
    - `make -B -f makefile.rtx5090 NVFLAGS="... -DCUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK" test`
  - 远端为隔离 worktree补齐测试数据：
    - `bench_smoke/d_obs` 目录用于 smoke 输出。
    - `perf_1gpu_6shots` 与 `profile_1gpu` velocity/d_obs 测试数据通过 symlink 或单文件复制接入隔离 worktree。
  - 远端运行：
    - `smoke_1gpu`
    - `correctness`
    - `profile_1gpu` debug dump step `0/1/2`
    - `perf_1gpu_6shots` A/B + repeat 3 轮
  - 本地取回 correctness compare 时，Windows Python 缺少 `paramiko`，未修改本机全局 Python 环境，改用 WSL Python 运行同一 `tools/remote_get.py` 脚本完成取回。
  - 自审发现未来 case 若所有 pressure-PML tiles 都被 len16 packed kernel 接管，residual p tile 数可能为 `0`；补充 host-side guard，避免 0-size `cudaMalloc` 或 0-grid residual launch。
  - guard 补丁后重新上传 `include/inc3D/single_solver.h`、`src/rem_fd.cu`、`src/single_solver.cu` 到远端隔离 worktree，并重跑 release rebuild + smoke。
  - 额外运行 macro-off build，确认不启用 `CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK` 时 direct-fill 组合仍可编译。
  - macro-off build 会覆盖远端 binary；随后重新以 len16 candidate flags 编译，确保远端隔离 worktree 的最终 `bin/cuda_3D_FM` 是 len16 candidate。
- 测试结果：
  - release build 通过。
  - smoke 通过，run：`benchmarks/runs/smoke_1gpu_len16_halfwarp_smoke_datafixed_20260608_153645`，returncode `0`，输出 `3` 个文件。
  - guard rebuild 通过，远端 build log：`reports/day_20260608/len16_halfwarp_guard_rebuild_make.log`，只有既有 unused warning。
  - guard rebuild 后 smoke 通过，run：`benchmarks/runs/smoke_1gpu_len16_halfwarp_guard_rebuild_20260608_155133`，returncode `0`，输出 `3` 个文件；smoke tile split：len16 `0`，residual p `240`。
  - macro-off build 通过，远端 build log：`reports/day_20260608/len16_halfwarp_guard_macro_off_make.log`。
  - final candidate rebuild 通过，远端 build log：`reports/day_20260608/len16_halfwarp_guard_candidate_final_make.log`。
  - correctness 通过：
    - baseline：`benchmarks/runs/correctness_directfill_base_for_len16_ab_20260608_152404`
    - candidate：`benchmarks/runs/correctness_len16_halfwarp_candidate_20260608_152436`
    - compare：`reports/day_20260608/len16_halfwarp_correctness_compare_20260608_152526`
    - 6 个输出 rel L2 全部 `0`
  - debug dump 通过：
    - root：`reports/day_20260608/len16_halfwarp_debug_profile_20260608_153419`
    - step `0`：pass，所有数组 rel L2 `0`
    - step `1`：pass，所有数组 rel L2 `0`
    - step `2`：pass，`p0` rel L2 `7.852061e-09`，其他数组 rel L2 `0`
  - `perf_1gpu_6shots` repeat 通过：
    - summary：`reports/day_20260608/len16_halfwarp_perf6_repeat_20260608_152944/summary.md`
    - 3 轮输出 compare 全部 pass
    - max rel L2 `6.384336e-07`
- 输出/哈希/误差摘要：
  - mean base WP vs direct-fill：`2.207751s`
  - mean candidate WP：`2.039080s`
  - mean WP speedup：`1.082719x`
  - mean base Gradient：`2.316433s`
  - mean candidate Gradient：`2.159948s`
  - mean Gradient speedup：`1.072448x`
  - round speedups：
    - round 1 WP `1.083596x`，Gradient `1.074611x`
    - round 2 WP `1.081255x`，Gradient `1.070784x`
    - round 3 WP `1.083304x`，Gradient `1.071949x`
  - candidate six-shot tile split：
    - shot 1：len16 `10816`，residual p `7168`
    - shot 2：len16 `12064`，residual p `8032`
    - shot 3：len16 `10816`，residual p `7648`
    - shot 4：len16 `10816`，residual p `7408`
    - shot 5：len16 `12064`，residual p `8300`
    - shot 6：len16 `10816`，residual p `7892`
  - final candidate binary SHA256 after guard rebuild：`c0d785d747f058e78b183d6b7d3984a4f04549c835d2ef6c994f5d7ce70becf7`
- 风险与下一步：
  - 决策：接受 `CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK` 为当前 RTX 5090 single-GPU best candidate。
  - guard rebuild 后只重跑 smoke；未重跑完整 perf repeat，因为 guard 只改变 host-side launch/allocation 条件，不改变当前 perf case 中 device kernel 数学路径。
  - 该结果不是正式 `perf_3gpu` 阈值存档；若要对外报告累计倍率，需要同机同 session 重跑 zmem/direct-fill/len16。
  - correctness case 本身 len16 tiles 为 `0`，因此 packed kernel 的数学等价主要由 `profile_1gpu` debug dump 与 `perf_1gpu_6shots` 输出对比覆盖。
  - 下一步应对 len16 candidate 做 NCU source/profile，判断剩余 bottleneck 是否变成 memory coalescing、shared-memory pressure、final `p0/mem_dzz` update，或可扩展到 length-23 active segment。
  - 不要回到已拒绝的 z-face direct/fusion/shared-VP、simple active-line list、z-cache fill 微调、`new_mem` 表达式、`p0` read-only load、ptxas cache-policy、inject/extract block-size、当前 tile 下 `vx/vy` split，除非有新的 profiler evidence。

## 2026-06-08 16:12:00 +08:00 - Len16 NCU profile and length-23 gate

- 操作目标：
  - 对已接受的 `CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK` 做 Nsight Compute A/B profile。
  - 判断 len16 收益是否在 kernel-level profile 中成立，以及下一步是否应该开 length-23 prototype。
- 修改文件：
  - 新增报告：`docs/day_20260608/len16_halfwarp_ncu_profile.md`。
  - 新增 NCU artifacts：
    - `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.md`
    - `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json`
    - `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/directfill_profile_ncu_details.csv`
    - `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/len16_profile_ncu_details.csv`
    - `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/directfill_lineinfo_bin.sha256`
    - `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/len16_lineinfo_bin.sha256`
    - `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/final_release_bin.sha256`
    - `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/directfill_profile_details_run.log`
    - `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/len16_profile_details_run.log`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端隔离 worktree：`/work/wenzhe/cuda3D/.codex_worktrees/sprint_0648`。
  - 检查环境：
    - Nsight Compute：`/usr/local/cuda-13.0/bin/ncu`，version `2025.3.0.0`。
    - GPU：RTX 5090，约 `481 MiB` 显存占用，`0%` util。
  - 先运行 syntax-check NCU，确认 kernel filter 能捕捉 `cuda_fd3d_p_pml_len16_halfwarp_ns`。
  - direct-fill profile build：
    - current direct-fill flags + `-lineinfo`
    - 不启用 `CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK`
  - len16 profile build：
    - current len16 flags + `-lineinfo`
    - 启用 `CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK`
  - NCU command 口径：
    - `--target-processes all`
    - `--csv --page details`
    - sections：`SpeedOfLight`、`MemoryWorkloadAnalysis`、`SchedulerStats`、`WarpStateStats`、`Occupancy`、`SourceCounters`
    - `--launch-skip 10`
    - `--launch-count 12`
    - kernel filter：`regex:.*cuda_fd3d_(p_core|v_pml_tile|p_pml_tile|p_pml_len16_halfwarp).*`
  - 使用 `tools/ncu_csv_summary.py` 生成 summary。
  - profile 后重建无 `-lineinfo` release len16 binary，并运行 smoke。
- 测试结果：
  - syntax-check NCU 成功，捕捉到 `cuda_fd3d_p_pml_len16_halfwarp_ns`。
  - direct-fill details NCU 成功，CSV `655` 行，run log 包含 `ALL DONE`。
  - len16 details NCU 成功，CSV `658` 行，run log 包含 `ALL DONE`。
  - summary 生成成功。
  - final release rebuild 成功。
  - post-profile smoke 通过：
    - run：`benchmarks/runs/smoke_1gpu_len16_after_ncu_restore_20260608_160633`
    - 输出包含 `ALL DONE`
- 输出/哈希/误差摘要：
  - NCU `profile_1gpu` sampled durations：
    - direct-fill `cuda_fd3d_p_core_ns`：`93.752us`
    - len16 `cuda_fd3d_p_core_ns`：`93.547us`
    - direct-fill `cuda_fd3d_v_pml_tile_ns`：`65.528us`
    - len16 `cuda_fd3d_v_pml_tile_ns`：`65.248us`
    - direct-fill pressure-PML total：`164.328us`
    - len16 residual pressure-PML：`72.683us`
    - len16 packed pressure-PML：`65.771us`
    - len16 pressure-PML total：`138.453us`
    - pressure-PML kernel-path speedup：约 `1.187x`
    - sampled main-kernel total：direct-fill `323.608us`，len16 `297.248us`，speedup `1.0887x`
  - direct-fill pressure-PML：
    - No Eligible `61.170%`
    - eligible warps/scheduler `0.775`
    - avg active threads/warp `19.680`
    - branch efficiency `75.530%`
  - len16 residual pressure-PML：
    - No Eligible `63.497%`
    - eligible warps/scheduler `0.733`
    - avg active threads/warp `22.950`
    - branch efficiency `83.320%`
  - len16 packed kernel：
    - No Eligible `73.827%`
    - eligible warps/scheduler `0.433`
    - warp cycles/issued instruction `34.210`
    - avg active threads/warp `26.380`
    - branch efficiency `65.220%`
  - final release binary SHA256：`2dd6c588c41f206adcb0121a755e17857ef1a862fc28d59d72c7434e64685b3a`
- 风险与下一步：
  - NCU 证实 len16 收益来自 pressure-PML active segment ownership，和 `perf_1gpu_6shots` repeat 方向一致。
  - 拒绝直接写简单 `CUDA3D_PML_PRESSURE_LEN23_*` prototype；length-23 单独只能移除约 `0.790M` inactive lanes，且不能两线合并进一个 warp，额外 launch/tile-list/control overhead 风险过高。
  - 下一步允许打开 Phase 4.11：exact active-point / compact descriptor budget。
  - 只有该预算证明 `>=5%` repeat speedup ceiling，才允许写新的 CUDA prototype；否则转向 packed len16 kernel source-level drill-down 或 v-PML memory layout/coalescing 设计。

## 2026-06-08 16:36:00 +08:00 - Phase 4.11 compact descriptor budget rejected

- 操作目标：
  - 执行 Phase 4.11 exact active-point / compact descriptor budget。
  - 以已接受的 len16 candidate 为当前基准，重新评估 compact descriptor 是否仍有 `>=5%` repeat speedup ceiling。
  - 若 gate 不足，明确禁止后续重复写 length-23/exact-point 小原型。
- 修改文件：
  - 新增工具：`tools/pml_compact_descriptor_budget.py`。
  - 新增报告：`docs/day_20260608/pml_compact_descriptor_budget.md`。
  - 新增 JSON：`reports/day_20260608/pml_compact_descriptor_budget.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地运行：
    - `python -m py_compile tools/pml_compact_descriptor_budget.py`
    - `python tools/pml_compact_descriptor_budget.py --json-out reports/day_20260608/pml_compact_descriptor_budget.json --md-out docs/day_20260608/pml_compact_descriptor_budget.md`
  - 上传工具到远端隔离 worktree：
    - `/work/wenzhe/cuda3D/.codex_worktrees/sprint_0648/tools/pml_compact_descriptor_budget.py`
  - 远端运行：
    - `python3 -m py_compile tools/pml_compact_descriptor_budget.py`
    - `python3 tools/pml_compact_descriptor_budget.py --json-out reports/day_20260608/pml_compact_descriptor_budget.json --md-out docs/day_20260608/pml_compact_descriptor_budget.md`
  - 取回远端生成的 `docs/day_20260608/pml_compact_descriptor_budget.md` 与 `reports/day_20260608/pml_compact_descriptor_budget.json`，确保报告路径为服务器路径。
- 测试结果：
  - 本地 py_compile 通过。
  - 本地预算生成成功。
  - 远端 py_compile 通过。
  - 远端预算生成成功。
  - 本地与远端 gate 一致：`reject_cuda_prototype`。
- 输出/哈希/误差摘要：
  - accepted len16 lanes：`19,908,928`。
  - active lanes：`19,118,944`。
  - remaining length-23 inactive lanes：`789,984`。
  - post-len16 pressure-PML sampled-main share：`46.58%`。
  - direct-fill -> len16 observed pressure-PML speedup：`1.1869x`。
  - direct-fill -> len16 lane ceiling：`1.4638x`。
  - observed lane-to-time efficiency factor：`0.811`。
  - `exact_length23_points_only`：
    - lane reduction vs len16：`3.97%`
    - p-PML lane ceiling：`1.0413x`
    - sampled-main ceiling：`1.0188x`
    - calibrated sampled-main estimate：`1.0153x`
    - descriptor traffic：`7.701 MiB/step aggregate-shots`
    - descriptor bytes per saved inactive lane：`10.23`
  - `exact_all_active_points`：
    - lane reduction vs len16：`3.97%`
    - sampled-main ceiling：`1.0188x`
    - calibrated sampled-main estimate：`1.0153x`
    - descriptor traffic：`72.933 MiB/step aggregate-shots`
    - descriptor bytes per saved inactive lane：`96.83`
- 风险与下一步：
  - 决策：拒绝 exact active-point / compact descriptor CUDA prototype。
  - 决策：拒绝简单 length-23 pressure-PML prototype。
  - 原因：len16 已消化主要 lane waste，剩余 length-23 lane savings 的 optimistic sampled-main ceiling 也不到 `2%`，低于 `>=5%` prototype gate；descriptor/control overhead 还未计入，真实收益会更低。
  - 只有新 descriptor/ownership 设计证明扣除 overhead 后仍有 `>=5%` `perf_1gpu_6shots` repeat speedup ceiling，才允许重开。
  - 下一步转向 `cuda_fd3d_p_pml_len16_halfwarp_ns` source-level drill-down 或 v-PML memory layout/coalescing 设计。

## 2026-06-08 16:58:00 +08:00 - Len16 source-level NCU profile and micro-route rejection

- 操作目标：
  - 对 accepted len16 packed kernel `cuda_fd3d_p_pml_len16_halfwarp_ns` 做 source-level Nsight Compute drill-down。
  - 找出 `No Eligible 73%+` 的具体源代码热点。
  - 决定是否打开 len16-only 微优化 prototype。
- 修改文件：
  - 新增报告：`docs/day_20260608/len16_halfwarp_source_profile.md`。
  - 新增 artifacts：
    - `reports/day_20260608/len16_source_profile_20260608_1646/details.csv`
    - `reports/day_20260608/len16_source_profile_20260608_1646/details_summary.md`
    - `reports/day_20260608/len16_source_profile_20260608_1646/details_summary.json`
    - `reports/day_20260608/len16_source_profile_20260608_1646/source_hotlines.md`
    - `reports/day_20260608/len16_source_profile_20260608_1646/source_hotlines.json`
    - `reports/day_20260608/len16_source_profile_20260608_1646/lineinfo_bin.sha256`
    - `reports/day_20260608/len16_source_profile_20260608_1646/final_release_bin.sha256`
    - `reports/day_20260608/len16_source_profile_20260608_1646/ncu_run.log`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端隔离 worktree：`/work/wenzhe/cuda3D/.codex_worktrees/sprint_0648`。
  - 检查 GPU：RTX 5090，约 `481 MiB` 显存占用，`0%` util。
  - 编译 accepted len16 candidate + `-lineinfo`。
  - 运行 Nsight Compute：
    - `--section SourceCounters`
    - `--section SchedulerStats`
    - `--section WarpStateStats`
    - `--section MemoryWorkloadAnalysis`
    - `--section Occupancy`
    - `--launch-skip 10`
    - `--launch-count 10`
    - `--kernel-name regex:.*cuda_fd3d_p_pml_len16_halfwarp.*`
  - 导出：
    - details CSV。
    - raw CSV。
    - source page。
    - `cuda,sass` source page with `--resolve-source-file src/single_solver.cu`。
  - 远端解析 `source_page_cuda_sass.txt`，生成 `source_hotlines.md/json`；大原始 `.ncu-rep` 和 source page 留在服务器，不提交。
  - profile 后重新编译无 `-lineinfo` release len16 binary，并运行 smoke。
- 测试结果：
  - NCU source profile 成功，profiled `cuda_fd3d_p_pml_len16_halfwarp_ns` 10 launches，每个 15 passes。
  - details CSV 生成成功，`431` 行。
  - source page 生成成功，`cuda,sass` source page 可解析到 `src/single_solver.cu` 行号。
  - final release rebuild 成功。
  - post-profile smoke 通过：
    - run：`benchmarks/runs/smoke_1gpu_len16_after_source_profile_restore_20260608_165211`
    - 输出包含 `ALL DONE`
- 输出/哈希/误差摘要：
  - lineinfo binary SHA256：见 `reports/day_20260608/len16_source_profile_20260608_1646/lineinfo_bin.sha256`。
  - final release binary SHA256：`77ba44c3f94fc5992b07b01ee786bfadf6c2a4671fc8e755dace2bcef9b31c58`。
  - kernel summary：
    - No Eligible：`73.545%`
    - issued warp/scheduler：`0.264`
    - active warps/scheduler：`8.986`
    - eligible warps/scheduler：`0.427`
    - warp cycles/issued instruction：`33.970`
    - avg active threads/warp：`26.380`
    - avg not-predicated threads/warp：`24.910`
    - branch efficiency：`65.220%`
    - avg divergent branches：`316.610`
    - achieved occupancy：`74.912%`
    - L1/TEX hit：`61.537%`
    - L2 hit：`54.157%`
    - NCU CPI stall：L1TEX scoreboard dependency 约 `24.6 cycles/warp`
  - parsed source hot lines，total parsed samples `15,712`：
    - line 1813 `p0[base]=2*__ldg(p1+base)-p0[base]`：`5,660` samples，`36.02%`
    - line 1814 `+__ldg(cw2+base)*dt*(c1+c2+c3);`：`3,890` samples，`24.76%`
    - line 1810 upper z `mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);`：`3,287` samples，`20.92%`
    - line 1804 lower z `mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);`：`927` samples，`5.90%`
    - z-cache shared-load lines each below `1%` parsed samples.
- 风险与下一步：
  - 决策：不写 len16-only `p0 __ldg`、local `new_mem`、branch-only lower/upper/margin specialization、或 z-cache/shared-memory 小修 prototype。
  - 原因：direct-fill 路线已经拒绝过 `p0 __ldg` 与 `new_mem`，当前 source profile 显示瓶颈仍是 final pressure writeback 与 CPML z-state dependency，不是简单表达式写法。
  - branch efficiency 虽低，但热点行不是 branch 本身；branch-only specialization 没有 `>=5%` speedup ceiling 且会增加 tile-list/launch overhead。
  - 下一步转向 v-PML memory layout/coalescing design，或提出更大粒度 pressure-PML memory-ownership design，但必须先证明 `>=5%` repeat speedup ceiling。

## 2026-06-08 17:23:20 +08:00 - V-PML coalescing/layout gate rejects CUDA prototype

- 操作目标：
  - 按照 Phase 4.12 的下一步，对 v-PML memory layout / coalescing 方向做 gate。
  - 判断是否值得写 v-only tile-layout CUDA prototype，还是继续禁止随机 `PmlTileBlockSize` sweep。
- 修改文件：
  - 新增工具：`tools/v_pml_coalescing_layout_budget.py`。
  - 新增报告：`docs/day_20260608/v_pml_coalescing_layout_budget.md`。
  - 新增 JSON：`reports/day_20260608/v_pml_coalescing_layout_budget.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地检查状态：`git status --short --branch`。
  - 读取既有证据：
    - `docs/day_20260608/v_pml_source_profile_gate.md`
    - `reports/day_20260608/directfill_v_pml_source_summary.md/json`
    - `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json`
    - `src/single_solver.cu` 中 `cuda_fd3d_v_pml_tile_ns`
    - `src/rem_fd.cu` 中 `build_pml_tile_list`
  - 初版逐 warp sector 精算模型超时：
    - `python tools/v_pml_coalescing_layout_budget.py --json-out reports/day_20260608/v_pml_coalescing_layout_budget.json --md-out docs/day_20260608/v_pml_coalescing_layout_budget.md`
    - 该进程被明确停止：`Stop-Process -Id 58816`
  - 将模型改为快速 gate：
    - 精确统计 tile/lane/component work。
    - 用 warp z-segment split factor 建模 coalescing 下界，而不是伪装成硬件 sector 计数。
  - 重新执行：
    - `python -m py_compile tools/v_pml_coalescing_layout_budget.py`
    - `python tools/v_pml_coalescing_layout_budget.py --json-out reports/day_20260608/v_pml_coalescing_layout_budget.json --md-out docs/day_20260608/v_pml_coalescing_layout_budget.md`
  - 远端只读状态检查：
    - Windows Python 直接运行 `tools/remote_exec.py` 失败：缺少本地 `paramiko` 模块。
    - 改用 WSL Python 与临时环境变量密码重试成功。
    - 命令：
      - `cd /work/wenzhe/cuda3D`
      - `git status --short --branch`
      - `git log --oneline -3`
- 测试结果：
  - `py_compile` 通过。
  - v-PML coalescing/layout budget 生成成功。
  - 本轮没有修改 CUDA 求解器源码，没有运行 correctness/perf。
  - 远端状态检查成功，但未修改服务器文件。
- 输出/哈希/误差摘要：
  - NCU anchor 使用 accepted len16 summary：
    - sampled main：`297.248us`。
    - `cuda_fd3d_v_pml_tile_ns`：`65.248us`。
    - v-PML sampled-main share：`21.95%`。
    - 若只优化 v-PML，要达到 `>=5%` sampled-main gain，v kernel 需要约 `1.2770x` speedup。
  - 当前 `32x4x2` 映射：
    - `threadIdx.x` 对应 z。
    - 一个 warp 是固定 x/y 的连续 32 个 z-lane。
    - 对 `p1`、`mem_dx`、`mem_dy` 主路径已经是有利 coalescing 形态。
  - best reasoned shape：`z8_x8_y4`。
    - launched lanes ratio vs current：`0.8830`。
    - warp z segments：`4`。
    - optimistic v-kernel speedup ceiling：`1.1325x`。
    - optimistic sampled-main ceiling：`1.0264x`。
  - 远端 `/work/wenzhe/cuda3D` 当前状态：
    - branch：`exp/wavestep-v2-shared-vp-night`。
    - 工作树有既有修改和未跟踪目录，包括 `AGENTS.md`、`AGENT_LOG.md`、`docs/day_20260608/`、`reports/day_20260608/`、`.codex_worktrees/`、`src/*.cu` 等。
    - 本轮未在远端执行 pull、reset、覆盖或文件写入。
- 风险与下一步：
  - 决策：拒绝 v-only tile-layout CUDA prototype。
  - 决策：继续禁止随机 `PmlTileBlockSize` sweep。
  - 决策：继续禁止 current-geometry vx/vy component split。
  - 原因：最佳 reasoned shape 的 optimistic sampled-main ceiling 只有 `2.64%`，低于 `>=5%` gate，而且还没计入 separate velocity tile-list plumbing、control overhead 和 pressure-path compatibility 成本。
  - 远端根目录当前不干净，不适合直接同步覆盖；后续若要服务器复现，应使用干净 worktree 或先整理远端未提交状态。
  - 下一步应转向更大粒度 memory ownership / wave-step scheduling 设计，重点是减少 `vx/vy` global round trip，或减少 pressure final writeback / CPML state dependency。

## 2026-06-08 18:00:29 +08:00 - Wave-step async streams prototype tested and rejected

- 操作目标：
  - 从更大粒度 wave-step scheduling 入手，评估 `p_core` 与 PML path 是否可以通过 CUDA streams 并行获得有意义提速。
  - 若模型放行，则实现宏默认关闭 prototype，进行 smoke、correctness、`perf_1gpu_6shots` repeat 验收。
- 修改文件：
  - 新增工具：`tools/wavestep_stream_overlap_model.py`。
  - 新增模型报告：`docs/day_20260608/wavestep_stream_overlap_model.md`。
  - 新增模型 JSON：`reports/day_20260608/wavestep_stream_overlap_model.json`。
  - 新增 prototype 报告：`docs/day_20260608/wavestep_async_streams_prototype.md`。
  - 新增远端取回 perf repeat summary：
    - `reports/day_20260608/wavestep_async_perf6_repeat_20260608_175407/summary.md`
    - `reports/day_20260608/wavestep_async_perf6_repeat_20260608_175407/summary.json`
    - `reports/day_20260608/wavestep_async_perf6_repeat_20260608_175407/base_r1.sha256`
    - `reports/day_20260608/wavestep_async_perf6_repeat_20260608_175407/async_r1.sha256`
  - 新增远端取回 correctness compare：
    - `reports/day_20260608/wavestep_async_correctness_compare_20260608_175029/comparison.md`
    - `reports/day_20260608/wavestep_async_correctness_compare_20260608_175029/comparison.json`
    - `reports/day_20260608/wavestep_async_correct_base.sha256`
    - `reports/day_20260608/wavestep_async_candidate.sha256`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 源码处理：
  - 临时在 `src/rem_fd.cu` 实现过 `CUDA3D_WAVESTEP_ASYNC_STREAMS`：
    - `p_core` 走 non-blocking core stream。
    - `v_pml -> p_pml_len16 -> p_pml_residual` 走 non-blocking PML stream。
    - default stream 在 source injection/extraction 前等待 core/PML stream。
  - prototype 通过 correctness 但未达到性能 gate，已从本地源码移除；最终待提交 diff 不包含 `src/rem_fd.cu`。
- 执行命令摘要：
  - 本地：
    - `python -m py_compile tools/wavestep_stream_overlap_model.py`
    - `python tools/wavestep_stream_overlap_model.py --json-out reports/day_20260608/wavestep_stream_overlap_model.json --md-out docs/day_20260608/wavestep_stream_overlap_model.md`
  - 远端：
    - 只读检查 `/work/wenzhe/cuda3D`，发现主目录 dirty 且在旧分支，不直接覆盖。
    - 创建干净 worktree：
      - `/work/wenzhe/cuda3D/.codex_worktrees/async_streams_20260608_1738`
    - 上传临时 `src/rem_fd.cu` 与模型报告文件到该 worktree。
    - 构建 async candidate：
      - `make -B -f makefile.rtx5090 test NVFLAGS="... -DCUDA3D_WAVESTEP_ASYNC_STREAMS"`
    - smoke：
      - `python3 tools/run_benchmark.py --case smoke_1gpu --tag async_streams_smoke_datafixed_flags`
    - correctness A/B：
      - base：`python3 tools/run_benchmark.py --case correctness --tag len16_base_for_async`
      - candidate：`python3 tools/run_benchmark.py --case correctness --tag async_streams_candidate`
      - compare：`tools/compare_outputs.py`
    - perf repeat 3 轮：
      - 每轮重建 base binary。
      - 每轮运行 `perf_1gpu_6shots` base。
      - 每轮重建 async binary。
      - 每轮运行 `perf_1gpu_6shots` async candidate。
      - 每轮输出 compare。
- 环境/流程问题记录：
  - 第一次远端 build 使用 `export NVFLAGS=...`，但 `makefile.rtx5090` 内部定义 `NVFLAGS = ...`，环境变量不会覆盖 makefile 变量；修正为 command-line override：`make ... NVFLAGS="..."`。
  - 新建 worktree 初始缺少 `bench_smoke/d_obs` 输出目录，旧程序写输出时会崩溃；补齐目录后 smoke 正常。
  - 新建 worktree 初始缺少 `perf_1gpu_6shots` velocity symlink；补 `vel_perf_1gpu_6shots_ny384_nx384_nz95.dir` symlink 后 perf case 正常。
  - 以上均为隔离 worktree 的测试数据布置问题，未修改远端主目录。
- 测试结果：
  - 模型 gate：
    - accepted len16 sampled main：`297.248us`。
    - `p_core`：`93.547us`。
    - `v_pml + pressure-PML serial path`：`203.701us`。
    - conservative two-stream ceiling：`1.4592x`。
    - 达到 `1.05x` sampled-main 只需 `15.13%` realized overlap，因此允许 prototype。
  - async build 通过：
    - binary SHA256：`78ea9c9ee37328ff913e9a403b8abe3ec3e3c2f232790ccb104e3dcdcd2e0f86`。
  - smoke 通过：
    - run：`benchmarks/runs/smoke_1gpu_async_streams_smoke_datafixed_flags_20260608_174937`
    - returncode `0`，outputs `3`，包含 `ALL DONE`。
  - correctness 通过：
    - base：`benchmarks/runs/correctness_len16_base_for_async_20260608_175017`
    - candidate：`benchmarks/runs/correctness_async_streams_candidate_20260608_175026`
    - compare：`reports/day_20260608/wavestep_async_correctness_compare_20260608_175029`
    - 6 个输出 rel L2 全部 `0`。
  - `perf_1gpu_6shots` repeat 通过数值 compare，但性能未达 gate：
    - round 1：WP `1.004085x`，Gradient `1.002251x`
    - round 2：WP `1.005962x`，Gradient `1.002730x`
    - round 3：WP `1.005502x`，Gradient `1.003585x`
    - mean WP speedup：`1.005183x`
    - mean Gradient speedup：`1.002855x`
    - all compare pass：`True`
- 输出/哈希/误差摘要：
  - correctness max rel L2：`0`。
  - perf repeat compare：3 轮全部 pass。
  - perf summary：`reports/day_20260608/wavestep_async_perf6_repeat_20260608_175407/summary.md`。
  - async correctness candidate SHA：见 `reports/day_20260608/wavestep_async_candidate.sha256`。
- 风险与下一步：
  - 决策：拒绝 `CUDA3D_WAVESTEP_ASYNC_STREAMS` prototype，不进入主线。
  - 原因：模型上界成立，但实际硬件资源 contention 几乎吃掉 overlap，repeat WP 只有约 `0.5%`，低于 `>=5%` meaningful prototype gate。
  - 禁止重复 two-stream `p_core` vs PML overlap。
  - 禁止基于本结果写 single-GPU CUDA Graph / launch aggregation。
  - 三 stream pressure residual/len16 fanout 只有在 Nsight Systems 证明有真实 concurrent execution headroom 且新模型证明扣除 contention 后仍有 `>=5%` repeat speedup ceiling 时才允许重开。
  - 下一步回到实际减少 global memory work 的 ownership 设计，重点关注 pressure-PML final `p0/cw2` writeback 与 CPML z-state dependency。

## 2026-06-08 18:17:54 +08:00

- 操作目标：
  - 根据上一轮 async overlap 拒绝后的下一步，审查 pressure-PML final writeback 与 CPML z-state dependency 是否值得打开新的 CUDA micro prototype。
  - 将 len16 SourceCounters 热点转成 Amdahl ceiling 与 stop-rule，避免后续重复已证伪的小修路线。
- 修改文件：
  - 新增 `tools/pressure_pml_writeback_state_model.py`。
  - 新增 `docs/day_20260608/pressure_pml_writeback_state_model.md`。
  - 新增 `reports/day_20260608/pressure_pml_writeback_state_model.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `git status --short --branch`
  - `git log --oneline -5`
  - `python -m py_compile tools\pressure_pml_writeback_state_model.py`
  - `python tools\pressure_pml_writeback_state_model.py --json-out reports\day_20260608\pressure_pml_writeback_state_model.json --md-out docs\day_20260608\pressure_pml_writeback_state_model.md`
  - `python -c "import json, pathlib; json.load(open('reports/day_20260608/pressure_pml_writeback_state_model.json', encoding='utf-8')); print('json ok')"`
- 测试结果：
  - 模型脚本 Python 编译通过。
  - 生成 JSON 可被标准 `json.load` 读取。
  - 本轮未修改 CUDA 源码，未运行远端构建或 benchmark。
- 输出/哈希/误差摘要：
  - accepted len16 sampled main：`297.248us`。
  - len16 packed pressure-PML：`65.771us`，sampled-main share `22.13%`。
  - total pressure-PML：`138.453us`，sampled-main share `46.58%`。
  - len16 source parsed samples：`15,712`。
  - final `p0/p1/cw2` update：`60.78%` len16 source samples。
  - CPML `mem_dzz` update：`26.82%`。
  - z-cache shared loads：`1.92%`。
  - packed len16 kernel speedup required for `1.05x` sampled-main：`1.2742x`。
  - final `p0/p1/cw2` group speedup required if alone：`1.5482x`。
  - CPML `mem_dzz` group speedup required if alone：`5.0614x`。
  - final + `mem_dzz` group speedup required：`1.3257x`。
- 风险与下一步：
  - 决策：拒绝 pressure-PML writeback/state micro CUDA prototype。
  - 禁止重试 len16 `p0 __ldg`、old-p0 read syntax、explicit local `new_mem`、ptxas cache-policy、branch-only lower/upper specialization、或 accepted len16 z-cache fill/shared-cache 微调。
  - 允许重开条件：必须是状态表示或时间推进设计，证明 old-`p0`/`cw2` 或 `mem_dzz` traffic 被真正移除，并且扣除额外 storage/control 后仍有 `>=5%` `perf_1gpu_6shots` repeat speedup ceiling。
  - 下一步建议：
    - 若继续底层重写，做 math-level pressure state representation / PML ownership design gate。
    - 或先同 session 重跑 zmem/direct-fill/len16/current-best，生成正式总提速表并固化当前 best。

## 2026-06-08 18:28:45 +08:00

- 操作目标：
  - 按 Phase 4.15 的下一步建议，重跑 `zmem`、`directfill`、`len16_current_best` 三者同机同 session formal speed table。
  - 固化当前 RTX 5090 single-GPU `perf_1gpu_6shots` formal best，并验证数值差异。
- 修改文件：
  - 新增 `reports/day_20260608/formal_current_best_table_20260608_182525/summary.md`。
  - 新增 `reports/day_20260608/formal_current_best_table_20260608_182525/summary.json`。
  - 新增 `reports/day_20260608/formal_current_best_table_20260608_182525/records.jsonl`。
  - 新增 `reports/day_20260608/formal_current_best_table_20260608_182525/remote_paths.txt`。
  - 新增 6 组 output comparison artifact：
    - `compare_directfill_r{1,2,3}_vs_zmem/comparison.{md,json}`
    - `compare_len16_r{1,2,3}_vs_zmem/comparison.{md,json}`
  - 新增 9 组轻量 run artifact：
    - `run_artifacts/{zmem,directfill,len16_current_best}_r{1,2,3}/run.log`
    - `run_artifacts/{zmem,directfill,len16_current_best}_r{1,2,3}/manifest.json`
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端只读检查：
    - `cd /work/wenzhe/cuda3D`
    - `git status --short --branch`
    - `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader`
  - 远端创建隔离 worktree：
    - `git -c http.proxy= -c https.proxy= fetch origin exp/day-20260608-cpml-compact-temporal`
    - `git worktree add --detach /work/wenzhe/cuda3D/.codex_worktrees/formal_table_20260608_182525 FETCH_HEAD`
  - 远端环境：
    - `source ./env_5090.sh`
    - 新建 case output dir：`benchmarks/cases/perf_1gpu_6shots/d_obs`
    - 如缺失则为 velocity 数据创建 symlink 到主目录已有数据文件。
  - 每轮 protocol：
    - `make -B -f makefile.rtx5090 test NVFLAGS="<zmem flags>"`
    - `python3 tools/run_benchmark.py --case perf_1gpu_6shots --tag formal_zmem_rN`
    - `make -B -f makefile.rtx5090 test NVFLAGS="<directfill flags>"`
    - `python3 tools/run_benchmark.py --case perf_1gpu_6shots --tag formal_directfill_rN`
    - `python3 tools/compare_outputs.py --baseline <same-round-zmem>/outputs --candidate <directfill>/outputs --out <report>/compare_directfill_rN_vs_zmem`
    - `make -B -f makefile.rtx5090 test NVFLAGS="<len16 flags>"`
    - `python3 tools/run_benchmark.py --case perf_1gpu_6shots --tag formal_len16_current_best_rN`
    - `python3 tools/compare_outputs.py --baseline <same-round-zmem>/outputs --candidate <len16>/outputs --out <report>/compare_len16_rN_vs_zmem`
  - 本地取回：
    - `tools/remote_get.py` 拉取 summary、records、comparison、run.log、manifest.json。
- 测试结果：
  - 远端 GPU 运行前基本空闲：RTX 5090 memory used `481 MiB`，utilization `0%`。
  - `zmem`、`directfill`、`len16_current_best` 各 3 轮均构建和运行完成。
  - 6 组 output comparison 全部 pass。
- 输出/哈希/误差摘要：
  - `directfill` vs `zmem`：
    - mean WP speedup：`1.099957x`。
    - mean Gradient speedup：`1.097977x`。
    - mean elapsed speedup：`1.105408x`。
    - mean candidate WP：`2.203315s`。
    - max rel L2：`0`。
    - max abs：`0`。
  - `len16_current_best` vs `zmem`：
    - mean WP speedup：`1.192835x`。
    - mean Gradient speedup：`1.179213x`。
    - mean elapsed speedup：`1.156108x`。
    - mean candidate WP：`2.031753s`。
    - max rel L2：`6.384336e-07`。
    - max abs：`4.768372e-06`。
  - report：`reports/day_20260608/formal_current_best_table_20260608_182525/summary.md`。
- 风险与下一步：
  - 决策：`CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK` + direct-fill z-cache + CPML vmem double-buffer scaffold 是当前 RTX 5090 single-GPU formal best。
  - 正式 WP speedup vs `zmem_reference`：`1.192835x`。
  - 该结果未达到 `1.5x` archive threshold，不创建 `archives/speedups/1.5x_*`。
  - 下一步若继续底层核心重写，应先做 math-level pressure state representation / PML ownership design gate；不要回到已拒绝的 micro routes。

## 2026-06-08 18:40:06 +08:00

- 操作目标：
  - 继续 Phase 4.16 之后的底层核心重写路线，对 pressure state representation 做 math-level gate。
  - 判断 `q=p/cw2`、delta pressure state、first-order full-domain velocity-pressure 等看似结构性改写是否值得打开 CUDA prototype。
- 修改文件：
  - 新增 `tools/pressure_state_representation_model.py`。
  - 新增 `docs/day_20260608/pressure_state_representation_model.md`。
  - 新增 `reports/day_20260608/pressure_state_representation_model.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `git status --short --branch`
  - `git log --oneline -5`
  - `Select-String -Path "src\single_solver.cu" -Pattern "__global__ void cuda_fd3d_p_core_ns|__global__ void cuda_fd3d_v_pml_tile_ns|__global__ void cuda_fd3d_p_pml_len16_halfwarp_ns" -Context 0,90`
  - `python -m py_compile tools\pressure_state_representation_model.py`
  - `python tools\pressure_state_representation_model.py --json-out reports\day_20260608\pressure_state_representation_model.json --md-out docs\day_20260608\pressure_state_representation_model.md`
  - `python -c "import json; json.load(open('reports/day_20260608/pressure_state_representation_model.json', encoding='utf-8')); print('pressure state json ok')"`
- 测试结果：
  - 模型脚本 Python 编译通过。
  - 生成 JSON 可被标准 `json.load` 读取。
  - 本轮未修改 CUDA 源码，未运行远端构建或 benchmark。
- 输出/哈希/误差摘要：
  - sampled main：`297.248us`。
  - `p_core` sampled-main share：`31.47%`。
  - `v_pml` sampled-main share：`21.95%`。
  - pressure-PML sampled-main share：`46.58%`。
  - len16 packed pressure-PML sampled-main share：`22.13%`。
  - formal current-best WP speedup vs zmem：`1.192835x`。
  - 当前二阶 pressure update 每点最小 state traffic：`16B`：
    - `p_prev_read` `4B`
    - `p_cur_read` `4B`
    - `cw2_read` `4B`
    - `p_next_write` `4B`
  - `delta_pressure_state`：traffic `16B -> 20B`，sampled-main effect 若全 pressure update 使用约 `0.8957x`。
  - `scaled_pressure_q_only` (`q=p/cw2`)：`p_core` 至少需要 `>=29` 个 pressure value reconstruction per output，`p_core+v_pml` 合计 `53.42%` sampled-main 处于风险区。
  - `scaled_pressure_dual_p_and_q`：traffic `16B -> 32B`。
  - `first_order_full_domain_velocity_pressure`：不是 bitwise 等价；每 core 点最多省 `4B` old-p read，但至少新增 `24B` velocity state read/write。
  - `half_or_compressed_cw2`：当前精度契约下拒绝；理想 len16 cw2-line sampled-main ceiling 仅 `1.0282x`。
  - `cpml_mem_dzz_rescaled_state`：`mem_dzz` alone 需要 `5.0614x` local speedup 才能触及 gate，代数重标定不减少 recursive state read/write。
- 风险与下一步：
  - 决策：拒绝 pressure state representation CUDA prototype。
  - 禁止实现 `q=p/cw2`、delta pressure state、dual `p/q`、first-order full-domain velocity-pressure、precomputed `cw2dt`、compressed `cw2` 或 CPML `mem_dzz` rescale prototype。
  - 下一步转向 PML `vx/vy` round-trip ownership design；必须先有 `>=5%` model 再写 CUDA。
  - source-aware multi-step/wavefront 只有先解决 synchronization/halo ownership 才允许重开。
  - precision-relaxation 只有用户明确给出新 tolerance policy 才允许研究。

## 2026-06-08 18:57:34 +08:00

- 操作目标：
  - 继续 Phase 4.17 后的下一条 CUDA-core exact 路线，对 PML `vx/vy` global round-trip ownership 做 gate。
  - 判断 CTA-local / macro-tile shared-cache 是否能在不重复过多 velocity/CPML work 的前提下去掉 `vx/vy` global write/read round trip。
- 修改文件：
  - 新增 `tools/pml_vxvy_roundtrip_ownership_model.py`。
  - 新增 `docs/day_20260608/pml_vxvy_roundtrip_ownership_model.md`。
  - 新增 `reports/day_20260608/pml_vxvy_roundtrip_ownership_model.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `Get-Content -Encoding UTF8 docs\day_20260608\v_pml_coalescing_layout_budget.md`
  - `Get-Content -Encoding UTF8 docs\day_20260608\pressure_state_representation_model.md`
  - `Select-String -Path AGENTS.md -Pattern "vx/vy|round-trip|component-owner|RECOMPUTE_X|shared ``vx" -Context 1,1`
  - `Select-String -Path src\single_solver.cu -Pattern "if \(need_vx\)|if \(need_vy\)|float c2=stencil|float c3=stencil|__ldg\(vx|__ldg\(vy" -Context 2,6`
  - `python -m py_compile tools\pml_vxvy_roundtrip_ownership_model.py`
  - `python tools\pml_vxvy_roundtrip_ownership_model.py --json-out reports\day_20260608\pml_vxvy_roundtrip_ownership_model.json --md-out docs\day_20260608\pml_vxvy_roundtrip_ownership_model.md`
  - `python -c "import json; json.load(open('reports/day_20260608/pml_vxvy_roundtrip_ownership_model.json', encoding='utf-8')); print('vxvy model json ok')"`
- 测试结果：
  - 模型脚本 Python 编译通过。
  - 生成 JSON 可被标准 `json.load` 读取。
  - 本轮未修改 CUDA 源码，未运行远端构建或 benchmark。
- 输出/哈希/误差摘要：
  - sampled main：`297.248us`。
  - `p_core`：`93.547us`。
  - `v_pml`：`65.248us`。
  - pressure-PML total：`138.453us`。
  - formal current-best WP speedup vs zmem：`1.192835x`。
  - generous savable-time model：
    - len16 unknown/unparsed source time 全部算给 `vx/vy`：`4.056us`。
    - residual pressure-PML 慷慨假设 `20%` 可省：`14.537us`。
    - total generous `vx/vy` round-trip savable time：`18.593us`。
  - 要达到 `1.05x` sampled-main speedup，duplicate velocity/CPML work factor 必须 `<=1.068`。
  - CTA-local / macro-tile 结果：
    - `4x2` current pressure tile：duplicate `4.085x`，sampled-main speedup `0.6193x`。
    - `8x4`：duplicate `2.606x`，sampled-main speedup `0.7752x`。
    - `16x8`：duplicate `1.866x`，sampled-main speedup `0.8868x`。
    - `16x16`：duplicate `1.620x`，shared cache `94,208B`，sampled-main speedup `0.9315x`。
    - `32x8` / `32x16`：duplicate `1.743x` / `1.497x`，shared cache `101,376B` / `174,080B`，超出 conservative `96KiB` limit。
    - ideal no-duplicate cross-CTA owner：speedup ceiling `1.0667x`，但 ordinary CUDA 不能实现跨 CTA register/shared exchange。
- 风险与下一步：
  - 决策：拒绝 PML `vx/vy` round-trip ownership CUDA prototype。
  - 禁止写 CTA-local `vx/vy` shared-cache fusion、`RECOMPUTE_X/Y/XYZ`、direct p1 x/y derivative replacement、current-geometry `vx/vy` component-owner split、或依赖 cross-CTA shared values 的 ordinary CUDA producer-consumer fusion。
  - source-aware multi-step / wavefront design 只有先证明 synchronization 和 halo ownership 后才允许重开。
  - precision-relaxation 只有用户明确给出新 tolerance policy 才允许研究。
  - 如果 exact CUDA-core 路线继续被 gate 掐掉，可以转向 application-level multi-shot batching。

## 2026-06-08 19:07:44 +08:00

- 操作目标：
  - 继续 Phase 4.18 后的最后一条 exact CUDA-core temporal 路线，对 source-aware K=2 wavefront / synchronization 做 current-best rebase gate。
  - 判断 source/receiver 已兼容后，普通 CUDA 是否仍存在可实现的 no-duplicate wavefront temporal prototype。
- 修改文件：
  - 新增 `tools/source_aware_wavefront_sync_model.py`。
  - 新增 `docs/day_20260608/source_aware_wavefront_sync_model.md`。
  - 新增 `reports/day_20260608/source_aware_wavefront_sync_model.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `git status --short`
  - `Get-ChildItem -Path tools -Filter source_aware_wavefront_sync_model.py`
  - `Get-Content -Encoding UTF8 tools\source_aware_wavefront_sync_model.py -TotalCount 80`
  - `python -m py_compile tools\source_aware_wavefront_sync_model.py`
  - `Test-Path reports\day_20260608\temporal_pipeline_model.json`
  - `Test-Path reports\day_20260608\source_aware_temporal_model.json`
  - `Test-Path reports\day_20260608\len16_vs_directfill_ncu_20260608_1600\summary.json`
  - `Test-Path reports\day_20260608\formal_current_best_table_20260608_182525\summary.json`
  - `python tools\source_aware_wavefront_sync_model.py --json-out reports\day_20260608\source_aware_wavefront_sync_model.json --md-out docs\day_20260608\source_aware_wavefront_sync_model.md`
  - `python -c "import json; json.load(open('reports/day_20260608/source_aware_wavefront_sync_model.json', encoding='utf-8')); print('json ok')"`
- 测试结果：
  - 模型脚本 Python 编译通过。
  - 生成 JSON 可被标准 `json.load` 读取。
  - 本轮未修改 CUDA 源码，未运行远端构建或 benchmark。
- 输出/哈希/误差摘要：
  - current-best sampled main：`297.248us`。
  - current-best `p_core`：`93.547us`，sampled-main share `31.47%`。
  - formal current-best WP speedup vs zmem：`1.192835x`。
  - ideal K=2 p_core pair reduction：`35.25%`。
  - ideal K=2 sampled-main speedup on current best：`1.1248x`。
  - 达到 `1.05x` sampled-main speedup 所需 p_core reduction：`15.13%`，即 ideal saving 的 `42.92%`。
  - aggregate K=2 deep-core share：`73.22%`。
  - source overlap shots：`0`。
  - receiver overlap shots：`0`。
  - p_core grid blocks：`70688`。
  - conservative resident block capacity：`1360`。
  - cooperative-grid over-capacity factor：`51.98x`。
  - candidate gate：
    - `safe_global_middle_two_kernel`：speedup ceiling `1.0000x`，拒绝。
    - `cooperative_grid_full_core_k2`：ideal `1.1248x`，但 grid 超 resident capacity 约 `52x`，拒绝。
    - `cta_local_diamond_k2`：需要 `11.29x` 到 `21.30x` baseline pair bytes，拒绝。
    - `multi_kernel_global_wavefront`：speedup ceiling `1.0000x`，拒绝。
    - `persistent_wavefront_without_global_barrier`：依赖普通 CUDA 不提供的跨 CTA shared/register ownership，拒绝。
    - `ideal_no_dup_source_aware_wavefront`：有 meaningful ceiling，但不是 ordinary CUDA implementation。
- 风险与下一步：
  - 决策：拒绝 source-aware K=2 wavefront CUDA prototype。
  - 禁止写 ordinary CUDA K=2 source-aware wavefront、multi-kernel global-middle wavefront、CTA-local diamond temporal、或依赖 cross-CTA shared/register values 的 persistent-kernel wavefront。
  - 今日 exact CUDA-core 结构性路线基本收口；后续建议转向 application-level multi-shot batching / scheduling。
  - precision relaxation 只有用户明确放宽 tolerance policy 后才允许研究。
  - no-duplicate wavefront temporal blocking 只有在发现具体 hardware/runtime cross-CTA ownership primitive 后才允许重开。

## 2026-06-08 19:36:29 +08:00

- 操作目标：
  - 继续 heartbeat 自动推进，将 exact CUDA-core route 收口后转向 application-level scheduling。
  - 在 RTX 5090 上做 same-GPU multi-rank probe，验证 `np=2/3` 共享同一张 GPU 分炮是否值得进入 repeat benchmark。
- 修改文件：
  - 新增 `reports/day_20260608/multirank_samegpu_sched_20260608_193042/summary.md`。
  - 新增 `reports/day_20260608/multirank_samegpu_sched_20260608_193042/summary.json`。
  - 新增 `reports/day_20260608/multirank_samegpu_sched_20260608_193042/binary.sha256`。
  - 新增 `reports/day_20260608/multirank_samegpu_sched_20260608_193042/velocity_link.txt`。
  - 新增 `reports/day_20260608/multirank_samegpu_sched_20260608_193042/compare_np2_vs_np1/comparison.md`。
  - 新增 `reports/day_20260608/multirank_samegpu_sched_20260608_193042/compare_np2_vs_np1/comparison.json`。
  - 新增 `reports/day_20260608/multirank_samegpu_sched_20260608_193042/compare_np3_vs_np1/comparison.md`。
  - 新增 `reports/day_20260608/multirank_samegpu_sched_20260608_193042/compare_np3_vs_np1/comparison.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端检查：
    - `cd /work/wenzhe/cuda3D`
    - `git status --short --branch`
    - `git log --oneline -3`
    - `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader`
  - 远端创建隔离 worktree：
    - `git -c http.proxy= -c https.proxy= fetch origin exp/day-20260608-cpml-compact-temporal`
    - `git worktree add --detach /work/wenzhe/cuda3D/.codex_worktrees/multirank_samegpu_20260608_193042 FETCH_HEAD`
  - 远端构建：
    - `source ./env_5090.sh`
    - `make -B -f makefile.rtx5090 test NVFLAGS="<current-best flags>"`
  - 远端 case 修复：
    - `mkdir -p benchmarks/cases/perf_1gpu_6shots/d_obs`
    - `ln -s /work/wenzhe/cuda3D/benchmarks/cases/perf_1gpu_6shots/vel_perf_1gpu_6shots_ny384_nx384_nz95.dir benchmarks/cases/perf_1gpu_6shots/vel_perf_1gpu_6shots_ny384_nx384_nz95.dir`
  - 远端 benchmark：
    - `python3 tools/run_benchmark.py --case perf_1gpu_6shots --tag sched_samegpu_np1_rerun --np 1 --gpus 0 --timeout 2400`
    - `python3 tools/run_benchmark.py --case perf_1gpu_6shots --tag sched_samegpu_np2_rerun --np 2 --gpus 0 --timeout 2400`
    - `python3 tools/run_benchmark.py --case perf_1gpu_6shots --tag sched_samegpu_np3_rerun --np 3 --gpus 0 --timeout 2400`
    - `python3 tools/compare_outputs.py --baseline <np1>/outputs --candidate <np2>/outputs --out <report>/compare_np2_vs_np1`
    - `python3 tools/compare_outputs.py --baseline <np1>/outputs --candidate <np3>/outputs --out <report>/compare_np3_vs_np1`
  - 本地拉取：
    - `tools/remote_get.py` 拉取 summary、comparison、binary sha 和 velocity symlink record。
- 测试结果：
  - 远端 GPU 运行前基本空闲：RTX 5090 memory used `481 MiB`，utilization `0%`。
  - 当前 best binary 构建成功。
  - 首次 `np=1` run 因新 worktree 缺少 large velocity file 失败；已用显式 symlink 指向主目录 velocity 后重跑成功。
  - `set -u` 会导致 conda/oneAPI activation 访问未定义变量时报错；远端环境脚本应使用 `set -eo pipefail` 或在 source 前关闭 nounset。
  - `np=1/2/3` rerun 均 returncode `0`，均输出 `6` 个文件。
  - `np=2` vs `np=1` 输出对比 pass。
  - `np=3` vs `np=1` 输出对比 pass。
- 输出/哈希/误差摘要：
  - binary sha256：`9b4cc826195df5b9b66c8b0281ca29dbf0301422ab7631e6878b2f0563569a3f`。
  - `np=1`：
    - elapsed：`2.990s`。
    - `Gradient TIME all`：`2.165543s`。
    - printed `WP computing time`：`2.048052s`。
    - shots seen：`[0, 1, 2, 3, 4, 5]`。
  - `np=2`：
    - elapsed：`3.370s`。
    - `Gradient TIME all`：`2.311468s`。
    - printed `WP computing time`：`2.443532s`。
    - elapsed speedup vs `np=1`：`0.8872x`。
    - Gradient speedup vs `np=1`：`0.9369x`。
    - correctness max rel L2：`0`。
  - `np=3`：
    - elapsed：`3.250s`。
    - `Gradient TIME all`：`2.328266s`。
    - printed `WP computing time`：`2.158150s`。
    - elapsed speedup vs `np=1`：`0.9200x`。
    - Gradient speedup vs `np=1`：`0.9301x`。
    - correctness max rel L2：`0`。
- 风险与下一步：
  - 决策：拒绝 same-GPU multi-rank oversubscription，不进入 repeat benchmark。
  - 多 rank scheduling 不能使用 root-rank printed `WP computing time` 作为正式 speedup 证据，必须看 elapsed 和 `Gradient TIME all`。
  - 下一步若继续 application-level scheduling，应转向 true multi-GPU / multi-job batching，确保每个 rank/job 拥有不同 GPU。
  - true multi-GPU 调度结果必须报告 elapsed、`Gradient TIME all`、correctness、GPU 数、rank 数和 shot 分配方式。

## 2026-06-08 19:50:53 +08:00

- 操作目标：
  - 继续 Phase 4.20 后的 application-level scheduling 路线，检查当前稳定 RTX 5090 平台是否能执行 true multi-GPU / multi-job batching 验证。
  - 审计现有 MPI shot 分配和 GPU 选择逻辑，固化未来多 GPU 验收协议。
- 修改文件：
  - 新增 `tools/multigpu_batching_protocol.py`。
  - 新增 `docs/day_20260608/true_multigpu_batching_protocol.md`。
  - 新增 `reports/day_20260608/true_multigpu_batching_protocol.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地：
    - `git status --short --branch`
    - `git log --oneline -5`
    - `Select-String -Path AGENTS.md -Pattern "Phase 4\\.20|true multi-GPU|multi-job|same-GPU" -Context 1,2`
    - `Get-Content src\main.cu | Select-Object -Skip 555 -First 95`
    - `Get-Content src\main.cu | Select-Object -Skip 380 -First 60`
    - `python -m py_compile tools\multigpu_batching_protocol.py`
    - `python tools\multigpu_batching_protocol.py --root . --available-gpus 1 --json-out reports\day_20260608\true_multigpu_batching_protocol.json --md-out docs\day_20260608\true_multigpu_batching_protocol.md`
    - `python -c "import json; d=json.load(open('reports/day_20260608/true_multigpu_batching_protocol.json', encoding='utf-8')); ..."`
  - 远端：
    - `cd /work/wenzhe/cuda3D`
    - `git status --short --branch | head -30`
    - `git log --oneline -3`
    - `nvidia-smi -L`
    - `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader`
- 测试结果：
  - 本地工具 Python 编译通过。
  - 生成 JSON 可读。
  - 远端当前稳定服务器仅暴露 `1` 张 GPU：`GPU 0: NVIDIA GeForce RTX 5090`。
  - 因此当前平台不能执行 true multi-GPU validation。
  - 本轮未修改 CUDA 源码，未运行新的 benchmark。
- 输出/哈希/误差摘要：
  - current-best anchor：
    - mean elapsed：`2.970s`。
    - mean `Gradient TIME all`：`2.155902s`。
    - mean WP：`2.031753s`。
    - WP speedup vs zmem：`1.192835x`。
  - source audit：
    - `src/main.cu` 从 input 读取 `gpus_p_node`。
    - `cudaSetDevice(mytid % gpus_p_node)` 决定 rank 到 GPU 的映射。
    - shot 分配使用 `sht_num[is * ntids + mytid]`。
  - true multi-GPU 配置要求：
    - `mpirun -np N`。
    - `CUDA_VISIBLE_DEVICES` 暴露 `N` 张卡。
    - 输入文件最后一行 `gpus_p_node=N`。
  - `perf_1gpu_6shots` shot-balance ideal upper bound：
    - `1` GPU：`[6]`，ideal `1.0000x`。
    - `2` GPUs：`[3,3]`，ideal `2.0000x`。
    - `3` GPUs：`[2,2,2]`，ideal `3.0000x`。
    - `4` GPUs：`[2,2,1,1]`，ideal `3.0000x`。
    - `6` GPUs：`[1,1,1,1,1,1]`，ideal `6.0000x`。
- 风险与下一步：
  - 决策：当前平台 defer true multi-GPU validation，不等同于拒绝 true multi-GPU batching。
  - 禁止把 `run_benchmark.py --gpus` 单独当成完整 true multi-GPU 配置；它只控制 `CUDA_VISIBLE_DEVICES`，还必须配套 input override。
  - 后续如果有 `>=2` GPU 平台，先生成 `input_perf_1gpu_6shots_gpusN.in`，最后一行改为 `N`，再用 `np=N` 和 `N` 张 visible GPUs 做 3 轮 repeat。
  - 多 GPU 调度验收必须用 elapsed 和 `Gradient TIME all`，root-rank printed WP 仅作诊断。

## 2026-06-08 20:10:58 +08:00

- 操作目标：
  - 继续 current-best 后的可提升空间审计，检查 formal elapsed 与 `Gradient TIME all` 之间的 host/setup overhead 是否值得进入 prototype。
  - 建立 host/setup route gate，明确何时可以优化，何时只能先 profile。
- 修改文件：
  - 新增 `tools/host_setup_overhead_gate.py`。
  - 新增 `docs/day_20260608/host_setup_overhead_gate.md`。
  - 新增 `reports/day_20260608/host_setup_overhead_gate.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `git status --short --branch`
  - `git log --oneline -5`
  - `Get-Content -Encoding UTF8 reports\day_20260608\formal_current_best_table_20260608_182525\summary.json`
  - `Get-Content -Encoding UTF8 reports\day_20260608\multirank_samegpu_sched_20260608_193042\summary.json`
  - `python -m py_compile tools\host_setup_overhead_gate.py`
  - `python tools\host_setup_overhead_gate.py --root . --json-out reports\day_20260608\host_setup_overhead_gate.json --md-out docs\day_20260608\host_setup_overhead_gate.md`
  - `python -c "import json; d=json.load(open('reports/day_20260608/host_setup_overhead_gate.json', encoding='utf-8')); ..."`
- 测试结果：
  - 模型脚本 Python 编译通过。
  - 生成 JSON 可被标准 `json.load` 读取。
  - 本轮未修改 CUDA 源码，未运行远端 benchmark。
- 输出/哈希/误差摘要：
  - current best：`len16_current_best`。
  - mean elapsed：`2.970s`。
  - mean `Gradient TIME all`：`2.155902s`。
  - mean WP：`2.031753s`。
  - elapsed - Gradient：`0.814098s`，占 elapsed `27.41%`。
  - elapsed - WP：`0.938247s`，占 elapsed `31.59%`。
  - current-best speedup vs zmem：
    - elapsed：`1.1560x`。
    - Gradient：`1.1792x`。
    - WP：`1.1928x`。
  - 若要让 current-best elapsed 再提升 `1.05x`：
    - 需要节省 `0.141429s`。
    - 相当于 `elapsed - Gradient` 的 `17.37%`。
  - scenario：
    - 移除 `10%` elapsed-Gradient：speedup `1.0282x`。
    - 移除 `25%` elapsed-Gradient：speedup `1.0736x`。
    - 移除 `50%` elapsed-Gradient：speedup `1.1588x`。
    - 移除 `100%` elapsed-Gradient：speedup `1.3776x`。
- 风险与下一步：
  - 决策：host/setup overhead 有潜力，但不允许盲改 prototype。
  - 只有 Nsight Systems、CPU sampling 或 targeted timers 证明某个具体 host/setup hotspot 具备 `>=5%` elapsed-speedup ceiling 后，才允许写 host/setup optimization。
  - 禁止移动计时点后声称加速。
  - 禁止跳过输出生成或 correctness 工作。
  - 禁止优化 `run_benchmark.py` output copy 来解释当前 elapsed metric，因为 output copy 在 `/usr/bin/time` 之后。
  - 下一步建议做 current-best host/setup profile 或 targeted timers，分解 `0.814s` 来源。

## 2026-06-08 20:39:48 +08:00

- 操作目标：
  - 执行 Phase 4.22 的下一步，添加 default-off targeted timers，分解 current-best `elapsed - Gradient` 的来源。
  - 用远端 RTX 5090 隔离 worktree 构建 timer binary，运行 `perf_1gpu_6shots`，并验证插桩不改变数值输出。
- 修改文件：
  - 修改 `src/main.cu`：新增 `CUDA3D_HOST_SETUP_TIMERS` 下的 main setup phase timers。
  - 修改 `src/optimization_cuda.cu`：新增 `cal_fwi_grad_3d` pre-Gradient init timer。
  - 新增 `tools/host_setup_timer_summary.py`。
  - 新增 `reports/day_20260608/host_setup_timer_probe_20260608_203508/run.log`。
  - 新增 `reports/day_20260608/host_setup_timer_probe_20260608_203508/manifest.json`。
  - 新增 `reports/day_20260608/host_setup_timer_probe_20260608_203508/run.stdout`。
  - 新增 `reports/day_20260608/host_setup_timer_probe_20260608_203508/run_dir.txt`。
  - 新增 `reports/day_20260608/host_setup_timer_probe_20260608_203508/binary.sha256`。
  - 新增 `reports/day_20260608/host_setup_timer_probe_20260608_203508/build.log`。
  - 新增 `reports/day_20260608/host_setup_timer_probe_20260608_203508/summary.md`。
  - 新增 `reports/day_20260608/host_setup_timer_probe_20260608_203508/summary.json`。
  - 新增 `reports/day_20260608/host_setup_timer_probe_20260608_203508/compare_vs_formal_len16_r1/comparison.md`。
  - 新增 `reports/day_20260608/host_setup_timer_probe_20260608_203508/compare_vs_formal_len16_r1/comparison.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地：
    - `git status --short --branch`
    - `git log --oneline -5`
    - `Select-String -Path src\main.cu,src\optimization_cuda.cu -Pattern "MPI_Init|MPI_Wtime|Gradient TIME all|cal_fwi_grad_3d|MPI_Finalize|read_dir_3d|cudaSetDevice|cudaGetDeviceCount" -Context 2,2`
    - `python -m py_compile tools\host_setup_timer_summary.py`
    - `python tools\host_setup_timer_summary.py --run-log reports\day_20260608\host_setup_timer_probe_20260608_203508\run.log --json-out reports\day_20260608\host_setup_timer_probe_20260608_203508\summary.json --md-out reports\day_20260608\host_setup_timer_probe_20260608_203508\summary.md`
  - 远端：
    - `git -c http.proxy= -c https.proxy= fetch origin exp/day-20260608-cpml-compact-temporal`
    - `git worktree add --detach /work/wenzhe/cuda3D/.codex_worktrees/host_setup_timers_20260608_203508 FETCH_HEAD`
    - `tools/remote_put.py` 上传 `src/main.cu` 和 `src/optimization_cuda.cu` 到隔离 worktree。
    - `make -B -f makefile.rtx5090 test NVFLAGS="<current-best flags> -DCUDA3D_HOST_SETUP_TIMERS"`
    - `python3 tools/run_benchmark.py --case perf_1gpu_6shots --tag host_setup_timers --np 1 --gpus 0 --timeout 2400`
    - `python3 tools/compare_outputs.py --baseline <formal len16 r1>/outputs --candidate <timer run>/outputs --out <report>/compare_vs_formal_len16_r1`
    - `make -B -f makefile.rtx5090 test NVFLAGS="<current-best flags>"` 验证 default-off build。
  - 本地拉取：
    - `tools/remote_get.py` 拉取 run log、manifest、build log、comparison 和 binary sha。
- 测试结果：
  - timer binary 构建通过。
  - timer run returncode `0`，输出 `6` 个文件。
  - timer binary vs formal len16 current-best r1 输出对比 pass。
  - 6 个输出 max rel L2 `0`，max abs `0`。
  - default-off current-best build 通过，确认 `CUDA3D_HOST_SETUP_TIMERS` 默认关闭不影响普通构建。
- 输出/哈希/误差摘要：
  - timer binary sha256：`d51f08cbb3d1f54276fddc8c357de9afa679c5a2e8c0a3e059471f748d084e9e`。
  - default-off binary sha256：`824bf4a383ba90c10fba9b76b42ad264a12b3865d3e3fa97460c733890633196`。
  - elapsed：`2.980s`。
  - `Gradient TIME all`：`2.162907s`。
  - WP：`2.046621s`。
  - elapsed - Gradient：`0.817093s`。
  - measured pre-Gradient setup：`0.238399s`。
  - unaccounted elapsed-minus-Gradient：`0.578694s`。
  - main timers：
    - `gpu_setup`：`0.174303s`。
    - `root_model_read`：`0.018118s`。
    - `shot_list`：`0.022419s`。
    - `total_pre_gradient`：`0.215846s`。
    - `gradient_call_total`：`2.188155s`。
    - `post_gradient_barrier_and_free`：`0.001178s`。
  - cal timer：
    - `pre_gradient_init`：`0.022553s`。
- 风险与下一步：
  - 决策：保留 `CUDA3D_HOST_SETUP_TIMERS` 作为默认关闭诊断路径。
  - 不写 host/setup optimization prototype；当前最大未解释 gap `0.578694s` 在 after-MPI timer 外部，可能包含 bash/oneAPI source、mpirun、`MPI_Init`、finalization。
  - `gpu_setup` 是最大 measured stage，但更像 CUDA device/context 初始化，一次性启动成本；移动或预热它不能作为 CUDA-core speedup。
  - 下一步若继续 wall-clock 路线，应添加 process-level timer around `MPI_Init` 或做 Nsight Systems OS/runtime profile。
  - CUDA-core 优化仍以 `Gradient TIME all` 和 WP 为主指标。

## 2026-06-08 21:07:08 +08:00

- 操作目标：
  - 继续 Phase 4.23 后的 wall-clock gap 分解，添加 process-level timer，定位 `MPI_Init`、`MPI_Finalize` 与 `/usr/bin/time` 外壳开销。
  - 判断 host/setup wall-clock route 是否仍值得作为 CUDA-core optimization 推进。
- 修改文件：
  - 修改 `src/main.cu`：在 `CUDA3D_HOST_SETUP_TIMERS` 下新增 `gettimeofday` process timer，计时 `MPI_Init`、main after-MPI to pre-finalize、`MPI_Finalize`、process total。
  - 修改 `tools/host_setup_timer_summary.py`：新增 process timer 解析与 known non-Gradient accounting。
  - 新增 `reports/day_20260608/process_timer_probe_20260608_205311/run.log`。
  - 新增 `reports/day_20260608/process_timer_probe_20260608_205311/manifest.json`。
  - 新增 `reports/day_20260608/process_timer_probe_20260608_205311/run.stdout`。
  - 新增 `reports/day_20260608/process_timer_probe_20260608_205311/run_dir.txt`。
  - 新增 `reports/day_20260608/process_timer_probe_20260608_205311/binary.sha256`。
  - 新增 `reports/day_20260608/process_timer_probe_20260608_205311/build.log`。
  - 新增 `reports/day_20260608/process_timer_probe_20260608_205311/summary.md`。
  - 新增 `reports/day_20260608/process_timer_probe_20260608_205311/summary.json`。
  - 新增 `reports/day_20260608/process_timer_probe_20260608_205311/compare_vs_formal_len16_r1/comparison.md`。
  - 新增 `reports/day_20260608/process_timer_probe_20260608_205311/compare_vs_formal_len16_r1/comparison.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地：
    - `git status --short --branch`
    - `git log --oneline -5`
    - `python -m py_compile tools\host_setup_timer_summary.py`
    - `python tools\host_setup_timer_summary.py --run-log reports\day_20260608\process_timer_probe_20260608_205311\run.log --json-out reports\day_20260608\process_timer_probe_20260608_205311\summary.json --md-out reports\day_20260608\process_timer_probe_20260608_205311\summary.md`
  - 远端：
    - `git -c http.proxy= -c https.proxy= fetch origin exp/day-20260608-cpml-compact-temporal`
    - `git worktree add --detach /work/wenzhe/cuda3D/.codex_worktrees/process_timers_20260608_205311 FETCH_HEAD`
    - `tools/remote_put.py` 上传 `src/main.cu` 与 `tools/host_setup_timer_summary.py`。
    - 第一次 `remote_put.py` 出现 `Connection reset by peer (104)`，重试后成功，未影响文件状态。
    - `make -B -f makefile.rtx5090 test NVFLAGS="<current-best flags> -DCUDA3D_HOST_SETUP_TIMERS"`
    - `python3 tools/run_benchmark.py --case perf_1gpu_6shots --tag process_timers --np 1 --gpus 0 --timeout 2400`
    - `python3 tools/compare_outputs.py --baseline <formal len16 r1>/outputs --candidate <process timer run>/outputs --out <report>/compare_vs_formal_len16_r1`
    - `python3 tools/host_setup_timer_summary.py --run-log <report>/run.log --json-out <report>/summary.json --md-out <report>/summary.md`
    - `make -B -f makefile.rtx5090 test NVFLAGS="<current-best flags>"` 验证 default-off build。
  - 本地拉取：
    - `tools/remote_get.py` 拉取 process timer run log、manifest、summary、comparison、binary sha 和 build log。
- 测试结果：
  - process timer binary 构建通过。
  - process timer run returncode `0`，输出 `6` 个文件。
  - process timer binary vs formal len16 current-best r1 输出对比 pass。
  - 6 个输出 max rel L2 `0`，max abs `0`。
  - default-off current-best build 通过，binary sha256 `ca8c64735b77db3c600a8132b818ef0d70b42007a49a1608a3ea1681103e297f`。
- 输出/哈希/误差摘要：
  - process timer binary sha256：`45df148a3c9584384d779d14cb9f82743855d119ca2b5cfe9e3f4bac9aed5da2`。
  - elapsed：`3.220s`。
  - `Gradient TIME all`：`2.161705s`。
  - WP：`2.045140s`。
  - elapsed - Gradient：`1.058295s`。
  - process timers：
    - `MPI_Init`：`0.254292s`。
    - main after-MPI to pre-finalize：`2.418194s`。
    - `MPI_Finalize`：`0.000283s`。
    - process total：`2.672769s`。
    - elapsed - process total：`0.547231s`。
  - in-program timers：
    - measured pre-Gradient setup：`0.250119s`。
    - `gpu_setup`：`0.186226s`。
    - `root_model_read`：`0.018050s`。
    - `shot_list`：`0.022546s`。
    - `cal pre_gradient_init`：`0.022299s`。
  - accounting：
    - known non-Gradient time：`1.053080s`。
    - residual after known non-Gradient timers：`0.005215s`。
- 风险与下一步：
  - 决策：host/setup wall-clock gap 已基本闭合，不继续作为 CUDA-core optimization route。
  - 最大非计算项为 `/usr/bin/time` command shell / `source setvars` / `mpirun` wrapper 约 `0.547s`、`MPI_Init` 约 `0.254s`、CUDA device/context setup 约 `0.186s`。
  - 这些属于 benchmark/deployment/startup policy 或 long-running service/multi-shot batching 议题，不能作为 CUDA kernel speedup。
  - 后续 CUDA-core 提速继续使用 `Gradient TIME all` 与 WP 指标。
  - wall-clock route 只在 true multi-GPU/multi-job batching 或明确 deployment benchmark 下重开。

## 2026-06-08 21:23:24 +08:00

- 操作目标：
  - 继续 Phase 4.24 后的 compute-metric route，分解 `Gradient TIME all - WP` 的 cal-loop 内部非 FD 开销。
  - 判断 host-side per-shot setup、`vc/vc_pad` preparation、output write、cleanup、copy/reduce 是否值得写 optimization prototype。
- 修改文件：
  - 修改 `src/optimization_cuda.cu`：在 `CUDA3D_HOST_SETUP_TIMERS` 下新增 `cal_loop` timers。
  - 修改 `tools/host_setup_timer_summary.py`：新增 `Cal Loop Timers` Markdown 输出。
  - 新增 `reports/day_20260608/cal_loop_timer_probe_20260608_212019/run.log`。
  - 新增 `reports/day_20260608/cal_loop_timer_probe_20260608_212019/manifest.json`。
  - 新增 `reports/day_20260608/cal_loop_timer_probe_20260608_212019/run.stdout`。
  - 新增 `reports/day_20260608/cal_loop_timer_probe_20260608_212019/run_dir.txt`。
  - 新增 `reports/day_20260608/cal_loop_timer_probe_20260608_212019/binary.sha256`。
  - 新增 `reports/day_20260608/cal_loop_timer_probe_20260608_212019/build.log`。
  - 新增 `reports/day_20260608/cal_loop_timer_probe_20260608_212019/summary.md`。
  - 新增 `reports/day_20260608/cal_loop_timer_probe_20260608_212019/summary.json`。
  - 新增 `reports/day_20260608/cal_loop_timer_probe_20260608_212019/compare_vs_formal_len16_r1/comparison.md`。
  - 新增 `reports/day_20260608/cal_loop_timer_probe_20260608_212019/compare_vs_formal_len16_r1/comparison.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地：
    - `git status --short --branch`
    - `git log --oneline -5`
    - `python -m py_compile tools\host_setup_timer_summary.py`
    - `git diff --check`
  - 远端：
    - `git -c http.proxy= -c https.proxy= fetch origin exp/day-20260608-cpml-compact-temporal`
    - `git worktree add --detach /work/wenzhe/cuda3D/.codex_worktrees/cal_loop_timers_20260608_212019 FETCH_HEAD`
    - `tools/remote_put.py` 上传 `src/optimization_cuda.cu` 与 `tools/host_setup_timer_summary.py`。
    - `make -B -f makefile.rtx5090 test NVFLAGS="<current-best flags> -DCUDA3D_HOST_SETUP_TIMERS"`
    - `python3 tools/run_benchmark.py --case perf_1gpu_6shots --tag cal_loop_timers --np 1 --gpus 0 --timeout 2400`
    - `python3 tools/compare_outputs.py --baseline <formal len16 r1>/outputs --candidate <cal-loop timer run>/outputs --out <report>/compare_vs_formal_len16_r1`
    - `python3 tools/host_setup_timer_summary.py --run-log <report>/run.log --json-out <report>/summary.json --md-out <report>/summary.md`
    - `make -B -f makefile.rtx5090 test NVFLAGS="<current-best flags>"` 验证 default-off build。
  - 本地拉取：
    - `tools/remote_get.py` 拉取 cal-loop timer run log、manifest、summary、comparison、binary sha 和 build log。
- 测试结果：
  - cal-loop timer binary 构建通过。
  - cal-loop timer run returncode `0`，输出 `6` 个文件。
  - cal-loop timer binary vs formal len16 current-best r1 输出对比 pass。
  - 6 个输出 max rel L2 `0`，max abs `0`。
  - default-off current-best build 通过，binary sha256 `48d4f4421b48fe4e3e8f2ba07c292de3178727dddb0181f1bda2e238aca50d1c`。
- 输出/哈希/误差摘要：
  - cal-loop timer binary sha256：`24e4dc2a31a05d76ef789b34a28600ffbd03e8e611b1f604caa353a47ea746be`。
  - elapsed：`2.990s`。
  - `Gradient TIME all`：`2.164033s`。
  - WP：`2.044622s`。
  - cal `pre_gradient_init`：`0.023527s`。
  - `cal_loop` across 6 shots：
    - `obs_setup`：`0.002679s`。
    - `domain_setup`：`0.000010s`。
    - `wavefield_prep`：`0.049816s`。
    - `fd_call`：`2.089376s`。
    - `output_write`：`0.004491s`。
    - `cleanup`：`0.002593s`。
    - `post_loop_sync`：`0.000004s`。
    - `copy_reduce`：`0.015053s`。
- 风险与下一步：
  - 决策：拒绝 host-side cal-loop micro optimization prototypes。
  - 不写 `vc/vc_pad` preparation optimization、output write / cleanup / copy-reduce micro prototype。
  - 最大非-FD cal-loop 项 `wavefield_prep=0.049816s`，即使理想消除也只有约 `2.4%` Gradient speedup ceiling，低于 `>=5%` prototype gate。
  - 后续 exact compute 优化应聚焦 `fd_3d_f` kernel/dataflow，或等待 true multi-GPU batching 平台。

## 2026-06-08 21:44:13 +08:00

- 操作目标：
  - 继续 Phase 4.25 后的 CUDA-core route，检查 accepted len16 half-warp packing 后，residual pressure-PML 中 length-32 full-active z-line 是否值得单独 full-warp specialization。
  - 在写 CUDA prototype 前建立 `>=5%` sampled-main gate，避免重复 branch/control-only 微调。
- 修改文件：
  - 新增 `tools/pml_len32_fullwarp_specialization_budget.py`。
  - 新增 `docs/day_20260608/pml_len32_fullwarp_specialization_budget.md`。
  - 新增 `reports/day_20260608/pml_len32_fullwarp_specialization_budget.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `python tools\pml_len32_fullwarp_specialization_budget.py --json-out reports\day_20260608\pml_len32_fullwarp_specialization_budget.json --md-out docs\day_20260608\pml_len32_fullwarp_specialization_budget.md`
  - `Get-Content -Encoding UTF8 docs\day_20260608\pml_len32_fullwarp_specialization_budget.md`
- 测试结果：
  - 模型工具运行成功。
  - 本轮不改 CUDA 执行路径，不需要远端 correctness/perf run。
- 输出/哈希/误差摘要：
  - sampled main：`297.248us`。
  - residual pressure-PML：`72.683us`。
  - packed len16 pressure-PML：`65.771us`。
  - length-32 line share：`75.00%`。
  - length-32 active-lane share：`80.67%`。
  - length-32 要让 sampled-main 达到 `>=1.05x`，本地需要约 `1.3182x` 到 `1.3507x` speedup。
  - 最乐观 branch/control 场景：
    - perfect branch-efficiency on entire residual：sampled-main `1.0425x`。
    - 20% full32 local speedup：sampled-main `1.0411x`。
- 风险与下一步：
  - 决策：拒绝 `CUDA3D_PML_PRESSURE_LEN32_FULL_WARP_SPECIALIZE` branch/control-only CUDA prototype。
  - 不把 full-active length-32 lines 当作新的 lane-compaction opportunity。
  - 只有 future source-level profile 能单独分离 length-32 residual，并证明扣除额外 launch/tile-list/control overhead 后仍有 `>=5%` repeat speedup ceiling，才允许重开。
  - 下一步继续寻找能真实减少 memory traffic / state ownership cost 的 `fd_3d_f` 结构路线，或等待 true multi-GPU 平台。

## 2026-06-08 22:20:57 +08:00

- 操作目标：
  - 建立 p-core shared-plane stencil budget，检查是否存在比 current z-only shared kernel 更强的 p1 global-load reduction 路线。
  - 对通过 budget gate 的 `[16,16,1]` z+x shared-plane prototype 进行远端 build、smoke、correctness 和 perf repeat 验证。
- 修改文件：
  - 新增 `tools/p_core_shared_plane_budget.py`。
  - 新增 `docs/day_20260608/p_core_shared_plane_budget.md`。
  - 新增 `reports/day_20260608/p_core_shared_plane_budget.json`。
  - 新增 `reports/day_20260608/p_core_zx_prototype_20260608_2158/summary.md`。
  - 新增 `reports/day_20260608/p_core_zx_prototype_20260608_2158/perf6_repeat_summary.json`。
  - 新增 `reports/day_20260608/p_core_zx_prototype_20260608_2158/build.log`。
  - 新增 `reports/day_20260608/p_core_zx_prototype_20260608_2158/binary.sha256`。
  - 新增 smoke/correctness/perf comparison 摘要文件于 `reports/day_20260608/p_core_zx_prototype_20260608_2158/`。
  - 曾临时修改 `include/inc3D/single_solver.h`、`src/single_solver.cu`、`src/rem_fd.cu` 实现 `CUDA3D_P_CORE_SHARED_ZX_PLANE`；因 perf gate 失败，本地源码已恢复，不保留失败 kernel。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地：
    - `python tools\p_core_shared_plane_budget.py --json-out reports\day_20260608\p_core_shared_plane_budget.json --md-out docs\day_20260608\p_core_shared_plane_budget.md`
    - `python -m py_compile tools\p_core_shared_plane_budget.py`
    - `git diff --check`
  - 远端：
    - `git -c http.proxy= -c https.proxy= fetch origin exp/day-20260608-cpml-compact-temporal`
    - `git worktree add --detach /work/wenzhe/cuda3D/.codex_worktrees/p_core_zx_20260608_2158 FETCH_HEAD`
    - 上传临时 prototype 源码到该 worktree。
    - `make -B -f makefile.rtx5090 test NVFLAGS="<current-best flags> -DCUDA3D_P_CORE_SHARED_ZX_PLANE"`
    - `python3 tools/run_benchmark.py --case smoke_1gpu --tag p_core_zx_smoke_fixed --np 1 --gpus 0 --timeout 300`
    - `python3 tools/run_benchmark.py --case correctness --tag p_core_zx_correctness --np 1 --gpus 0 --timeout 600`
    - `python3 tools/compare_outputs.py --baseline <len16 correctness baseline>/outputs --candidate <p_core_zx correctness>/outputs --out <report>/compare_correctness_vs_len16_base`
    - `python3 tools/run_benchmark.py --case perf_1gpu_6shots --tag p_core_zx_perf6_retry_r{1,2,3} --np 1 --gpus 0 --timeout 2400`
    - 每轮 perf 后用 `tools/compare_outputs.py` 对 formal current-best len16 r1 输出做比较。
- 测试结果：
  - budget gate：允许 prototype。
  - build：通过。
  - smoke：通过。
  - correctness：通过，6 个输出 rel L2 全部 `0`。
  - perf repeat：三轮输出对比全部通过，max rel L2 `0`。
  - performance gate：失败，明显慢于 current-best。
- 输出/哈希/误差摘要：
  - prototype binary sha256：`45213389d52df56c9ab433f2bb48b72517d3c301555f32a6bde7c16d172602fe`。
  - current p_core p1 global floats/output：`29.109375`。
  - `[16,16,1]` z+x shared-plane modeled p1 floats/output：`17.516`。
  - modeled sampled-main ceiling：`1.1282x`。
  - perf repeat rows：
    - r1：WP `2.589493s`，Gradient `2.731454s`。
    - r2：WP `2.597138s`，Gradient `2.734513s`。
    - r3：WP `2.583236s`，Gradient `2.727773s`。
  - mean WP：`2.589956s`。
  - mean Gradient：`2.731247s`。
  - WP speedup vs formal current-best：`0.784474x`。
  - Gradient speedup vs formal current-best：`0.789347x`。
- 风险与下一步：
  - 决策：拒绝当前 `CUDA3D_P_CORE_SHARED_ZX_PLANE` / `16x16x1` z+x shared-plane prototype。
  - 当前源码已恢复，不保留失败宏。
  - 不继续当前 p-core shared-plane 形态；模型高估是因为 shared tile fill、control overhead、warp mapping/coalescing 变化吞掉了 global-load reduction。
  - 只有新的 warp/coalescing 设计能先证明 shared-fill/control overhead 明显更低，才允许重开 p-core shared-plane CUDA prototype。

## 2026-06-08 22:30:01 +08:00

- 操作目标：
  - 用 Phase 4.27 失败 prototype 校准 p-core shared-plane byte model。
  - 判断是否还值得测试 `[32,8,1]`、`[16,8,2]`、`[64,2,2]` 等其它 shared-plane shape。
- 修改文件：
  - 新增 `tools/p_core_shared_plane_calibrated_gate.py`。
  - 新增 `docs/day_20260608/p_core_shared_plane_calibrated_gate.md`。
  - 新增 `reports/day_20260608/p_core_shared_plane_calibrated_gate.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `python tools\p_core_shared_plane_calibrated_gate.py --json-out reports\day_20260608\p_core_shared_plane_calibrated_gate.json --md-out docs\day_20260608\p_core_shared_plane_calibrated_gate.md`
  - `Get-Content -Encoding UTF8 docs\day_20260608\p_core_shared_plane_calibrated_gate.md`
- 测试结果：
  - 校准工具运行成功。
  - 本轮不改 CUDA 源码，不需要远端 build/run。
- 输出/哈希/误差摘要：
  - tested shape：`[16,16,1]` / `zx_shared_y_global`。
  - modeled p_core local speedup：`1.5651x`。
  - modeled sampled-main speedup：`1.1282x`。
  - observed WP global speedup：`0.7845x`。
  - observed Gradient global speedup：`0.7893x`。
  - inferred WP-local p_core speedup：`0.5339x`。
  - inferred Gradient-local p_core speedup：`0.5411x`。
  - WP model-to-observed factor：`0.3411x`。
  - Gradient model-to-observed factor：`0.3457x`。
  - calibrated best current shared-plane WP sampled speedup：`0.7845x`。
- 风险与下一步：
  - 决策：拒绝当前 p-core shared-plane shape family。
  - 不继续测试 `[32,8,1]`、`[16,8,2]`、`[64,2,2]` 等同类变体。
  - 只有 materially different warp/coalescing design，并且模型显式计入 shared fill、同步和控制开销后仍有 `>=5%` repeat speedup ceiling，才允许重开。

## 2026-06-08 23:03:18 +08:00

- 操作目标：
  - 检查 accepted pressure len16 half-warp packing 后，`v_pml` 是否存在类似 active z-segment packing 机会。
  - 对通过 gate 的 whole-tile length-16 velocity-PML half-warp prototype 进行远端 build、smoke、correctness、`perf_1gpu_6shots` repeat 验证。
- 修改文件：
  - 新增 `tools/v_pml_active_segment_packing_model.py`。
  - 新增 `docs/day_20260608/v_pml_active_segment_packing_model.md`。
  - 新增 `docs/day_20260608/v_pml_len16_halfwarp_prototype.md`。
  - 新增 `reports/day_20260608/v_pml_active_segment_packing_model.json`。
  - 新增 `reports/day_20260608/v_pml_len16_prototype_20260608_2238/` 远端测试报告。
  - 修改 `include/inc3D/single_solver.h`。
  - 修改 `src/single_solver.cu`。
  - 修改 `src/rem_fd.cu`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地：
    - `python tools\v_pml_active_segment_packing_model.py --json-out reports\day_20260608\v_pml_active_segment_packing_model.json --md-out docs\day_20260608\v_pml_active_segment_packing_model.md`
    - `python -m py_compile tools\v_pml_active_segment_packing_model.py`
    - `git diff --check`
  - 远端：
    - `git worktree add --detach /work/wenzhe/cuda3D/.codex_worktrees/v_pml_len16_20260608_2238 FETCH_HEAD`
    - 上传 `include/inc3D/single_solver.h`、`src/rem_fd.cu`、`src/single_solver.cu` 到该 isolated worktree。
    - `make -B -f makefile.rtx5090 test NVFLAGS="<current-best flags> -DCUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK"`
    - `python3 tools/run_benchmark.py --case smoke_1gpu --tag v_pml_len16_smoke --np 1 --gpus 0 --timeout 300`
    - 同 worktree 重建 current-best baseline 与 candidate，分别运行 `correctness`。
    - 同 worktree 重建 current-best baseline 与 candidate，分别运行 3 轮 `perf_1gpu_6shots`。
    - candidate 每轮输出用 `tools/compare_outputs.py` 对 baseline perf 第 1 轮输出比较。
- 测试结果：
  - model gate：允许 whole-tile len16 velocity-PML prototype。
  - build：通过。
  - smoke：通过，`outputs=3`；smoke 中 `len16_tiles=0`，只验证 wiring。
  - correctness：通过，candidate vs baseline 输出比较通过；correctness 中 `len16_tiles=0`，只验证 wiring。
  - `perf_1gpu_6shots` repeat：三轮输出对比全部通过，max rel L2 `0`。
- 输出/哈希/误差摘要：
  - model：
    - current launched lanes：`30,420,992`。
    - true vx/vy active-any lanes：`20,646,925`。
    - length-16 z-line slots：`506,974`。
    - whole length-16 tiles：`62,400`。
    - whole-tile len16 v lane ceiling：`1.3560x`。
    - sampled-main ceiling：`1.0612x`。
  - candidate perf repeat：
    - mean base WP：`2.052228s`。
    - mean candidate WP：`1.988482s`。
    - WP speedup：`1.032058x`。
    - mean base Gradient：`2.169915s`。
    - mean candidate Gradient：`2.109314s`。
    - Gradient speedup：`1.028730x`。
    - compare max rel L2：`0`。
  - binary hashes：
    - base perf retry：`bd21a45ced78362eb8f87b97651dd3e0e73fe3305f58487ae9e8f2f69f26c0c6`。
    - candidate perf retry：`6d06c99b7daa3fb13a26b8999a7212ee53efb18dda1ad3ff3e04b5fa1b1bfa86`。
- 风险与下一步：
  - 严格 `>=5%` breakthrough gate 未达到，因此不继续扩展本路线。
  - minor `>=2%` candidate gate 通过，保留 `CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK` 为 macro-default-off current-best flags 候选。
  - 不继续做 v-PML line descriptor / exact active-point descriptor prototype，除非有新的 descriptor/control overhead model 且仍证明 `>=5%` repeat speedup ceiling。
  - 继续禁止 random v-PML tile-shape sweep 与 current-geometry vx/vy component-owner split。
  - 远端经验记录：加载 oneAPI/conda 环境脚本时不要使用 `set -u`，否则环境脚本访问未定义变量会提前退出且可能无 build 输出；改用 `set -eo pipefail`。
  - isolated worktree 缺少 perf 大样例 velocity/source 数据文件时，使用指向 `/work/wenzhe/cuda3D/benchmarks/cases/perf_1gpu_6shots/` 的 symlink 补齐，不复制大文件、不删除文件。

## 2026-06-08 23:23:50 +08:00

- 操作目标：
  - 对 `CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK` candidate 做 Nsight Compute 短 profile。
  - 判断 v-PML len16 之后剩余瓶颈与下一步 CUDA-core 方向。
- 修改文件：
  - 新增 `docs/day_20260608/v_pml_len16_ncu_profile.md`。
  - 新增 `reports/day_20260608/v_pml_len16_ncu_short_20260608_2315/` 文本 profile artifacts。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端检查 GPU 与 isolated worktree 状态。
  - 给 isolated worktree 的 `benchmarks/cases/profile_1gpu` 补充指向根目录 profile 大样例 velocity/source 文件的 symlink。
  - 首次 NCU 使用全量匹配 kernel profile，运行超过 8 分钟仍未结束，生成 `.ncu-rep` 但未产出 CSV；停止该 profile 进程族。
  - 重新运行短 NCU：
    - `ncu --launch-count 5 --kernel-name "regex:cuda_fd3d_(p_core_ns|p_pml_tile_ns|p_pml_len16_halfwarp_ns|v_pml_tile_ns|v_pml_len16_halfwarp_ns)" ...`
    - 导出 CSV。
    - `python3 tools/ncu_csv_summary.py --profile v_pml_len16_short <csv> --json-out <summary.json> --md-out <summary.md>`
  - 本地删除明确路径的二进制 `.ncu-rep` 文件，只保留 CSV/JSON/Markdown/log 文本 artifacts。
- 测试结果：
  - 短 NCU profile 成功。
  - 本轮不改 CUDA 源码，不需要重新 correctness/perf repeat。
- 输出/哈希/误差摘要：
  - captured kernel durations，CSV 原始单位为 `us`：
    - `cuda_fd3d_v_pml_len16_halfwarp_ns`：`20.030us`。
    - `cuda_fd3d_v_pml_tile_ns`：`32.130us`。
    - `cuda_fd3d_p_core_ns`：`93.730us`。
    - `cuda_fd3d_p_pml_len16_halfwarp_ns`：`66.180us`。
    - `cuda_fd3d_p_pml_tile_ns`：`71.940us`。
  - sampled main total：`284.010us`。
  - p-core share：`33.00%`。
  - pressure-PML total share：`48.63%`。
  - velocity-PML total share：`18.37%`。
  - packed v kernel avg active threads/warp：`32.000`。
  - residual v kernel branch efficiency：`94.770%`。
- 风险与下一步：
  - 现有 `tools/ncu_csv_summary.py` 没有记录 `Metric Unit`，本轮文档明确 Duration 数字按 CSV 的 `us` 解释；未修改历史脚本。
  - 全量 NCU profile 太慢，后续这类 heartbeat profiling 默认使用 `--launch-count` 或更窄 kernel filter。
  - v-PML packed 后只剩 sampled-main `18.37%`，继续 v-PML descriptor / point-list 实验不应开启，除非 overhead model 仍证明 `>=5%` repeat-speedup ceiling。
  - 下一步 CUDA-core 方向应回到 pressure-PML 或 materially new p-core design，但必须先通过 model gate。

## 2026-06-08 23:36:02 +08:00

- 操作目标：
  - 基于 v-PML len16 后的最新 NCU 短 profile，重新 gate pressure-PML 下一步路线。
  - 判断是否应继续开 micro CUDA prototype，或先收口到正式同 session benchmark 与更大粒度 ownership model。
- 修改文件：
  - 新增 `tools/post_vlen16_pressure_next_gate.py`。
  - 新增 `docs/day_20260608/post_vlen16_pressure_next_gate.md`。
  - 新增 `reports/day_20260608/post_vlen16_pressure_next_gate.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `python -m py_compile tools\post_vlen16_pressure_next_gate.py`
  - `python tools\post_vlen16_pressure_next_gate.py --json-out reports\day_20260608\post_vlen16_pressure_next_gate.json --md-out docs\day_20260608\post_vlen16_pressure_next_gate.md`
  - `python -c "import json; p='reports/day_20260608/post_vlen16_pressure_next_gate.json'; d=json.load(open(p,encoding='utf-8')); print(d['gate']['decision']); print(d['inputs']['profile']['sampled_main_us']); print(d['derived']['required_local_speedup_by_region']['p_pml_total'])"`
- 测试结果：
  - Python 编译检查通过。
  - gate 工具运行成功并生成 Markdown/JSON。
  - 本轮不修改 CUDA 源码，不需要远端 build/correctness/perf repeat。
- 输出/哈希/误差摘要：
  - gate decision：`no_new_micro_cuda_prototype`。
  - post-vlen16 sampled main total：`284.010us`。
  - pressure-PML total：`138.120us`，sampled-main share `48.63%`。
  - velocity-PML total：`52.160us`，sampled-main share `18.37%`。
  - `>=5%` sampled-main 所需 local speedup：
    - pressure-PML total：`1.1085x`。
    - packed pressure len16 only：`1.2568x`。
    - residual pressure-PML only：`1.2315x`。
    - velocity-PML total：`1.3500x`。
  - packed pressure len16 source groups：
    - final `p0/p1/cw2` update：`60.78%`。
    - CPML `mem_dzz` update：`26.82%`。
    - final + `mem_dzz` group required speedup：`1.3043x`。
- 风险与下一步：
  - 决策：当前不启动新的 micro CUDA prototype。
  - 禁止继续重复 `p0 __ldg`、local `new_mem`、ptxas cache-policy、z-cache fill、shared-z-cache、pressure length-23/exact descriptor、v-PML descriptor expansion、direct z-face VP fusion/shared-VP retry，以及 rejected p-core shared-plane/block/register sweep。
  - 下一步优先同 session 正式重跑 `zmem`、direct-fill、pressure-len16、current-best，给出当前 RTX 5090 平台正式总提速表。
  - 然后只允许开启 design-level pressure/wave-step ownership model，并且必须先证明扣除 extra storage/control cost 后仍有 `>=5%` repeat-speedup ceiling。

## 2026-06-09 00:02:19 +08:00

- 操作目标：
  - 按 Phase 4.30 gate，正式同 session 重跑 `zmem`、`directfill`、`pressure_len16`、`current_best_v_pml_len16`。
  - 给出当前 RTX 5090 single-GPU 的正式 current-best 总提速表。
- 修改文件：
  - 新增 `reports/day_20260608/formal_vpmlen16_table_20260608_2359/summary.md`。
  - 新增 `reports/day_20260608/formal_vpmlen16_table_20260608_2359/summary.json`。
  - 新增 `reports/day_20260608/formal_vpmlen16_table_20260608_2359/records.jsonl`。
  - 新增 `reports/day_20260608/formal_vpmlen16_table_20260608_2359/remote_paths.txt`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端探测：
    - `git status --short` 显示根目录仍有旧实验脏状态，因此继续使用 isolated worktree。
    - `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader` 显示 RTX 5090 空闲：`481 MiB`，`0%`。
  - 第一次尝试：
    - `git worktree add --detach /work/wenzhe/cuda3D/.codex_worktrees/formal_vpmlen16_table_20260608_2352 FETCH_HEAD`
    - 构建 `zmem` 成功。
    - `perf_1gpu_6shots` 运行在第 1 炮后 segfault，returncode `255`，outputs `0`。
    - 检查发现 worktree case 下没有 case-local `d_obs/` 目录。
  - 正式 rerun：
    - `git worktree add --detach /work/wenzhe/cuda3D/.codex_worktrees/formal_vpmlen16_table_20260608_2359 FETCH_HEAD`
    - 创建 `benchmarks/cases/perf_1gpu_6shots/d_obs/`。
    - 只对缺失的大输入文件建立指向根目录 case 的 symlink；不 symlink `d_obs`。
    - 每个配置每轮运行前执行 `make -B -f makefile.rtx5090 test NVFLAGS="<flags>"`。
    - 运行 4 个配置 x 3 rounds 的 `python3 tools/run_benchmark.py --case perf_1gpu_6shots --np 1 --gpus 0 --timeout 2400`。
    - 每轮候选输出用 `tools/compare_outputs.py` 对同轮 `zmem` 输出比较。
    - 使用 `tools/remote_get.py` 拉回 summary/records/remote_paths。
- 测试结果：
  - 正式 rerun 全部 build 通过。
  - 12 个 perf run 全部 `ALL DONE`，每次输出 `6` 个 `.dir`。
  - 9 个 candidate-vs-zmem 输出比较全部通过。
- 输出/哈希/误差摘要：
  - remote worktree：`/work/wenzhe/cuda3D/.codex_worktrees/formal_vpmlen16_table_20260608_2359`。
  - commit：`33553596ab66a9090e39c04be2928d4029a99db5`。
  - mean speedup vs zmem：
    - `directfill`：WP `1.101172x`，Gradient `1.100029x`，elapsed `1.081287x`，max rel L2 `0`。
    - `pressure_len16`：WP `1.194495x`，Gradient `1.179869x`，elapsed `1.098568x`，max rel L2 `6.384336e-07`。
    - `current_best_v_pml_len16`：WP `1.222023x`，Gradient `1.206588x`，elapsed `1.118261x`，max rel L2 `6.384336e-07`。
  - current best flags：
    - `-O3 -arch=sm_120 --use_fast_math`
    - `-DCUDA3D_PML_RECOMPUTE_Z`
    - `-DCUDA3D_PML_TILE_LIST`
    - `-DCUDA3D_PML_ZMEM_IN_P`
    - `-DPmlTileBlockSize1=32`
    - `-DPmlTileBlockSize2=4`
    - `-DPmlTileBlockSize3=2`
    - `-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`
    - `-DCUDA3D_CPML_VMEM_DISABLE_MPI`
    - `-DCUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
    - `-DCUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK`
    - `-DCUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK`
- 风险与下一步：
  - 决策：接受 `current_best_v_pml_len16` 为当前 RTX 5090 single-GPU formal current-best。
  - 该版本未达到 `1.5x` 阈值存档线，不做 speedup archive。
  - 后续不继续 v-PML micro packing。
  - 下一步应转入 design-level pressure/wave-step ownership model，或先整理给 Pro/后续 agent 的正式反馈报告。
  - 环境经验：isolated worktree 运行 `perf_1gpu_6shots` 前必须创建 case-local `d_obs/`；缺失时可能在写输出阶段 segfault。

## 2026-06-09 00:22:16 +08:00

- 操作目标：
  - 对 formal current-best 下的 residual pressure-PML kernel 做 NCU profile。
  - 判断 residual `cuda_fd3d_p_pml_tile_ns` 是否值得开启 branch/control/descriptor 类 CUDA prototype。
- 修改文件：
  - 新增 `reports/day_20260608/residual_pressure_source_profile_20260609_0012/` 文本 profile artifacts。
  - 新增 `tools/residual_pressure_route_gate.py`。
  - 新增 `docs/day_20260608/residual_pressure_route_gate.md`。
  - 新增 `reports/day_20260608/residual_pressure_route_gate.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端在 `/work/wenzhe/cuda3D/.codex_worktrees/formal_vpmlen16_table_20260608_2359` 中操作。
  - 使用 current-best flags 加 `-lineinfo` 重建：
    - `make -B -f makefile.rtx5090 test NVFLAGS="<current-best flags + -lineinfo>" -C src`
  - NCU：
    - `ncu --target-processes all --force-overwrite --export residual_p_pml_tile_source.ncu-rep --kernel-name "regex:cuda_fd3d_p_pml_tile_ns" --launch-skip 10 --launch-count 10 --section SourceCounters --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --section Occupancy ...`
    - `ncu --import residual_p_pml_tile_source.ncu-rep --csv --page details > residual_p_pml_tile_source.csv`
    - `python3 tools/ncu_csv_summary.py --profile residual_p_pml_tile_source ...`
  - 本地：
    - 使用 `tools/remote_get.py` 拉回 Markdown/JSON/CSV/log 文本 artifacts；不拉回 `.ncu-rep`。
    - `python -m py_compile tools\residual_pressure_route_gate.py`
    - `python tools\residual_pressure_route_gate.py --json-out reports\day_20260608\residual_pressure_route_gate.json --md-out docs\day_20260608\residual_pressure_route_gate.md`
- 测试结果：
  - current-best `-lineinfo` rebuild 通过。
  - NCU profile 通过，`NCU_CODE=0`。
  - gate 工具编译和运行通过。
  - 本轮不修改 CUDA 源码，不需要 correctness/perf repeat。
- 输出/哈希/误差摘要：
  - NCU residual kernel metrics：
    - No Eligible：`63.162%`。
    - eligible warps/scheduler：`0.766`。
    - warp cycles/issued inst：`23.682`。
    - avg active threads/warp：`23.050`。
    - avg not-predicated threads/warp：`21.730`。
    - branch efficiency：`83.750%`。
    - achieved occupancy：`73.389%`。
  - residual gate：
    - residual pressure-PML：`71.940us`，sampled-main share `25.33%`。
    - required residual local speedup：`1.2315x`。
    - required local reduction：`18.80%`。
    - required saved time：`13.524us`。
    - perfect branch efficiency sampled-main ceiling：`1.0429x`。
    - predicate cleanup sampled-main ceiling：`1.0147x`。
    - exact length-23 descriptor calibrated sampled-main speedup：`1.0153x`。
- 风险与下一步：
  - 决策：拒绝 residual pressure-PML micro CUDA prototype。
  - 禁止 residual branch-only split、length-32 branch/control specialization retry、length-23/exact descriptor retry、residual `p0 __ldg` / local `new_mem` / cache-policy / z-cache 小修。
  - 下一步只允许真正减少 pressure writeback 或 CPML state traffic 的 pressure/wave-step ownership model，或明确 cross-CTA/cluster-level primitive study，或用户明确改变 tolerance policy 后研究 precision-relaxation。
  - NCU source page 已在远端临时导出，但未映射出 C++ source text；保留 `.ncu-rep` 和 source page 在远端 worktree，不提交到仓库。

## 2026-06-09 00:35:55 +08:00

- 操作目标：
  - 汇总 current-best 之后所有剩余 pressure/wave-step ownership 路线。
  - 判断 ordinary exact-CUDA 下是否还有可立即写的新 prototype。
- 修改文件：
  - 新增 `tools/ownership_frontier_gate.py`。
  - 新增 `docs/day_20260608/ownership_frontier_gate.md`。
  - 新增 `reports/day_20260608/ownership_frontier_gate.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `python -m py_compile tools\ownership_frontier_gate.py`
  - `python tools\ownership_frontier_gate.py --json-out reports\day_20260608\ownership_frontier_gate.json --md-out docs\day_20260608\ownership_frontier_gate.md`
  - `python -c "import json; d=json.load(open('reports/day_20260608/ownership_frontier_gate.json', encoding='utf-8')); print(d['gate']['decision']); print(d['gate']['ordinary_cuda_allowed_count']); print(d['derived']['additional_wp_speedup_needed_to_1_5x'])"`
- 测试结果：
  - Python 编译检查通过。
  - frontier gate 工具运行成功，生成 Markdown/JSON。
  - 本轮不修改 CUDA 源码，不需要远端 build/correctness/perf repeat。
- 输出/哈希/误差摘要：
  - current best：`current_best_v_pml_len16`。
  - formal speedup vs zmem：
    - WP `1.222023x`。
    - Gradient `1.206588x`。
    - elapsed `1.118261x`。
    - max rel L2 `6.384336e-07`。
  - additional WP speedup needed to reach `1.5x`：`1.2275x`。
  - sampled main：`284.010us`。
  - remaining region shares：
    - p-core `33.00%`。
    - pressure-PML total `48.63%`。
    - v-PML total `18.37%`。
  - ordinary CUDA allowed prototype count：`0`。
  - gate decision：`ordinary_exact_cuda_frontier_exhausted_for_micro_routes`。
- 风险与下一步：
  - 决策：关闭 ordinary exact-CUDA micro-prototype frontier。
  - 不再启动 residual pressure、v-PML descriptor、z-face fusion、current p-core shared-plane、K=2 temporal、host scheduling 类小原型。
  - 不声明 `1.5x` milestone；当前正式 WP speedup 是 `1.222023x`。
  - 下一步允许写 Pro/后续 agent handoff report，或先研究具体 cluster/cooperative persistent-kernel primitive，或在用户明确改变 tolerance policy 后做 precision-relaxation，或转向 application-level batching / multi-shot scheduling。

## 2026-06-09 01:20:20 +08:00

- 操作目标：
  - 将 current-best 与已关闭 exact ordinary-CUDA frontier 整理成给 Pro/后续 agent 的交接报告。
  - 在 `AGENTS.md` 中固化 handoff 入口和后续禁止路线，避免重复已失败方向。
- 修改文件：
  - 新增 `docs/day_20260609/pro_handoff_current_best_frontier.md`。
  - 更新 `AGENTS.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `Get-Content -Encoding UTF8 "docs/day_20260609/pro_handoff_current_best_frontier.md"`
  - `Select-String -Path "AGENTS.md" -Encoding UTF8 -Pattern "Phase 4|current_best_v_pml_len16|ownership_frontier|2026-06-08 day sprint"`
  - `Get-Content -Encoding UTF8 "docs/architecture_decision_log.md" | Select-Object -Last 80`
  - 本轮仅做文档收口，不连接远端服务器，不运行 CUDA build/perf。
- 测试结果：
  - 本轮不修改 CUDA 源码、构建文件或 benchmark 脚本。
  - 不需要 correctness/perf repeat。
- 输出/哈希/误差摘要：
  - handoff 报告记录 current-best：`current_best_v_pml_len16`。
  - formal speedup vs zmem：
    - WP `1.222023x`。
    - Gradient `1.206588x`。
    - elapsed `1.118261x`。
    - max rel L2 `6.384336e-07`。
  - handoff 报告入口：`docs/day_20260609/pro_handoff_current_best_frontier.md`。
- 风险与下一步：
  - 当前 best 不是 `1.5x` milestone archive。
  - exact ordinary-CUDA micro-prototype frontier 仍保持关闭。
  - 后续要继续提速，必须先选择 scope change：
    - 研究 cluster/cooperative persistent-kernel primitive。
    - 用户明确放宽 tolerance 后研究 precision-relaxation。
    - 转向 application-level multi-shot scheduling。
    - 或停止 CUDA-core sprint 并打包当前成果。

## 2026-06-09 01:41:01 +08:00

- 操作目标：
  - 验证 RTX 5090 / CUDA 13 上 cooperative launch 与 thread-block cluster primitive 是否真实可用。
  - 判断这些 primitive 是否足以重开之前被 global synchronization / cross-CTA ownership 卡住的 K=2 temporal route。
- 修改文件：
  - 新增 `tools/cuda_cluster_capability_probe.cu`。
  - 新增 `tools/cluster_cooperative_frontier_gate.py`。
  - 新增 `docs/day_20260609/cluster_cooperative_frontier_gate.md`。
  - 新增 `reports/day_20260609/cluster_cooperative_frontier_gate.json`。
  - 新增 `reports/day_20260609/cluster_probe_stdout_20260609_0132.txt`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端创建隔离 worktree：
    - `git worktree add .codex_worktrees/cluster_probe_20260609_0132 FETCH_HEAD`
  - 上传 probe：
    - `tools/cuda_cluster_capability_probe.cu -> /work/wenzhe/cuda3D/.codex_worktrees/cluster_probe_20260609_0132/tools/cuda_cluster_capability_probe.cu`
  - 远端编译运行：
    - `/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O2 -arch=sm_120 -lineinfo tools/cuda_cluster_capability_probe.cu -o tools/cuda_cluster_capability_probe`
    - `./tools/cuda_cluster_capability_probe | tee reports_cluster_probe_stdout.txt`
  - 拉回 stdout：
    - `/work/wenzhe/cuda3D/.codex_worktrees/cluster_probe_20260609_0132/reports_cluster_probe_stdout.txt -> reports/day_20260609/cluster_probe_stdout_20260609_0132.txt`
  - 本地生成 gate：
    - `python -m py_compile tools\cluster_cooperative_frontier_gate.py`
    - `python tools\cluster_cooperative_frontier_gate.py --json-out reports\day_20260609\cluster_cooperative_frontier_gate.json --md-out docs\day_20260609\cluster_cooperative_frontier_gate.md`
- 测试结果：
  - CUDA probe 在远端 RTX 5090 上编译通过。
  - CUDA probe 运行通过。
  - Python gate 编译和生成通过。
  - 本轮不修改主 CUDA 程序，不需要 correctness/perf repeat。
- 输出/哈希/误差摘要：
  - device：`NVIDIA GeForce RTX 5090`。
  - compute capability：`12.0`。
  - SM count：`170`。
  - cooperative launch：`1`。
  - cluster launch：`1`。
  - 128-thread block active blocks / SM：`12`。
  - cooperative grid block ceiling：`2040`。
  - previous K=2 required blocks：`70688`。
  - cooperative over-capacity factor：`34.6510x`。
  - cooperative launch `2040` blocks pass；`2041` blocks 返回 `too many blocks in cooperative launch`。
  - cluster size `1/2/4/8` launch pass；cluster size `16` 返回 cluster misconfiguration。
- 风险与下一步：
  - 决策：`reject_direct_cooperative_grid_k2_temporal_reopen`。
  - 决策：`design_only_until_cluster_local_ownership_model_passes`。
  - 不写 direct cooperative-grid K=2 temporal prototype。
  - 不写没有 cluster-local ownership model 的 cluster temporal / producer-consumer fusion kernel。
  - 下一步若继续该方向，只允许先做 cluster-local ownership byte/synchronization model，必须覆盖 `p_mid` / velocity / CPML ownership、source injection、receiver extraction、shell/PML reconciliation 和 cross-cluster boundary。

## 2026-06-09 01:58:16 +08:00

- 操作目标：
  - 在已确认 cluster primitive 可用后，建立 cluster-local K=2 temporal ownership byte/synchronization model。
  - 判断是否可以写 cluster-local temporal / producer-consumer fusion CUDA prototype。
- 修改文件：
  - 新增 `tools/cluster_local_ownership_model.py`。
  - 新增 `docs/day_20260609/cluster_local_ownership_model.md`。
  - 新增 `reports/day_20260609/cluster_local_ownership_model.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `python -m py_compile tools\cluster_local_ownership_model.py`
  - `python tools\cluster_local_ownership_model.py --json-out reports\day_20260609\cluster_local_ownership_model.json --md-out docs\day_20260609\cluster_local_ownership_model.md`
- 测试结果：
  - Python 编译检查通过。
  - cluster-local ownership model 生成 Markdown/JSON 成功。
  - 本轮不修改主 CUDA 程序，不需要远端 build/correctness/perf repeat。
- 输出/哈希/误差摘要：
  - current-best anchor：
    - WP speedup vs zmem：`1.222023x`。
    - sampled main：`284.010us`。
    - p_core：`93.730us`，share `33.00%`。
  - cooperative / cluster capacity：
    - cooperative grid ceiling：`2040` blocks。
    - previous K=2 required blocks：`70688` blocks。
    - over-capacity factor：`34.6510x`。
    - max passing cluster size：`8`。
  - optimistic DSM tile gate：
    - required p_core pair reduction for `>=5%` sampled-main：`14.43%`。
    - required local pair byte ratio：`<=0.8557`。
    - ideal no-dup sampled-main speedup：`1.1317x`。
    - best DSM tile：cluster size `8`，output z/x/y `40/44/48`，p_mid bytes `776736`。
    - best local pair byte ratio：`1.1602x`。
    - estimated sampled-main speedup：`0.9498x`。
- 风险与下一步：
  - 决策：`reject_cluster_local_temporal_cuda_prototype`。
  - 不写 direct cooperative-grid K=2 temporal prototype。
  - 不写 cluster-local K=2 temporal CUDA prototype with DSM `p_mid` tile。
  - 不写没有新 ownership byte model 的 cluster producer-consumer fusion。
  - 后续 CUDA-core 方向只剩：
    - 用户明确放宽 tolerance 后做 precision-relaxation。
    - 转向 application-level multi-shot scheduling / batching。
    - 或提出完全不同的 ownership representation，并先通过 byte/synchronization model。

## 2026-06-09 02:16:19 +08:00

- 操作目标：
  - 汇总当前 exact CUDA-core frontier 关闭后的 application-level scheduling 路线。
  - 使用正式 `current_best_v_pml_len16` 口径更新 true multi-GPU / same-GPU / host-setup 决策。
- 修改文件：
  - 新增 `tools/application_level_frontier_gate.py`。
  - 新增 `docs/day_20260609/application_level_frontier_gate.md`。
  - 新增 `reports/day_20260609/application_level_frontier_gate.json`。
  - 更新 `AGENTS.md`。
  - 更新 `docs/architecture_decision_log.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端平台复核：
    - `cd /work/wenzhe/cuda3D`
    - `nvidia-smi -L`
  - 本地生成 gate：
    - `python -m py_compile tools\application_level_frontier_gate.py`
    - `python tools\application_level_frontier_gate.py --json-out reports\day_20260609\application_level_frontier_gate.json --md-out docs\day_20260609\application_level_frontier_gate.md`
- 测试结果：
  - 远端 `nvidia-smi -L` 显示当前仍只有 `1` 张 `NVIDIA GeForce RTX 5090`。
  - Python 编译检查通过。
  - application-level frontier Markdown/JSON 生成通过。
  - 本轮不修改主 CUDA 程序，不需要 build/correctness/perf repeat。
- 输出/哈希/误差摘要：
  - formal current-best：
    - alias：`current_best_v_pml_len16`。
    - mean elapsed：`3.016667s`。
    - mean Gradient：`2.111930s`。
    - mean WP：`1.988905s`。
    - elapsed speedup vs zmem：`1.1183x`。
    - Gradient speedup vs zmem：`1.2066x`。
    - WP speedup vs zmem：`1.2220x`。
    - max rel L2：`6.384336e-07`。
  - same-GPU multi-rank：
    - decision：`reject_same_gpu_multirank_probe`。
    - best elapsed speedup：`0.9200x`。
    - best Gradient speedup：`0.9301x`。
  - true multi-GPU shot-balance theory：
    - `2` GPUs：`[3,3]`，ideal `2.0000x`。
    - `3` GPUs：`[2,2,2]`，ideal `3.0000x`。
    - `4` GPUs：`[2,2,1,1]`，ideal `3.0000x`。
    - `6` GPUs：`[1,1,1,1,1,1]`，ideal `6.0000x`。
  - host/setup：
    - outside process wrapper：`0.547231s`。
    - `MPI_Init`：`0.254292s`。
    - gpu setup/context：`0.186226s`。
    - cal-loop `wavefield_prep` ceiling：`1.0236x` Gradient。
- 风险与下一步：
  - 决策：`no_local_application_level_experiment_available_on_single_gpu`。
  - 不继续 same-GPU oversubscription。
  - 不用 root-rank printed WP 声称 multi-rank speedup。
  - 不写没有新 `>=5%` measured hotspot 的 host/setup 小修。
  - true multi-GPU batching 必须等 `>=2` visible GPUs 平台。
  - 后续只剩：
    - 多 GPU 平台上验收 true multi-GPU batching。
    - 用户明确放宽 tolerance 后做 precision-relaxation。
    - 或停止 CUDA-core sprint，打包 current-best 成果。

## 2026-06-09 02:34:50 +08:00

- 操作目标：
  - 为当前 `current_best_v_pml_len16` 生成可交接的 current-best package summary。
  - 明确该 package 不是 `1.5x` speed-threshold archive，避免误存档。
- 修改文件：
  - 新增 `tools/current_best_package_summary.py`。
  - 新增 `docs/day_20260609/current_best_package_summary.md`。
  - 新增 `reports/day_20260609/current_best_package_summary.json`。
  - 更新 `AGENTS.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `python -m py_compile tools\current_best_package_summary.py`
  - `python tools\current_best_package_summary.py --json-out reports\day_20260609\current_best_package_summary.json --md-out docs\day_20260609\current_best_package_summary.md`
  - `git log --oneline --decorate -12`
- 测试结果：
  - Python 编译检查通过。
  - current-best package Markdown/JSON 生成通过。
  - 本轮不修改主 CUDA 程序，不需要 build/correctness/perf repeat。
- 输出/哈希/误差摘要：
  - package status：`current_best_not_speed_threshold_archive`。
  - branch：`exp/day-20260608-cpml-compact-temporal`。
  - package 生成时 HEAD：`f637ba115d52852b493867ab4a957113a01142a5`。
  - candidate：`current_best_v_pml_len16`。
  - mean elapsed：`3.016667s`。
  - mean Gradient：`2.111930s`。
  - mean WP：`1.988905s`。
  - elapsed speedup vs zmem：`1.118261x`。
  - Gradient speedup vs zmem：`1.206588x`。
  - WP speedup vs zmem：`1.222023x`。
  - max rel L2：`6.384336e-07`。
  - max abs：`4.768372e-06`。
  - all compare pass：`true`。
  - additional WP speedup to `1.5x`：`1.227472x`。
- 风险与下一步：
  - 不写入 `archives/speedups/`，因为当前不是 `1.5x` milestone archive。
  - single-GPU RTX 5090 上已无继续自动推进的本地 CUDA/profiling experiment。
  - 后续继续提速需要至少一个外部条件：
    - `>=2` visible GPUs 进行 true multi-GPU batching validation。
    - 用户明确放宽 tolerance policy 后做 precision-relaxation。
    - 提出全新的 ownership representation，并先通过 byte/synchronization model。

## 2026-06-09 12:46:39 +08:00

- 操作目标：
  - 根据 Pro 反馈开展今日任务：固化 current-best、建立 micro-bank 政策、准备 multi-GPU batching、写 precision-relaxation proposal 和 next-scope menu。
  - 明确 `2.3x vs original` 只能作为估算，不能伪造成 direct original-vs-current formal table。
- 修改文件：
  - 新增 `docs/current_best_v_pml_len16_release.md`。
  - 新增 `docs/original_vs_current_best_20260609.md`。
  - 新增 `reports/original_vs_current_best_20260609/summary.json`。
  - 新增 `docs/micro_bank_policy.md`。
  - 新增 `docs/multigpu_shot_batching_plan.md`。
  - 新增 `tools/run_multigpu_batching.py`。
  - 新增 `docs/precision_relaxation_policy_proposal.md`。
  - 新增 `docs/next_scope_decision_menu.md`。
  - 新增 `reports/current_best_frontier_20260609/final_report.md`。
  - 更新 `AGENTS.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - current-best tag：
    - `git tag -a current-best-v-pml-len16-rtx5090-20260609 -m "current best v-pml len16 RTX5090 20260609"`
  - original baseline 可用性核对：
    - `git tag --list "current-best-v-pml-len16-rtx5090-20260609"`
    - `git branch --all --list "*orig*" "*baseline*" "*current*"`
    - `Get-ChildItem -Recurse -Directory -Depth 3 | Where-Object {$_.Name -match 'orig|original|baseline|archive'}`
    - `Select-String ... -Pattern "orig_code|original baseline|最原始|current_best_reference|1.8x"`
  - multi-GPU runner checks：
    - `python -m py_compile tools\run_multigpu_batching.py`
    - `python tools\run_multigpu_batching.py --case perf_1gpu_6shots --gpus 0`
    - `python tools\run_multigpu_batching.py --case perf_1gpu_6shots --gpus 0,1`
- 测试结果：
  - `tools/run_multigpu_batching.py` 语法检查通过。
  - `--gpus 0` 正确拒绝：true multi-GPU batching requires at least 2 requested GPUs。
  - `--gpus 0,1` 在当前可见 GPU 不足时正确拒绝：nvidia-smi reports 1 GPUs, but 2 were requested。
  - 本轮不修改主 CUDA 程序，不需要 build/correctness/perf repeat。
- 输出/哈希/误差摘要：
  - current-best release：
    - tag：`current-best-v-pml-len16-rtx5090-20260609`。
    - tag target commit：`f637ba115d52852b493867ab4a957113a01142a5`。
    - WP speedup vs zmem：`1.222023x`。
    - Gradient speedup vs zmem：`1.206588x`。
    - elapsed speedup vs zmem：`1.118261x`。
    - max rel L2：`6.384336e-07`。
  - current-best binary hashes：
    - round 1：`aa58035a8a084bfd34fc2336bfbbb10fb3586ba9352c75109e49fd2be7909278`。
    - round 2：`dd085284245186517599db50cb98f19affef2855ef8ed17cc13a54273c64162b`。
    - round 3：`881e9e35f0291bad9b63da90bf04f18d0ad07550325a8b808475b2a1264940b9`。
  - original-vs-current：
    - direct original-vs-current table：`unavailable`。
    - 原因：未找到可证明为“最原始版本”的可重建源码；`orig_code` 不能作为 original baseline。
    - estimated WP vs original：`2.308x`。
    - estimated Gradient vs original：`2.274x`。
- 风险与下一步：
  - 当前 `2.3x` 必须表述为 estimated speedup vs original，直到找到真正 original source/commit 并同机重跑。
  - 不创建 `archives/speedups/1.5x...`，因为当前 package 不是 speed-threshold archive。
  - 下一步由 scope 决策驱动：
    - `>=2` GPUs：运行 true multi-GPU batching。
    - 用户批准 Tier 1/2 tolerance：启动 precision-relaxation feasibility。
    - 新 ownership idea：先过 byte/synchronization model。
    - 否则使用 current-best package 收口。

## 2026-06-09 14:58:00 +08:00

- 操作目标：
  - 根据用户提供的 freeosc 集群指南和登录信息，进行只读集群巡检。
  - 判断 Multi-GPU Batching 阶段是否可立即在 RTX 5090 节点上开展。
  - 固化 Slurm/GRES 使用约束，避免后续错误占卡或在登录节点运行 benchmark。
- 修改文件：
  - 新增 `docs/freeosc_cluster_survey_20260609.md`。
  - 更新 `docs/multigpu_shot_batching_plan.md`。
  - 更新 `AGENTS.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 本地：
    - `git status --short`
    - `Get-Content -Encoding UTF8 reports/freeosc_202603_extracted.txt -TotalCount 80`
  - 远程只读：
    - `hostname; whoami; id; pwd; date`
    - `sbatch --version`
    - `sinfo -p gpu -eN -O "NodeHost:8,StateLong:14,CPUsState:18,Gres:90,GresUsed:90,Reason:40"`
    - `squeue -p gpu`
    - `squeue -u shengwz`
    - `scontrol show partition gpu`
    - `sinfo -R`
    - `scontrol show node gpu1..gpu6`
    - `module avail cuda`
    - `module avail oneapi`
    - `module show cuda/12.3`
    - `module show cuda/12.8-gpu6`
    - `module show cuda/13.1-gpu6`
    - `df -h . /home/scratch`
    - `beegfs-ctl --getquota --uid $(id -u)`
- 测试结果：
  - SSH 登录成功，登录节点为 `mu01`。
  - Slurm 可用，版本 `24.05.3`。
  - 本轮未提交 GPU 作业，未编译，未运行 CUDA benchmark。
- 输出/哈希/误差摘要：
  - `gpu` partition：`Nodes=gpu[1-6]`，`OverSubscribe=NO`，`ExclusiveUser=NO`。
  - `gpu1`：mixed，`4x V100S + 4x A40`，8 个 GPU GRES 全部已分配。
  - `gpu2`：down/not responding，`4x A100`，原因 `power_cut`。
  - `gpu3`：mixed，`3x A100 + 4x 10gb`，7 个 GPU GRES 全部已分配。
  - `gpu4`：down/not responding，`8x RTX4090`，原因 `power_cut`。
  - `gpu5`：down/not responding，`6x RTX5090`，原因 `power_cut`。
  - `gpu6`：down/not responding，`8x RTX5090`，原因 `power_cut`。
  - 可用 CUDA modules：`cuda/10.1`、`cuda/11.5`、`cuda/12.3`、`cuda/12.6-gpu5`、`cuda/12.8-gpu6`、`cuda/13.1-gpu6`。
  - oneAPI modules 包括 `oneapi/compiler-2021.4.0`、`oneapi/mpi-2021.4.0`、`oneapi/mkl-2021.4.0`。
- 风险与下一步：
  - 当前 RTX 5090 节点 `gpu5/gpu6` 都处于 `power_cut`，不能立即开展 `sm_120` formal multi-GPU batching。
  - 若只验证流程，可等待 `gpu1/gpu3` 资源释放后按 V100S/A40/A100 架构重编，但不能用作 RTX 5090 current-best 正式续跑。
  - 下一步建议先请管理员恢复 `gpu5` 或 `gpu6`，随后提交极短 `sbatch` GPU probe，再上传/构建项目并运行 multi-GPU batching gate。

## 2026-06-09 16:35:00 +08:00

- 操作目标：
  - 启动 single-GPU exact-FP32 next optimization：`PML len16 compact-state ownership`。
  - 根据用户新的精度偏好，建立 high-precision / relaxed-precision 双轨纪律。
  - 不把 Multi-GPU batching 计入 CUDA code optimization speedup。
- 修改文件：
  - 新增 `tools/pml_len16_state_traffic_audit.py`。
  - 新增 `docs/precision_tracks_policy.md`。
  - 更新 `AGENTS.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - 远端创建隔离 worktree：
    - `/work/wenzhe/cuda3D/.codex_worktrees/compact_state_20260609`
    - branch：`exp/pml-len16-compact-state`
  - current-best build：
    - `make -B -f makefile.rtx5090 test` with current-best flags。
  - current-best verification：
    - `python tools/run_benchmark.py --case smoke_1gpu --tag compact_current_best_smoke_data_dir`
    - `python tools/run_benchmark.py --case correctness --tag compact_current_best_correctness`
    - `python tools/run_benchmark.py --case perf_1gpu_6shots --tag compact_current_best_perf6_a2`
    - `python tools/run_benchmark.py --case perf_1gpu_6shots --tag compact_current_best_perf6_b2`
    - `python tools/run_benchmark.py --case perf_1gpu_6shots --tag compact_current_best_perf6_c2`
  - NCU：
    - `ncu --target-processes all --section SpeedOfLight --section MemoryWorkloadAnalysis --section SourceCounters --section WarpStateStats --kernel-name regex:".*(p_pml|v_pml).*" --launch-skip 10 --launch-count 20 ...`
- 测试结果：
  - current-best build 通过。
  - 初次 `smoke_1gpu` 失败，原因为新 worktree 缺少 `bench_smoke/d_obs` 目录；补目录后 smoke 通过。
  - `correctness` 通过，输出 6 个 `.dir`。
  - 初次 `perf_1gpu_6shots` 失败，原因为新 worktree 缺少大速度模型 symlink；补充指向主目录数据的 symlink 后 3 轮 perf 均通过。
  - NCU PML short profile 通过并导出 CSV。
- 输出/哈希/误差摘要：
  - remote worktree：`/work/wenzhe/cuda3D/.codex_worktrees/compact_state_20260609`。
  - build binary SHA256：`05eeb26793e444c96de7117e4f086dce3c7682cda654a9a7e771b5af69c6f8f9`。
  - smoke：returncode `0`，outputs `3`，WP `0.002238s`，Gradient `0.003071s`。
  - correctness：returncode `0`，outputs `6`，WP `0.012079s`，Gradient `0.013798s`。
  - perf repeat mean WP：`2.004982s`。
  - perf repeat mean Gradient：`2.118638s`。
  - NCU profile output：`benchmarks/profiles/compact_state/current_best_pml_state_ncu.ncu-rep`。
  - NCU CSV：`benchmarks/profiles/compact_state/current_best_pml_state_ncu.csv`。
- 风险与下一步：
  - 新 worktree 中大测试数据是未跟踪 symlink/目录，后续迁移环境时必须重建这些链接。
  - 继续 Phase 1：运行 `tools/pml_len16_state_traffic_audit.py` 生成 compact-state traffic gate。
  - 如果 estimated whole-job speedup ceiling `<3%`，停止 compact-state CUDA 实现；`3%..5%` 只允许 mirror；`>=5%` 才进入 design + mirror + prototype。

## 2026-06-09 16:43:00 +08:00

- 操作目标：
  - 完成 Phase 1 `PML len16 state traffic audit`。
  - 根据 gate 结果写 Phase 2 compact-state ownership design。
- 修改文件：
  - 新增 `docs/compact_state/pml_len16_state_traffic_audit.md`。
  - 新增 `reports/compact_state/pml_len16_state_traffic_audit.json`。
  - 新增 `reports/compact_state/current_best_phase0_summary.json`。
  - 新增 `docs/compact_state/pml_len16_compact_state_design.md`。
  - 追加本 `AGENT_LOG.md` 条目。
- 执行命令摘要：
  - `python3 tools/pml_len16_state_traffic_audit.py --case-dir benchmarks/cases/perf_1gpu_6shots --perf-log ... --ncu-csv ... --json-out ... --md-out ... --p-core-us 75.0 --p-len16-state-fraction 0.35`
  - `remote_get.py` 同步 audit/report artifacts 回本地分支。
- 测试结果：
  - audit tool 运行通过。
  - Phase 1 gate：`allow_commit_prototype_after_design`。
  - 本轮仍未修改 CUDA kernel；不需要重新 build。
- 输出/哈希/误差摘要：
  - sampled main kernel us：`266.710`。
  - `cuda_fd3d_p_pml_len16_halfwarp_ns` duration：`67.04 us`。
  - `cuda_fd3d_p_pml_tile_ns` duration：`72.99 us`。
  - `cuda_fd3d_v_pml_len16_halfwarp_ns` duration：`20.16 us`。
  - `cuda_fd3d_v_pml_tile_ns` duration：`31.52 us`。
  - pressure len16 tiles：`67392`。
  - pressure len16 active points：`8626176`。
  - pressure len16 compact lines：`539136`。
  - compact pressure-state bytes：`127.512 MiB`。
  - full pressure-related state bytes x shots：`274.324 MiB`。
  - compact/full ratio：`0.464821`。
  - assumed removable p_len16 state fraction：`0.35`。
  - modeled whole sampled-main speedup ceiling：`1.096462x`。
- 风险与下一步：
  - 该 gate 是 optimistic ceiling，不代表实际能达到 `1.096x`。
  - `v_len16` 当前不更新 CPML state，compact-state 首版只瞄准 pressure len16 的 `memory_dzz` 与 z-recompute `memory_dz` old/next。
  - 下一步先实现 `CUDA3D_PML_LEN16_COMPACT_STATE_MIRROR`，full-array path 仍权威；mirror 失败则停止 commit prototype。

## 2026-06-09 17:08:00 +08:00

- 操作目标：
  - 实现并验证 Phase 3 `CUDA3D_PML_LEN16_COMPACT_STATE_MIRROR`。
  - mirror 只验证 accepted pressure len16 compact line mapping，不改变输出数学路径。
- 修改文件：
  - `include/inc3D/single_solver.h`
  - `src/single_solver.cu`
  - `src/rem_fd.cu`
  - 新增 `docs/compact_state/pml_len16_compact_state_mirror_result.md`
  - 新增 `reports/compact_state/mirror_binary_sha256.txt`
  - 新增 `reports/compact_state/compare_mirror_correctness_vs_current_best/comparison.md`
  - 新增 `reports/compact_state/compare_mirror_correctness_vs_current_best/comparison.json`
  - 新增 `reports/compact_state/compare_mirror_perf6_probe_vs_current_best/comparison.md`
  - 新增 `reports/compact_state/compare_mirror_perf6_probe_vs_current_best/comparison.json`
- 执行命令摘要：
  - Build mirror binary with current-best flags plus `-DCUDA3D_PML_LEN16_COMPACT_STATE_MIRROR`。
  - `python tools/run_benchmark.py --case smoke_1gpu --tag compact_mirror_smoke`
  - `python tools/run_benchmark.py --case correctness --tag compact_mirror_correctness`
  - `python tools/run_benchmark.py --case perf_1gpu_6shots --tag compact_mirror_perf6_probe`
  - `python3 tools/compare_outputs.py --baseline ...current_best_correctness... --candidate ...mirror_correctness...`
  - `python3 tools/compare_outputs.py --baseline ...current_best_perf6_a2... --candidate ...mirror_perf6_probe...`
- 测试结果：
  - 初次 mirror build link 失败：mirror kernels 被错误放在 disabled fused-zface conditional block 内。
  - 修正条件编译位置后，mirror build 通过。
  - `smoke_1gpu` 通过，outputs `3`。
  - `correctness` 通过，outputs `6`。
  - `perf_1gpu_6shots` probe 通过，outputs `6`。
  - mirror internal check 覆盖 6 炮，每炮 `it=0/1/2/1500`。
- 输出/哈希/误差摘要：
  - mirror binary SHA256：`3d284fac86d066d8ce09c4d1f0a7126714f198555ecb087350b5e27c6ae636b3`。
  - mirror internal check：所有检查行均为 `rel_l2=0`、`max_abs=0`、`bad=0`。
  - correctness output compare vs Phase 0 current-best：pass，6 个输出 rel L2 全部 `0`。
  - perf probe output compare vs Phase 0 current-best：pass，6 个输出 rel L2 全部 `0`。
  - mirror perf probe WP：`2.079581s`。
  - mirror perf probe Gradient：`2.191727s`。
- 风险与下一步：
  - mirror 有额外 gather/compare 开销，不能作为性能数据。
  - mirror 当前证明 compact line mapping 和 23-slot z-window 映射与 full arrays 对齐。
  - 下一步才是 commit prototype：`CUDA3D_PML_LEN16_COMPACT_STATE`，让 accepted pressure len16 kernel 真正从 compact state 读写，并保留 residual full-array fallback。
