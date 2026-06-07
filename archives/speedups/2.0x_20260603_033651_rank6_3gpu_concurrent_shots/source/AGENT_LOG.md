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
