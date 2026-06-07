# PML Layout Design Report

## 1. 设计目标

将 `v_pml_ns` → `p_pml_ns` 的数据流从 **full-domain 数组读写** 重构为 **PML face-major contiguous buffer**，以：
- 消除 `p_pml_ns` 在 core 区域 launch 空线程的 overhead
- 降低 `p_pml_ns` 读取 vz/vx/vy 时的 L2 cache thrashing（当前大量访问跨越 core 边界）
- 为后续可能的 PML-only temporal blocking 提供紧凑的数据布局

## 2. 当前 `p_pml_ns` 的数据访问分析

### 2.1 PML shell 的拓扑划分

定义（以 `perf_1gpu` 参数为例：n1=119, n2=408, n3=408, npml=12, CorePmlMargin=4）：

- **core box**: z∈[16,103), x∈[16,392), y∈[16,392) — `p_pml_ns` 直接 return
- **PML shell**: full domain − core box，体积 ≈ 5.8M points
- **6-face 互不相交划分**：
  - `Z0`: z∈[0,16),  x∈[0,408), y∈[0,408)
  - `Z1`: z∈[103,119),x∈[0,408), y∈[0,408)
  - `X0`: z∈[16,103),x∈[0,16),  y∈[0,408)
  - `X1`: z∈[16,103),x∈[392,408),y∈[0,408)
  - `Y0`: z∈[16,103),x∈[16,392),y∈[0,16)
  - `Y1`: z∈[16,103),x∈[16,392),y∈[392,408)

### 2.2 `p_pml_ns` 真正读取的 vz/vx/vy 范围

对于 PML shell 中任意点 `(z,x,y)`，`p_pml_ns` 读取：

| 数组 | 读取范围 | 条件 | 与 `v_pml_ns` 写入区域的关系 |
|---|---|---|---|
| `vz` | `[z-4, z+3]` @ `(x,y)` | 8th-order z-diff | **完全在 `v_pml_ns` 写入域内**（已验证） |
| `vx` | `[x-4, x+3]` @ `(z,y)` | 8th-order x-diff | **完全在 `v_pml_ns` 写入域内**（已验证） |
| `vy` | `[y-4, y+3]` @ `(z,x)` | 8th-order y-diff | **完全在 `v_pml_ns` 写入域内**（已验证） |

**关键验证**：`v_pml_ns` 对 vz 的不写入区域为 `z∈[19,99) ∧ x∈[16,392) ∧ y∈[16,392)`。PML shell 中任意点需要的 vz 邻居 `[z-4,z+3]`：
- 若点在 Z-face（z<16 或 z≥103），则 `z+3<19` 或 `z-4≥99`，邻居范围完全避开不写入区域
- 若点在 X-face（x<16 或 x≥392），则 x 不在 `[16,392)` 内，`need_vz=true`
- 若点在 Y-face（y<16 或 y≥392），同理 `need_vz=true`

因此 **`p_pml_ns` 读取的每一个 vz/vx/vy 元素，都由 `v_pml_ns` 在上一步写入**，无未定义值风险。

### 2.3 当前 full-domain 访问的问题

`p_pml_ns` 当前启动 grid 覆盖 full domain（≈19.8M 点），core 内线程 `return`。问题：

1. **Launch overhead**：约 70% 的线程空跑，占用 SM warp slot 和指令 issue 带宽
2. **Cache thrashing**：X-face 点读取 `vx[x-4:x+3]` 时，x 邻居深入 core 区域（如 x=15 需要 vx[11:18]，其中 vx[16:18] 在 core 边缘）。这些访问与 core 内 `p_core_ns` 的 p1 访问竞争 L2 cache line
3. **地址计算**：`base = (y+radius)*stride3 + (x+radius)*stride2 + (z+radius)` 产生大量 64-bit 地址运算，且跨度大

## 3. Face-Major Contiguous Buffer 设计

### 3.1 Buffer 分配策略

为 6 个 face 各分配一个 **独立 contiguous buffer**。每个 buffer 存储该 face 内所有点需要的 **vz/vx/vy 原始值**（不是差分值）。

**Z0 buffer**（示例）：
- 逻辑形状：`(y, x, z)` = `(408, 408, 16)`
- 实际存储：按 `z * (408*408) + x * 408 + y`（或其他连续顺序）
- 大小：408×408×16×3(arrays)×4(bytes) ≈ 31.9 MB（vz+vx+vy）

但 6 个 face 总和会超过当前 full-domain 数组大小。需要优化：

**优化**：每个 face 只存储该 face 需要的数组。
- Z-faces 的 `vz` 在 z 方向差分，但 `vx` 在 x 方向、`vy` 在 y 方向。所有 3 个数组都需要。
- 实际上 6 个 face 都需要 3 个数组。

更关键：**face buffer 只需要存储 PML shell 内的点，不需要 core 内的点**。但 stencil 需要 halo...

### 3.2 Halo 问题与解决方案

对于 Z0 face 中的点 z=15：
- 需要 vz[11:18]。vz[16:18] 在 margin `[12,16)` 内，属于 Z0 face。
- **不需要 core 内的 vz 值**！

验证：Z0 face 中 z 最大为 15。`vz[z-4:z+3]` 最大到 z=18。z=16,17 在 margin 内，仍属于 PML shell（因为 core 从 16 开始但 `p_pml_ns` 处理 z<16 的所有点，而 margin [12,16) 内没有 core 排除，实际上 margin 是 PML 的一部分...等等）。

重新理清：
- `p_pml_ns` 排除的是 `z∈[16,103) ∧ x∈[16,392) ∧ y∈[16,392)`
- 所以 z=16, x=200, y=200 是 core 点，`p_pml_ns` 不处理
- 但 z=15, x=200, y=200 是 PML 点，`p_pml_ns` 处理。它需要 vz[11:18]。其中 vz[16] 在 z=16, x=200, y=200，这是 core 点！

**问题来了**：vz[16,200,200] 是 core 点，`v_pml_ns` 不写入它（因为 `need_vz=false` 在 z∈[19,99), x∈[16,392), y∈[16,392)），但 z=16 < 19，所以 `need_vz=true`！

vz[17,200,200]: z=17<19, need_vz=true
vz[18,200,200]: z=18<19, need_vz=true
vz[19,200,200]: z=19≥19, x=200∈[16,392), y=200∈[16,392), **need_vz=false**

所以对于 z=15 的 PML 点，需要 vz[11:18]，其中 vz[19] **不**需要（范围到 18 为止）。所有需要的 vz 值都在 `v_pml_ns` 写入域内。

但等等，如果 PML 点 z=103（Z1 face 起点）：
- 需要 vz[99:106]
- vz[99]: z=99, need_vz = !(99≥19 && 99<99 ...) = true（因为 99<99 不成立）
- vz[100,101,102]: 同理 true
- vz[103,104,105,106]: z≥103, 不在 [19,99) 内，true

所以对于 Z1 face，需要的 vz 也都在写入域内。

**结论**：每个 face 的 stencil 邻居范围**完全落在 `v_pml_ns` 的写入域内**，且**不需要 core 内部的值**。这意味着 face buffer **不需要存储 core 内部的数据**！

但是，face buffer 内部是否需要 halo？
- Z0 face 中的点 z=0 需要 vz[-4:3]。vz[-4:-1] 在 padding 内。`v_pml_ns` 是否写入 padding？
- `v_pml_ns` 条件：`if( gtid1 < n1 && gtid2 < n2 && gtid3 < n3)`，gtid1 是原始 z，范围 [0, n1)。padding 对应 padded z ∈ [0,4) 和 [n1+4, n1+8)。`v_pml_ns` 不处理这些点（gtid1 ≥ n1 时条件不满足）。
- 所以 padding 内的 vz 值**不**被 `v_pml_ns` 写入，保持为 0。

这意味着 `p_pml_ns` 在 z=0 时读取 vz[-4:3]，其中 vz[-4:-1] 为 0。这与当前实现一致（当前 `v_pml_ns` 也不写入 padding）。

### 3.3 精简后的 Face Buffer 方案

**方案 A：6-face contiguous buffers，无 core 数据**

每个 face buffer 只存储该 face 内的 vz/vx/vy。但 `p_pml_ns` 的 stencil 可能跨越 face 边界：

- Z0 点 z=15 需要 vz[11:18]。vz[16:18] 仍在 Z0（因为 Z0 = z<16）。vz[11:15] 也在 Z0。
- Z0 点 x=0 需要 vx[-4:3]。vx[-4:-1] 在 padding 中（值为 0）。vx[0:3] 在 Z0 内（因为 Z0 包含所有 x∈[0,408)）。

等等，Z0 是 z∈[0,16), x∈[0,408), y∈[0,408)。对于 Z0 中的点，其 vx 邻居 `[x-4,x+3]`：
- x=0 需要 [-4,3]，其中 [-4:-1] 是 padding
- x=15 需要 [11,18]，都在 [0,408) 内

vx 邻居完全在 Z0 内，因为 Z0 的 x 范围是完整的 [0,408)。

类似地，vy 邻居 `[y-4,y+3]` 也完全在 Z0 内（y 范围完整）。

vz 邻居 `[z-4,z+3]`：
- z=0 需要 [-4,3]，[-4:-1] 是 padding
- z=15 需要 [11,18]，但 Z0 只到 z<16。vz[16:18] 不在 Z0 内！

**这是关键问题**：Z0 face 中的点需要 vz 邻居到 z=18，但 Z0 只存储 z<16 的数据。

如果 face buffer 不包含 margin 数据，就需要从别处读取 vz[16:18]。但 vz[16:18] 在 margin 内（z∈[12,16) 是 PML，z=16 是 core 边界）。

等等，z=16 是 core 的起点。但 `p_pml_ns` 排除的是 z∈[16,103) ∧ x∈[16,392) ∧ y∈[16,392)。对于 x=200,y=200，z=16 是 core。但对于 x=10,y=200，z=16 不是 core（因为 x<16），`p_pml_ns` 会处理它。

这意味着 vz[16] 对于某些点是 PML 点，对于某些点是 core 点。

**这使得 face buffer 设计变得复杂**。如果 Z0 buffer 只包含 z∈[0,16)，那么：
- Z0 中 z=15, x=200, y=200 的点需要 vz[16:18]，这些值不在 Z0 buffer 中
- 但这些值在 margin 内，`v_pml_ns` 确实写入了它们

如果要在 face buffer 中提供这些值，需要：
1. 方案 A1：Z0 buffer 扩展为 z∈[0, 16+halo)，其中 halo=3（因为 z=15 需要到 z=18）。但这会引入 core 边缘的数据。
2. 方案 A2：保持 Z0 buffer 为 z∈[0,16)，对 vz[16:18] 的访问回退到 full-domain 数组。
3. 方案 A3：将 margin 数据归入相邻 face，重新定义 face 边界。

方案 A3 可能最干净：把 margin 也作为独立 face，或扩展 face 定义。

但实际上，对于 z=15 的点，它属于 Z0 face。vz[16:18] 在逻辑上是 Z0 的 "upper halo"。如果我们在 Z0 buffer 中预留 halo 区域（z∈[0, 16+3) = [0,19)），就可以避免跨 buffer 访问。

类似地，Z1 需要 lower halo（z∈[96, 103) 即 [103-4, 103)？不对，Z1 起点 z=103，z=103 需要 vz[99:106]，其中 vz[99:102] 不在 Z1 内，需要 halo）。

**结论：每个 face buffer 需要包含 3 个点的 stencil halo**。

但这会导致 face buffer 之间有重叠。例如 Z0 的 upper halo（z=16:18）和 X0 的 z 范围（z∈[16,103)）有重叠。

如果每个 face buffer 独立包含自己的 halo，总存储量会增加，且有重复数据。

### 3.4 更实际的方案：Single PML Shell Buffer

与其分 6 个 face buffer 各带 halo，不如：

**分配一个完整的 PML shell buffer**，形状与 full domain 相同，但只在 PML shell 位置有有效数据，core 内为 0 或未使用。

但这没有解决任何问题（layout 还是一样）。

**替代方案：2 个 buffer（Z-faces + XY-faces）**

将 PML shell 分为：
1. **Z-buffer**：z∈[0,16) ∪ [103,119), x∈[0,408), y∈[0,408) — 完整的 Z-faces，天然 contiguous in z（两个 slab）
2. **X-buffer**：z∈[16,103), x∈[0,16) ∪ [392,408), y∈[0,408) — 两个 x-slabs
3. **Y-buffer**：z∈[16,103), x∈[16,392), y∈[0,16) ∪ [392,408) — 两个 y-slabs

但这只是按 face 分，存储布局还是 strided。

**关键洞察**：当前 full-domain 布局已经是 z-fastest, x-middle, y-slowest。Z-face（固定 z 范围，完整 x,y）在这种布局下是最不连续的（因为 z 是 fastest，固定 z 意味着跨 stride）。X-face（固定 x 范围，完整 z,y）是中等连续。Y-face（固定 y 范围，完整 z,x）是最连续的（因为 y 是 slowest，固定 y 意味着在一个 contiguous plane 内）。

所以如果要把 face 数据紧凑化，Z-face 的收益最大（因为当前 Z-face 的 z-fastest 布局导致 Z-face 内的点 memory stride 巨大）。

但 `p_pml_ns` 的 stencil 在 z 方向只需要 8 个连续值（在 z-fastest 布局下是完美 coalesced）。问题不在 Z-face 的读取模式，而在**跨越 core 边界的读取**。

### 3.5 最终推荐方案：PML-only 紧凑索引 + 单 kernel launch

经过以上分析，最实际且符合审查意见约束的方案是：

**保持 `p_pml_ns` 为单 kernel launch，但改为只覆盖 PML shell 的线程，并通过预计算的索引映射表访问 full-domain 数组中的 vz/vx/vy。**

不对，审查意见明确说 "避免 compact shell 的整数除法/取模映射"。

**重新思考**。

审查意见说："把 PML face / edge / corner 的数据访问从 full-domain 思维改为 boundary-domain 思维"。

这意味着不是让 `p_pml_ns` 从 full-domain 数组中 "挑选" PML 点，而是让 PML 数据本身就存在于紧凑的 boundary-domain buffer 中。

具体方案：

#### Phase 1: `v_pml_ns` 双写

`v_pml_ns` 在计算完 vz/vx/vy 后，**同时写入两个位置**：
1. 原 full-domain 数组（供 `p_core_ns` 或其他用途使用）
2. 新的 **PML face buffer**

但 `p_core_ns` 不读 vz/vx/vy，所以 full-domain 的 vz/vx/vy 只被 `p_pml_ns` 读取。

如果 `p_pml_ns` 改为从 face buffer 读取，那么 `v_pml_ns` 可以**只写入 face buffer**，不需要写 full-domain 数组！

#### Phase 2: Face buffer layout

定义 6 个 face 的 compact buffer。关键问题是如何处理 stencil halo（edge/corner 的邻居跨越）。

**解决方案**：每个 face buffer 包含该 face 的 PML 点 + 3 点 halo。**Halo 数据通过 face 归属规则唯一确定**，不重复存储。

具体：
- Z0 buffer: z∈[0,19), x∈[0,408), y∈[0,408)。其中 z∈[0,16) 是 PML 点，z∈[16,19) 是 halo（来自 margin，属于 `v_pml_ns` 写入域）。
- Z1 buffer: z∈[96,119), ...。z∈[103,119) 是 PML，z∈[96,103) 是 halo。
- X0 buffer: z∈[16,103), x∈[0,19), y∈[0,408)。x∈[0,16) 是 PML，x∈[16,19) 是 halo。
- X1 buffer: z∈[16,103), x∈[389,408), ...
- Y0 buffer: z∈[16,103), x∈[16,392), y∈[0,19)
- Y1 buffer: z∈[16,103), x∈[16,392), y∈[389,408)

注意 Z0 的 halo（z∈[16,19)）和 X0 的 PML（z∈[16,103), x∈[0,16)）有重叠。例如点 (z=17,x=10,y=200)：
- 在 Z0 halo 内（因为 z∈[16,19)）
- 在 X0 PML 内（因为 x∈[0,16)）

这个点会被存储在两个 buffer 中。这是**数据重复**。

数据重复的问题：
- `v_pml_ns` 需要写入两个位置（Z0 halo 和 X0 PML）
- 但数据量很小（halo 只有 3 层），且 `v_pml_ns` 本来就是 full-domain launch，可以条件写入

更优雅的方案：**6 个 face buffer 的 PML 部分不重叠，但 halo 可以重叠。**

或者：**将 PML shell 分为 3 组（Z-faces, X-faces, Y-faces），每组有自己的 buffer，组内包含所有需要的 halo。**

- Z-face group buffer: z∈[0,19)∪[96,119), x∈[0,408), y∈[0,408)
  - 包含 Z0 PML + Z0 halo + Z1 PML + Z1 halo
  - 这个 buffer 覆盖所有 z 方向需要的数据
  - `p_pml_ns` 中所有需要 vz 的差分都可以从这个 buffer 读取
- X-face group buffer: z∈[16,103), x∈[0,19)∪[389,408), y∈[0,408)
  - 包含 X0 PML + X0 halo + X1 PML + X1 halo
  - `p_pml_ns` 中所有需要 vx 的差分都可以从这个 buffer 读取
- Y-face group buffer: z∈[16,103), x∈[16,392), y∈[0,19)∪[389,408)
  - 包含 Y0 PML + Y0 halo + Y1 PML + Y1 halo
  - `p_pml_ns` 中所有需要 vy 的差分都可以从这个 buffer 读取

但 `p_pml_ns` 处理一个点时需要同时读取 vz、vx、vy。如果这 3 个数组分别存在 3 个 group buffer 中，`p_pml_ns` 需要从 3 个不同 buffer 读取。

这没问题！当前 `p_pml_ns` 也从 3 个不同数组（vz, vx, vy）读取。

### 3.6 Group Buffer 详细设计

#### Z-face group buffer (for vz)

逻辑形状：(nz_pml_z, nx, ny) = (19+23=42? 不对)

Z0: z∈[0,19), Z1: z∈[96,119)。总 z 长度 = 19 + 23 = 42。
但中间 z∈[19,96) 不存（这是 core）。

存储布局：可以存为两个 slab（Z0 和 Z1），或一个紧凑数组带索引映射。

如果存为两个独立数组：
- `vz_z0`: shape (19, 408, 408)，layout (y, x, z) 连续或 (z, x, y) 连续
- `vz_z1`: shape (23, 408, 408)

`p_pml_ns` 读取 vz 时：
- 如果点在 Z0（z<16），从 `vz_z0` 读取，索引 `z`
- 如果点在 Z1（z≥103），从 `vz_z1` 读取，索引 `z-96`
- 如果点在 X-face 或 Y-face（z∈[16,103)），需要 vz[z-4:z+3]。由于 z∈[16,103)，vz 邻居也在 [12,106)。这些值：
  - z-4 到 z+3 可能跨越 Z0-halo、core、Z1-halo 边界
  - 但 X-face 和 Y-face 的点需要的 vz 值**不在 Z0/Z1 PML 内**，它们在 core 或 margin 内

等等，这里又遇到问题了。X-face 点（如 z=50, x=10, y=200）需要 vz[46:53]。这些值在 core 内！

但前面验证了 `v_pml_ns` 写入了这些值。如果 Z-face group buffer 只包含 Z0/Z1 的数据，那 X-face 点需要的 vz 值不在 Z-face buffer 中。

**结论**：`p_pml_ns` **所有**点需要的 vz 值分布在**整个** z 轴上（因为每个点都需要 z-4 到 z+3 的 vz）。如果 face 只存 PML 区域的 vz，那 core 区域的 vz 仍然需要。

但 `v_pml_ns` 只在非 core-inner 区域写入 vz。对于 core-inner 区域（z∈[19,99), x∈[16,392), y∈[16,392)），vz 不写入。

X-face 点（x<16）在 core-z（z∈[19,99)）和 core-y（y∈[16,392)）内时，需要 vz[z-4:z+3]。由于 x<16，`need_vz=true`，vz 被写入。所以这些 vz 值存在 full-domain 数组中。

如果要把这些 vz 值放到 face buffer 中，X-face buffer 需要包含 z∈[0,119) 的完整范围（因为 X-face 的点分布在所有 z 上）。

**这意味着按 face 分 buffer 对 vz 没有帮助**，因为 vz 是 z 方向差分，任何 PML 点都需要完整 z 线的 vz 值。

类似地：
- vx 是 x 方向差分，任何 PML 点都需要完整 x 线的 vx 值
- vy 是 y 方向差分，任何 PML 点都需要完整 y 线的 vy 值

### 3.7 根本性结论

由于 `p_pml_ns` 的 stencil 是**方向性差分**（vz 沿 z，vx 沿 x，vy 沿 y），每个 PML 点需要的一维邻居线**必然跨越 face 边界**。

例如：
- X-face 点需要 vx 的 x 邻居，这些邻居在 x 方向跨越 X0 到 core 到 X1
- 如果 vx 只存 X-face 的紧凑 buffer，X-face 点需要的 core 内 vx 值就不在 buffer 中

这意味着：**无法为每个 face 单独分配只包含该 face 数据的 buffer，同时满足 stencil 邻居需求，除非 buffer 包含整个维度线**。

### 3.8 修正方案：Directional Slice Buffers

不是按 3D face 分 buffer，而是按**方向性一维 slice** 分 buffer：

- **vz_full**: 保持原样（或不变）。因为所有 PML 点都需要任意 (x,y) 位置的完整 z 线 vz 值。vz 的 full-domain 布局已经是 z-fastest，访问 coalesced。
- **vx_pml_slab**: 只包含 x∈[0,16)∪[392,408) 的 vx 值，但**包含所有 z 和 y**。因为 X-face 点需要的 vx 邻居在 x 方向跨越 PML 和 core，但 vx 本身只在 X-face 需要（因为 core 内不更新 vx，但等等...）

不对，core 内的 vx 是否被 `v_pml_ns` 写入？
- `need_vx = false` 当 z∈[16,103) ∧ x∈[19,388) ∧ y∈[16,392)
- 所以 core 内 x∈[19,388) 的 vx 不写入
- 但 x=16,17,18 的 vx 写入（因为 x<19）
- x=388,389,390,391 的 vx 也写入（因为 x≥388 不在 [19,388) 内）

X-face 点 x=15 需要 vx[11:18]。其中 vx[16:18] 在 margin 内（x∈[16,19)），被写入。vx[11:15] 在 X0 PML 内。所有值都在 PML 或 margin 内，不需要 core 内的 vx！

验证：X-face 点 x 最大为 15（X0）或最小为 392（X1）。
- x=15 需要 [11,18]。18 < 19，所以完全在 PML+margin 内。
- x=392 需要 [388,395]。388 ≥ 388，不在 [19,388) 内。所以完全在 PML+margin 内。

**关键发现**：对于 X-face 的 PML 点，需要的 vx 邻居范围 `[x-4,x+3]` 完全落在 PML+margin 内（即不在 core-inner 的 x 范围内）！

类似地：
- Y-face 点需要的 vy 邻居完全在 PML+margin 内
- Z-face 点需要的 vz 邻居完全在 PML+margin 内

**但**：
- Z-face 点需要 vx 和 vy 吗？
  - `p_pml_ns` 对 PML shell 中所有点都计算 c1,c2,c3，然后应用 PML 条件。
  - 实际上看代码：c1,c2,c3 对所有 PML 点都计算，然后根据 z,x,y 位置应用不同的 PML 系数。
  - 所以 Z-face 点也需要 vx 和 vy！
  - Z-face 点（如 z=10, x=200, y=200）需要 vx[x-4:x+3] = [196:203]。其中 vx[196:203] 在 core-inner 内（x∈[19,388)）！
  - 而且 `v_pml_ns` 对 x=200, z=10, y=200 的 `need_vx`：
    - z=10∈[16,103)? 10<16，所以条件不满足，`need_vx=true`
  - 所以 vx[196:203] 被 `v_pml_ns` 写入。

这意味着 Z-face 点需要的 vx/vy 值在 core-inner 内，但 `v_pml_ns` 确实写入了它们（因为 Z-face 点的 z 或 y 不在 core 内，导致 need_vx/need_vy 为 true）。

但如果要 compact vx buffer，我们需要存哪些 x 位置的值？
- Z-face 点分布在所有 x∈[0,408) 和 y∈[0,408)
- 需要的 vx 范围覆盖所有 x
- 所以 vx 的 compact buffer 仍然需要覆盖整个 x 维度

**这彻底否定了 face buffer 方案对 vx/vy/vz 任何一个数组的可行性**。因为：
- 每个 PML 点需要 3 个方向的邻居
- 每个方向跨越的维度范围覆盖整个 domain
- 任何 compact buffer 要么需要包含整个维度（失去 compact 意义），要么需要多 buffer 查找（增加复杂度）

### 3.9 回归审查意见的真正意图

如果 face buffer 不可行，那审查意见说的 "boundary-domain 思维" 是什么意思？

重新读审查意见：
> 把 PML face / edge / corner 的数据访问从 full-domain 思维改为 boundary-domain 思维
> 避免 compact shell 的整数除法/取模映射
> 避免简单拆成过多 kernel launch

也许 "boundary-domain 思维" 不是指物理上分离 buffer，而是指**kernel 的线程映射和循环结构**。

当前 `p_pml_ns`：
```cpp
int gtid1 = blockIdx.x * blockDim.x + threadIdx.x; // z
int gtid2 = blockIdx.y * blockDim.y + threadIdx.y; // x
int gtid3 = blockIdx.z * blockDim.z + threadIdx.z; // y
if (core) return;
// 处理 PML
```

这导致大量线程空跑。改为 boundary-domain 思维：
- 只启动 PML shell 的线程数
- 每个线程通过 compact 1D 索引映射到 PML shell 中的一个点

但审查意见说 "避免 compact shell 的整数除法/取模映射"，因为之前的 compact shell 尝试失败了。

**也许方案是：按 face 分多个 kernel launch，每个 kernel 只处理一个 face，线程映射与该 face 的自然坐标对齐。**

例如：
- `p_pml_z0_ns`: grid 覆盖 Z0 face（z∈[0,16), x∈[0,408), y∈[0,408)），线程 (z,x,y)
- `p_pml_z1_ns`: 覆盖 Z1 face
- `p_pml_x0_ns`: 覆盖 X0 face
- ...

共 6 个 kernel launch。每个 kernel 内部不需要 `if (core) return`，所有线程都有效。

但这正是审查意见说的 "避免简单拆成过多 kernel launch" 的反面。

### 3.10 折中方案：3 kernel launch（Z/X/Y group）

将 6 个 face 合并为 3 个 group：
1. **Z-group kernel**: 处理 Z0 + Z1。线程映射为 (z, x, y)，但只覆盖两个 z-slab。
   - 需要读取 vz（z 邻居在 slab 内或跨越 slab 边界）
   - 需要读取 vx/vy（x/y 邻居可能深入 core）
2. **X-group kernel**: 处理 X0 + X1。线程映射为 (x, y, z)。
3. **Y-group kernel**: 处理 Y0 + Y1。线程映射为 (y, z, x)。

每个 kernel 可以有自己的线程布局优化，但数据仍从 full-domain 数组读取。

**这没有解决数据访问问题**，只是减少了空线程。

### 3.11 数据访问问题的量化

当前 `p_pml_ns` 读取的数据量（每 time step）：
- PML shell 点数：5.8M
- 每个点读取 24 float (vz×8 + vx×8 + vy×8) = 96 bytes
- 总读取：5.8M × 96 B = 556.8 MB

如果保持 full-domain 数组但只启动 PML shell 线程：
- 线程数从 19.8M 减少到 5.8M（减少 70% launch overhead）
- 数据读取量不变（556.8 MB）

556.8 MB / 1501 steps = 835 GB 总计。RTX 5090 内存带宽 ~1 TB/s，但这是理论峰值，实际有效带宽受 cache 影响。

如果使用 L2 cache：vz/vx/vy 在 `v_pml_ns` 写入后立刻被 `p_pml_ns` 读取，如果 grid 布局相似，cache hit 率可能很高。

### 3.12 新的可能方向：内存布局重组（transpose）

当前布局：z-fastest, x-middle, y-slowest (zyx)。

`p_pml_ns` 的访问模式：
- vz 沿 z 连续（完美）
- vx 沿 x 连续，步长 = stride2（不连续，但 warp 内相邻线程 z 不同，x 相同，所以 vx 访问跨 warp 不 coalesced）
- vy 沿 y 连续，步长 = stride3（最差）

如果改为 **x-fastest, z-middle, y-slowest** 或 **y-fastest, z-middle, x-slowest**？

但 `p_core_ns` 使用 14th-order z-diff，z-fastest 对 `p_core_ns` 最有利。改变布局会伤害 `p_core_ns`（当前只占 21%，但已经优化过）。

或者，为 PML 阶段单独 transpose 数据？这增加了额外的 memory traffic，不划算。

## 4. 对审查问题的正式回答

### Q1: 当前 PML shell 中哪些 vx/vy/vz 元素真的会被 `p_pml_ns` 读取？

对于 PML shell 中的点 `(z,x,y)`：
- **vz**: 读取 `vz[z-4:z+3]` 在固定 `(x,y)` 处，共 8 个元素
- **vx**: 读取 `vx[x-4:x+3]` 在固定 `(z,y)` 处，共 8 个元素
- **vy**: 读取 `vy[y-4:y+3]` 在固定 `(z,x)` 处，共 8 个元素

**全部 24 个元素都由 `v_pml_ns` 在上一步写入**，无未定义值。PML shell 中约 5.8M 点，每步总读取量 ≈ 5.8M × 24 × 4 B = 557 MB。

### Q2: 这些元素是否能用 face-major contiguous buffer 表示？

**严格意义上的纯 face buffer（不含 core 数据）不可行**。

原因：每个 PML 点需要的 vx/vy/vz 邻居线跨越 face 边界。例如：
- Z-face 点需要 vx[x-4:x+3]，其中 x 可深入 core-inner（x∈[19,388)）。这些 vx 值在 core-inner 内，但 `v_pml_ns` 写入了它们（因为 Z-face 点的 z 不在 core-z 内，need_vx=true）。
- 如果 vx face buffer 只存 X-face 数据，Z-face 点需要的 core-inner vx 值不在 buffer 中。

**可行的变体**：
1. **Slice buffer**: 每个方向数组存为 "PML + margin" 的紧凑 slab。例如 vx 存所有 x∈[0,19)∪[389,408) 的值，但保留完整的 z 和 y。这没有减少存储量（因为 z×y 跨度完整）。
2. **3D 子域 buffer**: 存 PML shell 的 3D 膨胀版本（含 4 点 halo）。但这本质上与 full-domain 数组大小相近（PML shell + halo ≈ full domain）。

**结论**：在 8th-order stencil 下，PML 点与 core 存在 4 点耦合深度，无法将 PML 数据完全剥离为独立的紧凑 buffer 而不引入 halo 重叠或 core 数据冗余。

### Q3: face / edge / corner 如何避免重复写或漏写？

如果采用 6-face kernel 方案（每个 face 一个 kernel）：
- **互斥划分**（如第 2.1 节定义）保证每个点只被一个 kernel 处理，无重复无遗漏
- 但 edge/corner 点的 stencil 邻居可能跨越 face 边界，需要每个 kernel 能读取邻居 face 的数据
- 这要求 face buffer 之间有 halo 重叠，或回退到 full-domain 数组读取

如果采用单 kernel + PML-only grid 方案：
- 通过紧凑 1D 索引映射 PML shell 中的点，但审查意见明确反对 compact shell 的整数除法/取模映射
- 或通过 3D grid 覆盖 PML shell 的 bounding box（非矩形，仍有空线程）

### Q4: 新 layout 是否会引入更多 integer mapping？

- **6-face kernel 方案**：每个 kernel 内部使用自然的 3D 索引，无额外 mapping。但线程到 global 数组的地址计算与当前相同（仍需 `base = (y+radius)*stride3 + (x+radius)*stride2 + (z+radius)`）。
- **Compact 1D 方案**：需要除法/取模或查表映射，审查意见明确禁止。
- **Slice buffer 方案**：需要 slab offset 加法（如 `x < npml ? x : x - core_nx + npml`），比 1D compact 简单，但仍引入分支。

### Q5: 新 layout 会增加几个 kernel launch？

- **6-face kernel 方案**：`p_pml_ns` 从 1 个拆成 6 个，每步增加 5 个 launch。`v_pml_ns` 保持 1 个。
- **3-group kernel 方案**：`p_pml_ns` 拆成 3 个（Z-group, X-group, Y-group），每步增加 2 个 launch。
- **任何方案都不增加 `v_pml_ns` 的 launch 数**（除非增加 gather kernel）。

### Q6: 每个 time step 预计减少多少 global load/store？

**悲观分析**：
- 如果仅拆分 kernel 但不改变数据布局，global load/store 量**不变**（仍是 557 MB 读取 + p0/cw2 读写）。
- 拆分 kernel 的好处是减少 launch overhead（19.8M → 5.8M 线程），但 RTX 5090 的 launch overhead 在 microsecond 级别，1501 步的总节省可能只有几毫秒。

**乐观分析（需数据 transpose）**：
- 如果能为每个 face group 做数据 transpose，使线程访问完全 coalesced，可减少 L2 miss。
- 但 transpose 本身需要额外的 global memory traffic（读取 full-domain + 写入 compact buffer）。
- 净收益取决于 L2 miss 的减少量是否超过 transpose 成本。

**量化估算**（假设 L2 miss 率从 30% 降到 10%）：
- 当前有效带宽 = 557 MB × 30% miss × 1501 ≈ 250 GB 实际从 DRAM 读取
- 优化后 = 557 MB × 10% miss × 1501 ≈ 84 GB
- 节省 ≈ 166 GB，在 1 TB/s 带宽下约 0.17s
- 但 transpose 成本：读取 3 个 full-domain 数组（vz/vx/vy）≈ 19.8M × 3 × 4 B = 237 MB/步，1501 步 = 356 GB
- **净收益为负**

### Q7: 如何建立 one-step PML debug comparison？

建立单步 PML debug 流程：

1. **创建 `debug_1step` case**: nt=1 或 nt=2，只执行一个 time step
2. **Dump hook**: 在 `rem_fd.cu` 中 `p_pml_ns` 之后插入 `cudaMemcpy`，将 `p0` 的 PML shell 区域拷贝到 host
3. **逐点比较**: 写 Python 脚本对比 baseline 和 candidate 的 PML shell 区域：
   - 对每个 face，检查每个点是否被更新 exactly once
   - 对比第一个不匹配点的 `(z,x,y)`、baseline 值、candidate 值、差值
4. **中间变量 dump**: 可选地 dump `vz/vx/vy` 在 PML 点的值，验证 `v_pml_ns` 的输出是否一致
5. **自动化**: 将上述流程集成到 `tools/compare_outputs.py` 中，增加 `--pml-only` 和 `--max-err-loc` 选项

## 5. 设计决策

### 5.1 核心结论

**在当前 8th-order stencil 和跨-face 邻居依赖下，纯粹的 face-major compact buffer 不可行。** PML shell 与 core 存在 4 点深度的数据耦合，任何剥离 PML 数据的尝试都需要携带相当于 full-domain 的 halo，失去 compact 意义。

### 5.2 可行的次优方案

如果仍需优化 `p_pml_ns`，以下方向有实现价值：

#### 方案 B1: PML-only grid（不推荐，但可实验）
- 保持 `p_pml_ns` 单 kernel，但 grid 只覆盖 PML shell
- 使用预计算的 **索引映射表**（大小 5.8M ints）将 compact thread ID 映射到 `(z,x,y)`
- 避免运行时除法/取模，用查表替代
- **风险**：5.8M × 4 B = 23 MB 映射表，读取映射表本身增加 memory traffic
- **收益**：减少约 70% 的线程 launch overhead

#### 方案 B2: 3-group kernel（中等复杂度）
- `p_pml_ns` 拆为 `p_pml_z_ns`（Z0+Z1）、`p_pml_x_ns`（X0+X1）、`p_pml_y_ns`（Y0+Y1）
- 每个 kernel 只处理一组 face，所有线程有效
- 使用不同的 block 尺寸优化每组 face 的内存访问模式
- **收益**：减少空线程，可能改善 cache locality（Z-kernel 的 vz 访问更集中）
- **风险**：增加 2 个 kernel launch/步，launch overhead 可能抵消收益

#### 方案 B3: `v_pml_ns` + `p_pml_ns` 局部 fusion（高风险）
- 在一个 kernel 内先计算 velocity 再计算 pressure，但只对 PML shell 的一个子集
- 消除 vz/vx/vy 的 global write + global read
- **风险**：跨 block 依赖。一个 block 计算的 velocity 邻居需要被相邻 block 的 pressure 使用。同一 kernel 内如果不做全局同步，无法满足。
- **除非**：用整个 PML shell 作为一个 huge block（不现实），或在一个 block 内同时计算 velocity 和 pressure（block 大小限制）

#### 方案 B4: 重新审视 `p_pml_ns` 的 branch divergence（推荐）
- 当前 `p_pml_ns` 的 PML 条件（`if (gtid1<npml)`, `if (gtid1>=n1-npml)` 等）导致每个 warp 内可能有多种分支
- 但 PML shell 的点在 face 上有空间连续性，相同 face 的点执行相同分支
- 如果按 face 排序线程，可减少 branch divergence
- 这需要 compact mapping，但收益可能只是 branch divergence 的减少

### 5.3 下一步建议

鉴于 face buffer 的不可行性，建议：

1. **先建立 one-step PML debug harness**（回答 Q7）。这是审查意见明确要求的，且对后续任何 PML 优化都必要。
2. **用 debug harness 验证 `p_pml_ns` 的每个 PML 条件分支的输出**，确认 branch divergence 的实际成本。
3. **在 debug harness 基础上，实验方案 B2（3-group kernel）**，因为其实现复杂度中等，且可直接验证 correctness。
4. **不实验方案 B1（compact mapping）**，因为审查意见明确反对整数除法/取模映射，且映射表的 memory traffic 可能抵消收益。

## 6. 风险评估

| 方案 | 复杂度 | 预计收益 | 主要风险 |
|---|---|---|---|
| Face buffer (本报告分析) | 高 | 理论低 | Stencil halo 需要 full-domain 数据，不可行 |
| 3-group kernel (B2) | 中 | 低~中 | Launch overhead 增加，代码维护成本 |
| PML-only grid (B1) | 中 | 低 | 映射表 traffic，审查意见反对 |
| v+p fusion (B3) | 极高 | 高 | 跨 block 依赖，几乎不可行 |
| Branch优化 (B4) | 低 | 低 | 收益有限 |

## 7. 结论

PML layout 的物理剥离在当前 stencil 深度下不可行。`p_pml_ns` 的数据访问本质上是 full-domain 的（每个 PML 点需要跨越 core 边界的 1D 邻居线）。任何 compact buffer 方案要么需要包含相当于 full-domain 的 halo，要么需要多 buffer 查找，净收益为负或不可维护。

建议优先建立 **one-step PML debug harness**，然后实验 **3-group kernel 拆分**（Z/X/Y face group），以消除空线程并可能改善 cache locality。

---
