下面以 NVIDIA CUDA 为背景。最核心的层级是：

```
一次 Kernel 启动
└─ Grid（网格）：用于分配
   ├─ Block 0（线程块）
   │  ├─ Thread 0
   │  ├─ Thread 1
   │  └─ ...
   ├─ Block 1
   └─ ...
```

其中：

- **Kernel**：在 GPU 上执行的函数；

- **Thread**：执行一份 Kernel 代码的最小逻辑执行单元；

- **Block**：一组可以协作的线程；

  GPU 先以 **Block 为单位**，把工作分配到不同 SM；

  Block 进入某个 SM 后，SM 再以 **Warp 为单位**选择下一批要执行的线程。

- **Grid**：一次 Kernel 启动所产生的全部线程块。

  Grid 表示一次 kernel launch 的全部工作，并提供 Block 的全局编号：

  ```
  Grid
  ├─ Block 0：处理数据 0～255
  ├─ Block 1：处理数据 256～511
  ├─ Block 2：处理数据 512～767
  └─ ...
  ```

  Grid 解决的是：

  - 整个任务有多大；
  - 需要多少 Blocks；
  - Blocks 如何覆盖全部数据；
  - 这些 Blocks 属于哪一次 kernel 执行。

## 1. CUDA Thread 是什么？

CUDA Thread 是一个逻辑线程。启动 kernel 时，GPU 会创建大量线程，每个线程运行同一段 kernel 代码，但处理不同的数据。

例如向量加法：

\[ C_i=A_i+B_i \]

CUDA kernel 可以写成：

```
__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

所有线程执行同一段代码，区别在于每个线程计算出的 `i` 不同：

```
Thread 0 → C[0] = A[0] + B[0]
Thread 1 → C[1] = A[1] + B[1]
Thread 2 → C[2] = A[2] + B[2]
...
```

GPU 可以用成千上万个线程覆盖大量数据元素。

### Thread 有自己的状态

每个线程有自己的：

- `threadIdx`；
- 局部变量；
- 程序计数器；
- 寄存器状态。

例如：

```
float x = A[i];
float y = B[i];
float z = x + y;
```

这里的 `x`、`y`、`z` 对每个线程都是独立的，通常尽量放在寄存器中。

但 CUDA Thread 不是 CPU 那种重量级操作系统线程。它由 GPU 硬件调度，创建和切换成本小得多。

## 2. Thread Block 是什么？

Thread Block 是一组共同执行 kernel、并且可以紧密协作的线程。

例如：

```
Block 0
├─ Thread 0
├─ Thread 1
├─ Thread 2
└─ ... Thread 255
```

常见配置是每个 Block 包含 128、256 或 512 个线程。在现代 NVIDIA GPU 上，一个 Block 通常最多包含 1024 个线程，但具体限制取决于硬件。

线程块有三个重要特点：

1. 同一 Block 的线程会被安排在同一个 SM 上执行；
2. 同一 Block 的线程可以共享 Shared Memory；
3. 同一 Block 的线程可以使用 `__syncthreads()` 同步。

### Block 内的共享内存

```
__shared__ float buffer[256];
```

这块内存由同一 Block 中的线程共享：

```
Block 0
├─ Thread 0 ─┐
├─ Thread 1 ─┤
├─ Thread 2 ─┼→ Shared Memory of Block 0
└─ Thread 3 ─┘
```

另一个 Block 有自己独立的 Shared Memory：

```
Block 1 → Shared Memory of Block 1
```

Block 0 不能直接访问 Block 1 的 Shared Memory。

Shared Memory 位于 GPU 芯片上，延迟远低于显存，常用于：

- 数据复用；
- 矩阵分块；
- reduction；
- Attention 中的分块计算；
- 线程间交换中间结果。

## 3. Grid 是什么？

Grid 是一次 kernel launch 产生的全部 Blocks。

例如：

```
vector_add<<<4, 256>>>(A, B, C, 1000);
```

表示：

```
Grid 中有 4 个 Blocks
每个 Block 有 256 个 Threads

总线程数 = 4 × 256 = 1024
```

对应关系如下：

```
Block 0：全局线程 0～255
Block 1：全局线程 256～511
Block 2：全局线程 512～767
Block 3：全局线程 768～1023
```

代码中的：

```
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

包含：

- `blockIdx.x`：当前线程块在 Grid 中的编号；
- `blockDim.x`：每个 Block 的线程数量；
- `threadIdx.x`：当前线程在线程块内的编号。

例如 Block 2 的 Thread 10：

\[ i=2\times256+10=522 \]

因此它处理 `A[522]`、`B[522]` 和 `C[522]`。

## 4. Grid 和 Block 可以是多维的

CUDA 的 Grid 和 Block 可以是一维、二维或三维。

向量通常使用一维：

```
kernel<<<num_blocks, threads_per_block>>>();
```

图像或矩阵经常使用二维：

```
dim3 block(16, 16);
dim3 grid(
    (width  + 15) / 16,
    (height + 15) / 16
);

kernel<<<grid, block>>>();
```

线程定位：

```
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

可以想象成：

```
二维 Grid
┌────────┬────────┬────────┐
│Block0,0│Block1,0│Block2,0│
├────────┼────────┼────────┤
│Block0,1│Block1,1│Block2,1│
└────────┴────────┴────────┘
```

每个 Block 内部又是二维线程阵列：

```
16 × 16 = 256 threads
```

多维结构主要是为了让代码中的索引更自然。硬件最终仍然会把线程线性地组织和执行。

## 5. Warp：GPU 真正的执行单位

虽然程序员看到的是 Thread，但 NVIDIA GPU 实际以 **Warp** 为单位调度线程。

一个 Warp 通常包含：

\[ 32\ threads \]

例如一个 Block 有 256 个线程：

\[ 256/32=8\ warps \]

结构可以写成：

```
Block：256 Threads
├─ Warp 0：Thread 0～31
├─ Warp 1：Thread 32～63
├─ Warp 2：Thread 64～95
├─ ...
└─ Warp 7：Thread 224～255
```

同一个 Warp 中的 32 个线程通常在同一时刻执行同一条指令，但每个线程处理自己的数据。这叫作：

> SIMT：Single Instruction, Multiple Threads

例如：

```
C[i] = A[i] + B[i];
```

一个 Warp 中的 32 个线程一起执行加法，但分别计算：

```
Thread 0：A[0] + B[0]
Thread 1：A[1] + B[1]
...
Thread 31：A[31] + B[31]
```

## 6. 分支发散 Warp Divergence

如果同一个 Warp 中的线程执行不同分支：

```
if (i % 2 == 0) {
    do_A();
} else {
    do_B();
}
```

那么：

```
偶数线程执行 A
奇数线程执行 B
```

但一个 Warp 通常只能在某一时刻执行一条共同指令路径，因此硬件可能先：

```
只激活偶数线程，执行 A
```

再：

```
只激活奇数线程，执行 B
```

两条路径近似串行执行，这叫 **Warp Divergence**。

因此 GPU 代码通常希望同一个 Warp 中的线程尽量走相同控制路径。

不过分支发散只发生在同一个 Warp 内。不同 Warp 走不同分支通常没有同样的问题。

## 7. SM 是什么？

SM 是 Streaming Multiprocessor，可以理解为 GPU 上执行 Thread Blocks 和 Warps 的主要硬件单元。

GPU 的物理结构可以简化为：

```
GPU
├─ SM 0
├─ SM 1
├─ SM 2
├─ ...
└─ SM N
```

每个 SM 内包含：

- CUDA Cores；
- Warp Schedulers；
- Registers；
- Shared Memory/L1 Cache；
- Tensor Cores；
- Load/Store Units；
- Special Function Units。

CUDA 的软件结构会被映射到 SM：

```
Grid
├─ Block 0 → SM 0
├─ Block 1 → SM 1
├─ Block 2 → SM 2
├─ Block 3 → SM 0
└─ ...
```

需要特别注意：

> 一个 Block 只能在一个 SM 上执行，不能把同一个 Block 的线程拆到多个 SM。

但一个 SM 可以同时驻留多个 Blocks，只要寄存器、Shared Memory 和线程槽位等资源足够。

## 8. Block 数量可以远大于 SM 数量

假设：

```
Grid 有 1000 个 Blocks
GPU 有 100 个 SM
```

GPU 不需要同时执行全部 1000 个 Blocks。它会分批调度：

```
第一批 Blocks → 各个 SM
完成后
第二批 Blocks → 各个 SM
...
```

因此 CUDA 程序不需要提前知道 GPU 有多少 SM。

这也是 CUDA 扩展性的基础：

```
较小 GPU：分更多批执行
较大 GPU：同时执行更多 Blocks
```

同一段 kernel 可以在不同规模的 GPU 上运行。

## 9. 一个 SM 为什么同时保存多个 Warps？

GPU 经常需要等待显存访问。例如某个 Warp 发出显存读取：

```
Warp 0：等待 HBM 数据
```

SM 不会一直空等，而是切换到其他已就绪 Warp：

```
Warp 0：等待内存
Warp 1：执行计算
Warp 2：执行计算
Warp 3：发起内存读取
```

这种方法通过大量线程隐藏内存延迟：

> GPU 不一定让单个线程跑得非常快，而是准备大量线程；一部分线程等待时，执行另一部分线程。

这也是 GPU 需要大量 Thread 和 Warp 的重要原因。

## 10. Occupancy 是什么？

Occupancy 通常描述：

\[ \text{SM 上实际驻留的活跃 Warps} \div \text{SM 支持的最大活跃 Warps} \]

例如一个 SM 最多支持 64 个活跃 Warp，现在驻留了 48 个：

\[ Occupancy=48/64=75\% \]

影响 Occupancy 的因素包括：

- 每个 Block 的线程数；
- 每个线程使用多少寄存器；
- 每个 Block 使用多少 Shared Memory；
- GPU 架构的资源限制。

例如一个 Block 使用大量 Shared Memory：

```
SM Shared Memory：128 KB
每个 Block 使用：64 KB
```

那么一个 SM 最多只能同时放两个这样的 Blocks，即使线程槽位还有剩余。

但 Occupancy 并非越高性能就一定越好。只要有足够 Warps 隐藏延迟，继续提高 Occupancy 未必有收益；有时减少寄存器使用以追求高 Occupancy，反而会导致寄存器溢出到显存。

## 11. 线程、Warp、Block、SM 的关系

这是最重要的关系图：

```
软件视角                              硬件视角

Kernel Launch
└─ Grid
   ├─ Block 0 ─────────────────────→ 某个 SM
   │  ├─ Warp 0：Threads 0～31  ──→ Warp Scheduler
   │  ├─ Warp 1：Threads 32～63 ──→ Warp Scheduler
   │  └─ ...
   │
   ├─ Block 1 ─────────────────────→ 某个 SM
   └─ ...

GPU
├─ SM 0
├─ SM 1
└─ ...
```

简单记忆：

- **Thread**：处理一个或少量数据元素；
- **Warp**：硬件实际调度的 32 个线程；
- **Block**：可以共享内存、同步的一组线程；
- **Grid**：一次 kernel 启动中的全部 Blocks；
- **SM**：执行 Blocks 和 Warps 的物理硬件单元。

## 12. 为什么需要 Thread Block，而不是一个巨大线程组？

主要有三个原因。

第一，GPU 的资源是分布在多个 SM 上的。Block 提供了可以独立调度的工作单元：

```
Block 0 → SM 0
Block 1 → SM 1
Block 2 → SM 2
```

第二，Block 限制同步范围。只允许 Block 内同步，可以避免全 GPU 同步的巨大开销。

第三，Block 提供局部数据协作。一个 Block 中的线程可以把某个数据块加载到 Shared Memory，再重复使用。

## 13. 矩阵乘法中的 Thread Block

计算：

\[ C=A\times B \]

朴素方法可以让每个线程计算 \(C\) 的一个元素：

```
Thread(row, col) → C[row, col]
```

但每个线程都直接从显存读取 A 和 B，会产生大量重复访问。

优化方式是分块：

```
C 的一个 Tile
       ↑
一个 Thread Block 负责
```

Block 内线程协作：

1. 从 HBM 加载 A 的一个 Tile；
2. 从 HBM 加载 B 的一个 Tile；
3. 把 Tile 放入 Shared Memory；
4. `__syncthreads()`；
5. 所有线程重复使用 Shared Memory 中的数据；
6. 计算对应的 C Tile。

示意：

```
A Tile ─┐
        ├─→ Shared Memory → Block 内线程计算 → C Tile
B Tile ─┘
```

这种分块复用是 GPU 高性能矩阵乘法的基础。实际 Tensor Core GEMM 会采用更复杂的层次：

```
Grid Tile
└─ Thread Block Tile
   └─ Warp Tile
      └─ Tensor Core MMA Tile
```

## 14. 内存层级与线程层级

不同线程层级对应不同内存作用域：

| 内存              | 通常由谁访问 | 特点                       |
| ----------------- | ------------ | -------------------------- |
| Registers         | 单个 Thread  | 最快，容量最小             |
| Local Memory      | 单个 Thread  | 名为 local，但可能位于显存 |
| Shared Memory     | 同一 Block   | 快，可用于线程协作         |
| Global Memory/HBM | 整个 Grid    | 容量大，延迟高             |
| Constant Memory   | 整个 Grid    | 适合只读广播               |
| L1/L2 Cache       | 硬件管理     | 缓存全局内存访问           |

其中最容易误解的是：

> Local Memory 并不一定是芯片上的高速本地内存。

它是线程私有的地址空间，但寄存器不足时，数据可能被放到显存中，形成 register spilling。

## 15. 同步范围

Block 内可以直接同步：

```
__syncthreads();
```

它表示同一 Block 中所有线程到达这里后才能继续。

但普通 kernel 内通常不能简单地让整个 Grid 的所有 Blocks 同步，因为不同 Blocks 可能尚未被调度。

常用的全局同步方法是：

```
Kernel A 完成
     ↓
Kernel B 启动
```

Kernel 边界天然形成全局同步点。

还有 cooperative groups 等高级机制，但需要满足额外的调度条件。

## 16. 与 CUDA Graph 中 launch parameters 的关系

前面提到 CUDA Graph 会捕获 kernel launch parameters，例如：

```
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args);
```

其中：

- `gridDim`：Grid 包含多少 Blocks；
- `blockDim`：每个 Block 有多少 Threads；
- `sharedMemBytes`：每个 Block 需要多少动态 Shared Memory；
- `args`：输入、输出指针和其他参数。

例如：

```
kernel<<<1024, 256>>>(input, output);
```

表示：

```
Grid：1024 Blocks
每个 Block：256 Threads
总逻辑线程：262144
```

CUDA Graph 捕获后，这些配置通常不会在每次 replay 时任意变化。这就是论文所说的 `freezing launch parameters`。

## 最终记忆框架

可以把 GPU 想成一家大型工厂：

```
Grid       = 一整个生产任务
Block      = 一个工作小组
Thread     = 小组内的一名工人
Warp       = 统一接受指令的 32 名工人
SM         = 容纳若干工作小组的生产车间
Shared Mem = 小组共享的工作台
Registers  = 每名工人自己的工具
HBM        = 全厂共享的大仓库
```

但这个比喻有一个关键限制：同一 Warp 的“工人”并不是完全独立行动，而是通常接受同一条指令、处理不同数据。这正是 GPU 能够高效执行大规模规则并行计算的基础。