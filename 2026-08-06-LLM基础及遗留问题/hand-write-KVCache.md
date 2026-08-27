## 1. KV Cache 手算公式

适用于普通 decoder-only Transformer：

\[ M_{KV}=2\times L\times H_{kv}\times D_{head}\times S\times \sum_i T_i \]

其中：

- `2`：同时保存 Key 和 Value
- `L`：Transformer 层数
- `H_kv`：KV Head 数量
- `D_head`：每个 Head 的维度
- `S`：每个 KV 元素的字节数
  - FP32：4
  - FP16/BF16：2
  - FP8/INT8：1
- `T_i`：第 i 个请求当前缓存的 token 数
  - `输入 prompt token + 已生成 token`
- 如果所有请求长度相同，则：

\[ \sum_iT_i=B\times T \]

其中 `B` 是并发序列数，不一定等于传统训练中的 batch size。

最后换算：

\[ M_{KV}(GiB)=\frac{M_{KV}(bytes)}{2^{30}} \]

需要从模型的 `config.json` 中找到：

```
num_hidden_layers
num_attention_heads
num_key_value_heads
hidden_size
head_dim
```

如果没有单独的 `head_dim`：

\[ D_{head}=\frac{hidden\_size}{num\_attention\_heads} \]

注意缓存使用的是 `num_key_value_heads`，不是简单使用 `num_attention_heads`。

| 注意力类型 | KV Head 数 |
| ---------- | ---------- |
| MHA        | `Hkv = Hq` |
| GQA        | `Hkv < Hq` |
| MQA        | `Hkv = 1`  |

因此 GQA/MQA 能大幅降低 KV Cache。

## 2. Llama 3.1 8B 手算示例

根据 [Meta Llama 3.1 配置](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json)：

```
层数 L                 = 32
Query Heads            = 32
KV Heads Hkv           = 8
hidden_size            = 4096
head_dim               = 4096 / 32 = 128
KV dtype               = BF16，2 bytes
```

每个 token、每个请求的 KV Cache：

\[ 2\times32\times8\times128\times2 =131072\ bytes =128\ KiB \]

所以：

| 上下文长度     | 单请求 KV Cache |
| -------------- | --------------- |
| 4K（4096）     | 0.5 GiB         |
| 8K             | 1 GiB           |
| 32K            | 4 GiB           |
| 64K            | 8 GiB           |
| 128K（131072） | 16 GiB          |

如果同时跑 4 个同样长度的请求，再乘 4。例如 4 个 128K 请求就是约 `64 GiB` KV Cache。

## 3. 再看 Llama 3.1 70B

其主要 KV 参数为：

```
L = 80
Hkv = 8
head_dim = 128
KV dtype = BF16
```

每 token：

\[ 2\times80\times8\times128\times2 =327680\ bytes =320\ KiB \]

| 上下文长度 | 8B KV Cache | 70B KV Cache |
| ---------- | ----------- | ------------ |
| 8K         | 1 GiB       | 2.5 GiB      |
| 32K        | 4 GiB       | 10 GiB       |
| 128K       | 16 GiB      | 40 GiB       |

这说明 KV Cache 不完全由模型参数量决定，而主要由层数、KV Heads 和 Head 维度决定。

## 4. 与总显存的关系

推理显存账单大致是：

\[ M_{GPU}\approx M_{weights}+ M_{KV}+ M_{activation/workspace}+ M_{runtime} \]

其中：

- `weights`：模型权重
- `KV`：随 token 和并发线性增长
- `activation/workspace`：prefill 阶段通常更高
- `runtime`：CUDA 上下文、CUDA Graph、临时张量、分配器碎片等

权重的理论下限：

\[ M_{weights}=\frac{参数量\times权重位数}{8} \]

| 权重精度  | 每 10 亿参数理论占用 |
| --------- | -------------------- |
| FP32      | 3.73 GiB             |
| FP16/BF16 | 1.86 GiB             |
| INT8      | 0.93 GiB             |
| INT4      | 0.47 GiB             |

因此按模型名称中的参数量粗估：

| 模型 | BF16 权重    | INT4 理论下限 |
| ---- | ------------ | ------------- |
| 8B   | 约 14.9 GiB  | 约 3.7 GiB    |
| 70B  | 约 130.4 GiB | 约 32.6 GiB   |

量化模型还需要 scale、zero-point 等元数据，实际会比理论下限大。

一个非常容易算错的地方：

> AWQ、GPTQ、GGUF Q4 等“4-bit 模型”通常只量化权重，KV Cache 默认仍可能是 FP16/BF16。

只有明确启用了 FP8、INT8 或量化 KV Cache，才能把公式里的 `S=2` 改成 `S=1` 或更低。[Transformers 官方文档](https://huggingface.co/docs/transformers/v4.50.0/kv_cache)也将权重量化和 KV Cache 量化作为不同功能。

### 例：70B INT4、128K 上下文

粗略最低占用：

```
INT4 权重：约 32.6 GiB
BF16 KV：  约 40 GiB
合计：     约 72.6 GiB
```

这还没有包含量化元数据、激活、工作区和运行时开销。因此：

- 单张 H100 80GB：非常紧张，通常不能只凭 `72.6 < 80` 就判断能运行
- H200 141GB：空间明显更充裕
- 如果权重是 BF16：约 `130.4 + 40 = 170.4 GiB`，单张 H200 已经放不下

## 5. 反推最多能放多少 token

先算每 token 的 KV：

\[ S_{token}=2LH_{kv}D_{head}S \]

再估计可用于 KV 的显存：

\[ M_{availableKV} =M_{GPU}-M_{weights}-M_{other} \]

则总缓存 token 容量：

\[ N_{max}\approx \frac{M_{availableKV}}{S_{token}} \]

如果有 `C` 个等长并发请求：

\[ T_{max/request}\approx\frac{N_{max}}{C} \]

实际部署建议再保留约 10%～20% 余量，并根据推理框架实测修正。

## 6. 当前常见 GPU 显存容量

以下是截至 2026-08-14 的官方标称单 GPU 容量：

| 类型            | GPU                                                          | 单 GPU 显存         |
| --------------- | ------------------------------------------------------------ | ------------------- |
| 消费级          | [RTX 4090](https://www.nvidia.com/en-gb/geforce/graphics-cards/40-series/rtx-4090/) | 24 GB               |
| 消费级          | [RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) | 32 GB               |
| 工作站          | [RTX PRO 6000 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/) | 96 GB               |
| NVIDIA 数据中心 | [A100](https://www.nvidia.com/en-us/data-center/a100/)       | 40 / 80 GB          |
| NVIDIA 数据中心 | [H100](https://www.nvidia.com/en-us/data-center/h100/)       | 80 GB；NVL 94 GB/卡 |
| NVIDIA 数据中心 | [H200](https://www.nvidia.com/en-us/data-center/h200/)       | 141 GB              |
| NVIDIA 数据中心 | [B200](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html) | 180 GB              |
| NVIDIA 数据中心 | [B300](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html) | 288 GB              |
| AMD 数据中心    | [MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) | 192 GB              |
| AMD 数据中心    | [MI325X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html) | 256 GB              |
| AMD 数据中心    | [MI350X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html) | 288 GB              |
| AMD 数据中心    | [MI455X](https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html) | 432 GB，新发布      |

多卡显存不能自动看作一块连续的大显存。只有推理框架正确采用张量并行、流水线并行或 KV 分片时，才可以按多卡容量做整体预算；而且 GQA 模型在张量并行度大于 KV Head 数时，可能出现 KV 复制，不能简单除以 GPU 数量。

## 7. 实测时的三个差异

- Dynamic Cache：随 token 逐渐增长。
- Static Cache：可能一开始就按最大上下文预留。
- Paged KV Cache：服务框架通常先预留一个 KV 内存池，再按 block 分配，因此 `nvidia-smi` 显示的已占用显存不一定等于当前实际有效 KV 数据。[分页缓存说明](https://huggingface.co/docs/transformers/en/continuous_batching_architecture)

如果你指的是查看自己机器的当前显存，可以运行：

```
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
```

把具体模型名称或 `config.json`、权重精度、KV 精度、最大上下文、并发数和 GPU 型号发给我，我可以按这套公式给你做一张完整显存账单。