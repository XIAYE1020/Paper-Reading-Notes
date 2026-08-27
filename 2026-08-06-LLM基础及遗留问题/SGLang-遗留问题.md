# SGLang 遗留问题

## RadixAttention 的 KV Cache 淘汰机制

**问题：**SGLang 的 `evict blocks` 是否使用 LRU？LRU 如何与 ref count 配合，是否只有叶节点才能成为淘汰候选？`ref_count == 0` 是否就代表叶节点？如果一个公共前缀的 `ref_count == 0`，是否说明整棵子树都没有被使用，可以直接删除？如果所有 block 的 ref count 都不为 0，又会如何处理？

**答：**SGLang 采用引用计数保护下的叶节点 LRU，**淘汰条件是“叶节点且 `ref_count == 0`”**：ref count 表示是否有运行请求使用，叶节点表示是否没有子节点，两者并不等价。由于子节点的引用会传播到祖先，公共前缀的 `ref_count == 0` 确实说明整棵子树没有运行请求使用；整棵删除虽然安全，但可能误删近期使用的分支并释放过多空间，因此系统从最旧叶子开始逐步淘汰，空间仍不足时才可能最终删完整棵子树。如果所有 block 的 ref count 都不为 0，LRU 就没有候选节点：新请求需要等待；若当前 decode 也没有足够空间，当前实现会撤回部分运行请求并释放其 KV，单个请求仍无法容纳时则返回 OOM。

## FSM 与压缩 FSM 总结

### 1. FSM 解决什么问题？

FSM 用于**约束模型输出格式**。系统将正则表达式或 JSON Schema 转换成状态机：节点代表状态，边代表允许出现的字符或字符串；生成内容只有沿合法边前进并到达终止状态，才是合法输出。

提示词中的格式要求只是软约束，模型可以不遵守；FSM 在解码阶段直接限制候选 token，属于硬约束。

### 2. FSM mask 如何工作？

一次 decode forward 会为词表中的每个 token 输出一个 logit。每个 logit 只是一个实数分数，表示当前上下文对该 token 的相对倾向；经过 softmax 后才成为概率。

运行时根据当前 FSM 状态构造覆盖整个词表的 mask：

```text
mask[i] = 0       token i 合法
mask[i] = -∞      token i 非法
概率 = softmax(logits + mask)
```

随后只从合法 token 中采样，用生成 token 的字符串更新 FSM 状态，再为下一步生成新的 mask。FSM 不修改模型参数，只在模型计算 logits 与采样之间动态限制候选集合。

### 3. 一个完整例子

要求输出 `{"grade":"A"}`，其中评分只能为 `A/B/C/D`：

```text
q0 --固定文本 {"grade":"--> q1
q1 --A/B/C/D-----------> q2
q2 --固定文本 "}--------> q3（终止状态）
```

在 `q1`，模型可能给出：

| token | 原 logit | mask 后 |
|---|---:|---:|
| `A` | 1.0 | 1.0 |
| `B` | 2.0 | 2.0 |
| `X` | 4.0 | `-∞` |

虽然模型最想生成 `X`，但其概率被置为 0。假设最终选择 `B`，FSM 转到 `q2`，补上固定结尾并到达 `q3`，最终输出 `{"grade":"B"}`。

### 4. 普通 FSM 与压缩 FSM

- **普通 FSM**：即使一段文本只有唯一合法写法，也逐 token 调用模型和应用 mask。
- **压缩 FSM**：把连续的唯一分支合并成一条边，通过 jump-forward 一次处理多个确定 token；只有遇到分支时才让模型选择。

压缩 FSM 不改变格式和长度约束，只减少不必要的逐 token forward。由于 FSM 通常建立在字符或字符串上，而模型使用 token，SGLang 还需要重新分词，处理字符边界与 token 边界不一致的问题。

可以把 jump-forward 近似理解为一次**增量 prefill/extend**：运行时把格式已经确定的字符串重新分词，作为已知的续写 token 追加到当前序列，并复用已有 KV Cache，只对新增部分执行一次多 token forward。它不是重新 prefill 整个上下文，也不是让模型重新选择这些确定 token；模型仍需计算新增 token 在各层的表示和 KV。之后遇到有多个合法选择的状态，再恢复普通 decode 和 FSM mask。

### 5. FSM 如何控制长度？

FSM 通过“何时能够到达终止状态”间接控制长度：`\d{4}` 必须生成 4 个数字，`\d+` 则允许循环，长度不固定。`max_new_tokens` 是 FSM 之外的硬上限。字符数量也不一定等于 token 数量。

### 6. 为什么 SGLang 不能在闭源 API 外部执行 FSM？

FSM 需要在每个解码步骤读取和修改 logits。自托管模型的运行时可以做到；黑盒 API 客户端通常只能发送提示词并接收结果，无法介入内部解码。即使把固定 JSON 前缀放进提示词，也只能提前 prefill 开头部分，不能动态屏蔽后续非法 token。

闭源模型本身仍然可以使用 FSM：API 提供商可以接收 JSON Schema、正则或 grammar，并在服务器内部执行约束解码。此时控制权属于 API 提供商，而不是外部的 SGLang。

## SGLang 代码模板与 Agent 分工总结

论文 Figure 2 展示的是一个 **LM program**，不一定是自主 Agent。SGLang 原语代码位于 Agent 的**工作流编排/模型调用层**，用于描述模型调用之间的控制结构。

| 角色 | 主要职责 |
|---|---|
| 开发者或高层编译器 | 编写可复用的工作流模板 |
| **规则路由器或规划模型** | **根据用户问题选择模板并填写参数** |
| 任务模型 | 在 `gen/select` 位置生成判断和文本 |
| SGLang Runtime | 执行模板，并优化并行、缓存和约束解码 |

模板抽象的是“串行、并行、分支、循环、合并、约束输出”等**任务流结构**；提示词、评审维度、工具、分支数和 schema 是动态参数。因此三维评审改为四维评审只需更换 `dimensions`，不需要新代码。

```text
用户问题 → 规则或模型路由器 → 模板 + 参数 → SGLang Runtime → 模型
```

SGLang 本身不负责选择模板。固定应用可由普通路由代码选择；通用 Agent 可让规划模型通过 tool calling 返回模板名称和参数，再由宿主程序验证并调用。

不需要预写成千上万段代码，少量参数化模板可以覆盖大量请求。只有任务流拓扑真正改变时才可能需要新模板；如果没有合适模板，系统仍可退回单次模型调用或通用 Agent 循环，只是可能无法充分利用针对该结构的多调用优化。

模型可以辅助开发者生成模板，DSPy 等高层系统也可以编译到 SGLang，但论文默认方式不是任务模型在执行过程中临时生成并运行 SGLang 源码。普通用户通常只使用已经开发好的 Agent。

## 资料来源

- [SGLang 论文](../2026-07-20-SGLang-Efficient-Execution-of-Structured-Language-Model-Programs/SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.pdf)
- [RadixCache 实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- [请求撤回实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L2524-L2574)
