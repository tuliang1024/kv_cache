# DeepInfer 阶段一异步 KV Get 方案草稿

> 状态：方案草稿，不是最终设计文档。  
> 范围：只讨论阶段一 Block/Global KV 池化。  
> 明确不包含：SWA KV、`gpu_states` 持久化、SWA replay、SWA anchor、PD 数据面改造。

## 1. 目标与结论

阶段一只复用 `gpu_blocks` 中已经完成的完整 KV block。上游在请求进入 DeepInfer 前完成前缀匹配，并把命中结果作为 `PrefixPlan` 传入。DeepInfer 不重新实现 hash、前缀扫描或 Store LRU，只负责：

1. 在调度接纳时为命中的 prefix block 分配本地 HBM block。
2. 启动后台异步 Store `get`，直接写入 `gpu_blocks`。
3. get 期间不执行该请求的 prefill，继续调度其他请求。
4. get 完成后补充分配 suffix block，再按已有 prefill 流程计算未命中部分。
5. get 失败时释放未接管的本地资源，按 miss 重新计算。

推荐的总体方案是：

```text
PrefixMatcher/Router
    -> PrefixPlan
    -> DeepInfer.add_request
    -> Scheduler 资源接纳
    -> prefix-only block allocation
    -> KV loading 状态
    -> Worker 后台 Store.get
    -> Scheduler 主线程消费完成事件
    -> suffix block allocation
    -> 现有 prefill
```

## 2. 与 vLLM 的对应关系

vLLM 的相关流程可归纳为：

```text
get_num_new_matched_tokens()
    -> allocate_slots(... num_external_computed_tokens=hit)
    -> load_kv_async=True
    -> num_new_tokens=0
    -> WAITING_FOR_REMOTE_KVS
    -> connector metadata
    -> worker start_load_kv()
    -> 后台 get
    -> finished_recving
    -> 恢复 WAITING
    -> 下一轮 prefill
```

对应代码：

- external prefix lookup：`vllm/vllm/v1/core/sched/scheduler.py:1148-1180`
- async load admission：`vllm/vllm/v1/core/sched/scheduler.py:1325-1339`
- connector state update：`vllm/vllm/v1/core/sched/scheduler.py:1525-1548`
- `WAITING_FOR_REMOTE_KVS`：`vllm/vllm/v1/core/sched/scheduler.py:1576-1607`
- completion handling：`vllm/vllm/v1/core/sched/scheduler.py:3616-3666`
- blocked request handling：`vllm/vllm/v1/core/sched/scheduler.py:3671-3684`
- vllm-ascend worker load：`vllm_ascend/.../pool_worker.py:1345-1384`
- vllm-ascend async Store get：`vllm_ascend/.../kv_transfer.py:979-1139`

DeepInfer 的上游已经提供命中结果，因此不需要复用 vLLM 的 `get_num_new_matched_tokens()` 查询动作，但需要复用其后半段的生命周期：**分配外部 block -> 异步 get -> 完成确认 -> 恢复 prefill**。

## 3. PrefixPlan

### 3.1 输入

阶段一建议定义一个请求级 `PrefixPlan`，由 Entrypoint/Router 生成并随请求传输：

```text
PrefixPlan {
    request_id
    hit_tokens                 # 按 PE 实际 block_size 对齐
    model_id
    layout_version
    block_size
    block_bytes
    plan_epoch / expire_at     # 可选，用于失效检测
}
```

阶段一不携带 SWA 字段、`state_id` 或完整 Store key。`state_id` 不属于阶段一的加载目标；Store key 在 PE 侧根据请求 token/hash、实际 `block_size` 和本 rank 拓扑生成。阶段一只要求 prefix block 写入 `gpu_blocks`。

### 3.2 输入校验

Engine 接收 PrefixPlan 时只做结构和版本校验，不做重新匹配：

- `request_id` 必须匹配当前请求；
- `hit_tokens >= 0`；
- `hit_tokens` 必须不超过 prompt 长度减去需要重新计算的尾部；
- `hit_tokens` 必须按当前 PE `CacheEngine` 的实际 `block_size` 对齐；
- `model_id/layout_version/block_size/block_bytes` 必须与当前 Engine 一致。

PrefixPlan 只是 metadata 命中结果，不代表 Store 对象在真正 get 时仍然存在。Store get 返回失败时必须按 miss 处理。

上游不需要传递完整 Store key。PE 侧的 Python connector/worker 保存请求 token 和 PrefixPlan，在真正生成 load task 时复用 vLLM/vllm-ascend 的 hash/key builder：先按实际 block size 生成连续 block hash，再按本 rank 的 `tp/dp/pp` 或其他 shard 信息生成本 rank 的 key。各 rank 的 `hit_tokens` 必须一致；key 内容可以不同。

## 4. DeepInfer 队列和状态

### 4.1 保留现有 waiting 语义

DeepInfer 当前 `_schedule_prefill()` 在遇到 `schedulable=false` 时直接 `break`，并且 NORMAL 引擎的 `_has_prefill_waiting()` 只检查 waiting 队首。这套语义与当前 FIFO、chunked prefill 和 DP ALL/ANY 投票相关，阶段一不直接修改为“遇到异步请求就 continue”。

### 4.2 ALL/ANY 投票与异步 load

DeepInfer 的 NORMAL Scheduler 当前按 `prefill_attempt_interval` 和
`prefill_any_attempt_interval` 选择投票点：

```text
ALL 点：所有 DP rank 都有可执行 prefill 时才进入 prompt step
ANY 点：任意 DP rank 有可执行 prefill 时，全组进入 prompt step
```

实现位置是 `scheduler.hpp:_schedule()` 的 `_dp_allreduce()`，投票输入来自
`_has_prefill_waiting()`，并且还有第二轮 `any_scheduled` 投票防止空 prompt step。

阶段一必须区分两类操作：

1. **KV load admission**：分配 prefix blocks、生成 `KVLoadTask`、提交后台 get。不产生模型 token，不改变 `decode_step`，不参与 ALL/ANY 投票。
2. **suffix prefill**：get 完成、suffix blocks 分配成功后，才把请求放回普通 `waiting`，由现有 `_has_prefill_waiting()` 和 ALL/ANY 投票决定是否执行。

因此每个 step 的推荐顺序是：

```text
poll/apply KV completion
  -> 把可执行的 kv_ready 请求转换回 waiting
  -> 旁路 admission 新的 KV load（不改变步型）
  -> 执行原有 NORMAL ALL/ANY 投票
  -> prompt 或 decode
```

`kv_loading` 请求不应被计入 `_has_prefill_waiting()`；只有已经完成 get、suffix block 已准备好并回到 `waiting` 的请求才算 prefill work。若某 rank 只有 KV loading、没有可执行 prefill，它仍按原有 ALL/ANY 语义投票；异步 get 的后台进度不能单独触发 prompt step。

KV load admission 如果依赖 rank 本地空闲 block，可以各 rank 独立进行，因为它不执行 collective forward；但进入 suffix prefill 前，所有参与该请求的 rank 必须得到相同的 `hit_tokens` 和 suffix 起点。任一 rank 的加载失败都必须把该请求降级到统一的有效连续前缀。

### 4.3 新增 KV 专用状态

建议在 Scheduler 中新增两个队列：

```cpp
std::deque<SequencePtr> kv_loading;
std::deque<SequencePtr> kv_ready;
```

职责如下：

| 队列 | 含义 | 是否可执行模型计算 |
|---|---|---|
| `waiting` | 普通请求，或已准备好进行 prefill 的请求 | 是，按原有规则 |
| `kv_loading` | 已分配命中 prefix blocks，Store get 进行中 | 否 |
| `kv_ready` | Store get 完成，等待 suffix block 分配 | 尚未进入模型计算 |
| `running` | 已进入当前模型 batch | 是 |

`kv_loading` 对应 vLLM 的 `kv_holding_waiting`；`kv_ready` 是 DeepInfer 为明确两阶段交接而增加的显式状态。

不要复用现有 `schedule.schedulable` 表示 KV loading。该字段已经被多模态异步处理使用，混用会把两种异步生命周期耦合起来。

### 4.4 Sequence 状态

建议在 `SequenceScheduleState` 或独立的 KV 状态中增加：

```cpp
enum class KVLoadState {
    NONE,
    WAITING_TO_START,
    IN_PROGRESS,
    READY,
    FAILED,
};

KVLoadState kv_load_state{KVLoadState::NONE};
int prefix_hit_tokens{0};
uint64_t kv_load_id{0};
```

`SequenceStatus` 可以增加 `WAITING_FOR_REMOTE_KVS`，但如果增加，必须放在所有 `FINISHED_*` 状态之前，因为 DeepInfer 使用 `status >= FINISHED_STOPPED` 判断 finished。

## 5. 端到端时序

### 5.1 add_request 阶段

```text
1. add_request(request, prefix_plan)
2. 保存 PrefixPlan
3. hit_tokens == 0 -> 进入普通 waiting
4. hit_tokens > 0 -> 进入 waiting，标记 WAITING_TO_START
5. 不分配 HBM block，不启动 Store.get
```

不在 add_request 时启动 get 的原因是：请求可能尚未被调度接纳、可能被取消、可能暂时没有 block 资源，且异步 get 需要绑定本地 block 的生命周期。

### 5.2 Scheduler 接纳 prefix load

在每个 `schedule()` 开始处，先消费上一步的 KV 完成事件，再进行新的 load admission。

从 waiting 选择带有 PrefixPlan 的请求，检查：

1. 命中 block 数是否超过本地 block manager 容量；
2. 当前空闲 block 是否足够；
3. 并发 KV load 数是否达到上限；
4. 请求是否已经 abort/finished；
5. PrefixPlan layout 是否有效。

资源不足时保留在 waiting，不能分配部分 block。

资源足够时：

```text
1. 分配 hit_tokens / block_size 个 block
2. 把 local block IDs 与请求的 block 顺序建立一一映射；key 不进入 C++ task，由 PE worker 按 block 顺序生成
3. 写入 seq.cache.block_table 的 prefix 部分
4. seq.cache.prefix_hit_tokens = hit_tokens
5. seq.kv_load_state = IN_PROGRESS
6. 生成 KVLoadTask
7. 从 waiting 移到 kv_loading
8. 不把该请求放入本轮 scheduled_seqs
```

阶段一不分配 suffix block，因此不会因为一个长 prompt 的完整长度而阻塞 prefix load。只要命中部分 block 有空间，异步 get 就可以开始。

### 5.3 Worker 启动异步 get

Scheduler 输出或 Engine 侧 connector metadata 携带：

```text
KVLoadTask {
    request_id
    load_id
    local_block_ids
    hit_tokens
    block_size
    block_bytes
    layout_version/model_id
}
```

由于 DeepInfer 的 C++ `Scheduler::schedule()` 在 `_schedule()` 中使用
`py::gil_scoped_release`，C++ Scheduler 不应直接调用 Python Store API。推荐使用
“C++ 产出任务，Python 提交 get”的边界：

```text
C++ Scheduler:
    分配 local_block_ids
    生成 KVLoadTask
    通过 pybind 暴露 SchedulerOutputs.kv_load_tasks

Python LLMEngine.step:
    读取 kv_load_tasks
    交给 KVPoolWorker / Python Mooncake Store client
    worker 后台线程执行 get
```

Python worker 根据 `request_id` 找到请求 token 和 `hit_tokens`，在 PE 侧生成 key，随后为每个命中 block 构造一个 Store item：

```text
for each local block index:
    key = PEKeyBuilder(request_tokens, block_size, layout_version,
                       rank_topology, block_index)
    dst = (gpu_blocks.data_ptr() + local_block_id * block_bytes, block_bytes)
    items.append(GetItem(key=key, segments=[dst]))
results = Store.get(items)
```

目标地址直接指向 `gpu_blocks` 对应 block，不经过 host staging，也不做独立 swap-in。后台线程只写完成结果队列，不直接修改 Sequence 或 Scheduler。

这里的“异步”由 worker 的后台线程/任务池提供：`MooncakeStoreClient.get()` 本身可以在 worker 线程中同步等待本批 RPC 返回，不能在 Scheduler 线程中直接调用。Store backend 负责将每个 item 的返回字节数严格校验为 `block_bytes`；短读、对象不存在、连接异常都转成该 item 的失败结果。worker 再把逐 block 结果和 `load_id` 投递到完成队列。

与 `mooncake_store.md` 的接口对应关系如下：

| 阶段一对象 | Mooncake Store 调用 |
|---|---|
| 启动时的 `gpu_blocks` | `register_buffers([BufferRegion(ptr, nbytes)])` |
| 一个 prefix block 的目标 | `GetItem(key, segments=[(ptr + local_block_id * block_bytes, block_bytes)])` |
| 一批 prefix blocks | `batch_get_into_multi_buffers(keys, addrs, sizes)` |
| get 结果 | 每个 item 的 `bytes_read == block_bytes` 才算成功 |

因此 C++ task 只需携带 local block ID 和布局元数据；不携带地址和完整 key。Python worker 负责把已注册的 `gpu_blocks` 地址转换成 Store 的 segment，并在提交前验证 `block_id`、`block_bytes` 和布局版本。

该边界对齐 vLLM 的 connector 设计：Scheduler 只构造 connector metadata，Python worker 在
`start_load_kv()` 中启动 transfer，完成后通过 `get_finished()` 返回 request ID。DeepInfer
不需要让 C++ 调用 Python 回调，也不需要把完整 Store key 放进 C++ Sequence。

### 5.4 get 进行中

`kv_loading` 中的请求不参与 prefill/decode。Scheduler 仍然可以：

- 调度普通 waiting 请求；
- 调度已有 running 请求的 decode；
- 启动其他未超过并发上限的 KV load。

如果 DeepInfer 当前 NORMAL 调度必须在 prompt/decode 之间二选一，KV load admission 不应伪造一个带 prompt token 的 SchedulerOutput，也不应清零 `decode_step`。它应作为不产生模型 token 的旁路操作。

### 5.5 完成事件

在下一次 `schedule()` 开始时，Engine 从 worker 取得完成事件：

```text
KVLoadCompletion {
    request_id
    load_id
    success
    loaded_blocks
    failed_blocks
    error_code
}
```

主线程验证 `load_id`，防止旧请求或旧任务完成事件误更新新请求。

成功：

```text
1. 确认所有 prefix block 写入完成
2. kv_load_state = READY
3. 从 kv_loading 移到 kv_ready
```

失败：

```text
1. 如果允许部分命中，只保留连续成功的 block
2. 否则释放全部 prefix blocks
3. 清除 block_table 的外部命中部分
4. hit_tokens = 0 或缩短为有效连续前缀
5. kv_load_state = FAILED/NONE
6. 重新放入 waiting，走普通 prefill
```

Mooncake backend 的 `get(items)` 返回逐 item 的布尔结果；阶段一应按“一 item = 一 block”组织请求，从而可以保留从 block 0 开始的连续成功前缀。只有未来接入的 Store 适配层退化为请求级结果时，才按整次 miss 处理。

### 5.6 suffix 分配和 prefill

`kv_ready` 请求优先于普通新请求处理：

```text
1. 检查 prefix block 数量和 block_table 顺序
2. 计算完整 prompt 所需 block 数
3. 分配 suffix blocks
4. 设置 num_cached_tokens = hit_tokens
5. 将请求转为 DeepInfer 原有 WAITING/pre-fill-ready 状态
6. 让现有 _schedule_prefill 执行 suffix
```

suffix 分配失败时不能进入 prefill。请求保持 `kv_ready`，下一轮重试；已经加载的 prefix block 继续 pin，或者根据资源压力取消该 load 并回退为 miss。首版建议保留 prefix block 并重试，避免重复 Store get。

## 6. 与 DeepInfer 现有调度的适配

### 6.1 `scheduler.hpp`

需要增加：

- `kv_loading`、`kv_ready` 队列；
- prefix allocation admission；
- suffix block extension；
- KV load completion 的主线程应用接口；
- `kv_load_id` 校验；
- abort 时从两个 KV 队列移除并释放资源；
- `get_num_unfinished_seqs()` 和 pending 统计纳入 KV 队列。
- NORMAL 投票保持现有 ALL/ANY 逻辑；KV load task 不计入 `want`、`num_batched_tokens`、`seqlens_q` 或 `decode_step`。

建议接口形态：

```cpp
bool try_allocate_prefix(Sequence &seq, int hit_tokens);
bool try_extend_suffix_blocks(Sequence &seq);
void mark_kv_load_started(const std::string &request_id, uint64_t load_id);
void mark_kv_load_finished(const std::string &request_id, uint64_t load_id);
void mark_kv_load_failed(const std::string &request_id, uint64_t load_id);
```

### 6.2 `block_manager.hpp`

现有 `try_allocate_seq()` 面向完整序列，不足以支持两阶段分配。需要增加：

```cpp
bool try_allocate_prefix_blocks(Sequence &seq, int prefix_blocks);
bool try_extend_blocks(Sequence &seq, int target_blocks);
```

要求：

- prefix allocation 同时处理 block manager 资源；
- 阶段一不分配 suffix block；
- suffix extension 失败时不修改已有 prefix block；
- free 操作必须与后台 get 生命周期隔离；
- Store get 未完成时，相关 block 必须 pin，不能被抢占复用。

### 6.3 `sequence.hpp` / pybind

需要暴露给 Python/worker 的最小字段：

```text
request_id
prefix_hit_tokens
kv_load_state
kv_load_id
block_table
prompt token lookup handle or request_id
```

不建议让 Python 直接修改 `block_table`。block_table 应由 Scheduler/BlockManager 维护，Python 只读取用于生成 KVLoadTask。

### 6.4 `LLMEngine.step()`

建议调整为：

```text
step()
  1. Python worker.poll_completions()
  2. scheduler.apply_kv_load_completions()
  3. scheduler.schedule()  # 内部完成 load admission 和原有 decode/prefill
  4. 读取 scheduler_outputs.kv_load_tasks
  5. Python worker.submit_get_tasks()
  6. 执行模型
```

`SchedulerOutputs` 增加独立的 `kv_load_tasks` 数组。该数组不计入 `scheduled_seqs`、
`num_batched_tokens`、`seqlens_q` 或 `prompt_run`。Python 只把 task 提交给 worker，不修改
Scheduler 的 block_table。

### 6.5 `CacheEngine` / worker

阶段一需要导出：

```text
gpu_blocks data pointer
block_bytes
num_gpu_blocks
block_size
layout_version
```

Worker 负责：

- 注册 `gpu_blocks` 到 Store/Transport；
- 后台 batch get；
- 有界任务队列；
- 每个请求一个 load_id；
- 失败结果上报；
- 完成队列线程安全；
- 关闭时停止线程并等待正在访问 HBM 的任务结束。

## 7. NORMAL / PROMPT / COMPLETION 引擎

### NORMAL

- 异步 get admission 不改变本轮 prompt/decode 类型；
- completion 应用和 suffix block 扩展成功后，`kv_ready` 请求回到 `waiting`，再参与现有 ALL/ANY 投票；
- DP rank 必须使用一致的 `hit_tokens` 和 suffix 起点；
- 不用异步 get 的本地完成状态改变 `decode_step`。

### PROMPT

- 允许 prefix load 和 suffix prefill；
- `send_processed_prompts` 只能在 suffix prefill 完成后执行；
- get 未完成或 suffix block 未分配成功时不能向 CE 发布请求完成信息。

### COMPLETION

阶段一的 pool prefix load 主要面向 PE/PROMPT。COMPLETION 不应因为 Store pool hit 走独立 prefill load 路径；CE 仍按 PD 设计通过 Mooncake TransferEngine 拉取 PE HBM。若未来允许 CE 直接读取 pool，需单独定义，因为 CE 无法依赖重算兜底。

## 8. 异常和边界场景

### 8.1 Store get 失败

按 vLLM 的 invalid-block fallback 处理：

```text
Store 返回每个 block 的结果码
  -> 找到第一个失败 block
  -> 保留其前面的连续有效 prefix
  -> 释放失败 block 及其后的 prefix blocks
  -> hit_tokens = first_failed_block * block_size
  -> 从该边界继续普通 prefill
```

如果第一个 block 失败，等价于完整 miss。Store 失败不应把请求永久卡在 `kv_loading`，也不应让未写入 block 进入 attention。所有 rank 必须采用相同的有效连续前缀边界。

### 8.2 metadata 命中但对象被淘汰

PrefixPlan 只代表查询时存在。真正 get 时对象可能已经被淘汰，必须按 get 结果处理，不能假设 metadata 命中等于数据可读。

### 8.3 请求取消

取消时：

1. 从 `waiting`、`kv_loading`、`kv_ready` 移除；
2. 标记 load_id cancelled；
3. 后台线程完成后丢弃结果；
4. 确认 Store 不再访问目标 HBM block 后再 free/unpin；
5. 不能让旧完成事件影响新请求。

### 8.4 抢占和资源回收

正在 get 的 prefix block 不能直接回收到 free list。需要：

- pin count 覆盖整个 Store get；
- 取消或失败后等待后台访问结束；
- 再释放 block；
- suffix 尚未分配的请求不应进入普通 running 抢占逻辑。

### 8.5 并发 load 上限

需要限制：

- 同时进行的请求数；
- 同时进行的 block 数；
- Store backend outstanding operation 数；
- 完成队列长度。

达到上限时，PrefixPlan 继续保存在 waiting，不提前分配资源。

### 8.6 部分命中

PrefixPlan 的 `hit_tokens` 必须是连续前缀。Store 只返回请求级结果时，任意失败按完整 miss；Store 返回逐 block 结果时，保留从 block 0 开始的连续成功 block，后续部分按普通 prefill 计算。

### 8.7 新请求和 load 请求竞争 HBM

prefix load admission 只能使用当前可用 block，不能预留完整 prompt 的 suffix block。suffix 分配失败时请求保持 `kv_ready`，并由 Scheduler 在资源释放后重试。

### 8.8 多 DP rank

上游只需要提供请求级 `hit_tokens`；PE 的每个 rank 根据本地 tokenizer/hash 结果和 rank 拓扑生成自己的 shard key，但必须保证：

```text
同一请求的 hit_tokens 一致
同一 block 的 shard 全部可用，或者统一按 miss
suffix prefill 的起点一致
```

不能出现 rank 0 加载 8 个 block、rank 1 只加载 7 个 block 后继续 collective prefill。任一 rank 的 get 失败都要按第一个失败 block 统一截短命中范围。

## 9. 不采用的方案

### 9.1 add_request 时直接分配完整 prompt blocks

不采用。这样会把尚未被接纳的请求变成资源持有者，容易造成 KV cache 被 waiting 请求提前占满。

### 9.2 把 loading 请求留在现有 waiting 队首并继续使用 `break`

不采用。DeepInfer 当前队首不可调度会停止整个 prefill 扫描，会阻塞后续请求。

### 9.3 直接把 `schedulable=false` 当作 KV loading 状态

不采用。该字段已用于多模态异步处理，且无法表达 get、ready、failed、cancelled 等生命周期。

### 9.4 后台线程直接修改 Sequence/Scheduler

不采用。后台线程只写 HBM 和完成队列；Scheduler 状态变化统一在 Engine 主线程/调度线程应用。

### 9.5 用一个“空 prompt step”启动 get

首版不采用。异步 get 不产生模型 token，不应伪造 `prompt_run=true` 或改变 DeepInfer 的 decode_step。应通过独立 `KVLoadTask` 旁路提交。

### 9.6 在 C++ Scheduler 中直接调用 Python Store

不采用。当前 Scheduler 释放 GIL 执行调度，且 C++ 不应承担 Python Store client 的异常和线程生命周期。采用 `SchedulerOutputs.kv_load_tasks -> Python worker.submit_get_tasks()` 的显式边界。

## 10. 最小实现切分

### P1-1：协议和本地状态

- 定义 `PrefixPlan` 和 `KVLoadTask/KVLoadCompletion`；
- 为请求保存 PrefixPlan；
- 新增 `KVLoadState`、`prefix_hit_tokens`、`kv_load_id`；
- 完成 request cancel 清理。

### P1-2：两阶段 block 分配

- `BlockManager` 新增 prefix-only allocation；
- 新增 suffix extension；
- 保证 prefix block pin 和失败回滚；
- 增加单元测试覆盖资源不足、取消和重复完成事件。

### P1-3：异步 worker

- 注册 `gpu_blocks`；
- 后台 Store get；
- 有界任务/完成队列；
- load_id、失败码和线程退出处理。

### P1-4：Scheduler/Engine 交接

- step 开头消费 completion；
- admission 生成 KVLoadTask；
- `kv_loading -> kv_ready`；
- ready 后扩展 suffix 并进入原有 prefill；
- 验证 NORMAL 的 ALL/ANY 投票不被 KV load admission 改变。

### P1-5：异步写回

- prefill 产生完整 block 后异步 put；
- 请求结束时写入完整 block；
- put 前 pin，Store 读取完成后 unpin；
- 写回失败不影响当前请求结果。

## 11. 验证场景

至少覆盖：

1. 无命中，完全走原有 prefill；
2. 命中一个完整 block；
3. 命中多个完整 block，同时存在其他 waiting 请求；
4. get 慢，后续请求仍能调度；
5. get 完成后 suffix block 资源不足；
6. get 失败并回退完整 prefill；
7. Store metadata 命中但对象已淘汰；
8. 请求在 get 中途取消；
9. 同一 request 的旧完成事件晚到；
10. 多 DP rank 的命中长度不一致；
11. NORMAL decode 中启动异步 get，不改变 decode step 类型；
12. chunked prefill、抢占、MTP 和多模态请求共存；
13. Engine 关闭时仍有 get/put 任务；
14. HBM block pin/unpin 无泄漏、无重复释放。

## 12. 已确认事项与仍需外部确认事项

### 12.1 已确认的方案

1. **Store key**：上游不传完整 key；PE 侧 Python connector/worker 根据请求 token、实际 block size、hash 规则和本 rank 拓扑生成 key。vllm-ascend 的 `ChunkedTokenDatabase.process_token_key_strings*()` 和 worker 侧 key 生成逻辑可直接作为参考。
2. **hit_tokens 对齐**：由 PE `CacheEngine` 的运行时 `block_size` 决定，不固定写死 128。上游 PrefixPlan 必须带或协商 `block_size`；不匹配直接拒绝该计划并按 miss 处理。
3. **Python Store 适配**：C++ Scheduler 只生成 task descriptor；Python `LLMEngine`/`KVPoolWorker` 提交异步 get 并消费完成队列。C++ 不直接回调 Python Store。
4. **ALL/ANY 投票**：异步 load admission 不参与投票，不改变 prompt/decode 步型；只有 suffix ready 后回到 waiting 的请求参与原有投票。
5. **DP shard**：请求级 `hit_tokens` 在参与该请求的 rank 间一致；每个 PE rank 生成自己的 shard key。任一 rank 失败，按第一个失败 block 统一截短命中范围。
6. **部分失败**：采用 vLLM 风格的连续有效前缀 fallback，而不是首版整次 load 失败即完全 miss。
7. **suffix 资源不足**：保留已完成 prefix blocks，放在 `kv_ready` 等待资源；为防止长期占用，可配置 ready 超时或内存压力阈值，超时后释放 prefix 并回退 miss。

### 12.2 仍需外部确认

1. `PrefixPlan.layout_version/model_id` 由哪个 Router/PE 配置源生成，以及 engine 重启后的 epoch 失效规则。
2. `SchedulerOutputs.kv_load_tasks` 的 pybind 表达方式；建议使用本轮输出拥有的 task 容器，Python 在 `step()` 内立即转换为 worker 自有对象，不能借用下一轮会复用的 clone 数组。
3. `get_num_pendings()` 是否需要分别报告普通 waiting、kv_loading 和 kv_ready；已有 `get_num_prompt_len_waiting()` 不能把 loading 请求误报为可执行 prefill。

以下事项已由 `设计文档/mooncake_store.md` 给出实现约束，不再作为方案阻塞项：

- `gpu_blocks` 在启动阶段通过 `register_buffers` 注册；共享 TE 时注入同一个注册函数，Python worker 只使用已注册 region 内的地址。
- 每个 block 以一段 `(gpu_blocks.data_ptr() + block_id * block_bytes, block_bytes)` 传给 `batch_get_into_multi_buffers`；Store 返回字节数必须等于 `block_bytes`，短读按该 block 失败。
- backend 的 `_run()` 已负责分块、锁、重连和熔断；这些故障处理不应进入 Scheduler 的状态机。

### 12.3 基于 Mooncake Store 方案的本次修订点

1. **目标内存已确定**：启动时注册 `gpu_blocks`，get 直接写入对应 HBM block；不增加 host staging 或额外 swap-in 层。
2. **task 不携带 key 和裸地址**：C++ 只传 request/load 标识、local block IDs、实际 `block_size/block_bytes` 和布局元数据；Python worker 生成 key 并计算已注册 region 内的 segment 地址。
3. **get 粒度改为逐 block**：一个 block 对应一个 `GetItem`，便于使用 backend 的逐 item 结果实现连续有效前缀 fallback；不能把整请求拼成一个不可分辨的 item。
4. **异步边界明确**：Scheduler 只产生 task；worker 线程调用 `MooncakeStoreClient.get()`，主调度线程消费完成事件并校验 `load_id`。Store 的重连、熔断、短读处理留在 backend。
5. **成功条件明确**：每个 item 必须读回完整 `block_bytes`；对象不存在、短读、连接异常均为该 block 失败，不得把部分写入的 block 交给模型。
6. **外部确认项收缩**：剩余问题只涉及布局/版本来源、task 容器生命周期和 pending 统计，不再阻塞“Python 是否能读写 HBM”或“Store 是否支持逐 block 结果”的总体方案。

## 13. 方案结论

阶段一建议采用以下边界：

```text
上游：前缀匹配和 PrefixPlan
Scheduler：资源接纳、请求状态、队列、两阶段分配时机
BlockManager：prefix block 分配、suffix block 扩展、pin/free
Worker：后台 Store get/put、HBM 写入、完成事件
CacheEngine：gpu_blocks 布局和地址注册
```

关键实现选择是：

- PrefixPlan 在 `add_request()` 前提供；
- HBM 分配和异步 get 在 Scheduler 接纳请求时开始；
- 普通 waiting FIFO 保持不变；
- loading 请求放入独立 `kv_loading`，避免 DeepInfer 当前队首 `break` 阻塞后续请求；
- get 完成后先进入 `kv_ready`，suffix block 分配成功后才进入 prefill；
- 异步 get admission 不参与 NORMAL 的 ALL/ANY 投票，不改变 prompt/decode 步型；
- PE 侧按实际 block size 和本 rank 拓扑生成 Store key；
- Python worker 通过 `SchedulerOutputs.kv_load_tasks` 提交 Store get；
- Store 失败沿用 vLLM 的连续有效前缀 fallback；
- 任何后台完成都必须经过主线程完成队列应用。

