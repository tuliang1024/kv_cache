# 分析 KV Cache 异步池化流程

> 目前有一个需求是借鉴vllm：/Users/l/L/Code/TuringProject/D/vllm或者vllm-ascend：/Users/l/L/Code/TuringProject/D/vllm-ascend中关于kv cache池化的实现代码逻辑，在deepinfer/Users/l/L/Code/TuringProject/D/DeepInfer中实现类似的功能
>
> 具体需求：
> 目前deepinfer是离线推理框架，希望增加online模式的kv cache池话功能，具体的主要是在prefill阶段执行请求前，根据上游提供的前缀匹配结果，先分配前缀匹配的block hbm内存，并进行异步的store，get操作，然后不阻塞调度器同时执行其他请求，然后等一下step开始执行，查看这个get操作是否完成，如果完成也是可以调度的状态，然后在执行相应的后续block分配并等待调度。
>
> 有一个总体设计方案草稿，/Users/l/L/Code/TuringProject/D/设计文档/设计文档草稿.md，里面有涉及上述异步get的需求，但是主要实现逻辑还是参考vllm的对应实现，因为草稿可能有描述不准确或者错误的地方需要从代码里检查校验，所以：
> - 实现逻辑优先借鉴vllm的代码，
> - 可以查看方案草稿的，如果和vllm的代码实现逻辑有出入，以vllm代码逻辑为主，并需要指出
>
> 现在需要做的事情：
> 1。查看vllm和vllm-ascend的代码实现逻辑，确认前缀匹配的异步get流程，和时机
> 2.查看deepinfer仓的现有代码逻辑，确认可以如何借鉴vllm的实现，支持可行性
> 3.确认方案草稿和vllm对应代码实现逻辑
>
> 注：先不生成对应设计文档

<details><summary>4 previous messages</summary>

> 我会先沿着三条线核对：vLLM/vllm-ascend 的 prefix cache 异步读写与调度时机、DeepInfer 当前调度和 KV block 生命周期、以及草稿中对应描述。最后给出逐项结论和不一致点，暂不改代码或生成设计文档。
>
> <details><summary>Explored a file, a search, ran 2 commands</summary>
>
> - Ran `pwd && printf '%s\n' '--- top-level ---' && rg --files -g '!*__pycache__*' | head -200`
>
> <details><summary>Explored a file, a search</summary>
>
> - Read `./设计文档/设计文档草稿.md`
> - Searched for `block|kv_cache|cache|prefix|scheduler|allocate|swap|store|load|get` in `deepinfer`
>
> </details>
>
> - Ran `rg -n --hidden -S "async|wait_for|prefix.*cache|kv.*cache|Load|Store|BlockPool|external.*cache" vllm/vllm vllm-ascend/vllm_ascend 2>/dev/null | head -400`
>
> </details>
>
> 初步检索显示草稿已经按 `WAITING_FOR_REMOTE_KVS` 和“两阶段 block 分配”描述了目标流程，但还需要用具体源码确认：vLLM 主线的 connector API、vllm-ascend 的落地实现是否完全同义，以及 DeepInfer 是否已有可复用的 waiting/异步完成状态。接下来我会只读这些关键文件并标注行号。
>
> <details><summary>Explored 12 files, 16 searches, a list, ran 2 commands</summary>
>
> <details><summary>Explored 28 files, 10 searches</summary>
>
> - Searched for `class.*Block|BlockManager|Scheduler|schedule|waiting|RUNNING|BLOCKED|try_allocate` in `engine`
> - Searched for `try_allocate_prefix|extend_seq_blocks|waiting_for_remote|apply_finished|num_cached_tokens|cache.*block|block_table` in `deepinfer`
> - Searched for `WAITING_FOR_REMOTE_KVS|waiting_for_remote_kvs|remote_kv|kv_connector|prefix_cache` in `v1`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/block_manager.hpp`
> - Read `./DeepInfer/csrc/engine/sequence.hpp`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
> - Searched for `WAITING_FOR_REMOTE_KVS|waiting_for_remote_kvs|async_loading|load_async|finished_recving|process_finished` in `pool_scheduler.py`
> - Searched for `_update_waiting_for_remote_kv|WAITING_FOR_REMOTE_KVS|get_num_new_matched_tokens|update_state_after_alloc` in `scheduler.py`
> - Searched for `_update_waiting_for_remote_kv|WAITING_FOR_REMOTE_KVS` in `scheduler.py`
> - Searched for `def update_state_after_alloc|def get_finished|finished_req|_async_loading_req_ids|kv_recv_thread` in `pool_scheduler.py`
> - Searched for `class KVCacheStoreRecvingThread|def add_request|Store.get|gpu_blocks|set_num_tokens|extend` in `ascend_store`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/deepinfer/worker/cache_engine.py`
> - Read `./DeepInfer/csrc/engine/sequence.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Searched for `block_table|num_tokens|hit_tokens|state_id|allocate` in `worker`
> - Searched for `try_allocate_prefix|extend_seq_blocks|waiting_for_remote_kvs|apply_finished_recving|do_save_kvcache|kv_transfer|prefix` in `DeepInfer`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/deepinfer/sequence.py`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/deepinfer/sequence.py`
>
> </details>
>
> - Ran `sed -n '1,220p' vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/transfer_thread.py 2>/dev/null || true`
>
> <details><summary>Explored 9 files, 3 searches, a list</summary>
>
> - Listed files in `ascend_store`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
> - Read `./DeepInfer/csrc/engine/sequence.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Searched for `class KVCacheStoreRecvingThread|class KVTransferThread|def add_request|def _run|batch_get|load_gvas` in `kv_transfer.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Searched for `class KVCacheStoreRecvingThread|def _load|def load|batch_get_get|batch_get` in `kv_transfer.py`
> - Searched for `allocate_slots\(|num_external_computed_tokens|load_kv_async|defer_load` in `kv_cache_manager.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
>
> </details>
>
> - Ran `rg -n -C 8 "def _update_waiting_for_remote_kv|update_waiting_for_remote" vllm/vllm/v1/core/sched/scheduler.py vllm/vllm/v1/core/sched/*.py`
>
> <details><summary>Explored 2 files, 3 searches</summary>
>
> - Searched for `waiting_for_remote|finished_recving|set_schedulable|set_num_tokens|hit_tokens` in `deepinfer`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Searched for `def allocate_slots` in `kv_cache_manager.py`
> - Read `./vllm/vllm/v1/core/kv_cache_manager.py`
> - Searched for `WAITING_FOR_REMOTE|try_allocate_prefix|extend_seq_blocks|后台|异步|Entrypoint|前缀匹配|Engine|host staging|swap-in|Store.get|Store put|block_table|pool_next` in `设计文档草稿.md`
>
> </details>
>
> </details>
</details>

已完成代码核对，结论如下。

**1. vLLM / vllm-ascend 的异步 get 流程**

vLLM 的核心流程是：

1. 调度器首次接纳请求时调用 connector 的 `get_num_new_matched_tokens()`，获取外部命中 token 数和 `load_kv_async` 标志：[scheduler.py:1168](./vllm/vllm/v1/core/sched/scheduler.py:1168)。
2. 根据外部命中长度分配对应的 prefix KV block。`allocate_slots()` 明确把外部命中 block 与后续待计算 token 分开处理：[kv_cache_manager.py:401](./vllm/vllm/v1/core/kv_cache_manager.py:401)、[kv_cache_manager.py:488](./vllm/vllm/v1/core/kv_cache_manager.py:488)。
3. 异步 load 时本步不执行 prefill，`num_new_tokens=0`，请求进入 `WAITING_FOR_REMOTE_KVS`：[scheduler.py:1336](./vllm/vllm/v1/core/sched/scheduler.py:1336)、[scheduler.py:1576](./vllm/vllm/v1/core/sched/scheduler.py:1576)。
4. connector 在 `update_state_after_alloc()` 中记录已经分配的外部 block，并提交异步加载：[pool_scheduler.py:339](./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:339)。
5. vllm-ascend 的 `KVCacheStoreRecvingThread` 后台调用 Store `get()`，目标地址直接指向 KV cache 的 HBM block：[kv_transfer.py:979](./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:979)、[kv_transfer.py:1033](./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:1033)、[kv_transfer.py:1085](./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:1085)。
6. 每个 step 检查完成集合；未完成继续跳过，完成后才将请求恢复为 `WAITING`，缓存有效 block，然后再调度后续 prefill：[scheduler.py:3671](./vllm/vllm/v1/core/sched/scheduler.py:3671)、[scheduler.py:3616](./vllm/vllm/v1/core/sched/scheduler.py:3616)。

因此，异步 get 的关键点是：

- 先分配外部命中的 block；
- get 期间不执行 prefill；
- 不阻塞其他可调度请求；
- get 完成后才恢复请求；
- 失败时只保留成功加载的 prefix，失败部分重新计算。

**2. DeepInfer 当前实现情况**

DeepInfer 目前已有可复用的基础：

- `KvCacheState` 已有 `block_table`、`state_id`、`hit_tokens`、`swa_valid_start`、`set_num_tokens()` 等字段：[sequence.hpp:200](./DeepInfer/csrc/engine/sequence.hpp:200)。
- `BlockManager` 支持 block 分配、释放和 pin，但只有“按完整序列分配”的 `try_allocate_seq()`，没有 prefix-only 分配或 suffix 扩展接口：[block_manager.hpp:207](./DeepInfer/csrc/engine/block_manager.hpp:207)。
- 当前 `_schedule_prefill()` 对空 `block_table` 会按完整 prompt 预检查资源，并在真正调度时一次性分配整段序列 block：[scheduler.hpp:374](./DeepInfer/csrc/engine/scheduler.hpp:374)、[scheduler.hpp:517](./DeepInfer/csrc/engine/scheduler.hpp:517)。
- 已有 `schedulable` 标志和 `set_schedulable()`，目前主要被多模态异步处理使用：[scheduler.hpp:1054](./DeepInfer/csrc/engine/scheduler.hpp:1054)。
- `LLMEngine` 已有异步 D2D state copy 和 KV 保存钩子，但没有通用 KV pool connector、Store get worker、异步完成队列或远端 KV 状态：[llm_engine.py:787](./DeepInfer/deepinfer/engine/llm_engine.py:787)、[llm_engine.py:815](./DeepInfer/deepinfer/engine/llm_engine.py:815)。

所以 DeepInfer 的可行改造方向是：

- 新增 prefix-only block/state 分配；
- 新增 suffix block 扩展；
- 新增请求级异步 KV 状态；
- 新增后台 Store get/put worker；
- 在每次调度前消费完成队列；
- get 未完成时跳过该请求，完成后恢复为可调度状态；
- HBM 目标地址可复用 `gpu_blocks` / `gpu_states`，但需要新增注册、地址映射、生命周期 pin 和失败回滚逻辑。

**3. 草稿与实际 vLLM 逻辑的差异**

草稿中以下部分与 vLLM 基本一致：

- `WAITING_FOR_REMOTE_KVS` 语义；
- 异步 get 不阻塞其他请求；
- get 前只分配命中 prefix block；
- get 完成后再扩展 suffix block；
- 直接写 HBM，不经过 host staging；
- Store put 异步化；
- Store 失败按 miss 或部分命中处理。

需要修正或特别注意的地方：

1. 草稿写“请求留在 Scheduler `waiting`，用标志跳过”。
   vLLM 实际上使用独立的 blocked/deferred 状态和队列；`WAITING_FOR_REMOTE_KVS` 请求不会被普通 waiting 逻辑当作可调度请求处理。DeepInfer 当前 `_schedule_prefill()` 遇到 `schedulable=false` 会直接 `break`：[scheduler.hpp:369](./DeepInfer/csrc/engine/scheduler.hpp:369)。如果异步请求位于队首，会阻塞后续请求。因此不能只设置布尔标志，必须改成“跳过当前请求继续扫描”、独立队列，或调整 waiting 队列语义。

2. 草稿说“前缀匹配只在 Entrypoint 做”。
   这是目标架构选择，不是 vLLM 主线的固定实现。vLLM 的外部匹配入口实际是 scheduler 调用 connector 的 `get_num_new_matched_tokens()`；vllm-ascend connector 自己维护 lookup/load spec：[pool_scheduler.py:231](./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:231)。
   如果 DeepInfer 采用 Entrypoint 预匹配，需要把 `PrefixPlan` 作为明确的请求协议传入 Engine，并保证 hash、shard、block 边界和失效处理一致。

3. 草稿中的 `try_allocate_prefix()`、`extend_seq_blocks()`、`waiting_for_remote_kvs` 目前都不是 DeepInfer 主线已有接口或字段，而是待新增设计。

4. vllm-ascend 的异步 get 不是简单的“get 完成回调直接修改 Sequence”。后台线程只记录完成/失败，scheduler/worker 主线程在 step 边界应用状态。这一点应保留，避免后台线程直接操作 C++ scheduler 对象。

5. `Store.get()` 成功并不等于请求立即可执行。还需要完成 block/state 所有权确认、失败 block 处理、suffix block 扩展以及 GPU 写入可见性同步。

**总体判断**

DeepInfer 具备实现该功能所需的 block manager、state cache、GPU buffer 和 scheduler 基础，但当前仍是离线调度模型，没有 vLLM connector 对应的异步远端 KV 生命周期。最关键的结构性改造不是 Store API，而是：

- prefix-only allocation；
- async request state；
- 非阻塞 waiting 扫描；
- get 完成后的主线程状态交接；
- suffix allocation 与失败回滚；
- HBM block/state 的 pin/unpin 生命周期。

草稿的“两阶段 block 分配”和“异步直写 HBM”方向与 vllm-ascend 一致；“普通 waiting 队列中用标志跳过”需要结合 DeepInfer 当前 `break` 语义重新设计。

> 那当前的流程主要是在deepinfer的Scheduler里进行流程上的修改吗？

是，但不是只改 `DeepInfer Scheduler`。

Scheduler 是流程改造的核心，主要负责：

1. 请求首次进入时接收上游 `hit_tokens/PrefixPlan`。
2. 只分配命中的 prefix blocks 和 state slot。
3. 将请求置为异步加载状态，并跳过本次 prefill。
4. 后续每个 step 检查异步 get 是否完成。
5. 未完成时继续调度其他请求。
6. 完成后补充分配 suffix blocks，设置 `num_cached_tokens`，再进入 prefill。
7. 失败时释放或缩短有效命中范围，回退到本地计算。

但还需要配套修改以下模块：

| 模块 | 作用 |
|---|---|
| `scheduler.hpp` / `block_manager.hpp` | prefix-only 分配、suffix 扩展、异步状态、非阻塞队列扫描 |
| `sequence.hpp` / pybind | 增加 loading 状态、命中长度、失败状态等字段 |
| KV pool connector | 生成 load plan、记录 block 映射、提交异步 get |
| worker / transfer thread | 后台 Store `get()`，直接写入 `gpu_blocks` / `gpu_states` |
| `LLMEngine.step()` | 每步消费 get 完成队列，并在主线程更新 Scheduler |
| `CacheEngine` | 暴露 block/state 布局、地址和容量信息 |
| 保存逻辑 | 异步 Store `put()`，处理 pin/unpin 和生命周期 |

需要特别注意 DeepInfer 当前代码：

```cpp
if (!seq->schedule.schedulable) {
    break;
}
```

这意味着仅增加 `schedulable=false` 还不够，否则异步加载请求位于 `waiting` 队首时，会阻塞后续请求。需要把它改成跳过当前请求继续扫描，或者增加独立的异步 loading 队列。

因此可以概括为：

> **Scheduler 负责状态机和调度时机；KV connector/worker 负责异步 get/put；BlockManager 负责两阶段内存分配；Engine 负责在 step 边界完成状态交接。**

> 那请确认一下当前deepinfe的schedule为什么需要设计为waiting的堆头阻塞状态？那如果增加的修改了这个推头阻塞的机制，会不会对现有deepinfer的其他模块有影响？

<details><summary>2 previous messages</summary>

> 我会结合 DeepInfer 的调度循环、DP 投票和 prefill 队列语义确认这个 `break` 是否是有意的队首阻塞，以及改成可跳过后会影响哪些同步和资源假设。
>
> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Searched for `waiting|FCFS|队首|break|_dp_allreduce|ALL|ANY|prefill_attempt` in `scheduler.hpp`
> - Searched for `get_sum_prompt_len_waiting|waiting|set_schedulable|schedule\(` in `deepinfer`
>
> </details>
</details>

当前的队首阻塞是有意设计的，不是偶然实现。

**为什么要阻塞队首**

1. **FIFO 调度假设**

   `_schedule_prefill()` 默认 waiting 队列中的 `schedulable` 状态是单调的：

   ```cpp
   if (!seq->schedule.schedulable) {
       break;
   }
   ```

   代码注释明确说明：队首不可调度时，后续请求也被认为不可调度：[scheduler.hpp:270](./DeepInfer/csrc/engine/scheduler.hpp:270)。

   这保证了请求基本按进入顺序处理，不会因为后来的请求更容易调度而绕过前面的请求。

2. **避免资源检查顺序失真**

   `_schedule_prefill()` 会按照 waiting 队列顺序检查：

   - 最大 prompt 数；
   - KV block 是否足够；
   - prefill token budget；
   - chunked prefill 限制；
   - 多模态切分点。

   这些检查遇到不足时也会 `break`，例如 block 不足或 chunk 不能满足要求：[scheduler.hpp:386](./DeepInfer/csrc/engine/scheduler.hpp:386)、[scheduler.hpp:504](./DeepInfer/csrc/engine/scheduler.hpp:504)。

   也就是说，当前 scheduler 的整体策略是“队首请求当前无法调度时，等待下一轮资源变化”，而不是扫描后面的请求。

3. **保证 DP rank 调度一致**

   NORMAL 引擎会通过 `_dp_allreduce()` 决定这一轮全组做 prefill 还是 decode：[scheduler.hpp:295](./DeepInfer/csrc/engine/scheduler.hpp:295)。

   当前 `_has_prefill_waiting()` 只检查队首：

   ```cpp
   return !waiting.empty() && waiting.front()->schedule.schedulable;
   ```

   这是因为所有 rank 都采用同样的队首判断，能够保持投票语义一致。代码还通过第二轮投票处理“投票认为有 prefill，但实际一个请求也没排出来”的情况：[scheduler.hpp:328](./DeepInfer/csrc/engine/scheduler.hpp:328)。

4. **避免调度状态和队列状态错位**

   chunked prefill 未完成的请求会重新放回 waiting 队首：[scheduler.hpp:1012](./DeepInfer/csrc/engine/scheduler.hpp:1012)。抢占重算的请求也会放回队首：[scheduler.hpp:783](./DeepInfer/csrc/engine/scheduler.hpp:783)。

   这说明 waiting 队列不仅是普通请求队列，还承担了“下一步必须优先继续处理”的语义。

**修改队首阻塞会有什么影响**

会有影响，但影响集中在调度层和分布式步型一致性，不会自动影响所有模块。

主要风险如下：

| 影响点 | 风险 |
|---|---|
| FIFO 语义 | 后续请求可能绕过先到请求，改变公平性和延迟分布 |
| DP 投票 | `_has_prefill_waiting()` 不能再只看队首，否则可能错误进入 decode 或错误触发 prefill |
| 多 rank 步型 | 各 rank 扫描出的可调度请求数量和 token 数可能不同，导致 collective 输入不一致 |
| chunked prefill | 被跳过的请求和续跑请求可能失去原来的优先级 |
| `get_sum_prompt_len_waiting()` | 统计仍会包含异步 loading 请求，可能导致 pending 数和实际可调度数不一致 |
| abort/清理 | 当前 decode 分支会扫描 waiting 清理已结束请求；新增状态也必须纳入清理 |
| block 资源 | 异步请求已经占用 prefix blocks，后续请求继续分配可能增加内存压力和抢占概率 |
| 多模态异步 | 当前 `schedulable=false` 已被多模态处理使用，不能简单改变它的全局含义 |

**推荐的改法**

不要直接把现有：

```cpp
if (!seq->schedule.schedulable) {
    break;
}
```

改成无条件 `continue`。

更稳妥的方式是引入独立的异步 KV 状态或队列：

```text
waiting
  ├─ 普通可调度请求
  ├─ WAITING_FOR_REMOTE_KVS 请求
  └─ 多模态异步等待请求
```

每轮调度前：

1. 消费异步 get 完成队列；
2. 成功的请求补齐 suffix block，并重新放回 waiting；
3. 失败的请求回滚或缩短命中范围；
4. 普通 waiting 仍保持原有队首 FIFO 逻辑；
5. `_has_prefill_waiting()` 改为检查“是否存在可调度的普通请求”，并重新验证 DP 投票语义。

这样可以把异步 KV 请求从原有 FIFO 队列中隔离出来，降低对现有 chunked prefill、抢占、MTP 和多模态流程的影响。

> 那这部分是不是deepinfer的独有逻辑？vllm就不需要这方面的考量？

不是 vLLM 完全不需要考虑，而是 **vLLM 已经把这些情况抽象进了自己的队列和状态机**；DeepInfer 当前把这些语义压缩在一个 `waiting` 队列中，所以看起来更容易出现队首阻塞问题。

**vLLM 的处理方式**

vLLM 中异步 KV 请求不会和普通可调度请求简单混在一起：

- `WAITING_FOR_REMOTE_KVS` 属于 blocked 状态：[scheduler.py:2755](./vllm/vllm/v1/core/sched/scheduler.py:2755)。
- 请求会被放入不同队列，例如：
  - `waiting`
  - `async_kv_waiting`
  - `kv_holding_waiting`
  - `deferred_waiting`

  队列选择逻辑见：[scheduler.py:2776](./vllm/vllm/v1/core/sched/scheduler.py:2776)。

- 调度时，vLLM 可以通过 `skip_request()` 跳过当前不能调度的请求，继续检查其他请求。
- 对于 `WAITING_FOR_REMOTE_KVS`，每轮只检查异步 get 是否完成；未完成就跳过，完成后恢复为 `WAITING`：[scheduler.py:3671](./vllm/vllm/v1/core/sched/scheduler.py:3671)。

所以 vLLM 也需要处理：

- 异步加载期间不能执行 prefill；
- 不能阻塞其他请求；
- block 资源已被部分占用；
- get 完成后才能恢复调度；
- 加载失败后的 block 回滚；
- DP 场景下的步型同步。

只是这些逻辑已经由独立状态、多个 waiting queue 和 connector 接口承担了。

**DeepInfer 的特殊点**

DeepInfer 的特殊之处在于：

1. 当前只有一个主要 `waiting` 队列。
2. `_schedule_prefill()` 默认队首不可调度时直接 `break`。
3. `_has_prefill_waiting()` 只看队首。
4. NORMAL 引擎还叠加了 DeepInfer 自己的 DP `ALL/ANY` prefill 投票机制：[scheduler.hpp:295](./DeepInfer/csrc/engine/scheduler.hpp:295)。

因此，DeepInfer 不能直接照搬 vllm-ascend 的：

```text
请求置为 WAITING_FOR_REMOTE_KVS
后台 get
完成后恢复
```

而必须先决定异步请求如何脱离现有队首 FIFO 语义。

**准确的对比是：**

| 问题 | vLLM | DeepInfer |
|---|---|---|
| 异步请求状态 | 已有 `WAITING_FOR_REMOTE_KVS` | 需要新增 |
| 异步请求队列 | 有独立 waiting/deferred/async 队列 | 当前主要是单一 waiting |
| 队首阻塞 | 普通队列仍可能有队首约束，但可通过多个队列和 skip 规避 | `waiting.front()` 不可调度直接 `break` |
| DP 同步 | 有自己的 DP 调度逻辑 | 有 DeepInfer 特有的 ALL/ANY 投票 |
| suffix block 扩展 | KV cache manager 已支持外部 block 分配模型 | 需要新增 prefix-only allocation 和 suffix 扩展 |
| 完成处理 | connector 输出驱动 scheduler 状态转换 | 需要新增 Engine/Scheduler 完成队列 |

所以这不是 DeepInfer 独有的业务问题，而是 **DeepInfer 当前调度器尚未具备 vLLM 已有的异步 KV 调度抽象**。真正需要适配 DeepInfer 独有逻辑的部分，主要是：

- C++ `waiting` 队列的队首阻塞语义；
- `ALL/ANY` DP 投票；
- DeepInfer 的 chunked prefill、MTP 和抢占队列；
- 当前 `schedulable` 标志被多模态异步流程复用的情况。

> 那想确认一下vllm的流程，比如说add-request之后，请求进入waiting队列，那是什么时候判断有前缀匹配的结果并进行异步get的？是在add进入waiting队列的时候就开始了，还是说在调度这个请求到了对头准备进行prefill的时候发现再进行异步调度，然后此次step会跳过这个，进行后续的可执行队列调度任务？

<details><summary>2 previous messages</summary>

> 我会把“入队、前缀查询、block 分配、异步 get 提交、后续请求继续调度”这几个时点拆开，避免把 scheduler admission 和 worker load 的时机混在一起。
>
> <details><summary>Explored a file, 2 searches</summary>
>
> - Searched for `def add_request|_enqueue_waiting_request|add_request\(` in `scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Searched for `build_connector_meta|start_load_kv|kv_connector_metadata` in `v1`
>
> </details>
</details>

准确流程是：**不是在 `add_request()` 入队时就开始前缀匹配和异步 get，而是在调度器实际处理该请求、准备为它安排 prefill 时才进行。**

时间线如下：

```text
add_request
  -> on_new_request()
  -> 放入 waiting / async_kv_waiting
  -> 本阶段通常不执行 Store lookup，也不启动 get

某个 schedule step
  -> 取出待处理请求
  -> 判断 num_computed_tokens == 0
  -> 查询本地 prefix cache
  -> 调用 connector.get_num_new_matched_tokens()
  -> 得到外部命中长度和 load_kv_async
  -> 分配命中的外部 KV blocks
  -> 当前请求本 step 不执行 prefill
  -> 提交 connector metadata

worker 处理本 step metadata
  -> start_load_kv()
  -> 后台线程提交 Store.get()
  -> scheduler 继续调度其他请求

后续 step
  -> 检查 get 是否完成
  -> 未完成：继续跳过
  -> 已完成：恢复请求为 WAITING
  -> 下一次调度时执行 suffix prefill
```

**1. `add_request()` 阶段只做入队和 connector 初始化**

vLLM 的 `add_request()` 主要调用：

```python
self.connector.on_new_request(request)
self._enqueue_waiting_request(request)
```

见：[scheduler.py:3080](./vllm/vllm/v1/core/sched/scheduler.py:3080)。

`on_new_request()` 可以做请求记录、hash 初始化、决定进入哪个等待队列，但不会根据完整调度上下文分配 block，也不会启动本次请求的 Store get。

**2. 前缀匹配发生在调度循环中**

当 scheduler 处理一个尚未计算 token 的请求时：

```python
if request.num_computed_tokens == 0:
```

它先查询本地 prefix cache，然后调用：

```python
self.connector.get_num_new_matched_tokens(...)
```

见：[scheduler.py:1148](./vllm/vllm/v1/core/sched/scheduler.py:1148)、[scheduler.py:1168](./vllm/vllm/v1/core/sched/scheduler.py:1168)。

因此，外部前缀匹配是在“请求准备被调度”时进行的，而不是入队瞬间进行的。

**3. 异步命中时本 step 不执行这个请求的 prefill**

如果 connector 返回 `load_kv_async=True`：

```python
num_new_tokens = 0
```

表示该请求本 step 不计算新 token，只为外部命中的 KV 准备 block 和加载计划：[scheduler.py:1336](./vllm/vllm/v1/core/sched/scheduler.py:1336)。

之后 scheduler 将请求设置为：

```python
request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
```

并跳过当前请求：[scheduler.py:1576](./vllm/vllm/v1/core/sched/scheduler.py:1576)。

**4. 其他请求可以继续调度**

这里的“跳过”不是结束整个 schedule step。vLLM 会把当前请求放回对应的等待队列，继续处理其他请求；本轮结束时再恢复被跳过的请求：[scheduler.py:1675](./vllm/vllm/v1/core/sched/scheduler.py:1675)。

异步 Store get 的真正提交发生在 scheduler 生成 connector metadata 之后，由 worker 的：

```python
start_load_kv(metadata)
```

处理；vllm-ascend 中该方法把请求放入 `KVCacheStoreRecvingThread`：[pool_worker.py:1345](./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1345)、[pool_worker.py:1380](./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1380)。

所以更精确地说：

> **调度时发现命中并创建异步加载计划；当前请求本 step 不执行 prefill；scheduler 继续调度其他请求；worker 在本 step 的 connector metadata 阶段启动后台 Store get。**

另外，vLLM 还支持 `defer_async_loads`，在某些异步调度场景下会把加载计划延迟到后续 step，但这不改变总体语义：[scheduler.py:1530](./vllm/vllm/v1/core/sched/scheduler.py:1530)。

> 请详细介绍一下vllm调度器有哪些队列，以及这些队列在有前缀匹配过程中，队列的流程。

<details><summary>1 previous message</summary>

> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Searched for `self\.(waiting|async_kv_waiting|kv_holding_waiting|deferred_waiting|running)|class Scheduling|RequestQueue|waiting =|async_kv` in `scheduler.py`
> - Searched for `class RequestQueue|def skip_request|def prepend_requests|def pop_request|class SchedulingBudget` in `core`
>
> </details>
</details>

vLLM 的请求并不是始终只存在于一个 `waiting` 队列中。它根据请求是否持有 KV block、是否可能进行异步 KV load、是否暂时被跳过等情况，在多个队列之间流转。

## 1. 主要队列和集合

Scheduler 初始化了这些结构：

```python
self.waiting
self.kv_holding_waiting
self.async_kv_waiting
self.deferred_waiting
self.running
```

定义位置：[scheduler.py:229](./vllm/vllm/v1/core/sched/scheduler.py:229)。

### `waiting`

普通等待队列，保存：

- 新到达、尚未开始 prefill 的请求；
- 没有占用 KV block 的普通请求；
- prefix lookup 后决定本地计算或同步加载的请求；
- 异步 KV 请求加载完成后重新恢复的请求。

它使用配置的调度策略，通常是 FCFS，也可以是 priority queue。

### `async_kv_waiting`

可能进行异步 KV load、但尚未开始 load 的请求。

只有启用了 `max_num_inflight_kv_loads`，且 connector 判断该请求可能支持异步加载时，才会进入这个队列：

```python
elif (
    self.connector is not None
    and self.scheduler_config.max_num_inflight_kv_loads
    and self.connector.may_load_async(request)
):
    self.async_kv_waiting.add_request(request)
```

见：[scheduler.py:2776](./vllm/vllm/v1/core/sched/scheduler.py:2776)。

这个队列的意义是：即使当前 running batch 或 token budget 已满，也可以继续启动异步 KV load，因为异步 load 本身不需要立即占用 running slot。

### `kv_holding_waiting`

已经持有 KV blocks，但当前还不能进入 running 的请求。

典型情况就是：

```text
prefix blocks 已分配
Store.get 仍在进行
请求状态为 WAITING_FOR_REMOTE_KVS
```

vLLM 对这类请求设置了较高优先级：

```python
# Waiting requests that hold KV blocks are always drained first.
```

见：[scheduler.py:231](./vllm/vllm/v1/core/sched/scheduler.py:231)。

原因是它们已经占用 KV cache 资源，必须优先处理，否则可能出现：

```text
异步请求占用 block
-> 后续请求继续分配 block
-> KV cache 资源耗尽
-> 异步请求无法继续恢复或释放
```

### `deferred_waiting`

这是一个 `set`，不是实际的 FIFO 队列。

它记录本轮调度中被暂时跳过的请求，例如：

- 异步 get 尚未完成；
- encoder 输入尚未就绪；
- LoRA 数量超限；
- stale output 尚未清理；
- 当前 step token budget 不足。

本轮结束后，真正的请求会被重新放回对应队列；`deferred_waiting` 主要用于状态和容量统计。

### `running`

已经进入当前运行 batch 的请求。

这些请求通常执行：

- decode；
- chunked prefill 的后续 chunk；
- 已经完成 prefix KV load 后的 suffix prefill。

调度器首先处理 `running`，再处理等待队列：[scheduler.py:809](./vllm/vllm/v1/core/sched/scheduler.py:809)。

---

## 2. 新请求进入哪个队列

调用 `add_request()` 时，vLLM 不会立刻执行 prefix match，也不会立刻启动 Store get。

流程是：

```python
connector.on_new_request(request)
_enqueue_waiting_request(request)
```

见：[scheduler.py:3080](./vllm/vllm/v1/core/sched/scheduler.py:3080)。

`_enqueue_waiting_request()` 根据请求状态和 connector 能力选择队列：

```python
if request 持有 KV blocks:
    kv_holding_waiting

elif connector 允许异步 load:
    async_kv_waiting

else:
    waiting
```

所以，新请求可能一开始就进入 `async_kv_waiting`，但这只是“它具备异步 load 资格”，并不代表已经完成 prefix match 或已经启动 get。

---

## 3. 调度器如何选择 waiting 队列

调度器处理等待请求时，顺序大致是：

```text
1. kv_holding_waiting
2. waiting 和 async_kv_waiting 中按 arrival_time/policy 选择
3. 如果 running slot 或 token budget 已满，只处理 async_kv_waiting
```

相关逻辑：[scheduler.py:1091](./vllm/vllm/v1/core/sched/scheduler.py:1091)。

普通情况下：

```python
request_queue = self.kv_holding_waiting

if not request_queue:
    request_queue = self._next_waiting_queue(take_async_kv)
```

当没有可用 running slot 或 token budget 已用完时：

```python
request_queue = self.async_kv_waiting
```

这实现了一个重要优化：

> 当前 batch 已经满了，也可以继续启动远端 KV load，不必等已有 batch 完成。

`waiting` 和 `async_kv_waiting` 之间的选择由 `_next_waiting_queue()` 决定，通常按请求策略或 arrival time 选择较早的请求：[scheduler.py:2830](./vllm/vllm/v1/core/sched/scheduler.py:2830)。

---

## 4. 前缀匹配发生在哪一步

请求从 `waiting` 或 `async_kv_waiting` 中取出后，scheduler 才开始判断 prefix cache：

```python
if request.num_computed_tokens == 0:
    ...
    self._get_local_prefix_cache_hit(...)
    self.connector.get_num_new_matched_tokens(...)
```

见：[scheduler.py:1148](./vllm/vllm/v1/core/sched/scheduler.py:1148)。

这里会同时判断：

1. 本地 vLLM prefix cache 命中；
2. 外部 KV connector 命中；
3. 命中的 token 数；
4. 是否需要异步加载。

因此 prefix match 发生在“该请求实际被 scheduler 取出来准备调度”时。

---

## 5. 发生异步 prefix load 时的队列流转

假设新请求 `R1` 在 `async_kv_waiting` 中被取出，并发现远端命中 1024 tokens。

### 第一步：查询并计算命中范围

```text
async_kv_waiting
    -> connector.get_num_new_matched_tokens()
    -> 返回 ext_tokens=1024, load_kv_async=True
```

### 第二步：只分配外部命中的 block

scheduler 计算：

```python
num_new_tokens = 0
```

也就是说当前 step 不计算 suffix，只为外部 KV 命中的 block 分配空间。

### 第三步：设置异步等待状态

请求变为：

```python
request.status = WAITING_FOR_REMOTE_KVS
```

见：[scheduler.py:1576](./vllm/vllm/v1/core/sched/scheduler.py:1576)。

请求不会进入 `running`，本 step 不执行 prefill。

### 第四步：移动到 `kv_holding_waiting`

因为该请求已经拥有 KV blocks，调度循环结束后会把它加入 `kv_holding_waiting`：

```python
for request in step_started_kv_loads:
    self.kv_holding_waiting.add_request(request)
```

见：[scheduler.py:1675](./vllm/vllm/v1/core/sched/scheduler.py:1675)。

此时状态是：

```text
kv_holding_waiting
    R1: WAITING_FOR_REMOTE_KVS
```

### 第五步：worker 启动后台 Store.get

scheduler 构造 connector metadata 后，worker 调用：

```python
start_load_kv(metadata)
```

vllm-ascend 中会把请求放入 `KVCacheStoreRecvingThread`：

```python
self.kv_recv_thread.add_request(request)
```

见：[pool_worker.py:1345](./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1345)、[pool_worker.py:1380](./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1380)。

后台线程调用 Store `get()`，将数据写入已经分配的目标 KV blocks。

---

## 6. get 未完成时

下一次 scheduler 处理 `kv_holding_waiting` 的队首请求时，会调用：

```python
_handle_blocked_waiting_request(request)
```

如果请求仍处于：

```python
WAITING_FOR_REMOTE_KVS
```

并且没有出现在完成集合中，则：

```text
本轮跳过 R1
R1 加入 deferred_waiting
其他请求继续调度
```

对应逻辑：[scheduler.py:3671](./vllm/vllm/v1/core/sched/scheduler.py:3671)。

本轮结束后，R1 被重新放回 `kv_holding_waiting`，不会丢失它已经持有的 block。

---

## 7. get 完成时

当 worker 报告 R1 的异步 get 已完成：

```text
kv_holding_waiting
    -> get_finished
    -> _update_waiting_for_remote_kv
```

`_update_waiting_for_remote_kv()` 会：

1. 将已经成功加载的 block 标记为有效；
2. 处理失败 block；
3. 必要时释放无效 block；
4. 更新 `num_computed_tokens`；
5. 将状态从 `WAITING_FOR_REMOTE_KVS` 恢复为 `WAITING`。

见：[scheduler.py:3616](./vllm/vllm/v1/core/sched/scheduler.py:3616)。

恢复后的请求会继续进入当前调度流程，或者在后续 step 中从 `waiting` 重新被取出。此时 scheduler 才会为剩余 suffix 分配 block，并执行 prefill。

完整状态转移可以表示为：

```text
add_request
    -> waiting / async_kv_waiting
    -> prefix lookup
    -> allocate external prefix blocks
    -> WAITING_FOR_REMOTE_KVS
    -> kv_holding_waiting
    -> Store.get

Store.get 未完成:
    kv_holding_waiting
        -> deferred_waiting
        -> kv_holding_waiting

Store.get 完成:
    kv_holding_waiting
        -> restore WAITING
        -> waiting
        -> allocate suffix blocks
        -> prefill
        -> running
```

## 8. 一个重要细节：vLLM 也不是所有请求都能无限跳过队首

vLLM 的“可跳过”是通过多个队列和 `skip_request()` 实现的，并不是所有约束都简单 `continue`。

例如：

- `async_kv_waiting` 可以在无 token budget 时继续启动异步 load；
- `kv_holding_waiting` 优先于普通 waiting；
- 某些资源不足时，会停止扫描当前队列；
- 被跳过的请求会暂存，step 结束后按原队列重新插入。

`skip_request()` 的实现：[scheduler.py:1076](./vllm/vllm/v1/core/sched/scheduler.py:1076)。

因此，vLLM 的核心不是“完全取消队首约束”，而是：

> **把不同资源状态的请求拆到不同队列中，再针对每个队列定义不同的阻塞和跳过规则。**

这也是 DeepInfer 如果增加异步 prefix KV load 时，最值得借鉴的部分。

> 前缀匹配的结果应该是在add之前，上游已经前缀匹配计算了吧？

要区分两种“前缀匹配结果”。

## 1. 如果采用你们需求中的上游 PrefixPlan

如果上游 Router/Entrypoint 已经完成：

```text
token prefix hash
-> Store metadata 查询
-> 得到 hit_tokens / block keys / state 信息
-> 把 PrefixPlan 随请求传给 DeepInfer
```

那么这个结果确实应该在 `add_request()` 之前或随 `add_request()` 一起提供。

此时 DeepInfer 的流程可以是：

```text
上游完成 PrefixPlan
    -> add_request(request, prefix_plan)
    -> 请求进入等待队列
    -> scheduler 调度到该请求
    -> 根据 prefix_plan 分配命中 blocks/state
    -> 提交异步 get
    -> 当前请求跳过 prefill
```

也就是说：

- **匹配计算**可以在 add 之前完成；
- **HBM block 分配和异步 get 启动**仍建议在 scheduler 正式接纳请求时进行。

不能在 add 时直接启动 get，除非同时解决：

- HBM block 是否已经分配；
- 请求是否被取消；
- 请求是否因为资源限制长期无法调度；
- Store 元数据在匹配后是否已经失效；
- block 生命周期如何 pin/unpin；
- DP rank 是否都准备好；
- 请求是否允许占用异步 load 配额。

## 2. vLLM 当前代码不完全是这种模式

vLLM 主线的普通 KV connector 流程中，前缀命中判断通常是在 scheduler 调度请求时调用：

```python
self.connector.get_num_new_matched_tokens(...)
```

位置：[scheduler.py:1168](./vllm/vllm/v1/core/sched/scheduler.py:1168)。

因此 vLLM 的典型流程是：

```text
add_request
    -> waiting / async_kv_waiting
    -> scheduler 取出请求
    -> connector 做 external prefix lookup
    -> 决定命中长度和是否异步 load
```

`add_request()` 中的：

```python
self.connector.on_new_request(request)
```

主要是登记请求、初始化 connector 状态，不等于已经完成 prefix match：[scheduler.py:3080](./vllm/vllm/v1/core/sched/scheduler.py:3080)。

另外，vLLM 中请求携带的 `kv_transfer_params`，例如 PD 分离的远端 prefill 信息，也不一定等同于 Ascend Store 的 prefix pool 命中结果。不能把这两种信息混为一谈。

## 3. 对 DeepInfer 的建议

如果 DeepInfer 的明确前提是：

> 上游已经完成前缀匹配，并把命中结果提供给 Engine

那么可以设计成：

```text
PrefixPlan:
    request_id
    hit_tokens
    block_keys / block references
    state reference
    layout/version
    expiry or epoch

add_request(request, prefix_plan)
    -> 保存 PrefixPlan
    -> 放入等待队列
```

但在 scheduler 调度到该请求时，仍然需要执行：

```text
1. 校验 PrefixPlan 是否有效
2. 分配命中 prefix blocks/state
3. 建立 request -> local block 的映射
4. 提交异步 get
5. 设置 WAITING_FOR_REMOTE_KVS
6. 本轮不执行该请求的 prefill
```

因此最终结论是：

> **如果是你们的上游匹配架构，匹配结果可以在 `add_request()` 前获得；但 vLLM 当前 connector 的匹配通常发生在 scheduler 调度阶段。无论匹配何时完成，异步 get 的启动都应绑定到 scheduler 的资源接纳和 block 分配阶段。**

> vllm为什么要这样设计这么多队列，不能都知道waiting队列里，如果遇到这种不同异步get状态的，就跳后从后面的中取请求任务吗？

可以做成一个 `waiting` 队列，然后遇到不可执行请求就继续扫描后面的请求，但这会带来几个问题。vLLM 选择多个队列，是把这些资源和优先级约束显式化。

## 1. 不同请求的资源状态不同

普通等待请求通常还没有占用 KV block：

```text
waiting
    -> 等待 block
    -> 可以直接被调度或继续等待
```

异步 KV 请求则可能已经占用了部分 block：

```text
kv_holding_waiting
    -> 已经占用 prefix blocks
    -> 正在等待 Store.get
```

这两种请求不能使用同一种调度优先级。

如果异步请求已经占用 block，却让后面的普通请求不断分配剩余 block，可能出现：

```text
异步请求占用部分 block
后续请求持续分配
KV cache 被耗尽
异步请求无法恢复或释放
```

所以 vLLM 让 `kv_holding_waiting` 优先处理。

## 2. KV load 可以在 running slot 满时继续启动

这是 `async_kv_waiting` 单独存在的主要原因。

假设：

```text
running batch 已满
token budget 已用完
```

普通请求不能再进入当前 batch，但异步 Store.get 不需要立即执行模型计算，因此仍然可以启动：

```text
async_kv_waiting
    -> 启动 Store.get
    -> 不占用 running slot
    -> 等待下一步使用已加载 KV
```

如果只有一个 waiting 队列，扫描逻辑中就必须额外判断：

```text
当前请求需要计算，不能放入
当前请求只需要异步 load，可以放入
```

这实际上就是在一个队列里重新实现 `async_kv_waiting` 的语义。

## 3. 简单跳过会破坏顺序和公平性

如果只有一个队列：

```text
waiting = [R1, R2, R3]
```

R1 暂时不能调度，直接跳到 R2、R3，后续需要决定：

- R1 是否仍保持原顺序；
- R2、R3 是否可以连续多轮绕过 R1；
- R1 恢复后放到队首还是队尾；
- 被跳过请求是否会永久饥饿；
- 优先级调度时是否还能保持正确顺序。

vLLM 通过“临时跳过，step 结束后重新插回原队列”处理这个问题：

```python
step_skipped_waiting
step_skipped_async_kv
step_skipped_kv_holding
```

见：[scheduler.py:1070](./vllm/vllm/v1/core/sched/scheduler.py:1070) 和 [scheduler.py:1675](./vllm/vllm/v1/core/sched/scheduler.py:1675)。

## 4. 单队列扫描会增加复杂的组合条件

一个统一队列至少要同时处理：

- 是否持有 KV blocks；
- 是否正在异步 get；
- 是否可以启动新的异步 get；
- 是否已经完成 get；
- 是否占用 running slot；
- 是否满足 token budget；
- 是否受 pause 状态限制；
- 是否因为 LoRA、encoder、structured output 而暂时不可调度；
- 是否需要优先处理已持有 block 的请求。

最终会变成类似：

```python
for request in waiting:
    if request.holds_blocks:
        ...
    elif request.can_start_async_load:
        ...
    elif running_full:
        ...
    elif token_budget_insufficient:
        ...
```

这并不会减少状态复杂度，只是把多个队列转换成一个带大量分支的扫描器。

## 5. 多队列还能表达明确的优先级

vLLM 的优先级大致是：

```text
kv_holding_waiting
    > 可启动的 async_kv_waiting / waiting
    > 暂时 deferred 的请求
```

同时 `waiting` 和 `async_kv_waiting` 之间还会根据 arrival time 或调度策略选择队首：[scheduler.py:2830](./vllm/vllm/v1/core/sched/scheduler.py:2830)。

这比在一个队列里扫描每个请求的状态更容易验证。

## 6. 能不能只用一个 waiting 队列？

可以，但需要同时增加：

```text
请求状态字段
跳过后的临时列表
已持有 block 的优先级
异步 load 配额
本轮是否只允许启动 async load
恢复时的插入策略
取消和失败时的统一清理
```

最终代码结构通常会变成：

```text
一个 waiting 队列
+ 多种 request state
+ 多个临时 skipped 列表
+ 多套扫描和重排规则
```

从逻辑上看，它等价于 vLLM 的多个队列，只是边界更加隐含。

## 对 DeepInfer 的启示

DeepInfer 不一定要完全复制 vLLM 的四个队列，但至少需要区分这三类状态：

```text
1. 普通 waiting：
   没有加载任务，可以执行 prefill

2. KV loading：
   已分配 prefix blocks，等待异步 get

3. KV ready：
   get 已完成，等待 suffix 分配和 prefill
```

如果只保留一个 `waiting` 队列，至少要保证：

- KV loading 请求不会阻塞其他请求；
- 已经占用 block 的请求不会无限期滞留；
- get 完成请求优先被恢复；
- 异步 load 可以在 batch 满时继续启动；
- DP rank 的调度结果仍保持一致；
- 请求取消和 Store 失败时能正确释放 block。

所以，vLLM 的多队列不是因为单队列绝对不可行，而是因为它把不同资源生命周期和调度优先级显式分开，避免一个统一扫描器承担所有状态组合。

> 如果说，前缀匹配的结果是上游给入到prefill之前的调度，那请结合deepinfer已有的调度策略，借鉴vllm成熟的异步get操作的合理方案

如果前缀命中结果由上游在 `add_request()` 之前提供，那么 DeepInfer 不需要在 Scheduler 内重新做 hash 或 Store metadata 查询。推荐把问题拆成：

```text
上游完成匹配
    -> DeepInfer 接收 PrefixPlan
    -> Scheduler 负责资源接纳和状态流转
    -> Worker 负责异步 Store.get
```

重点是：**前缀匹配可以提前完成，但 prefix block 的 HBM 分配和异步 get 启动仍然要由 Scheduler 控制。**

## 一、推荐的队列结构

尽量保留 DeepInfer 当前 `waiting` 的语义，不直接修改现有队首阻塞逻辑。

新增两个与 KV pool 相关的队列或状态：

```text
waiting
    普通等待请求，或者已经准备好执行 prefill 的请求

kv_loading
    已经分配 prefix blocks，正在异步 Store.get

kv_ready
    Store.get 已完成，等待 suffix block 分配并执行 prefill
```

现有队列职责：

```text
waiting      -> 普通 prefill/FIFO 语义
kv_loading   -> 类似 vLLM 的 kv_holding_waiting
kv_ready     -> 已完成异步加载，优先恢复
running      -> 已经进入模型执行 batch
```

不建议复用现有的 `schedulable` 表示 KV loading，因为这个字段已经被多模态异步流程使用。建议增加独立状态，例如：

```cpp
enum class KVLoadState {
    NONE,
    WAITING_TO_START,
    IN_PROGRESS,
    READY,
    FAILED,
};
```

或者增加 `SequenceStatus::WAITING_FOR_REMOTE_KVS`，再配合独立的 KV 队列。

## 二、请求进入 DeepInfer 时

上游传入：

```text
PrefixPlan {
    request_id
    hit_tokens
    block_keys
    state_key / state reference
    block_size
    layout_version
    model/version information
    plan_epoch or expiration information
}
```

`add_request()` 阶段只做：

```text
1. 保存 PrefixPlan
2. 校验 request_id 和基本 layout 信息
3. 如果 hit_tokens == 0，直接进入普通 waiting
4. 如果 hit_tokens > 0，进入 waiting 或 pending-load 状态
```

此阶段不建议立即：

- 分配 GPU blocks；
- 启动 Store.get；
- 修改 `seq.cache.num_tokens`；
- 把请求放入 running。

原因是请求可能还没有被 Scheduler 接纳，且可能被取消、超过容量限制或等待其他资源。

## 三、Scheduler 每个 step 的处理顺序

建议把 DeepInfer 的 `_schedule()` 拆成四个逻辑阶段。

### 阶段 1：消费异步 get 完成结果

在真正决定本轮是 prefill 还是 decode 之前，先从 worker 拉取完成事件：

```text
poll_kv_load_completions()
```

对于每个完成请求：

```text
Store.get 成功
    -> 校验加载 block
    -> KVLoadState = READY
    -> 放入 kv_ready

Store.get 失败
    -> 记录失败 block
    -> 释放 prefix blocks/state
    -> 清除 PrefixPlan
    -> 回退到普通 prefill
```

后台线程不应直接修改 C++ `Sequence` 或 `Scheduler`，而是通过完成队列在 Engine/Scheduler 主线程应用结果。

### 阶段 2：启动新的异步 prefix load

从普通 `waiting` 中选择带有 PrefixPlan 的请求。

需要检查：

```text
1. prefix blocks 是否有足够 HBM 空间
2. state slot 是否可用
3. 当前异步 load 数是否超过上限
4. 请求是否已取消
5. PrefixPlan layout 是否匹配当前 engine
```

资源足够时：

```text
1. 分配 hit_tokens 对应的 prefix blocks
2. 分配 state slot
3. 将 block IDs 和 PrefixPlan 组成 KVLoadTask
4. seq.cache.hit_tokens = hit_tokens
5. KVLoadState = IN_PROGRESS
6. 从 waiting 移到 kv_loading
7. 不执行该请求的 prefill
```

注意此时不要把 `seq.cache.num_tokens` 直接当作已完成 token。异步 get 尚未完成，KV 内容还不能使用。

### 阶段 3：继续处理其他请求

启动异步 get 的请求不能阻塞普通请求。

因此：

```text
R1: prefix load in progress
R2: ordinary prefill
R3: ordinary decode
```

本轮可以继续调度 R2/R3。

这和 vLLM 的核心语义一致：

```text
异步 KV load 不占用模型计算 token budget
但会占用部分 KV cache block
```

### 阶段 4：调度 kv_ready 请求

`kv_ready` 请求应优先于普通新请求处理，因为它已经占用了 prefix blocks。

处理流程：

```text
kv_ready
    -> 检查 prefix blocks 已经可见
    -> 扩充分配 suffix blocks
    -> 设置 num_cached_tokens = hit_tokens
    -> 设置 swa_valid_start 等状态
    -> 清除 KVLoadState
    -> 执行剩余 suffix prefill
```

对应草稿中的：

```text
try_allocate_prefix(hit)
    -> async get
    -> extend_seq_blocks(full_prompt)
    -> suffix prefill
```

这部分应该是 DeepInfer 新增的核心接口。

## 四、建议的调度时序

完整流程可以表示为：

```text
上游 PrefixMatcher
    -> PrefixPlan(hit_tokens, block_keys)
    -> add_request()

DeepInfer waiting
    -> Scheduler 选择请求
    -> 分配命中 prefix blocks/state
    -> 请求进入 kv_loading
    -> 生成 KVLoadTask
    -> Worker 后台 Store.get
    -> Scheduler 继续处理其他请求

下一 step:
    -> poll completion

get 成功:
    kv_loading
        -> kv_ready
        -> 扩展 suffix blocks
        -> 设置 num_cached_tokens
        -> suffix prefill
        -> running

get 失败:
    kv_loading
        -> 释放已分配资源
        -> hit_tokens = 0 或缩短到有效前缀
        -> 普通 prefill
```

## 五、与 DeepInfer 现有 NORMAL/DP 调度的结合

DeepInfer 的 NORMAL 引擎有自己的 ALL/ANY prefill 投票：

```cpp
_dp_allreduce(...)
```

见：[scheduler.hpp:295](./DeepInfer/csrc/engine/scheduler.hpp:295)。

这里需要区分两种操作：

### 异步 get 启动

异步 get 不执行模型 forward，因此不应该强制把本轮变成 prompt step，也不应该清零 `decode_step`。

更合理的方式是：

```text
Scheduler 先处理 KV load admission
    -> 产生 KVLoadTask
    -> 本轮仍按原逻辑决定 decode/prefill
```

这样异步 get 可以和当前 decode 或其他 prefill 重叠。

### suffix prefill

只有 `kv_ready` 请求真正进入 suffix prefill 时，才参与原有的 ALL/ANY prefill 投票。

因此：

```text
KV loading:
    不参与 prompt token 计算，不改变 prefill/decode 步型

KV ready:
    作为普通 prefill 请求参与现有 DP 投票
```

如果各 DP rank 的 prefix plan 不完全一致，需要在上游或调度前统一 hit length。至少要保证同一个请求在不同 rank 上：

```text
hit_tokens 一致
suffix 起点一致
prefill token 数一致
```

否则可能导致不同 rank 的 forward shape 不一致。

## 六、DeepInfer 需要新增或修改的接口

### Scheduler

建议增加：

```cpp
bool try_allocate_prefix(Sequence &seq, int hit_tokens);
bool extend_seq_blocks(Sequence &seq);
void mark_kv_load_started(request_id, load_id);
void mark_kv_load_finished(request_id, bool success, int valid_tokens);
void mark_kv_load_failed(request_id);
```

同时增加：

```cpp
std::deque<SequencePtr> kv_loading;
std::deque<SequencePtr> kv_ready;
```

或者使用一个 `kv_loading` 队列加完成后的 ready list。

### Sequence

建议增加：

```cpp
enum class KVLoadState {
    NONE,
    WAITING_TO_START,
    IN_PROGRESS,
    READY,
    FAILED,
};

int prefix_hit_tokens;
int kv_load_id;
KVLoadState kv_load_state;
```

现有的：

```cpp
schedulable
```

不要直接复用来表示 KV load。

### SchedulerOutputs

增加异步加载任务信息：

```text
request_id
local block ids
hit_tokens
block keys
state id
layout/version
```

这些信息由 Python/worker 层提交给后台 Store worker。

### LLMEngine

`step()` 的前后顺序建议是：

```text
1. poll KV load completion
2. 更新 Scheduler 中的 kv_ready / failed 请求
3. Scheduler 进行 load admission
4. Scheduler 执行原有 decode/prefill 调度
5. Worker 提交新的 KVLoadTask
6. 执行本轮模型计算
```

如果当前运行时不方便在一次 step 中同时返回 KVLoadTask 和 SchedulerOutputs，也可以先由 Scheduler 生成任务，再由 Engine 在模型执行前提交后台线程。

## 七、为什么不直接把请求放入现有 waiting 并跳过

DeepInfer 当前有：

```cpp
if (!seq->schedule.schedulable) {
    break;
}
```

而且 `_has_prefill_waiting()` 只检查 waiting 队首：[scheduler.hpp:270](./DeepInfer/csrc/engine/scheduler.hpp:270)。

如果把正在异步 get 的请求继续放在 waiting 队首，会导致：

```text
R1 正在 get
R2 可执行
R3 可执行

waiting.front() = R1
break
R2/R3 无法调度
```

因此推荐保留普通 `waiting` 队列语义，使用独立的 `kv_loading` / `kv_ready` 队列承载异步 KV 状态。这样对现有以下逻辑影响最小：

- FIFO；
- chunked prefill；
- MTP；
- 抢占和重算；
- NORMAL 的 ALL/ANY 投票；
- 多模态 `schedulable` 机制。

## 八、推荐的最终方案

对于“上游已经提供 prefix match 结果”的前提，建议采用：

```text
上游:
    PrefixMatcher -> PrefixPlan

DeepInfer:
    add_request(PrefixPlan)
        -> waiting

    Scheduler:
        1. poll 完成事件
        2. 从 waiting 接纳 prefix load
        3. 分配 prefix blocks/state
        4. 生成 KVLoadTask
        5. 请求进入 kv_loading
        6. 继续调度其他请求

    Worker:
        后台 Store.get -> 直接写 HBM

    Scheduler:
        get 完成
        -> kv_ready
        -> extend suffix blocks
        -> 进入原有 prefill 流程
```

核心原则是：

> **上游负责匹配，Scheduler 负责接纳和状态机，BlockManager 负责两阶段内存分配，Worker 负责异步 get，原有 prefill/decode 调度只处理已经 ready 的请求。**

这样既能借鉴 vLLM 的异步 KV 生命周期，也能保留 DeepInfer 现有的队列、DP 投票和 prefill/decode 步型语义。

> 很好，目前的需求是从设计文档草稿中的阶段一，也就是理解不需要考虑swa相关的内容。那请优先结合vllm这部分的异步get逻辑，以及deepinfer的现有代码架构，设计出合理的方案。刚刚上述考虑到的，这个方案也是一个草稿，不需要做出完善的设计文档最终版，主要是敲定主要设计方案，和可能方案是否考虑到异步get的一起特殊场景是否考虑全，以及也需要结合好deepinfer的现有逻辑做新增适配，输出到/Users/l/L/Code/TuringProject/D/设计文档

<details><summary>6 previous messages</summary>

> 我会在 `设计文档` 下新增一份“阶段一异步 KV Get 方案草稿”，范围限定为 Block/Global KV，不引入 SWA。内容会把 vLLM 的异步加载状态机映射到 DeepInfer 当前 C++ Scheduler、BlockManager、LLMEngine 和 worker 边界，并明确需要新增的接口、队列、时序和异常处理。
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./设计文档/设计文档草稿.md`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
>
> </details>
>
> 现有草稿 v1.2 把阶段一、阶段二和 PD 设计混在一起，且其中“请求留在 waiting 用标志跳过”与 DeepInfer 当前队首 `break` 语义不一致。新草稿会明确采用阶段一独立 KV loading/ready 状态、保留普通 waiting FIFO，并把异步加载作为不改变模型步型的旁路流程。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md` (+564 -0)
>
> </details>
>
> 草稿已写入，当前先做一致性检查：确认文件内容、引用的 DeepInfer 现有接口，以及是否仍误带入 SWA/阶段二语义。之后我会给出文件位置和本轮方案的关键结论。
>
> <details><summary>Explored 2 searches, ran 2 commands</summary>
>
> - Ran `wc -l '设计文档/DeepInfer阶段一异步KV_Get方案草稿.md'`
>
> <details><summary>Explored 2 searches</summary>
>
> - Searched for `try_allocate_prefix|try_extend|KVLoadState|kv_loading|kv_ready|PrefixPlan|NORMAL` in `DeepInfer阶段一异步KV_Get方案草稿.md`
> - Searched for `SWA|state_size_to_disk|copy_state|阶段二|gpu_states` in `DeepInfer阶段一异步KV_Get方案草稿.md`
>
> </details>
>
> - Ran `git status --short -- '设计文档/DeepInfer阶段一异步KV_Get方案草稿.md'`
>
> </details>
</details>

已完成阶段一异步 KV Get 方案草稿，文件位于：

[DeepInfer阶段一异步KV_Get方案草稿.md](./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md)

方案核心结论：

- 上游在 `add_request()` 前提供 `PrefixPlan`；
- DeepInfer 不重复做前缀匹配；
- Scheduler 接纳请求后只分配命中 prefix blocks；
- 新增独立 `kv_loading` / `kv_ready` 状态，避免阻塞现有 `waiting` 队列；
- Worker 后台 Store `get` 直接写入 `gpu_blocks`；
- get 完成后再扩展 suffix blocks；
- suffix 分配成功后才进入原有 prefill；
- Store 失败按 miss 回退；
- 异步 load admission 不改变 NORMAL 的 prompt/decode 步型；
- 覆盖取消、失败、资源不足、并发上限、DP rank 不一致、旧完成事件等边界场景。

本次只新增了方案草稿，没有修改 DeepInfer 代码，也没有改动原有设计文档。

> 有几个问题需要待确认一下？刚刚生生成的阶段一草稿方案，请检查是否有考虑到提到的all或者any相关的投票流程？
>
> 2. 上游会提供mooncake store 的get接口，但是是python的，此时调度器是c++的，需要考虑到代码开发的适配性
> 3. 上游不传递完整的store key，是直接在pe再生成
> 4. hit-tokens由实际的block-size决定，
> 5. 第12章节的一些其他待确认点，尽量可以参考vllm的实现里面是否有现成的解决方案，如果有请解决并查看是否能适配到deepinfer中

<details><summary>6 previous messages</summary>

> 我会按这五点回查草稿和源码：先核对 DeepInfer 的 ALL/ANY 投票入口与异步 load admission 的放置位置，再核对 vLLM/vllm-ascend 的 Python connector 边界、PE 侧 key 生成、实际 block size 语义，以及第 12 节待确认项。确认后直接更新草稿，把能由现有实现确定的内容改成结论，只保留真正需要外部约定的事项。
>
> <details><summary>Explored 4 files, 5 searches, ran a command</summary>
>
> <details><summary>Explored 2 searches</summary>
>
> - Searched for `block_size|block_hashes|process_tokens|PoolKey|to_string|lookup_external|_generate_store_query_keys|generate.*key` in `ascend_store`
> - Searched for `python.*get|Store.get|batch_get|LookupKeyClient|start_load_kv|KVConnector` in `kv_transfer`
>
> </details>
>
> - Ran `sed -n '90,230p;260,390p;520,565p' '设计文档/DeepInfer阶段一异步KV_Get方案草稿.md'`
>
> <details><summary>Explored 4 files, 3 searches</summary>
>
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Searched for `_dp_allreduce|prefill_attempt_interval|prefill_any_attempt_interval|_has_prefill_waiting` in `sched`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Searched for `def _lookup_external_hit_tokens|def lookup\(|process_token_key_strings|PoolKey|block_hashes` in `ascend_store`
> - Searched for `class KVConnectorBase_V1|def build_connector_meta|def get_finished|start_load_kv` in `v1`
> - Read `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md`
>
> </details>
>
> </details>
>
> 源码核对后有几项可以直接定案：DeepInfer 的 ALL/ANY 投票只应决定真正的 prefill step，异步 load admission 不参与投票；C++ Scheduler 不能在 `py::gil_scoped_release` 期间直接调用 Python Store，应由 C++ 产出 load task、Python Engine/worker 提交 get；Store key 由 PE 侧按请求 token、实际 block size 和本 rank 拓扑生成；部分 get 失败可以沿用 vLLM 的“保留连续有效前缀、后续重算”策略。现在更新草稿。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md` (+110 -31)
>
> </details>
>
> 草稿已按确认内容更新。重点变化是：补入 ALL/ANY 的具体参与边界；明确 C++/Python 的 task descriptor 适配方式；删除上游传完整 key 的假设；将命中长度绑定 PE 实际 block size；并用 vLLM 的连续有效前缀 fallback、完成事件和并发 load 语义收敛原先第 12 节的大部分待确认项。现在做最终一致性检查。
>
> <details><summary>Edited a file, explored a file, 2 searches</summary>
>
> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Read `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md`
> - Searched for `block_keys|128|整次 get|完整 Store key|_has_prefill_waiting|ALL/ANY|kv_load_tasks|py::gil_scoped_release|hit_tokens` in `DeepInfer阶段一异步KV_Get方案草稿.md`
> - Searched for `当前草稿仍需确认|第12|SWA|gpu_states|state_size_to_disk` in `DeepInfer阶段一异步KV_Get方案草稿.md`
> - Read `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md`
>
> </details>
>
> - Wrote `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md` (+3 -3)
>
> </details>
</details>

已检查并更新阶段一方案草稿：

[DeepInfer阶段一异步KV_Get方案草稿.md](./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md)

本次已确认并写入：

1. **ALL/ANY 投票**

   - 异步 KV load admission 不参与 ALL/ANY 投票。
   - 不改变 `decode_step`、`prompt_run`、`num_batched_tokens`。
   - 只有 get 完成、suffix block 分配成功并回到普通 `waiting` 的请求，才参与现有 prefill 投票。
   - `kv_loading` 不计入 `_has_prefill_waiting()`。
   - DP rank 必须以一致的 `hit_tokens` 和 suffix 起点进入 collective prefill。

2. **C++ Scheduler 与 Python Mooncake Store 的适配**

   - C++ Scheduler 不直接调用 Python Store API。
   - Scheduler 只生成 `KVLoadTask`，通过 `SchedulerOutputs.kv_load_tasks` 暴露。
   - Python `LLMEngine/KVPoolWorker` 接收 task 后提交后台 Store `get`。
   - 完成结果通过 Python completion queue，在下一次 step 的主线程中交回 Scheduler。
   - 这样适配了 DeepInfer 当前 `_schedule()` 使用 `py::gil_scoped_release` 的架构。

3. **Store key 生成位置**

   - 上游只提供请求级 `PrefixPlan` 和 `hit_tokens`。
   - 不传完整 Store key。
   - PE 侧 Python worker 根据请求 token、实际 block size、hash 规则和本 rank 的 TP/DP/PP 拓扑生成 key。
   - C++ `Sequence` 不保存完整 Store key。

4. **hit_tokens 对齐**

   - 不再固定写死 128。
   - `hit_tokens` 按 PE `CacheEngine` 的实际 `block_size` 对齐。
   - PrefixPlan 中的 block size/layout 必须与当前 Engine 一致，否则按 miss 处理。

5. **vLLM 可复用的异常处理**

   - 如果 Store 返回逐 block 结果，保留从第一个 block 开始的连续有效 prefix，释放失败 block 及其后续 block。
   - 如果 Store 只能返回请求级成功/失败，则整个请求按 miss 回退。
   - 旧完成事件通过 `load_id` 校验，避免误更新新请求。
   - suffix block 暂时不足时保留已加载 prefix，放入 `kv_ready` 等待资源。

第 12 节已从“待确认问题”改为“已确认方案 + 少量真实外部依赖”。目前仍需要实际确认的主要是 Mooncake Python API 是否支持将已注册的 HBM 地址作为异步 `get` 目标，以及 Store 是否提供逐 block 返回码。

> /Users/l/L/Code/TuringProject/D/设计文档/mooncake_store.md这个是上游mooncake store相关的设计方案，当前异步拉去需要调用store的相关操作，可以先借鉴上游store的方案直接调用，请再确认方案修改点

<details><summary>9 previous messages</summary>

> 我会先读取 `mooncake_store.md`，重点确认上游 Store 的 Python API、异步任务模型、HBM buffer 注册和返回结果语义，然后对阶段一草稿中“C++ task -> Python worker -> Store.get”的边界做针对性修订，删除已经由上游方案确定的待确认项。
>
> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Searched for `Store.get|batch_get|mooncake|KVPoolWorker|kv_load_tasks|Python worker` in `DeepInfer阶段一异步KV_Get方案草稿.md`
> - Read `./设计文档/mooncake_store.md`
> - Read `./设计文档/mooncake_store.md`
> - Searched for `get|put|batch|async|异步|Python|HBM|register|buffer|返回|callback|线程|future|地址` in `mooncake_store.md`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我会把 `mooncake_store.md` 的接口约束逐项映射回阶段一草稿，重点修正三处：key 生成边界、按 block 组织 `get` 的粒度，以及 C++ 输出到 Python worker 的生命周期。然后检查草稿中仍然互相矛盾的表述并直接更新文件。
>
> <details><summary>Explored 10 files, 2 searches</summary>
>
> - Searched for `5\.2|5\.3|5\.5|8\.6|12\.2|KVLoadTask|block_keys|整次|get|HBM|未决|待确认` in `DeepInfer阶段一异步KV_Get方案草稿.md`
> - Read `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md`
> - Read `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md`
> - Read `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md`
> - Read `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md`
> - Read `./设计文档/mooncake_store.md`
> - Read `./设计文档/mooncake_store.md`
> - Searched for `register_buffers|block_segment|data_ptr|gpu_blocks|Segment|GetItem|batch_get` in `mooncake_store.md`
> - Read `./设计文档/mooncake_store.md`
> - Read `./设计文档/mooncake_store.md`
> - Read `./设计文档/mooncake_store.md`
> - Read `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md`
>
> </details>
>
> `mooncake_store.md` 已经把关键前提定死了：Store 接收的是已注册 region 内的 `(addr, len)`，而不是 Python tensor 对象；`get` 按 item 返回结果，item 可以按 block 拆分。基于此，我会把草稿中的“仍需确认 HBM 是否可作为目标”改成具体接入方案，并把逐 block fallback 设为推荐实现。
>
> <details><summary>Edited a file, ran a command</summary>
>
> - Wrote `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md` (+19 -12)
> - Ran `git diff -- '设计文档/DeepInfer阶段一异步KV_Get方案草稿.md'; rg -n "block IDs 与|整次 miss|是否能把|按 block 返回|block_keys|Store key" '设计文档/DeepInfer阶段一异步KV_Get方案草稿.md'`
> - Wrote `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md` (+13 -0)
>
> </details>
>
> 草稿已更新：现在明确由 Python worker 的后台线程提供异步语义，Mooncake client 的 `get()` 可以同步等待本批 RPC，但绝不能运行在 Scheduler 线程；每个 block 单独作为一个 Store item，结果按 block 回传并做连续前缀截断。接下来我做一次全文一致性检查，确保没有残留的旧结论。
>
> <details><summary>Edited a file, explored a file, a search</summary>
>
> <details><summary>Explored a search</summary>
>
> - Searched for `仍需外部确认|HBM tensor|请求级|整次 miss|逐 block|block_keys|block IDs 与|Store\.get\(|batch_get|register_buffers|借用数组|owned|异步` in `DeepInfer阶段一异步KV_Get方案草稿.md`
>
> </details>
>
> - Wrote `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md` (+9 -0)
>
> <details><summary>Explored a file</summary>
>
> - Read `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md`
>
> </details>
>
> </details>
</details>

已根据 [mooncake_store.md](./设计文档/mooncake_store.md) 修订阶段一草稿：

- 明确 `gpu_blocks` 启动时通过 `register_buffers` 注册，Store `get` 直接写入 HBM。
- C++ `KVLoadTask` 不携带完整 key 和裸地址，只传 request/load 标识、block ID、`block_size/block_bytes`、布局元数据。
- Python worker 根据请求 token、实际 block size 和 rank 拓扑生成 key。
- 一个 prefix block 对应一个 Store item，支持逐 block 结果和连续有效前缀 fallback。
- `MooncakeStoreClient.get()` 可同步等待 RPC，但只允许在 worker 后台线程执行；Scheduler 线程只负责生成任务和消费完成事件。
- 短读、对象不存在、连接异常都按对应 block 失败处理，不能将不完整 KV 交给模型。
- Store 的注册、重连、熔断、分块和加锁留在 backend，不进入 Scheduler 状态机。
- 删除了“是否能把 HBM 作为 get 目标”和“是否支持逐 block 结果”这两个已被上游方案解决的阻塞项。

剩余待确认项收敛为：

1. `layout_version/model_id` 的配置来源和重启失效规则。
2. `SchedulerOutputs.kv_load_tasks` 的 pybind 生命周期。
3. pending 统计是否拆分 `waiting`、`kv_loading`、`kv_ready`。

修改已写入 [DeepInfer阶段一异步KV_Get方案草稿.md](./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md)。本次未修改 DeepInfer 源代码。

> /Users/l/L/Code/TuringProject/D/cann-deepinfer-recipe这个是具体进行deepinfer 异步kv-get方案的开发代码仓，请按照这个代码仓里的设计文档要求设计文档，新建一个文件夹，里面有img文件夹和md的设计文档，
>
> 设计文档要求：
>
> 1。格式要按照ann-deepinfer-recipe仓库的设计文档要求来（应该在/Users/l/L/Code/TuringProject/D/cann-deepinfer-recipe/CONTRIBUTING.md）
>
> 2. 设计文档里的格式要求要专业合理，描述准确精炼简洁，不必要的描述不说
> 3. 在介绍异步get的流程的时候，最好可以正常专业的流程图，来配合说明，流程图就生成svg格式就好，风格要专业且按照github已有风格来
> 4. 设计文档需要设计相关的伪代码和ST用例
> 5. 伪代码里如果涉及的已有Deepinfer代码需要准确，如果新的代码逻辑最好参考vllm已有的逻辑来
> 6. ST用例需要表述清楚测试目的、测试步骤和测试点
> 7. 整个设计文档的格式要考虑清楚设计紧密

<details><summary>6 previous messages</summary>

> 我先读取 `cann-deepinfer-recipe` 的贡献规范和现有设计文档样例，再对照当前阶段一异步 KV_Get 草稿整理成仓库风格的设计文档。之后会补充 SVG 流程图、伪代码和可执行的 ST 用例，最后检查目录结构和 Markdown 引用。
>
> <details><summary>Explored 5 files, 2 lists</summary>
>
> - Read `./CONTRIBUTING.md`
> - Read `./设计文档/DeepInfer阶段一异步KV_Get方案草稿.md`
> - Listed files in `.`
> - Listed files
> - Read `./docs/style-guide.md`
> - Read `./docs/modules/kv-cache-pooling/README.md`
> - Read `./docs/requirements.yaml`
>
> </details>
>
> 规范确认了：该需求应使用 `REQ-KVP-04`，正式文档必须有 YAML frontmatter、固定章节、伪代码和 `TC-KVP-04-*` 用例；模块 README 还要求文档放在 `docs/modules/kv-cache-pooling`。我会按需求编号建立独立目录 `REQ-KVP-04-async-kv-get/`，其中放 Markdown 和 `img/`，并把依赖写成已有的 `REQ-KVP-01/02/03/05/07`。
>
> <details><summary>Explored 3 files, 2 searches, ran a command</summary>
>
> - Ran `find docs/modules -maxdepth 3 -type f -name '*.md' -print -exec sed -n '1,80p' {} \;`
>
> <details><summary>Explored 9 files, 2 searches</summary>
>
> - Searched for `class BlockManager|allocate|free|block_table|gpu_blocks|block_size|CacheEngine` in `engine`
> - Searched for `class Scheduler|_schedule\(|_schedule_prefill|_has_prefill_waiting|waiting|running|SequenceScheduleState|SchedulerOutputs|schedule\(` in `scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/sequence.hpp`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
>
> </details>
>
> </details>
>
> 现有代码确认了几个必须在正式文档中写准的边界：当前 C++ 入口是 `Scheduler::add_sequence()` 和 `Scheduler::schedule()`，`SchedulerOutputs` 的 clone 数组是借用存储；Python `LLMEngine.step()` 先调度、再执行 `post_schedule()`。因此新方案会把 KV load task 设计成 SchedulerOutputs 的拥有型字段，并在 `step()` 中按“消费完成事件 → 调度 → 提交新 get → 模型执行”接入，避免伪造 prompt step。现在开始创建正式目录和文档。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./cann-deepinfer-recipe/docs/modules/kv-cache-pooling/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg` (+106 -0)
>
> </details>
</details>

> 还是按照上述要求生成，但是我希望生成的文档和对应svg图片先在/Users/l/L/Code/TuringProject/D/设计文档这路径下创建新的文件夹，然后文档格式要求其他的都参考一下就行。请继续

<details><summary>6 previous messages</summary>

> 收到，目录位置改为 `/Users/l/L/Code/TuringProject/D/设计文档`。我会在该路径下新建独立目录，文档结构、命名和内容仍参考 `cann-deepinfer-recipe` 的规范，但不会把文件写入 recipe 仓库；先检查上一轮是否留下了部分文件，再继续生成。
>
> <details><summary>Explored a list, ran 3 commands</summary>
>
> - Ran `find 'cann-deepinfer-recipe/docs/modules/kv-cache-pooling' -maxdepth 4 -type f -print 2>/dev/null | sort`
>
> - Ran `test -f 'cann-deepinfer-recipe/docs/modules/kv-cache-pooling/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg' && sed -n '1,8p' 'cann-deepinfer-recipe/docs/modules/kv-cache-pooling/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg' || true`
>
> <details><summary>Explored a list</summary>
>
> - Listed files in `设计文档`
>
> </details>
>
> - Ran `mkdir -p '设计文档/REQ-KVP-04-async-kv-get/img' && cp 'cann-deepinfer-recipe/docs/modules/kv-cache-pooling/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg' '设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg'`
>
> </details>
>
> 目录已创建，SVG 已放入新目录。文档将以 `REQ-KVP-04-async-kv-get.md` 命名，包含摘要、目标边界、流程设计、伪代码、接口、ST 用例、依赖约束和遗留问题；流程图通过相对路径引用 `img/async-kv-get-flow.svg`。
>
> <details><summary>Created a file, explored a file, a search, ran 2 commands</summary>
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+393 -0)
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
>
> </details>
>
> - Ran `find '设计文档/REQ-KVP-04-async-kv-get' -maxdepth 3 -type f -print -exec wc -c {} \;`
>
> - Ran `xmllint --noout '设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg'`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `^---$|^#|^##|^###|TC-KVP-04|未确认|待确认|不适用` in `REQ-KVP-04-async-kv-get.md`
>
> </details>
>
> </details>
>
> 检查发现一个会影响现有引擎生命周期的点：如果只新增 `kv_loading/kv_ready`，而不更新 `has_unfinished_requests()`、abort 扫描和资源回收，Engine 可能把仍在异步加载的请求误认为已结束。因此我会补充这一组必须同步修改的调度器边界，并再做最终校验。
>
> <details><summary>Edited a file, deleted a file, explored a search, a list, ran 2 commands</summary>
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+13 -0)
>
> - Deleted `./cann-deepinfer-recipe/docs/modules/kv-cache-pooling/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg` (+0 -106)
>
> - Ran `test ! -e 'cann-deepinfer-recipe/docs/modules/kv-cache-pooling/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg' && echo RECIPE_CLEAN`
>
> <details><summary>Explored a list</summary>
>
> - Listed files in `REQ-KVP-04-async-kv-get`
>
> </details>
>
> - Ran `xmllint --noout '设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg' && echo SVG_OK`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `get_num_unfinished|abort_sequence|free_finished|get_sum_prompt|3\.8|KVPoolWorker|async-kv-get-flow` in `REQ-KVP-04-async-kv-get.md`
>
> </details>
>
> </details>
</details>

已完成，文件已按你的要求创建在 `/Users/l/L/Code/TuringProject/D/设计文档` 下：

- [REQ-KVP-04-async-kv-get.md](./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md)
- [async-kv-get-flow.svg](./设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg)

文档内容包括：

- `REQ-KVP-04` YAML frontmatter
- 需求摘要、目标和非目标
- PrefixPlan、队列和状态设计
- ALL/ANY 投票与异步 get 的调度边界
- HBM block 生命周期和异常处理
- 对现有 `Scheduler`、`LLMEngine.step()` 生命周期的影响
- C++ Scheduler 与 Python worker 接口
- 关键流程伪代码
- 15 条 ST 用例，覆盖正常、并发、异常、取消、资源不足和多 DP 场景
- 依赖、约束和遗留问题
- SVG 专业流程图

已使用 `xmllint` 校验 SVG 格式通过。当前文档状态标记为“草稿”，符合尚未完成 RFC/committer 确认的情况。

> 设计文档有几个地方需要检查是否修改：
>
> 1. 摘要不需要，1.3的非目标不需要，2的约定用语不需要
> 2. 3.1的总体流程图风格很好，就是能不能体现一下get的异步拉取性
> 3. 3.2的`PrefixPlan`应该也不需要，这个是上游传入的，而且上游传入的前缀匹配字段应该只需要命中数一个字段就行？所以可以先不强调上游给什么样的数据结构
> 4. 3.3的`KV 状态`的字段最好和vllm的对齐，如果vllm有类似逻辑的状态字段，就不要自己再设计，能借用就借用
> 5. 3.4需要确认的一个描述点“\`kv_ready\` 只有在 suffix block 分配成功后才回到 \`waiting\`”，那这么说是get异步加载完后的任务会pop再重新入队到waiting的队列队尾部吗？还是这个描述什么？如果重新进入队列尾部会不会无法按照之前说的优先处理的？这个地方请重新确认vllm的代码流程和之前定的/Users/l/L/Code/TuringProject/D/设计文档/DeepInfer阶段一异步KV_Get方案草稿.md
> 6. 3.5章节应该也可以先不要
> 7. 第4张的伪代码最好还是可读的代码结构，然后加上中文注释就行，不需要“    while waiting has a request with kv_load_state == WAITING_TO_START:”类似这种的
> 8. 第5章的对外接口这部分也可以先不介绍，因为我感觉这个`KVPoolWorker`相关的流程操作也应该是当前需求的一部分？这种最好也设计到伪代码里

<details><summary>6 previous messages</summary>

> 我会按这 8 点重构文档，而不是只删标题：先核对 vLLM 的远端 KV 状态和完成后的队列位置，再同步修正 DeepInfer 方案。重点会确认 `kv_ready` 是否需要重新进入 `waiting`、进入哪一端，以及如何保证优先处理；同时把 worker 交接从“对外接口”移入可读伪代码流程。
>
> <details><summary>Explored 10 files, 2 searches</summary>
>
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Searched for `WAITING_FOR_REMOTE_KVS|kv_holding|finished_recving|remote_kvs|waiting\.append|waiting\.popleft|connector` in `scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Searched for `async_kv_waiting|WAITING_FOR_REMOTE_KVS` in `scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
>
> </details>
>
> vLLM 的实际行为已确认：异步加载请求使用独立的 `async_kv_waiting`，完成后不是简单地追加到普通 waiting 队尾；请求保留在“持有 KV block 的等待队列”中，该队列优先于普通 waiting，被调度循环优先检查。完成事件只把状态从 `WAITING_FOR_REMOTE_KVS` 恢复为 `WAITING`，随后优先队列中的请求继续做 suffix 计算。DeepInfer 的 `kv_ready` 应按这个语义实现为“优先持有队列”，必要时放回 `waiting` 队首，而不是队尾。
>
> <details><summary>Edited 2 files, explored 2 files, 2 searches, ran a command</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `./设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg`
>
> </details>
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg` (+10 -2)
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+25 -66)
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+3 -13)
>
> <details><summary>Explored a file</summary>
>
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
>
> </details>
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+98 -125)
>
> <details><summary>Explored 2 files, a search</summary>
>
> - Searched for `摘要|非目标|约定用语|PrefixPlan|kv_load_state|WAITING_FOR_REMOTE|waiting.push_back|对外接口|3\.|4\.|5\.|6\.|7\.|async_kv|kv_holding|队尾|队首|接口` in `REQ-KVP-04-async-kv-get.md`
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
>
> </details>
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+5 -9)
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+2 -1)
>
> <details><summary>Explored a file</summary>
>
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
>
> </details>
>
> - Ran `xmllint --noout '设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg' && echo SVG_OK`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `摘要|非目标|约定用语|PrefixPlan|对外接口|### [0-9]+\.[0-9]|## [0-9]` in `REQ-KVP-04-async-kv-get.md`
>
> </details>
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+1 -0)
>
> </details>
>
> 文档结构和流程已按要求收敛：删除摘要、非目标、术语和独立接口章节；`WAITING_FOR_REMOTE_KVS` 直接对齐 vLLM，`kv_ready` 改成类似 vLLM `kv_holding_waiting` 的优先持有队列，不再描述为重新追加到普通 waiting 队尾。伪代码已改成带中文注释的可读流程，并包含 worker 后台 get。现在做最后一致性检查。
>
> <details><summary>Edited a file, explored a search, ran 2 commands</summary>
>
> - Ran `xmllint --noout '设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg' && echo SVG_OK`
>
> - Ran `find '设计文档/REQ-KVP-04-async-kv-get' -maxdepth 2 -type f -print; test ! -e 'cann-deepinfer-recipe/docs/modules/kv-cache-pooling/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg' && echo RECIPE_CLEAN`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `PrefixPlan|摘要|非目标|约定用语|对外接口|kv_load_state|WAITING_FOR_REMOTE_KVS|kv_holding_waiting|waiting\.push_front|waiting\.push_back|后台异步|Scheduler 继续运行` in `REQ-KVP-04-async-kv-get.md`
>
> </details>
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg` (+3 -3)
>
> </details>
</details>

已按 8 点完成修改：

- 删除摘要、非目标、约定用语和独立“对外接口”章节。
- 流程图新增 `Scheduler 继续运行` 并行路径，明确 Store `get` 在后台异步执行，不阻塞其他请求。
- 删除 `PrefixPlan` 的具体结构设计，只保留“上游提供命中 token 数”。
- 状态设计对齐 vLLM：
  - `WAITING_FOR_REMOTE_KVS`
  - `kv_loading` 对应 `async_kv_waiting`
  - `kv_ready` 对应 `kv_holding_waiting`
- 明确 `kv_ready` 不是普通 `waiting` 的尾部队列：
  - get 完成后进入优先队列；
  - suffix block 和预算满足时优先调度；
  - 资源不足时保持队首等待；
  - 不重复提交 get，也不追加到普通 waiting 队尾。
- 删除独立的 HBM 生命周期章节。
- 将伪代码改为带中文注释的可读代码结构，并纳入 `LLMEngine`、Scheduler 和 `KVPoolWorker` 流程。
- 测试用例同步调整，不再验证不存在的 `PrefixPlan` 字段。

文件位置：

- [设计文档](./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md)
- [异步流程图](./设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg)

已通过 SVG 格式校验。

> 我觉得有些还可以修改：
>
> 1. 2.1的总体流程图，我觉得是不是直接画时序图会更能体现异步和流程步骤？如果是图的风格还是保持一致
> 2. 2.3的调度时机和投票我觉得也可以画图表示更直观，至于是流程图还是什么图，请你给出专业的意见，图的风格和2.1章节的图保持一致
> 3. 第4章的伪代码，我觉得如果是一个类的，可以写在一个类里，然后给出这个类涉及的方法和方法里的主要流程
> 4. ST测试用例我感觉太多了，我觉得可以先看看有没有复用的公共测试用例，把几个点都覆盖到。如果不行也没事，先设计个6-8条ST的主要重要用例
> 5. 第6章和第7章的内容可以放在同文件夹里的一个新md文档中，作为自己设计文档串讲的备忘文档
> 6. 变更记录也不需要

<details><summary>6 previous messages</summary>

> 我会把文档再收敛一轮：2.1 改成带主线程/worker 并行泳道的时序图，2.3 增加独立的调度投票时序图；第 4 章改成一个 Scheduler/Worker 协作类的伪代码；ST 收敛到 8 条主用例。第 6、7 章会迁移到同目录的备忘文档，并删除变更记录。先检查仓库现有测试文档是否有可复用的公共场景。
>
> <details><summary>Explored a file, a list, ran a command</summary>
>
> <details><summary>Explored a file, a list</summary>
>
> - Listed files in `REQ-KVP-04-async-kv-get`
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
>
> </details>
>
> - Ran `rg -n "测试用例|ST|公共|Store|异步|取消|短读|多 DP|资源不足" cann-deepinfer-recipe/docs cann-deepinfer-recipe/DeepInfer/tests 2>/dev/null | head -160`
>
> </details>
>
> 现有仓库没有可直接复用的异步 KV Get 公共 ST 用例，只有 Store/state cache 的局部测试，无法覆盖 Scheduler 与 worker 的交接。因此主文档保留 8 条组合用例，合并重复的生命周期、队列和失败场景。接下来生成两张 SVG：一张端到端异步时序图，一张调度投票时序图。
>
> <details><summary>Created 3 files, edited a file, deleted a file</summary>
>
> - Deleted `./设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg` (+0 -114)
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg` (+52 -0)
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-sequence.svg` (+47 -0)
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+6 -0)
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+92 -108)
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+1 -39)
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/设计串讲备忘.md` (+36 -0)
>
> </details>
>
> 两张图已替换为时序图：第一张展示 Scheduler 提交任务后与 worker/Store 并行，第二张展示 completion、`kv_ready` 优先处理和 ALL/ANY 投票的先后关系。依赖与遗留问题已迁移到同目录备忘文档，主文档保留主流程、伪代码和 8 条 ST 用例。最后检查 Markdown 结构、链接和 SVG。
>
> <details><summary>Edited a file, explored a search, ran 2 commands</summary>
>
> - Ran `xmllint --noout '设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg' && xmllint --noout '设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-sequence.svg' && echo SVG_OK`
>
> - Ran `find '设计文档/REQ-KVP-04-async-kv-get' -maxdepth 2 -type f -print -exec wc -c {} \;`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `^##|^###|变更记录|对外接口|PrefixPlan|TC-KVP-04-|设计串讲备忘|\.svg` in `REQ-KVP-04-async-kv-get.md`
>
> </details>
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+13 -0)
>
> </details>
</details>

已按要求完成调整。

主要变化：

1. **2.1 总体流程**
   - 改为主线程、Python worker、Mooncake Store、其他请求之间的时序图。
   - 明确黄色虚线为异步路径。
   - 展示 Scheduler 提交 `KVLoadTask` 后继续调度其他请求。

2. **2.3 调度时机与投票**
   - 新增独立时序图：
     [scheduler-vote-sequence.svg](./设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-sequence.svg)
   - 明确 `kv_loading` 不参与 ALL/ANY 投票，`kv_ready` 准备好 suffix 后才进入投票。

3. **第 4 章伪代码**
   - 改为统一的 `AsyncKVGetScheduler` 类。
   - 将 `step`、异步 load admission、worker get、完成事件处理、`kv_ready` 优先调度统一放入类中。
   - 保留中文注释和关键异常分支。

4. **ST 用例**
   - 从 15 条收敛为 8 条主用例。
   - 覆盖异步并发、优先队列、ALL/ANY、部分失败、整体 miss、取消竞态、DP rank 一致性和生命周期。

5. **依赖与遗留问题**
   - 已迁移到：
     [设计串讲备忘.md](./设计文档/REQ-KVP-04-async-kv-get/设计串讲备忘.md)
   - 主设计文档只保留链接。

6. **其他**
   - 删除变更记录。
   - 主文档新增第 3 章“实现边界”，说明 C++ Scheduler 与 Python worker 的职责。
   - 两张 SVG 均已通过 `xmllint` 校验。
   - 检查了 recipe 仓库现有测试，没有找到可直接覆盖该异步 Scheduler 流程的公共 ST 用例。

> 待确认和可能修改点：
>
> 1. 2.1总体流程图，感觉可以右上角先说明有几种队列，时序图里waiting和kv_loading等不是很清晰；其次kv-loading到其他请求这个线是不是有点问题？我理解就是在scheduler里继续调度其他可以进行的请求就行吧？（至于如何在时序图里好描述，请自行确定设计专业）；最后，下一轮step poll动作的发起是调度器还是kv worker？这个需要确认
> 2. 2.3章的图有点不明白是在介绍什么？请先确认2.3章需要介绍的东西，然后再结合需要介绍的内容设计图？我感觉这种是不是时序图不是很直观？其他的uml图呢？
> 3. 第4章的伪代码职责边界不明确感觉，如果是调度器的就只说明调度器里需要修改什么，但是我看AsyncKVGetScheduler类里还有        self.execute_model(metadata, outputs)         self.process_model_outputs(outputs)，这些deepinfer里应该不是在调度器里的吧？如果是调度器里的就调度器里，如果是kv pool work的就重新写一个，分别进行说明。整体伪代码重新设计分布
> 4. 设计文档有一些话感觉是说个我听的，但是设计文档主要是给其他人看设计的，不需要一些不专业或者无关精要的描述，请重新全篇检查描述
> 5. 参考文献章节不需要

<details><summary>6 previous messages</summary>

> 我会先把职责边界和调用方向校准：`LLMEngine.step()` 发起 worker 的 completion poll，Scheduler 只接收并应用完成事件；Scheduler 自己不直接调用 Python worker。2.3 将改成 UML 活动图，专门表达“完成事件处理 → `kv_ready` 优先资源检查 → 新 load admission → ALL/ANY 投票 → prompt/decode”的决策关系。之后拆分第 4 章伪代码，并清理面向讨论过程的表述。
>
> <details><summary>Explored a file, 2 searches</summary>
>
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
> - Searched for `def step|scheduler\.schedule|post_schedule|poll|completion|run_workers` in `llm_engine.py`
> - Searched for `poll|finished|completion|start_load_kv|get_finished|WAITING_FOR_REMOTE_KVS` in `pool_worker.py`
>
> </details>
>
> 校准结果：vLLM 的完成通知由 worker/connector 产生，Scheduler 在下一次调度前消费；在 DeepInfer 中，具体实现应由 `LLMEngine.step()` 调用 `KVPoolWorker.poll_completions()`，再把事件交给 C++ Scheduler 应用。Scheduler 不负责发起 Python poll，也不包含模型执行和输出处理。2.3 的图会改成 UML 活动图，更适合表达分支和投票条件。
>
> <details><summary>Created a file, edited 2 files, deleted a file, explored 2 files, a search, ran 2 commands</summary>
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg` (+3 -0)
>
> - Deleted `./设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-sequence.svg` (+0 -47)
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-sequence.svg` (+59 -0)
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+3 -3)
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+78 -71)
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+3 -4)
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
> - Read `./设计文档/REQ-KVP-04-async-kv-get/设计串讲备忘.md`
>
> </details>
>
> - Ran `xmllint --noout '设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg' && xmllint --noout '设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-sequence.svg' && echo SVG_OK`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `参考文献|变更记录|建议|该图|本需求|不需要|感觉|听|对外接口|execute_model|process_model|poll_completions|set_kv_completions|SchedulerOutputsBuilder|##` in `REQ-KVP-04-async-kv-get.md`
>
> </details>
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+2 -2)
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+1 -1)
>
> - Ran `mv '设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-sequence.svg' '设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-activity.svg'`
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+1 -1)
>
> </details>
>
> 职责和图示已完成重构：第一张图明确队列和 Scheduler 继续调度其他请求；第二张图改为 UML 活动图；第 4 章拆成 Scheduler、KVPoolWorker 和 LLMEngine 三部分。现在进行最终格式和一致性检查。
>
> <details><summary>Explored a search, a list, ran a command</summary>
>
> <details><summary>Explored a list</summary>
>
> - Listed files in `REQ-KVP-04-async-kv-get`
>
> </details>
>
> - Ran `xmllint --noout '设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg' && xmllint --noout '设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-activity.svg' && echo SVG_OK`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `参考文献|变更记录|建议调整|建议按|AsyncKVGetScheduler|execute_model\(|process_model_outputs\(|scheduler-vote-sequence|scheduler-vote-activity|poll_completions|##` in `REQ-KVP-04-async-kv-get.md`
>
> </details>
>
> </details>
</details>

已完成本轮修改。

主要调整：

- **2.1 总体流程图**
  - 改为更清晰的异步时序图。
  - 右上角标注 `waiting`、`kv_loading`、`kv_ready` 三类队列。
  - 明确 `Scheduler` 提交异步任务后继续调度其他请求。
  - 明确下一轮 `step` 的 poll 由 `LLMEngine` 发起，调用 `KVPoolWorker.poll_completions()`，再交由 Scheduler 应用完成事件。

- **2.3 调度时机与投票**
  - 改为 UML 活动图：
    [scheduler-vote-activity.svg](./设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-activity.svg)
  - 重点表达完成事件处理、`kv_ready` 优先检查、新 KV load admission、ALL/ANY 投票和 prompt/decode 的决策顺序。

- **第 4 章伪代码**
  - 拆分为三个职责边界：
    - `Scheduler`
    - `KVPoolWorker`
    - `LLMEngine.step()`
  - Scheduler 不再包含模型执行和输出处理。
  - Worker 只负责后台 Store get 和 completion queue。
  - `LLMEngine` 负责 poll、调用 Scheduler 和提交新 load task。

- **ST 用例**
  - 保留 8 条核心用例，覆盖异步并发、优先队列、投票边界、部分失败、整体 miss、取消竞态、DP rank 一致性和生命周期。

- **文档清理**
  - 删除参考文献章节。
  - 保留依赖、约束和遗留问题在 [设计串讲备忘.md](./设计文档/REQ-KVP-04-async-kv-get/设计串讲备忘.md)。
  - 删除变更记录。
  - 清理了不必要的讨论性描述。

两张 SVG 均已通过 `xmllint` 校验。

> 2.3的活动图线有点交叉了，那这样是不是可以直接用流程图代替就行？同时流程的线和框需要画清楚，不能交叉重叠

<details><summary>3 previous messages</summary>

> 我会把 2.3 改成单主路径的 UML 流程图：主流程从上到下，只有两个明确的左右分支，异步 load admission 作为独立步骤放在投票前，避免时序图中多条回线交叉。同步更新章节说明，突出该图表达的是每轮调度的决策顺序。
>
> <details><summary>Created a file, edited a file, deleted a file, ran 2 commands</summary>
>
> - Deleted `./设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-activity.svg` (+0 -59)
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-activity.svg` (+90 -0)
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+1 -1)
> - Ran `command -v rsvg-convert || command -v inkscape || command -v convert || true`
> - Ran `rsvg-convert '设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-activity.svg' -o /tmp/scheduler-vote-activity.png && rsvg-convert '设计文档/REQ-KVP-04-async-kv-get/img/async-kv-get-flow.svg' -o /tmp/async-kv-get-flow.png`
>
> </details>
>
> Viewed 2 images
</details>

已将 2.3 改为清晰的单主路径流程图：

- 主流程自上而下排列。
- `kv_ready` 资源判断使用左右分支。
- `admit 新 KV load`、NORMAL 投票点、ALL/ANY 分支均使用独立节点。
- 连接线已重新布局，避免交叉和框体重叠。
- 保留 GitHub 风格的蓝色同步、黄色异步、绿色完成状态。

文件已更新：

[调度投票流程图](./设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-activity.svg)

同时通过 SVG 格式校验，并渲染检查了图形布局。

> 请再确认一下vllm是不是也是优先处理kv_ready队列里的请求的？

<details><summary>4 previous messages</summary>

> 我会直接核对 vLLM 的队列选择和完成恢复代码，区分“优先处理已持有 block 的请求”和“异步 get 尚未完成的请求”，这两个队列在 vLLM 中不是同一个队列。确认后我会说明当前文档中的 `kv_ready` 对应关系是否准确。
>
> <details><summary>Explored a file</summary>
>
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
>
> </details>
>
> 确认结果：vLLM 没有名为 `kv_ready` 的队列，但确实优先处理“已经持有 KV block”的队列 `kv_holding_waiting`。需要区分两个阶段：
>
> - `async_kv_waiting`：等待启动异步 KV load 的请求。
> - `kv_holding_waiting`：已经分配/持有 KV block 的请求，包含正在接收和接收完成等待继续 Prefill 的请求。
>
> 调度时只要有 `kv_holding_waiting`，vLLM 在普通 `waiting` 前优先检查它；load 未完成时跳过该请求，完成后在同一优先队列中恢复，不追加到普通 waiting 队尾。我会把文档中的对应关系补准确。
>
> <details><summary>Edited 2 files, ran a command</summary>
>
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md` (+5 -3)
> - Wrote `./设计文档/REQ-KVP-04-async-kv-get/设计串讲备忘.md` (+1 -1)
> - Ran `xmllint --noout '设计文档/REQ-KVP-04-async-kv-get/img/scheduler-vote-activity.svg' && rg -n "async_kv_waiting|kv_holding_waiting|kv_ready|WAITING_FOR_REMOTE_KVS" '设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md' '设计文档/REQ-KVP-04-async-kv-get/设计串讲备忘.md'`
>
> </details>
</details>

是，但需要准确区分 vLLM 的两个队列：

- `async_kv_waiting`：等待启动异步 KV load 的请求。
- `kv_holding_waiting`：已经分配并持有 KV block 的请求，包括 KV 接收中和接收完成等待 Prefill 的请求。

vLLM 在调度时优先检查 `kv_holding_waiting`，然后才检查普通 `waiting`，对应代码：

- [scheduler.py:1098-1103](./vllm/vllm/v1/core/sched/scheduler.py:1098)
- [scheduler.py:1675-1679](./vllm/vllm/v1/core/sched/scheduler.py:1675)

异步 load 未完成时，请求保持 `WAITING_FOR_REMOTE_KVS`，被暂时跳过；完成事件到达后，Scheduler 在同一个优先队列中恢复该请求，不追加到普通 `waiting` 队尾：

- [scheduler.py:3671-3683](./vllm/vllm/v1/core/sched/scheduler.py:3671)

因此，当前 DeepInfer 文档中的对应关系是正确的，但更准确的说法是：

- `kv_loading`：表示 get 在途。
- `kv_ready`：表示 get 完成且仍持有 prefix block，语义对应 vLLM 的 `kv_holding_waiting`。
- `kv_ready` 应优先于普通 `waiting` 处理，不能重新追加到普通 waiting 队尾。

我已将这一点补充到主设计文档和设计串讲备忘中。

> 还有一个问题需要确认一下，暂不修改设计文档内容。请确认deepinfer里有咩有chunk prefill的操作，如果有的话，当前的异步get操作是否支持或者有遗漏？同时对齐检查vllm中是否有对应的操作

<details><summary>2 previous messages</summary>

> 我会只做代码核对，不修改设计文档。重点检查 DeepInfer 的 chunked prefill 入口、`cache.num_tokens`/`block_table` 的分配关系，以及异步 KV load 完成后是否会与 chunk 边界冲突；同时对照 vLLM 的 `num_computed_tokens`、chunk budget 和远端 KV 恢复路径。
>
> <details><summary>Explored 6 files, 4 searches, ran a command</summary>
>
> <details><summary>Explored 10 files, 3 searches</summary>
>
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Searched for `chunked_prefill|chunk_size|token_budget|num_computed_tokens|num_new_tokens|load_kv_async|WAITING_FOR_REMOTE_KVS|kv_holding_waiting` in `scheduler.py`
> - Searched for `chunked_prefill|chunk_checkpoints|chunked_rest_len|_adjust_chunk_size|num_batched_tokens|seqlen_q|prompt_limit` in `scheduler.hpp`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/deepinfer/engine/arg_utils.py`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Searched for `get_num_blocks|num_tokens|block_table` in `sequence.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/sequence.hpp`
>
> </details>
>
> - Ran `rg -n "_append_slot\(" DeepInfer/csrc/engine/scheduler.hpp DeepInfer/csrc/engine | cat; rg -n "hit_tokens|block_table" DeepInfer/csrc/engine/scheduler.hpp DeepInfer/csrc/engine/sequence.hpp | head -120`
>
> <details><summary>Explored 4 files, a search</summary>
>
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Searched for `allocate_seq` in `scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/block_manager.hpp`
> - Read `./DeepInfer/csrc/engine/block_manager.hpp`
>
> </details>
>
> </details>
</details>

确认结果：**DeepInfer 已经支持 chunked prefill，但当前异步 KV Get 方案还存在几个需要补齐的适配点。vLLM 对 chunked prefill 有完整支持。**

## 1. DeepInfer 是否支持 chunked prefill

支持，入口和逻辑在：

- `DeepInfer/deepinfer/engine/config.py`
- `DeepInfer/deepinfer/engine/arg_utils.py`
- `DeepInfer/csrc/engine/scheduler.hpp`

核心逻辑：

1. `chunked_prefill` 控制是否允许切分。
2. `_schedule_prefill()` 根据 `max_num_batched_tokens` 计算本轮 `seqlen_q`。
3. 当本轮只能处理部分 prompt 时：

```cpp
is_chunked = seqlen_q < seq_workload
```

4. chunk 未完成时，Sequence 保持 `WAITING`。
5. `process_outputs()` 处理完当前 chunk 后，将请求重新放回：

```cpp
waiting.push_front(seq_ptr);
```

相关代码：

- [`scheduler.hpp:460-510`](./DeepInfer/csrc/engine/scheduler.hpp:460)
- [`scheduler.hpp:522-545`](./DeepInfer/csrc/engine/scheduler.hpp:522)
- [`scheduler.hpp:1011-1020`](./DeepInfer/csrc/engine/scheduler.hpp:1011)

DeepInfer 还支持：

- `max_num_batched_tokens` 限制；
- `chunk_checkpoints`；
- 多模态 token 边界调整；
- CP 对齐；
- `chunked_rest_len` 记录剩余长度。

## 2. 异步 Get 与 chunked prefill 是否兼容

整体流程是兼容的：

```text
上游命中 prefix
→ 异步 Get
→ cache.num_tokens = 命中 token 数
→ 计算剩余 prompt
→ 按 max_num_batched_tokens 分 chunk Prefill
```

异步 Get 本身不产生模型 token，因此不能设置 `prompt_run`，也不能推进 chunk 计数。Get 完成后，才进入普通 chunked prefill。

但是当前方案有以下遗漏。

### 2.1 Prefix block 后的 suffix block 扩展需要单独处理

DeepInfer 当前 `_schedule_prefill()` 只有在：

```cpp
seq->cache.block_table.empty()
```

时才调用：

```cpp
acquire.allocate_seq(*seq);
```

而异步 Get 完成后，`block_table` 已经包含 prefix block，因此不会进入这条分配路径。

这意味着：

```text
prefix block 已存在
→ suffix chunk 开始执行
→ block_table 可能没有足够的 suffix block
```

需要增加 suffix block 扩展逻辑。扩展方式可以是：

- 在进入第一个 suffix chunk 前，一次性补齐 prompt 所需 block；
- 或者参考 vLLM，按每个 chunk 增量分配当前所需 block。

vLLM 采用第二种方式，通过 `allocate_slots(request, num_new_tokens)` 按当前 chunk 增量分配。

这是当前 DeepInfer 异步 Get 适配中最明确的遗漏点。

### 2.2 `hit_tokens == prompt_len` 的完整命中场景

DeepInfer 当前 `_schedule_prefill()` 使用：

```cpp
num_prompt_tokens = seq->get_len() - seq->cache.num_tokens;
```

如果异步 Get 后：

```text
cache.num_tokens == prompt_len
```

则 `num_prompt_tokens == 0`，当前 Prefill 路径没有可执行 token，后续可能触发 `seqlen_q == 0` 或相关断言。

vLLM 对完整命中有专门处理。异步接收完成后，如果已经覆盖完整请求，会将：

```python
request.num_computed_tokens = request.num_tokens - 1
```

保留最后一个 token 重新执行，避免请求完全没有 Prefill step。

相关代码：

- [`scheduler.py:3661-3665`](./vllm/vllm/v1/core/sched/scheduler.py:3661)

因此 DeepInfer 需要明确：

- 命中数最多允许到 `prompt_len - 1`；
- 或者完整命中时保留最后一个 token 做一次 Prefill；
- 不能直接让 `cache.num_tokens == prompt_len` 进入现有 `_schedule_prefill()`。

### 2.3 chunk 完成后的队列回放

DeepInfer chunk 完成后会：

```cpp
waiting.push_front(seq_ptr);
```

这与异步 Get 设计可以兼容，前提是：

1. 首个 suffix chunk 已经从 `kv_ready` 优先队列中取出；
2. suffix block 已经准备好；
3. 请求不再处于 `kv_loading` 或 `kv_ready`；
4. 后续 chunk 使用普通 `waiting.push_front()` 继续执行。

也就是说：

```text
kv_ready
→ suffix block 准备
→ 执行首个 chunk
→ 未完成时回到 waiting 队首
→ 后续 chunk
```

不能在 suffix block 尚未准备好时直接把请求放入普通 waiting。

### 2.4 chunk checkpoint 的偏移

DeepInfer 的 chunk checkpoint 使用：

```cpp
start_offset = seq->cache.num_tokens;
```

异步 Get 完成后，`cache.num_tokens` 应设置为有效的连续命中 token 数。这样后续 checkpoint 才会从正确的 prefix 偏移继续计算。

需要注意：

- Get 失败后截短命中范围时，checkpoint 也必须基于新的 `cache.num_tokens` 重新判断；
- 不能继续使用异步 Get 之前的零偏移；
- 多 DP rank 必须使用同一个有效命中前缀。

## 3. vLLM 的对应处理

vLLM 的 chunked prefill 与异步 KV load 是直接兼容的。

vLLM 使用：

```text
request.num_computed_tokens
```

表示已经具备 KV 的 token 数。

调度时：

```python
num_new_tokens = request.num_tokens - request.num_computed_tokens
num_new_tokens = min(num_new_tokens, token_budget)
```

所以：

- 异步 Get 进行中：`num_new_tokens = 0`；
- Get 完成后：恢复 `WAITING`；
- 下一轮根据剩余 `num_computed_tokens` 计算 chunk；
- 每个 chunk 通过 `allocate_slots()` 增量分配 block。

相关代码：

- [`scheduler.py:1336-1339`](./vllm/vllm/v1/core/sched/scheduler.py:1336)
- [`scheduler.py:1353-1356`](./vllm/vllm/v1/core/sched/scheduler.py:1353)
- [`scheduler.py:1396-1397`](./vllm/vllm/v1/core/sched/scheduler.py:1396)
- [`scheduler.py:1497-1504`](./vllm/vllm/v1/core/sched/scheduler.py:1497)
- [`scheduler.py:1585-1595`](./vllm/vllm/v1/core/sched/scheduler.py:1585)

vLLM 还会在异步 KV load 时延后 speculative lookahead block 的分配，避免远端 prefix block 数量和本地 speculative block 数量不一致：

```python
limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
```

相关代码：

- [`scheduler.py:1462-1468`](./vllm/vllm/v1/core/sched/scheduler.py:1462)

## 4. 结论

DeepInfer 的 chunked prefill **可以支持异步 Get**，但当前设计至少需要额外确认以下三点：

1. **异步 prefix block 完成后，如何扩展 suffix block table。**
2. **完整 prompt 命中时如何保留一个 token 进行 Prefill。**
3. **chunk checkpoint 和 `cache.num_tokens` 如何在部分命中、部分失败时保持一致。**

其中第 1 点是当前 DeepInfer 与 vLLM 的主要实现差异：DeepInfer 普通 Prefill 假设首次调度时 `block_table` 为空并一次性分配，而异步 Get 会提前填充 prefix block，因此必须增加 prefix block 之后的 suffix block 扩展路径。

> 好的，现在还有一个需求“PE Prefill/CE结束时提交Block Put Task，利用pin/unpin保证HBM生命周期”，请结合上述需求的方式，从vllm中确认是否有成熟的代码流程和代码实现，以及deepinfer的现有代码实现是否适配需求，最后可以参考设计文档草稿里的针对这个需求的说明借鉴（可以借鉴这个设计文档草稿，但是需要确认这个文档里是否有问题，最优先以vllm和deepinfer的代码为主），最后也是在/Users/l/L/Code/TuringProject/D/设计文档中生成一个对应方案的草稿

<details><summary>5 previous messages</summary>

> 我会先核对三部分：vLLM/vllm-ascend 的异步 Block Put、DeepInfer 当前 Prefill/CE 生命周期和 pin/unpin 实现、以及现有设计草稿中关于 Put Task 的描述。确认完接口边界和资源释放时机后，再在 `设计文档` 下新建独立方案草稿和必要的流程图。
>
> <details><summary>Explored 7 files, 3 searches, a list</summary>
>
> - Listed files in `设计文档`
> - Searched for `put|store|pin|unpin|kv.*transfer|save_kvcache|kvcache` in `设计文档草稿.md`
> - Searched for `pin|unpin|save_kvcache|kvcache|do_save|pe_do_save|post_process|finished|free_seq|process_outputs|block_table` in `engine`
> - Searched for `put|send|pin|unpin|finished_sending|kv.*store|start_load_kv|store` in `v1`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
> - Read `./DeepInfer/csrc/engine/block_manager.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/csrc/engine/pybind_scheduler.cpp`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我会先把 vLLM/vllm-ascend 的 Put 提交、完成通知和 pin 生命周期逐段对齐，再核对 DeepInfer 的 Prefill/CE 保存路径及释放逻辑，最后按现有设计文档风格生成独立草稿和 SVG。当前只做设计产物，不改运行时代码。
>
> <details><summary>Explored 10 files, 9 searches, a list, ran a command</summary>
>
> <details><summary>Explored 13 files, 8 searches, a list</summary>
>
> - Read `./cann-deepinfer-recipe/CONTRIBUTING.md`
> - Listed files in `REQ-KVP-04-async-kv-get`
> - Searched for `save_kv|wait_for_previous_save|finished_sending|pin|unpin|put\(` in `v1`
> - Searched for `Put|put|pin|unpin|Prefill|CE|Block` in `设计文档草稿.md`
> - Searched for `pe_prepare_save_kvcache|pe_do_save_kvcache|do_save_kvcache|finalize_seq|send_processed_prompts|post_process|free_seq|pin_blocks|unpin_blocks` in `deepinfer`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
> - Searched for `Block Put|Put Task|pin|unpin|提交|Prefill|CE结束|put` in `设计文档草稿.md`
> - Searched for `save_kv|save_kv_layer|wait_for_previous_save|finished_sending|request_finished|start_load_kv|update_connector_output` in `vllm`
> - Searched for `def (do_save_kvcache|pe_prepare_save_kvcache|pe_do_save_kvcache)|pe_prepare_save|pe_do_save|do_save_kvcache|post_process|free_seq_nolock|pin_blocks|unpin_blocks` in `deepinfer`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./cann-deepinfer-recipe/docs/style-guide.md`
> - Searched for `REQ-KVP-06|Block Put|pin/unpin|Put Task` in `docs`
> - Searched for `def save_kv_layer|def request_finished|def wait_for_previous_save|finished_sending|add_save_batch|save_kv` in `vllm`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
>
> </details>
>
> - Ran `rg -n "class Block|pin_blocks|unpin_blocks|free_seq_nolock|free_seq\(" DeepInfer/csrc/engine/block_manager.* DeepInfer/csrc/engine/scheduler.* DeepInfer/csrc/engine/sequence.* DeepInfer/csrc/engine/pybind_scheduler.cpp`
>
> <details><summary>Explored 9 files, a search</summary>
>
> - Searched for `def (do_save_kvcache|pe_prepare_save_kvcache|pe_do_save_kvcache)|pe_prepare_save_kvcache|pe_do_save_kvcache|do_save_kvcache|post_process\(` in `llm_engine.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
> - Read `./vllm/vllm/v1/worker/gpu/kv_connector.py`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/csrc/engine/block_manager.hpp`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
>
> </details>
>
> </details>
</details>

> 请继续检查，但是不生成对应设计草稿，先直接输出

<details><summary>2 previous messages</summary>

> 我继续核对具体调用边界，重点确认两件事：vLLM 的异步 Put 是在请求结束前还是结束时接管 block 释放，以及 DeepInfer 当前 `finalize_seq/free_seq` 是否已经有可复用的延迟释放点。确认后我会直接给出结论和适配缺口，不生成文档。
>
> <details><summary>Explored 15 files, 14 searches</summary>
>
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py`
> - Searched for `free_seq\(|finalize_seq\(|finished_seqs|SchedulerOutputs` in `llm_engine.py`
> - Searched for `def finish_store_job|finished_requests|completed_events|done_task_lock|request_queue.get` in `kv_transfer.py`
> - Searched for `request_finished_all_groups|request_finished\(` in `scheduler.py`
> - Searched for `finished_sending` in `scheduler.py`
> - Searched for `def request_finished` in `connector.py`
> - Searched for `pin|unpin|block_ids|_async_saving|save` in `pool_scheduler.py`
> - Searched for `_connector_finished|delay_free|request_finished` in `scheduler.py`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Searched for `pin_blocks|unpin_blocks|free_blocks` in `deepinfer`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`
> - Searched for `retain_blocks_for_save|release_blocks|pin` in `store`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/in_flight_save_blocks.py`
> - Searched for `InFlightSaveBlockTracker|retain_blocks_for_save|release.*save|block.*pin` in `ascend_store`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py`
> - Searched for `delay_free|finished_store_req|_free_blocks` in `worker.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
> - Searched for `_in_flight_save|completed_events|complete\(` in `pool_scheduler.py`
> - Searched for `BlockManager|block_manager|state_manager` in `llm_engine.py`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/csrc/engine/block_manager.hpp`
> - Read `./DeepInfer/csrc/engine/block_manager.hpp`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
> - Searched for `get_completed_events|completed_events` in `kv_transfer.py`
>
> </details>
</details>

结论：DeepInfer 具备实现基础，但当前代码还没有完整支持“异步 Block Put 期间保持 HBM block 生命周期”。

**1. vLLM 的成熟流程**

vLLM 的 Put 生命周期分为两类：

- `save_kv_layer()`：模型执行过程中逐层提交 KV 保存任务。
- `request_finished()`：请求结束、Scheduler 即将释放 block 时调用。

当 Connector 判定保存是异步的，`request_finished()` 返回 `True`，Scheduler 不立即释放 block，直到后续收到 `finished_sending`，再执行 `_free_blocks()`。

关键逻辑位于：

- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- `vllm/vllm/v1/core/sched/scheduler.py:3170`
- `vllm/vllm/v1/core/sched/scheduler.py:3706`

vLLM 的核心原则是：

```text
请求结束
  -> Connector 接管 block 所有权
  -> 异步 Store Put
  -> 收到完成通知
  -> Scheduler 释放 block
```

vllm-ascend 的 Store 实现还增加了：

- `add_save_batch()`：请求任务后追加 batch fence。
- 下一轮 `wait_for_save()` 等待上一批 fence。
- worker 完成后记录 `completed_events`。
- Scheduler 侧根据所有 worker 的完成数量释放保留 block。

`InFlightSaveBlockTracker` 通过 `block_pool.touch()` 保持 block 不被淘汰，并在所有 worker 完成后释放。

需要注意：当前 vllm-ascend 的普通 Store 保存路径中，`request_finished()` 返回 `False`，因为保存完成同步点放在 `wait_for_save()`；`InFlightSaveBlockTracker` 主要服务特定的 hybrid/Mamba blockwise 路径。因此它不能直接作为 DeepInfer “请求结束后完全异步 Put” 的完整实现，但其任务跟踪、批次 fence、全 worker completion 计数方式可以借鉴。

**2. DeepInfer 当前可复用的流程**

DeepInfer 已有明确的 Prefill/CE 保存钩子：

```text
pre_sampler_sync()
  -> Prefill: pe_prepare_save_kvcache()
  -> Decode:  do_save_kvcache()
  -> finished sequence: finalize_seq()
       -> scheduler.free_seq()
```

Prefill 模型执行完成后：

```text
post_process()
  -> pe_do_save_kvcache()
```

因此：

- `pe_prepare_save_kvcache()` 适合准备 block ID、记录 NPU event、执行 pin。
- `pe_do_save_kvcache()` 适合在模型写入完成后提交 Put Task。
- `do_save_kvcache()` 适合 CE 结束或 Decode block 完成时提交 Put Task。
- `finalize_seq()` 是请求释放 block 前的关键拦截点。

现有 C++ `BlockManager` 已支持引用计数：

```cpp
allocate()       // ref_count +1
pin_block()      // ref_count +1
free()           // ref_count -1
```

并且 Python 已暴露：

```python
block_manager.pin_blocks(block_ids)
block_manager.unpin_blocks(block_ids)
```

当前 `unpin_blocks` 实际映射到 `free_blocks`，命名上容易误导，但引用计数语义是可用的。

**3. 当前 DeepInfer 的关键缺口**

当前 `Scheduler::free_seq_nolock()` 会直接释放请求 block：

```cpp
block_manager->free_blocks_nolock(seq.cache.block_table);
seq.cache.block_table.clear();
```

它没有判断：

- block 是否正在执行 Store Put；
- 是否已经被某个 Put Task pin；
- 是否已有多个 Put Task 同时持有；
- Put 是否由多个 rank 分片完成。

因此如果直接在 `pe_do_save_kvcache()` 或 `do_save_kvcache()` 中提交异步任务，而不增加生命周期记录，会出现：

```text
Put worker 仍在读取 HBM
  -> finalize_seq()
  -> free_seq()
  -> block 被新请求重新分配
  -> Store Put 读取到被覆盖的数据
```

这是当前方案必须解决的问题。

**4. 推荐的 DeepInfer 适配方式**

建议采用“任务级 pin + 完成级 unpin”，不修改 waiting 队列机制。

```text
Prefill/CE 产生完整 block
  -> 记录 block_ids
  -> pin_blocks(block_ids)
  -> 记录任务 ID、block_ids、NPU event
  -> 提交 PutTask 到 KVPoolWorker
  -> Scheduler/Engine 正常 free_seq
       只释放请求自身的引用
  -> worker 等待 event 后执行 Store.put
  -> worker 返回 PutTaskCompleted(task_id, success)
  -> Engine 主线程 unpin_blocks(block_ids)
```

这样 `free_seq()` 执行后，Put pin 仍保留一份引用，block 不会进入 free list。

必须满足以下约束：

1. `pin` 必须发生在 `free_seq()` 之前。
2. PutTask 必须保存独立的 `block_ids` 快照，不能依赖 `seq.cache.block_table`。
3. Put worker 必须等待模型写入完成的 NPU/CUDA event。
4. 每个 PutTask 只能对应一次 unpin。
5. 同一 block 被多个 PutTask 使用时，每个任务都要独立 pin/unpin。
6. 多 TP/PP rank 必须等本 rank 所有相关 Put 完成后再 unpin；如果由统一任务管理，则需统计所有分片完成状态。
7. Put 失败也必须 unpin，失败只影响池化命中，不得阻塞推理主流程。
8. Mooncake `put` 返回 `0` 或 `-705` 都应视为成功。

**5. Prefill 与 CE 的具体时机**

Prefill：

```text
Scheduler.schedule()
  -> 模型执行 Prefill
  -> pe_prepare_save_kvcache()
       记录 block、pin、准备 event
  -> 模型写 KV 完成
  -> pe_do_save_kvcache()
       提交异步 PutTask
  -> 继续完成请求转移或释放
```

如果 `pe_prepare_save_kvcache()` 发生在模型执行前，则只能做元数据准备和 pin，不能执行真正的 Store Put。

CE：

```text
模型执行 Decode
  -> do_save_kvcache()
       针对已完成的完整 block 提交 PutTask
  -> finalize_seq()
       free_seq() 只释放请求持有的引用
  -> Put 完成后 unpin
```

对于 CE 的最后一个 block，必须以“已完成 token 数和完整 block 数”为准，不能把未写满的尾块提交为完整 block。

**6. 设计草稿中需要修正的地方**

现有草稿中“driver 线程提交前 pin，完成回调 unpin”的方向是正确的，但还需要补充：

- `free_seq()` 与 Put pin 的引用计数关系；
- PutTask 必须保存 block ID 快照；
- NPU event 必须在 worker 执行 Store.put 前等待；
- 多 rank completion 的聚合规则；
- 请求 abort、异常、worker 崩溃时的 unpin；
- 重复 Put 的幂等处理；
- `pe_prepare_save_kvcache()` 和 `pe_do_save_kvcache()` 的职责边界。

另外，草稿中把 `send_processed_prompts`、`pe_do_save_kvcache`、`finalize_seq` 都列为可能的 Put 入口，实际实现时应避免同一批 block 被重复提交。建议统一由一个 Put task builder 去重，再由不同生命周期钩子调用。

**最终判断**

DeepInfer 可以复用现有的：

- `BlockManager` 引用计数；
- `pin_blocks/unpin_blocks`；
- `pre_sampler_sync()`；
- `pe_prepare_save_kvcache()`；
- `pe_do_save_kvcache()`；
- `do_save_kvcache()`；
- `finalize_seq()`；
- `DeferredRelease` 的事件延迟释放模式。

但需要新增：

- PutTask 状态和唯一 ID；
- in-flight block 记录表；
- 异步 worker completion 队列；
- 主线程 poll completion 并执行 unpin；
- `free_seq()` 前的 Put pin 接入；
- 多 rank 完成聚合及异常清理。

所以这部分不是只修改 DeepInfer `Scheduler`。主要改动边界应是：

```text
LLMEngine：决定提交时机、poll 完成、调用 pin/unpin
KVPoolWorker：异步执行 Store.put、等待 event、返回完成状态
Scheduler/BlockManager：保证 free_seq 与 Put pin 的引用计数兼容
```

目前未生成或修改设计文档。

> 请重新检查vllm和deepinfer的代码实现和可复用或者进行put的地方，设计文档草稿只能借鉴（有问题还是以vllm代码实现借鉴为主），重新生成deepinfer适配增加put异步方案，当前不生成文档

<details><summary>1 previous message</summary>

> <details><summary>Explored 4 files, 8 searches</summary>
>
> - Searched for `def request_finished` in `scheduler.py`
> - Searched for `register_finished_lookahead_save|async_save|request_finished` in `scheduler.py`
> - Searched for `block_table|num_cached_tokens|cache\.block|block_size|prompt_run` in `llm_engine.py`
> - Searched for `def (pe_prepare_save_kvcache|pe_do_save_kvcache|do_save_kvcache|finalize_seq|send_processed_prompts)` in `DeepInfer`
> - Searched for `pe_prepare_save_kvcache|pe_do_save_kvcache|do_save_kvcache|finalize_seq|send_processed_prompts` in `DeepInfer`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`
> - Searched for `class .*Engine|send_processed_prompts|finalize_seq|do_save_kvcache|pe_do_save` in `deepinfer`
> - Searched for `_pinned_saves|completed_saves|_reference_save_blocks` in `scheduler.py`
> - Searched for `def process_outputs|process_outputs` in `scheduler.hpp`
>
> </details>
</details>

重新核对后，建议把方案调整为下面的实现方向。

## 1. vLLM 的实际 Put 生命周期

vLLM 的成熟实现不是简单地在 `request_finished()` 中直接执行 Store Put，而是分为两条路径。

### 1.1 运行中保存

vLLM Connector 在调度输出中生成保存元数据，随后：

```text
SchedulerOutput
  -> Connector metadata
  -> worker.save_kv_layer / store worker
  -> 异步 Put
```

Mooncake Store scheduler 会为每个 Store job 建立：

```text
store_job_id -> (block_ids, remaining_worker_count)
```

对应代码：

- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:530`
- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:582`

提交任务前，vLLM 会为 block 增加一个独立引用：

```python
pool.touch([pool.blocks[block_id] for block_id in block_ids])
```

Store worker 完成后上报：

```text
store_job_id -> completed_count
```

Scheduler 只有在所有 worker 都完成后，才释放这份额外引用：

```python
pool.free_blocks(...)
```

对应代码：

- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:625`
- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:755`

这正是 DeepInfer 需要借鉴的核心：**Put task 要拥有独立的 block 引用，不能只依赖 request 的 block_table。**

### 1.2 请求结束时保存

vLLM Scheduler 在释放请求前调用：

```python
connector.request_finished(request, block_ids)
```

如果 Connector 返回延迟释放，Scheduler 不立即执行 `_free_blocks()`，而是等 `finished_sending` 后释放。

对应代码：

- `vllm/vllm/v1/core/sched/scheduler.py:3170`
- `vllm/vllm/v1/core/sched/scheduler.py:3483`
- `vllm/vllm/v1/core/sched/scheduler.py:3730`

Mooncake Store 的请求结束路径会记录待发送 block，然后由 worker 异步提交。

### 1.3 vllm-ascend 的差异

vllm-ascend 的 `KVCacheStoreSendingThread` 使用：

```text
request -> batch fence
```

`wait_for_save()` 提交当前 batch，下一次保存时等待上一批 fence。

但当前普通 Store 保存路径中：

```python
request_finished(...) -> return False
```

也就是说，它依赖保存流程自己的引用跟踪，并不完全依赖 Scheduler 的 `finished_sending` 延迟释放。

因此 vllm-ascend 的实现可以借鉴：

- `store_job_id`
- batch fence
- completed event
- 多 worker 完成计数
- request generation 防止 request ID 重用
- worker 失败时也发送完成事件

但不能直接照搬其 `request_finished()` 返回值。

---

## 2. DeepInfer 当前调用位置

DeepInfer 当前的保存钩子位于 `LLMEngine`：

[llm_engine.py](./DeepInfer/deepinfer/engine/llm_engine.py:804)

```python
finalize_seq()
do_save_kvcache()
pe_prepare_save_kvcache()
pe_do_save_kvcache()
send_processed_prompts()
```

调用顺序是：

### Prefill

```text
schedule()
  -> pre_sampler_sync()
       -> pe_prepare_save_kvcache()
  -> model execution
  -> post_process()
       -> pe_do_save_kvcache()
       -> process_model_outputs()
  -> PROMPT engine: send_processed_prompts()
```

### Decode / CE

```text
schedule()
  -> pre_sampler_sync()
       -> do_save_kvcache()
       -> finalize_seq(finished)
            -> scheduler.free_seq()
  -> model execution
  -> post_process()
```

因此职责应调整为：

| 位置 | 责任 |
|---|---|
| `pe_prepare_save_kvcache()` | 筛选可保存请求、生成 block 快照、pin、准备 event |
| `pe_do_save_kvcache()` | Prefill 模型写入完成后，提交 PutTask |
| `do_save_kvcache()` | 提交 CE 已完成的完整 block |
| `finalize_seq()` | 释放 request 引用；不能释放 Put task 持有的引用 |
| `post_process()` | 轮询 Put completion 并执行 unpin，或调用统一 completion 处理器 |

---

## 3. DeepInfer 当前最重要的适配点

### 3.1 `BlockManager` 已具备引用计数能力

[block_manager.hpp](./DeepInfer/csrc/engine/block_manager.hpp:82)

```text
allocate  -> ref_count +1
pin       -> ref_count +1
free      -> ref_count -1
```

Python 已暴露：

```python
block_manager.pin_blocks(block_ids)
block_manager.unpin_blocks(block_ids)
```

因此可以采用：

```text
request 引用：1
Put task 引用：1

free_seq()       -> 释放 request 引用
Put 完成后 unpin -> 释放 Put 引用
```

当前 `Scheduler::free_seq_nolock()` 不需要知道 Store 细节，只要 Put pin 在它之前完成即可。

但 `unpin_blocks` 当前实际绑定的是 `free_blocks`，命名和语义不完全一致，后续实现时应明确这是“释放一份引用”，不能重复调用。

### 3.2 当前 `free_seq()` 没有 Put 感知

[scheduler.hpp](./DeepInfer/csrc/engine/scheduler.hpp:738)

```cpp
block_manager->free_blocks_nolock(seq.cache.block_table);
seq.cache.block_table.clear();
```

所以 Put task 必须在 `free_seq()` 前完成：

```text
snapshot block_ids
  -> pin_blocks(block_ids)
  -> enqueue PutTask
  -> free_seq()
```

不能在 `free_seq()` 之后再从 `seq.cache.block_table` 获取 block ID，因为该表已经被清空。

### 3.3 必须避免 Prefill chunk 重复提交

DeepInfer 支持 chunked prefill。`pe_do_save_kvcache()` 会在每个 prompt step 触发，而不是只在完整 prompt 结束时触发。

`llm_engine.py` 已经通过：

```python
seq_metadata.is_last_prompt_chunk
```

判断最后一个 prompt chunk。

因此 Prefill Put 必须满足：

```text
仅最后一个 prompt chunk 提交完整 prompt block
```

不能在每个 chunk 结束时把当前 block 当作最终 block 提交，否则会产生：

- 同一个 key 多次 Put；
- 未完成 block 被错误写入；
- hash 与 block 内容不一致；
- 后续请求命中不完整 KV。

推荐使用：

```text
is_last_prompt_chunk
且 block 是完整 block
且尚未提交过
```

作为提交条件。

---

## 4. 推荐的 DeepInfer 新方案

### 4.1 新增统一的 Put task ledger

不建议将 Put 状态分散在 `Scheduler`、`LLMEngine` 和 worker 中。建议在 Engine 侧增加统一记录：

```text
PutTaskLedger:
    task_id
    request_id
    request_generation
    block_ids
    npu_event
    expected_rank_count
    completed_rank_count
    state
```

状态至少包括：

```text
CREATED
PINNED
QUEUED
RUNNING
SUCCEEDED
FAILED
CANCELLED
RELEASED
```

其中：

- `block_ids` 必须是独立快照；
- `request_generation` 防止 request ID 重用时误释放旧任务；
- `completed_rank_count` 用于多 rank 聚合；
- `RELEASED` 是幂等保护，确保 unpin 只执行一次。

### 4.2 Prefill 流程

```text
pe_prepare_save_kvcache()
  -> 只选择最后一个 prompt chunk
  -> 计算完整 block 范围
  -> 复制 block_ids
  -> pin_blocks(block_ids)
  -> 创建 PutTask
  -> 记录模型写入 event

pe_do_save_kvcache()
  -> 对 PutTask 绑定 event
  -> 提交到 KVPoolWorker
  -> 不等待 Store.put
```

KV worker 执行：

```text
worker:
  -> wait(event)
  -> 根据 block_ids 计算地址和 key
  -> Store.put(...)
  -> 生成 PutCompleted(task_id, rank, result)
```

### 4.3 CE 流程

CE 的 `do_save_kvcache()` 不能默认保存全部 block，而应只保存：

```text
[上一次已保存位置, 当前已提交 token 对齐位置)
```

需要保存：

- 已经写满的 block；
- 当前请求结束时的所有完整 block；
- 不保存未完成尾块。

建议维护：

```text
request_id -> next_unsaved_block
```

这样可以避免 Decode 阶段每一步重复扫描和重复 Put。

### 4.4 请求结束流程

```text
请求结束
  -> 生成最后一个 PutTask
  -> pin_blocks(block_ids)
  -> enqueue
  -> finalize_seq()
       -> scheduler.free_seq()
       -> 只释放 request 引用
  -> 后台 Store.put
  -> completion poll
  -> unpin_blocks(block_ids)
```

这与 vLLM 的 `_reference_save_blocks()` 和 `_pinned_saves` 语义一致。

---

## 5. completion poll 应放在哪里

completion 不应由 Store worker 直接调用 `BlockManager.unpin_blocks()`，原因是：

- worker 线程不应直接修改 Scheduler 的请求生命周期；
- 多 rank completion 需要聚合；
- abort 和 request ID 重用需要主线程判定；
- `BlockManager` 与 Scheduler 需要保持统一锁顺序。

建议由 Engine 主线程在每个 step 的固定位置轮询：

```text
step 开始或 post_process()
  -> poll_put_completions()
  -> 聚合 task_id / rank
  -> task 完成后 unpin_blocks()
  -> 删除 ledger 记录
```

同时，Engine 的继续运行条件必须包含：

```text
scheduler.has_unfinished_seqs()
或
put_task_ledger.has_pending_tasks()
```

否则请求已经结束、队列为空时，Engine 可能停止 step，导致 completion 永远不能被消费，block 永久保持 pin。

vLLM 对应的机制是：

```python
has_pending_push_work()
```

Mooncake Store scheduler 通过该状态保证仍有 Store job 时 Engine 继续 step。

---

## 6. abort、失败和重复场景

### Store Put 失败

```text
Put failed
  -> 记录失败指标
  -> unpin
  -> 不影响推理结果
  -> 后续请求按 cache miss 处理
```

### abort

如果 Put 尚未入队：

```text
取消 task
  -> unpin
  -> 删除 ledger
```

如果 Put 已经开始：

```text
不能强行释放 block
  -> 等 worker 返回完成或失败
  -> 再 unpin
```

### worker 异常

vllm-ascend 已有处理方式：即使 worker 进入 fatal 状态，也必须为已注册任务发送 completion，避免 pin 永久不释放。

DeepInfer 也必须保证：

```text
worker 退出 / Store 初始化失败 / Put 抛异常
  -> 每个已 pin task 都产生失败 completion
  -> 主线程统一 unpin
```

### 重复 Put

Mooncake `-705` 应视为幂等成功。

同一个 request 的重复提交需要通过：

```text
request_generation + next_unsaved_block
```

去重，不能只依赖 `request_id`。

---

## 7. 是否需要修改 Scheduler

当前推荐方案中，Scheduler 不负责调用 Store，也不负责轮询 Store completion。

Scheduler 需要的改动较小：

1. 保持 `free_seq()` 的 request 引用释放语义。
2. 如果 Engine 的运行条件只检查 Scheduler 队列，需要新增一个由 Engine 提供的 pending Put 状态，或增加类似 `has_pending_put_tasks()` 的查询。
3. 如果后续希望由 Scheduler 统一接管请求结束异步释放，可以仿照 vLLM 增加：
   ```text
   request_finished(seq) -> delay_free
   ```
   但这不是当前最小适配路径的必要条件。

推荐职责边界：

```text
Scheduler:
    管理 request block 引用

LLMEngine:
    决定 Prefill/CE 提交时机
    pin
    poll completion
    unpin

KVPoolWorker:
    等待 event
    执行 Store.put
    返回 task completion
```

## 8. 最终适配判断

DeepInfer 可以直接复用：

- `BlockManager` 引用计数；
- `pin_blocks/unpin_blocks`；
- `pe_prepare_save_kvcache()`；
- `pe_do_save_kvcache()`；
- `do_save_kvcache()`；
- `finalize_seq()`；
- `DeferredRelease` 的 event 延迟释放模式。

需要新增：

- Put task ledger；
- block 快照；
- event 传递；
- worker completion；
- 多 rank 完成聚合；
- pending Put 驱动的 Engine stepping；
- chunked prefill 最后 chunk 过滤；
- abort、异常和重复提交清理。

因此，适配重点是 `LLMEngine + KVPoolWorker + BlockManager 引用管理`，不是单独修改 DeepInfer `Scheduler`。

> 请先说明为什么需要新增Put task ledger；
>
> 1. block 快照；
> 2. event 传递；
> 3. worker completion；
> 4. 多 rank 完成聚合；
> 5. pending Put 驱动的 Engine stepping；这些？

需要这些新增对象，是因为异步 Put 会把“请求生命周期”和“Store 任务生命周期”分开。请求可以结束并释放自己的引用，但 Store worker 仍可能读取同一块 HBM。必须有一套独立的任务状态来管理这段额外生命周期。

## 为什么需要 Put task ledger

Put task ledger 是 Engine 侧的任务账本，负责记录：

```text
task_id
request_id
block_ids
event
已完成 rank 数
任务状态
```

它解决的是“这块 block 由哪个异步任务持有、何时可以释放”的问题。

没有 ledger 时会出现：

- completion 无法找到对应的 block；
- 同一个请求多次 Put 时无法区分任务；
- request 已结束后，`seq` 对象可能已释放；
- worker 失败或重复回调时容易重复 unpin；
- request ID 重用时，旧任务可能误释放新请求的 block。

所以 ledger 不是为了替代 Scheduler，而是为了保存 Scheduler 生命周期之外的 Put 任务状态。

---

## 1. 为什么需要 block 快照

`seq.cache.block_table` 是请求当前使用的 block 表，会继续被 Scheduler 修改。

DeepInfer 的 `free_seq_nolock()` 还会：

```cpp
block_manager->free_blocks_nolock(seq.cache.block_table);
seq.cache.block_table.clear();
```

因此 Put worker 不能异步持有：

```text
seq
seq.cache.block_table
```

而必须在提交时复制：

```text
task.block_ids = copy(seq.cache.block_table)
```

流程是：

```text
复制 block_ids
  -> pin(block_ids)
  -> 提交 PutTask
  -> free_seq() 清理 seq
  -> worker 使用 task.block_ids
```

block 快照保证：

- `seq` 被释放后任务仍然有完整地址索引；
- 请求后续扩展或重建 block table 不影响当前 Put；
- abort、chunked prefill、request ID 重用不会修改正在执行的任务。

vLLM 的 `_reference_save_blocks()` 也是基于保存任务自己的 `block_ids`，而不是异步读取 request 当前 block table。

---

## 2. 为什么需要 event 传递

模型写 KV 和 Store Put 可能属于不同线程或不同设备 stream。

如果模型还没有完成写入，worker 就执行：

```text
Store.put(HBM address)
```

Store 可能读到未完成或不一致的 KV 数据。

因此需要传递模型写入完成 event：

```text
模型执行
  -> record event
  -> PutTask 携带 event
  -> worker wait(event)
  -> Store.put
```

event 的作用是建立：

```text
GPU KV 写入完成
    happens-before
Store DMA 读取 HBM
```

这比在 Engine 主线程直接同步更适合异步 Put，因为主线程可以继续处理调度和其他请求。

vllm-ascend 的保存线程也会在真正调用 Store 前等待 `current_event`，这是同类机制。

---

## 3. 为什么需要 worker completion

Put 是异步提交的，调用 `enqueue()` 返回时只代表：

```text
任务已经进入队列
```

不代表：

```text
Store.put 已经完成
```

因此必须由 worker 返回明确的 completion：

```text
PutCompleted(task_id, rank, success)
```

Engine 只有收到 completion 后，才能执行：

```text
unpin(block_ids)
```

completion 还需要携带失败结果，因为以下情况都必须释放 pin：

- Store 返回失败；
- Store 抛异常；
- worker 进入 fatal 状态；
- 任务被取消；
- 重复 Put 被判定为幂等成功。

如果没有 completion，Engine 无法判断什么时候释放额外引用，最终会出现：

- 过早 unpin，导致 HBM 被复用；
- 永不 unpin，导致 block 泄漏；
- Store 失败后 block 永久占用。

---

## 4. 为什么需要多 rank 完成聚合

在 TP、PP 或多 rank 部署中，同一个逻辑 block 通常对应多个 rank 的本地 HBM 数据。

例如：

```text
逻辑 block 0
  -> rank 0 的 KV
  -> rank 1 的 KV
  -> rank 2 的 KV
```

每个 rank 可能独立执行：

```text
Store.put(local_rank_block)
```

如果 rank 0 先完成就立即 unpin，而 rank 1、rank 2 仍在读取自己的 HBM，就会破坏生命周期保证。

因此任务需要记录：

```text
expected_rank_count
completed_rank_count
```

只有：

```text
completed_rank_count == expected_rank_count
```

才释放该 Put task 的 block 引用。

vLLM Mooncake Store 的 `_pinned_saves` 就采用了：

```text
store_job_id -> (block_ids, remaining_worker_count)
```

vllm-ascend 的 `InFlightSaveBlockTracker` 也使用 expected worker count。

单卡单 rank 场景可以把聚合数量固定为 1，但保留这个抽象可以避免后续 TP 适配时重新设计。

---

## 5. 为什么需要 pending Put 驱动 Engine stepping

当前 DeepInfer 的 completion 处理需要挂在 Engine 主循环中，例如：

```text
step()
  -> poll Put completion
  -> unpin 完成任务
```

但请求结束后，Scheduler 可能已经没有：

- waiting 请求；
- running 请求；
- prompted 请求。

如果 Engine 的继续运行条件只依赖 Scheduler：

```text
has_unfinished_seqs() == false
```

Engine 就可能停止 step，导致：

```text
Put 已完成
但 completion 没有被 poll
block 一直保持 pin
```

所以需要把 pending Put 纳入 Engine 的继续条件：

```text
继续 step =
    Scheduler 有未完成请求
    或存在 pending Put task
```

这和 vLLM 的：

```python
has_pending_push_work()
```

作用相同，保证异步任务完成通知最终能回到调度主线程。

这里的“驱动 stepping”不是让 Scheduler 等待 Put，也不是阻塞当前请求，而是让 Engine 在没有推理请求时继续运行少量轮询步骤，直到释放所有 Put 引用。

如果未来改成独立 completion 回调线程，并且该线程可以安全地完成多 rank 聚合和 `unpin`，则可以不依赖 stepping。但按 DeepInfer 当前 Engine、Scheduler、BlockManager 的线程边界，主线程 poll 是更容易保持一致性的方案。

## 关系总结

```text
block 快照
    保证异步任务仍知道读哪些 HBM block

event 传递
    保证模型写入完成后再读 HBM

worker completion
    告诉主线程 Store 任务何时结束

多 rank 聚合
    保证所有 rank 都结束后再释放

pending Put stepping
    保证 completion 一定会被主线程消费

Put task ledger
    把上述信息绑定到同一个 task，并保证状态和释放操作幂等
```

这五项共同组成异步 Put 的生命周期管理；缺少其中任意一项，都可能出现数据未写完、block 过早复用或 pin 永久不释放。

> 不是，我理解是不是没有这么复杂？比如说写入时机是两个是
>
> 1. 在prefill结束后（如果是chunk prefill，需要考虑所有prefill都完成）
> 2. 在decode阶段满一个block size就
>
> 在上述两个写入时机的是偶发起一个异步put的任务，同时会进行pin，然后在kv work 线程中异步执行就行，执行完后的会进行unpin的操作，这样是不是就完成了？

你的理解基本正确。这个需求可以先按“两个触发点 + 一个异步任务队列”实现，不需要一开始引入很复杂的状态机。

## 最小可行流程

```text
满足写入条件
  -> 确定本次要写的 block_ids
  -> 复制 block_ids
  -> pin_blocks(block_ids)
  -> 创建 PutTask 并入队
  -> KV worker 线程等待必要的 GPU event
  -> 执行 Store.put
  -> Put 完成后 unpin_blocks(block_ids)
```

两个写入时机分别是：

### 1. Prefill 完成

普通 Prefill：

```text
Prefill 完成
  -> 提交所有完整 prompt block
```

Chunk Prefill：

```text
每个 chunk 完成
  -> 暂不提交最终 Prefill Put

最后一个 prompt chunk 完成
  -> 提交完整 prompt 中已经完成的 block
```

DeepInfer 已有 `seq_metadata.is_last_prompt_chunk`，可以用它判断是否已经完成全部 Prefill。

### 2. Decode 写满 block

Decode 阶段每当累计完成一个完整 block：

```text
block 尚未提交
且 token 数达到 block_size
  -> 提交该完整 block
```

必须维护一个很小的保存进度，例如：

```text
request_id -> next_unsaved_block
```

否则每个 Decode step 都可能重复提交同一个 block。重复 Put 虽然可以依赖 Mooncake 的幂等行为，但会增加 Store 请求和 HBM pin 压力。

## 为什么仍然需要任务对象

不需要单独设计复杂的 Put ledger，但 `PutTask` 本身至少要带：

```text
task_id
block_ids
store keys
HBM 地址和长度
GPU 完成 event
```

其中最重要的是 `block_ids` 快照：

```text
task.block_ids = copy(seq.cache.block_table[...])
```

不能让 worker 在异步执行时再读取 `seq.cache.block_table`，因为请求可能已经结束，Scheduler 可能已经执行：

```cpp
free_seq()
```

并清空 block table。

因此：

```text
PutTask 本身就是最小生命周期记录
```

不一定需要额外的 `PutTaskLedger` 容器。

## unpin 由谁执行

逻辑上是：

```text
Store.put 完成
  -> unpin(block_ids)
```

但建议由 Engine 主线程消费 worker 的完成通知后执行：

```text
worker:
  -> Put
  -> completion_queue.push(task_id, result)

Engine:
  -> poll completion
  -> unpin(task.block_ids)
```

原因是 Engine 主线程掌握 Scheduler、BlockManager 和请求生命周期，比较容易处理：

- 请求已经结束；
- 请求被 abort；
- worker 失败；
- 重复 completion；
- Engine shutdown。

如果确认 KV worker 与 `BlockManager` 的生命周期严格绑定，也可以让 worker 直接调用 `unpin_blocks`，但需要额外保证并发锁、对象生命周期和异常路径，当前不建议作为第一版方案。

## GPU event 是否必须

如果 Store worker 直接读取 HBM，则必须保证模型已经写完本次 KV：

```text
模型写 KV
  -> record event
  -> PutTask 携带 event
  -> worker wait(event)
  -> Store.put
```

如果提交 Put 的位置本身已经同步完成 GPU 写入，则可以不额外传 event。但异步线程通常不能默认认为主线程的 GPU 操作已经完成，所以最好保留 event 字段。

vllm-ascend 的保存线程也是在实际 Store 操作前等待对应 event。

## 多 rank 是否必须做复杂聚合

第一版可以按每个 rank 独立处理：

```text
每个 rank：
  pin 本 rank 的 block
  执行本 rank 的 Store.put
  完成后 unpin 本 rank 的 block
```

如果 DeepInfer 的 block 只表示本 rank 的 HBM block，这样就足够。

只有在以下情况下才需要额外的多 rank 聚合：

- 一个逻辑 block 必须等所有 rank 都写成功后才算完成；
- block 的释放由一个统一进程管理；
- Store completion 只在 rank 0 汇总；
- 需要发布“整个逻辑 block 已可命中”的事件。

因此多 rank 聚合不是最小异步 Put 的必选项，但后续做跨 rank 命中事件时需要补上。

## pending Put 是否必须驱动 Engine stepping

这取决于 completion 的消费方式。

如果 Engine 主循环是：

```text
step()
  -> poll completion
  -> unpin
```

那么需要保证有 pending Put 时 Engine 仍会继续 step，否则请求都结束后可能不再 poll。

如果 worker 能够独立安全地完成 unpin，则不需要 pending Put 驱动 Engine stepping，但需要处理好 worker 线程与 BlockManager 的生命周期。

结合当前 DeepInfer 架构，建议采用：

```text
worker 只执行 Put 并产生 completion
Engine 每轮 poll completion 并 unpin
```

然后把：

```text
pending_put_tasks > 0
```

加入 Engine 的继续运行条件。

## 最小方案总结

你的方案可以收敛为：

```text
Prefill 最后一个 chunk 完成
或
Decode 新写满一个 block
  -> 生成 block 快照
  -> pin
  -> PutTask 入队

KV worker:
  -> 等待 GPU event
  -> Store.put
  -> 返回完成通知

Engine:
  -> 消费完成通知
  -> unpin
```

第一版不必引入完整的复杂 ledger，但不能省略：

1. PutTask 中保存独立的 block ID；
2. 每个 block 只在满足条件时提交一次；
3. Store 操作完成后再 unpin；
4. Put 失败也必须 unpin；
5. Chunk Prefill 只在最后一个 chunk 触发最终 Prefill Put；
6. 如果由 Engine poll completion，则需要 pending Put 保证 Engine 继续运行。

> 请确认这个最小方案和vllm实现以及是否可以正常适配到deepinfer里，同时对比一下设计文档草稿，确认是否有出入？

<details><summary>2 previous messages</summary>

> 我会把“最小方案”逐项对照 vLLM 的实际保存引用、完成通知和释放路径，再对照 DeepInfer 当前的 Prefill/Decode 调用点，最后列出与 `设计文档草稿.md` 不一致或需要收敛的地方。此次仍只输出核对结论，不修改文档。
>
> <details><summary>Explored 7 files</summary>
>
> - Read `./设计文档/设计文档草稿.md`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/csrc/engine/scheduler.hpp`
> - Read `./vllm/vllm/v1/core/sched/scheduler.py`
> - Read `./vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py`
> - Read `./vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`
>
> </details>
</details>

结论：这个最小方案与 vLLM 的核心生命周期设计一致，也可以适配 DeepInfer；但它是 vLLM 方案的简化子集，需要明确几个边界。

## 1. 与 vLLM 实现的对应关系

| 最小方案 | vLLM 对应实现 | 结论 |
|---|---|---|
| Prefill 完成后提交 Put | Store scheduler 为当前保存任务生成 metadata 并提交 worker | 一致 |
| Decode 满完整 block 后提交 Put | vLLM 按新增可保存 block 生成保存任务 | 一致 |
| 提交前 pin | `_reference_save_blocks()` 对每个 Store job 增加 block 引用 | 一致，DeepInfer 可用 `pin_blocks()` |
| Put worker 异步执行 | Mooncake worker / vllm-ascend sending thread | 一致 |
| Put 完成后 unpin | `completed_saves` 到达 scheduler 后释放保存任务引用 | 一致 |
| 保存进度 | vLLM 使用 `store_job_id`、saved offset、request tracker | 最小方案至少需要 `next_unsaved_block` |
| Engine 继续轮询 | vLLM 使用 `has_pending_push_work()` 保证完成通知被消费 | 当前 DeepInfer 需要补充 |

vLLM 的核心不是“提交一个 Put 就结束”，而是：

```text
保存任务拥有独立 block 引用
  -> worker 异步读取 HBM
  -> worker 返回完成信息
  -> 任务引用释放
```

你的最小方案保留了这个必要部分，因此方向正确。

## 2. 最小方案可以不引入完整 ledger

不需要复制 vLLM 的完整 `_pinned_saves`、request tracker 等结构。

第一版可以把必要状态直接放在 `PutTask` 中：

```text
PutTask:
    task_id
    request_id
    block_ids
    keys
    addrs
    sizes
    ready_event
```

再配合一个简单的：

```text
request_id -> next_unsaved_block
```

即可满足阶段一需求。

但 `PutTask` 本身不能只保存 `request_id`，必须保存 `block_ids` 快照。因为 DeepInfer 的 `free_seq_nolock()` 会释放并清空 `seq.cache.block_table`：

[scheduler.hpp](./DeepInfer/csrc/engine/scheduler.hpp:738)

## 3. DeepInfer 的适配点

### Prefill

DeepInfer 当前流程：

```text
pre_sampler_sync()
  -> pe_prepare_save_kvcache()
  -> 模型执行
post_process()
  -> pe_do_save_kvcache()
  -> process_model_outputs()
```

因此建议：

```text
pe_prepare_save_kvcache()
  -> 判断是否为最后一个 prompt chunk
  -> 计算完整 block
  -> 复制 block_ids
  -> pin_blocks()
  -> 准备 PutTask 和 GPU event

pe_do_save_kvcache()
  -> 绑定模型写入完成 event
  -> 将 PutTask 放入 KV worker 队列
```

Chunk Prefill 不能在每个 chunk 结束时提交最终 Prefill Put。应使用已有的：

```python
seq_metadata.is_last_prompt_chunk
```

只在最后一个 chunk 提交完整 prompt 中的完整 block。

`send_processed_prompts()` 可以继续负责 PD 发送，但不建议同时负责相同 block 的 Put，否则容易与 `pe_do_save_kvcache()` 重复提交。Prefill Put 应明确由一个入口负责。

### Decode

Decode 满一个 block 后：

```text
已提交 token 数达到 block_size
  -> 找到尚未保存的完整 block
  -> 复制 block_ids
  -> pin
  -> PutTask 入队
  -> next_unsaved_block 前移
```

这里必须使用实际已经提交的 token 数，不能使用 ZOS 的乐观 token 数。设计草稿中关于这一点的描述是正确的。

DeepInfer 当前 `do_save_kvcache()` 在 `pre_sampler_sync()` 调用：

[llm_engine.py](./DeepInfer/deepinfer/engine/llm_engine.py:830)

因此需要确认它拿到的是上一轮已经完成写入并提交的 token 状态。若要严格按“当前 Decode step 写满后立即提交”，则应让 `do_save_kvcache()` 使用上一轮完成状态，或在 `post_process()` 增加 Decode block Put 触发点。

## 4. unpin 的实现方式

推荐：

```text
KV worker:
    等待 ready_event
    执行 Store.put
    写入 completion_queue

Engine:
    poll completion_queue
    根据 task.block_ids 调用 unpin_blocks()
```

这样可以保持：

- Scheduler 和 BlockManager 由 Engine 主线程管理；
- worker 只负责 Store 操作；
- Put 失败、异常、取消都统一经过 completion 清理。

如果未来确认 worker 可以安全持有 `BlockManager`，也可以由 worker 直接 unpin，但这需要额外处理对象生命周期和关闭流程，第一版不建议这样做。

## 5. 多 rank 是否需要完整聚合

如果每个 rank 只负责自己的 HBM block，那么第一版可以按 rank 独立处理：

```text
rank 0 pin -> put -> unpin
rank 1 pin -> put -> unpin
```

但如果要发布“逻辑 block 已完整可命中”的事件，则必须等待所有 rank 的 Put 完成。

设计草稿第 4.3 的描述：

> 每个 rank 写自己的 shard；所有 shard key 都存在才算命中

这是正确的。因此：

- block 生命周期可以先按本 rank 独立 unpin；
- 全局命中判定仍然要求所有 shard 存在；
- 后续如果增加 BlockStored 事件，需要增加 rank 聚合。

## 6. pending Put 是否必须保留

如果 completion 由 Engine 主循环 poll，则必须保留 pending Put 状态。

原因是当前 DeepInfer 的 Engine 可能在 Scheduler 没有请求时停止 step。如果停止后不再 poll：

```text
Put 已完成
但 completion 未消费
block 一直保持 pin
```

因此需要将：

```text
scheduler 有未完成请求
或存在 pending Put
```

作为 Engine 继续运行的条件。

这对应 vLLM 的：

```python
has_pending_push_work()
```

如果以后改为独立 completion 线程直接 unpin，则可以不依赖 pending Put stepping，但那不是当前 DeepInfer 最自然的线程模型。

## 7. 与设计文档草稿的出入

### 一致的部分

设计草稿以下内容与 vLLM 和最小方案一致：

- Prefill 完成后异步 Put；
- Decode 完整 block 可以异步 Put；
- 提交前 pin；
- Store 操作完成后 unpin；
- Put 不阻塞推理；
- 只保存完整 block；
- 使用 `next_block` 或等价状态避免重复提交；
- Put 失败不影响推理；
- 多 rank 分片独立写入，命中要求所有 shard 存在；
- block Put 不应写入 SWA。

对应草稿位置：

- `设计文档/设计文档草稿.md:365-398`
- `设计文档/设计文档草稿.md:411-420`

### 需要收敛的部分

#### 1. 草稿默认是请求结束写入

草稿中：

```text
CE_WRITE_MODE=end
```

是默认策略，Decode 满 block 的 `block` 模式是可选策略。

你的最小方案是：

```text
Prefill 完成
Decode 每满一个 block
```

这等价于选择 `CE_WRITE_MODE=block`，与草稿默认值不同，需要在方案中明确。

#### 2. 原始需求标题强调 CE 结束

原始需求是“PE Prefill/CE 结束时提交 Block Put Task”，这更接近草稿的 `end` 模式。

如果现在确定 Decode 满 block 也要写，需要明确这是需求范围扩展：

```text
PE：Prefill 完成时写
CE：可选按请求结束写，或按完整 block 增量写
```

否则“CE 结束时写”和“Decode 满 block 就写”会被理解为两个不同策略。

#### 3. 草稿中 Put 入口过多

草稿同时列出：

- `send_processed_prompts`
- `pe_do_save_kvcache`
- `do_save_kvcache`
- `finalize_seq`

作为 Put 可能入口。

这会带来重复提交风险。建议收敛为：

```text
Prefill Put：pe_do_save_kvcache
Decode Put：do_save_kvcache
finalize_seq：只负责释放 request 引用
send_processed_prompts：只负责 PD 发送
```

如果 `do_save_kvcache()` 只保留请求结束策略，则按 block 的 Decode Put 需要单独增加一个明确入口。

#### 4. 草稿对 pin 的描述基本正确，但没有写清引用关系

草稿说：

> driver 线程提交 PutTask 前 pin，完成回调 unpin

这个结论正确，但还应明确：

```text
free_seq() 释放 request 引用
unpin() 释放 Put task 引用
```

否则实现者可能误以为 pin 后必须延迟调用 `free_seq()`，从而重新引入请求级阻塞。

#### 5. 草稿中的 `block_pool.touch()` 不能原样搬到 DeepInfer

vLLM 使用 BlockPool 的引用机制，DeepInfer 当前对应机制是：

```python
block_manager.pin_blocks(block_ids)
block_manager.unpin_blocks(block_ids)
```

适配时应使用 DeepInfer 的 BlockManager API，而不是照搬 vLLM 的 `BlockPool.touch()` 名称。

## 最终确认

这个最小方案：

```text
Prefill 最后一个 chunk 完成
或
Decode 新写满一个 block
  -> 复制 block_ids
  -> pin
  -> PutTask 入队
  -> worker 等待 event 并执行 Store.put
  -> completion
  -> unpin
```

与 vLLM 的核心实现一致，可以适配 DeepInfer。

需要保留的最小状态只有：

```text
PutTask.block_ids
PutTask.task_id
PutTask.ready_event
request.next_unsaved_block
pending Put 数量
```

不需要第一版就实现完整的 vLLM request tracker、复杂 Store job ledger 或请求级延迟释放机制。最需要修改和确认的是：

1. Prefill 只在最后一个 chunk 提交；
2. Decode 保存进度不能重复提交；
3. `do_save_kvcache()` 的实际调用时机要对应已提交 token；
4. Put task 的 pin 必须早于 `free_seq()`；
5. completion 必须保证失败路径也能 unpin；
6. Engine 在 pending Put 存在时继续 poll；
7. 明确选择 CE `end` 模式还是 `block` 模式，不能同时作为默认行为。

> 有几个点需要确认一下
>
> 1. 是普通prefill完成的时候一次性写入put，或者如果是chunk prefill，那就是最后一次preill再写入
> 2. prefill写入也是按照block size的写入，如果不满block size的没有写入，这个不用传递给decode，可以在decode端算出prefill结尾不满block的和decode新生成的一起满block就进行put任务提交
> 3. 提交put的时机需要确认在deepinfer的具体代码的哪里增加这些写入流程？

<details><summary>1 previous message</summary>

> <details><summary>Explored 3 files, a search</summary>
>
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/deepinfer/worker/model_runner.py`
> - Read `./DeepInfer/deepinfer/worker/worker.py`
> - Searched for `is_last_prompt_chunk|chunked_rest_len|num_cached_tokens` in `scheduler.hpp`
>
> </details>
</details>

确认后，这三个判断可以成立，但需要区分“池化 Store Put”和“PE 到 CE 的 PD 传输”。

## 1. Prefill 的 Put 时机

### 普通 Prefill

普通 Prefill 只执行一次：

```text
Prefill forward 完成
  -> 计算完整 block
  -> 提交一次 Put
```

### Chunk Prefill

Chunk Prefill 不能每个 chunk 都提交最终 Prefill Put：

```text
chunk 1 完成 -> 不提交最终 Put
chunk 2 完成 -> 不提交最终 Put
最后一个 chunk 完成 -> 提交完整 block
```

DeepInfer 已有判断条件：

```python
seq_metadata.is_last_prompt_chunk
```

位置在：

[llm_engine.py](./DeepInfer/deepinfer/engine/llm_engine.py:910)

因此 Prefill Put 的条件应是：

```text
is_last_prompt_chunk
且 block 已完整
```

## 2. Prefill 的部分 block 是否写入

你的理解正确：

```text
Prefill 只写完整 block
Prefill 末尾未满 block 的部分不提交 Store Put
```

例如：

```text
block_size = 128
prompt = 300 tokens

Prefill 写入：
  block 0: 128
  block 1: 128
  block 2: 44 -> 不写
```

后续 Decode 继续使用同一个 block：

```text
Prefill 44 tokens
+ Decode 新生成 84 tokens
= 128 tokens
-> 提交 block 2
```

所以不能把最后的部分 block 当作独立 Put 对象，也不需要在 Store 中保存一个不完整 block。

但需要注意边界：

- 对于同一个 DeepInfer 实例内连续 Decode，这种方式成立。
- 如果 PE 和 CE 是不同实例，CE 不能直接使用 PE 的 HBM block。此时部分尾块要么通过 PD 传输，要么由 CE 重算，不能简单认为“不需要传给 Decode”。

这里讨论的“部分 block 不写入”只针对 KV pool 的 Store Put。

## 3. DeepInfer 具体增加位置

### 3.1 Prefill：`pe_prepare_save_kvcache`

当前入口：

[llm_engine.py](./DeepInfer/deepinfer/engine/llm_engine.py:818)

建议职责：

```text
pe_prepare_save_kvcache()
  -> 判断最后一个 prompt chunk
  -> 计算完整 block 数
  -> 复制 block_ids
  -> pin_blocks(block_ids)
  -> 创建待提交 PutTask
```

由于当前函数签名只有：

```python
pe_prepare_save_kvcache(scheduled, finished)
```

没有 `seq_metadata_list`，无法直接判断 `is_last_prompt_chunk`。

建议二选一：

```python
pe_prepare_save_kvcache(
    scheduled,
    finished,
    seq_metadata_list,
)
```

或者在 `LLMEngine.step()` 中先根据：

```python
zip(scheduler_outputs.scheduled_seqs, seq_metadata_list)
```

过滤出最后一个 chunk 的 sequence，再传给现有保存钩子。

不建议通过 `seq.schedule.chunked_rest_len == 0` 代替，因为该字段表示调度状态，不能完全替代当前 step 的 `is_last_prompt_chunk` 语义。

### 3.2 Prefill：`pe_do_save_kvcache`

当前入口：

[llm_engine.py](./DeepInfer/deepinfer/engine/llm_engine.py:988)

调用位置在模型执行完成、worker 同步采样 event 之后，因此适合提交 Put：

```text
模型执行完成
  -> pe_do_save_kvcache()
  -> PutTask 入 KV worker 队列
```

其职责应是：

```text
pe_do_save_kvcache()
  -> 使用 prepare 阶段保存的 block_ids
  -> 绑定 GPU 写入完成 event
  -> 提交异步 PutTask
```

`send_processed_prompts()` 不建议再次提交相同 block 的 Put。它应继续负责 Prompt Engine 的请求转移或 PD 发布。

当前调用顺序是：

[llm_engine.py](./DeepInfer/deepinfer/engine/llm_engine.py:951)

```text
model execution
  -> post_process()
       -> pe_do_save_kvcache()
       -> process_model_outputs()
  -> send_processed_prompts()
```

因此 Prefill Put 的唯一提交入口建议放在 `pe_do_save_kvcache()`。

### 3.3 Decode：`do_save_kvcache`

当前入口：

[llm_engine.py](./DeepInfer/deepinfer/engine/llm_engine.py:834)

```python
if not scheduler_outputs.prompt_run:
    self.do_save_kvcache(...)
```

该调用发生在 `pre_sampler_sync()` 中，并且会先同步 `copy_state_event`。

Decode 阶段建议由 `do_save_kvcache()` 处理：

```text
本轮开始
  -> 查看上一轮已经提交的 token 数
  -> 找到新形成的完整 block
  -> pin
  -> 提交 PutTask
```

这里虽然调用点在当前模型执行前，但它处理的是上一轮已经完成的 KV。这样还可以保证请求结束前完成 pin，避免后续：

```text
finalize_seq()
  -> scheduler.free_seq()
```

把正在被 Store worker 读取的 block 释放掉。

需要维护：

```text
seq -> next_unsaved_block
```

避免每个 Decode step 重复提交同一个 block。

### 3.4 `finalize_seq` 不负责提交 Put

当前：

[llm_engine.py](./DeepInfer/deepinfer/engine/llm_engine.py:804)

```python
def finalize_seq(self, seq):
    self.scheduler.free_seq(seq)
```

建议保持它只做 request 自身 block 引用的释放。

Put 必须在 `finalize_seq()` 前完成：

```text
do_save_kvcache()
  -> pin + 创建 PutTask
finalize_seq()
  -> free_seq() 释放 request 引用
worker 完成
  -> unpin() 释放 Put 引用
```

这样不需要修改 Scheduler 的 Store 逻辑。

## 4. 最小实现的完整时序

### Prefill

```text
schedule()
  -> 生成 seq_metadata_list
  -> 判断 is_last_prompt_chunk
  -> 模型执行
  -> pe_do_save_kvcache()
       只选完整 block
       pin
       PutTask 入队
  -> send_processed_prompts()
```

### Decode

```text
schedule()
  -> pre_sampler_sync()
       do_save_kvcache()
         找到新完成的完整 block
         pin
         PutTask 入队
       finalize_seq()
         free_seq()
  -> 模型执行下一步
```

### Worker

```text
PutTask
  -> 等待 GPU event
  -> Store.put()
  -> 返回 task completion
```

### Engine

```text
下一轮 step 或 completion poll
  -> 收到 task completion
  -> unpin(task.block_ids)
```

## 5. 与设计文档草稿的出入

### 一致部分

草稿中以下内容与当前结论一致：

- Prefill 只写完整 block；
- Prefill 末尾不完整 block 不写；
- Decode 可以在完整 block 形成后写入；
- Put 前 pin，完成后 unpin；
- Put 失败不阻塞推理；
- 使用 `next_block` 或等价字段避免重复提交。

对应位置：

- `设计文档/设计文档草稿.md:365-398`
- `设计文档/设计文档草稿.md:411-420`

### 需要调整或明确部分

1. 草稿将 CE 的 `end` 模式作为默认值，Decode 满 block 是可选的 `block` 模式。

   当前讨论的方案实际选择的是：

   ```text
   Prefill 最后一个 chunk 写完整 block
   Decode 每形成一个完整 block 就写
   ```

   这等价于选择 `CE_WRITE_MODE=block`，需要明确是阶段一默认策略，还是保留为配置项。

2. 草稿把 `send_processed_prompts`、`pe_do_save_kvcache`、`do_save_kvcache`、`finalize_seq` 都列为可能的 Put 入口。

   建议收敛为：

   ```text
   pe_do_save_kvcache：Prefill Put
   do_save_kvcache：Decode block Put
   send_processed_prompts：只做 PD/Prompt 发送
   finalize_seq：只释放 request 引用
   ```

3. 草稿中“Prefill 结束时提交 Put”需要补充：

   ```text
   Chunk Prefill 只有最后一个 chunk 才触发
   ```

4. 草稿中“Prefill prompt 的所有完整 block”是正确的，但需要明确不包括最后未满的 block。

5. 草稿中使用 `seq.num_cached_tokens` 计算 block 数的伪代码，需要确认这个字段已经反映实际提交 token。Decode Put 应基于 `process_outputs()` 已提交的 token 数，不能使用 ZOS 的乐观值。

## 最终结论

这三个判断可以确定：

```text
普通 Prefill：
  一次完成后提交完整 block

Chunk Prefill：
  只有最后一个 chunk 完成后提交完整 block

Decode：
  每形成一个新的完整 block 就提交 Put
```

DeepInfer 的主要新增位置是：

```text
pe_prepare_save_kvcache()
  -> Prefill block 筛选和 pin 准备

pe_do_save_kvcache()
  -> Prefill PutTask 提交

do_save_kvcache()
  -> Decode 完整 block PutTask 提交

KVPoolWorker
  -> 异步 Store.put

Engine completion poll
  -> unpin
```

不需要把 Store Put 逻辑放进 C++ Scheduler；Scheduler 继续负责 request block 的正常分配和释放，Put task 通过额外 pin 保持 HBM 生命周期。

> 草稿只是借鉴，如果最终方案主要是和vllm流程一致就行。上述不是说pd之间不用传输，是说prefill如果结尾33个token没有写入put任务，这个数可以不用传递给decode，可以在decode直接算出来然后和decod新生成的一个满block一起put。然后还需要确认的就是deepinfer预留了一些虚函数做kv cache的保存，也需要明确这个需求实现的时候，虚函数是接入到deepinfer主流程代码的哪个地方。
>
> 确认清楚后可以在/Users/l/L/Code/TuringProject/D/设计文档里设计当前put需求的方案草稿

<details><summary>7 previous messages</summary>

> 已确认最终边界：Prefill 只提交最后一个 chunk 中的完整 block，末尾未满 block 留在本地，由 Decode 继续填满后再提交；PD 传输仍按原协议处理。现在我会按现有 `REQ-KVP-04` 的文档风格，新增 `REQ-KVP-06` 草稿，明确 DeepInfer 虚函数在主流程中的接入位置，并补充异步 Put 时序图、伪代码和主要 ST 用例。
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
> - Read `./cann-deepinfer-recipe/docs/requirements.yaml`
>
> </details>
>
> 我会新增 `REQ-KVP-06-block-put` 目录，文档中把两类触发条件固定为：最后一个 Prefill chunk 的完整 block，以及 Decode 期间新形成的完整 block；未满 block 不单独写入，后续由同一请求继续填满。虚函数接入会按 `pre_sampler_sync()`、模型执行、`post_process()` 的真实顺序分别说明。
>
> <details><summary>Created 2 files, edited a file, explored 2 files, 3 searches, ran 3 commands</summary>
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+289 -0)
> - Wrote `./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg` (+71 -0)
>
> - Ran `xmllint --noout 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg && rg -n "chunk或|def poll_completions|未满尾 block不" 设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md`
>
> - Ran `git diff --stat -- 设计文档/REQ-KVP-06-block-put; sed -n '1,260p' 设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md`
>
> - Ran `wc -l 设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md; tail -80 设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md`
>
> <details><summary>Explored 2 files, 3 searches</summary>
>
> - Searched for `pre_sampler_sync\(` in `deepinfer`
> - Searched for `def step\(|pre_sampler_sync_fn` in `llm_engine.py`
> - Read `./DeepInfer/deepinfer/worker/worker.py`
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Searched for `poll_put_completions|PutTask|next_unsaved_block|submit_put` in `REQ-KVP-06-block-put.md`
>
> </details>
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+10 -2)
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+3 -3)
>
> </details>
>
> 文档和 SVG 已创建。现在做一轮结构校验：检查 XML、伪代码与 DeepInfer 当前回调顺序是否一致，并确认文档没有把 PD 传输和 Store Put 混为一条路径。
>
> <details><summary>Explored a search, ran 3 commands</summary>
>
> - Ran `command -v rsvg-convert || true; find 设计文档/REQ-KVP-06-block-put -maxdepth 2 -type f -print`
>
> - Ran `xmllint --noout 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `chunk 或|pre_sampler_sync|pe_prepare|pe_do|do_save|send_processed|PD|未满` in `REQ-KVP-06-block-put.md`
>
> </details>
>
> - Ran `rsvg-convert -o /tmp/async-block-put-sequence.png 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg`
>
> </details>
>
> Viewed an image
</details>

已生成方案草稿：

- [REQ-KVP-06-block-put.md](./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md)
- [async-block-put-sequence.svg](./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg)

文档已明确：

- 普通 Prefill 一次提交完整 block。
- Chunk Prefill 仅最后一个 chunk 提交完整 block。
- Prefill 尾部未满 block 不写 Store，后续 Decode 填满后提交。
- Decode 按 `next_unsaved_block` 增量提交完整 block。
- `pe_prepare_save_kvcache` 负责筛选、复制 block ID 和 pin。
- `pe_do_save_kvcache` 负责 Prefill PutTask 提交。
- `do_save_kvcache` 负责 Decode 完整 block Put。
- `finalize_seq` 只释放请求自身引用。
- KV worker 异步执行 Store Put，Engine 消费 completion 后 unpin。
- PD HBM 传输与 Store Put 保持独立。

SVG 已通过 `xmllint` 校验，并完成渲染检查。

> 有几个修改或确认点：
>
> 1. 2.4节不需要
> 2. 需要确认一下LLMEngine的post_process是否有被调用？

<details><summary>2 previous messages</summary>

> 我先确认调用链：`LLMEngine.post_process()` 不是闲置虚函数，它通过 `EngineCallbacks.post_process_fn` 注册，并由 Worker 的 `_post_process_step()` 在采样完成事件同步后调用。随后我会删除文档的 2.4 小节并保留其必要结论到对应章节。
>
> <details><summary>Edited a file, explored 3 files, 2 searches, ran a command</summary>
>
> <details><summary>Explored 3 files, a search</summary>
>
> - Read `./DeepInfer/deepinfer/engine/llm_engine.py`
> - Read `./DeepInfer/deepinfer/worker/worker.py`
> - Searched for `^### 2\.4|^## 3\.|与 vLLM 的对应|DeepInfer 主流程` in `REQ-KVP-06-block-put.md`
> - Read `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md`
>
> </details>
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+0 -13)
>
> - Ran `xmllint --noout 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `^### 2\.4|^## 3\.|post_process_fn|_post_process_step|pe_do_save_kvcache` in `REQ-KVP-06-block-put.md`
>
> </details>
>
> </details>
</details>

已完成两项修改和确认。

1. 已从 [REQ-KVP-06-block-put.md](./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md) 删除原 2.4 小节。

2. `LLMEngine.post_process()` 确实会被调用：

```python
# llm_engine.py:399
EngineCallbacks(post_process_fn=self.post_process, ...)
```

Worker 在 `_post_process_step()` 中：

```python
# worker.py:262-271
step_state.sample_end_evt.synchronize()
self.model_runner.post_process(...)
self.engine_callbacks.post_process_fn(step_state)
```

因此实际调用链是：

```text
Worker._post_process_step()
  -> 等待 sample_end_evt
  -> ModelRunner.post_process()
  -> LLMEngine.post_process()
       -> pe_do_save_kvcache()
       -> process_model_outputs()
```

这也确认了 `pe_do_save_kvcache()` 位于模型执行完成之后，适合作为 Prefill PutTask 的提交位置。SVG 仍通过 `xmllint` 校验。

> 1. 2.2的异步block生命周期图有点虚线没画到对应位置上，不确定作用，如果不行看是否时序图明确对应流程或者调用伪代码的方法的步骤流程，确保图没有重叠或者流程不清晰的地方
> 2. 全文中“`​LLMEngine​ 在 Prefill 最后一个 chunk `”这种描述是不是不太合理？还是是普通prefill不是chunk prefill的模式的，请重新看看这个表述如何准确专业
> 3. 3.1中调用顺序需要说明，现在deepinfer的主流程是哪些，以及虚函数需要新增的异步put逻辑是哪些，要突出当前需求新增的，表现形式看看如何专业体现
> 4. 3.2decode阶段的接入也需要突出说明，已有主流程和对应的新增逻辑加入点在哪
> 5. 4.1和4.2都有class LLMEngine，看是不是按照类来进行伪代码，要不然两处都有class LLMEngine:，感觉有点给人看着疑惑
> 6. 我删除了第5章，请把第六章后续的章节序号提前
> 7. 第7章删除，第八章移入到串讲设计备忘录中

<details><summary>8 previous messages</summary>

> 我会按真实 DeepInfer 调用链重画 2.2：把主线程、BlockManager、KV worker、Store 的调用和返回分别对齐，去掉容易误读的曲线虚线。正文会统一使用“Prefill 完成”与“最后一个 Prefill chunk 完成”的条件表达，分别说明普通 Prefill 和 Chunk Prefill；同时重写 3.1、3.2 和伪代码职责边界，并按你的要求删除第 5、7 章及迁移第 8 章内容。
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./设计文档/REQ-KVP-04-async-kv-get/设计串讲备忘.md`
> - Read `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md`
> - Read `./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg`
>
> </details>
>
> 我会把图改成标准时序图：每个参与者一条垂直生命线，主流程按时间向下，异步 Put 和 completion 用同一条虚线消息分别指向明确的接收者，不再使用跨泳道曲线。正文会把“已有调用”和“新增逻辑”用表格与标注分开，避免读者误以为这些虚函数当前已经实现了 Put。
>
> <details><summary>Created 2 files, edited a file, deleted a file, explored a file, a search, a list, ran 2 commands</summary>
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+1 -1)
>
> <details><summary>Explored a file</summary>
>
> - Read `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md`
>
> </details>
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+12 -9)
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+11 -26)
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+1 -22)
>
> - Deleted `./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg` (+0 -71)
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg` (+83 -0)
>
> <details><summary>Explored a file, a list</summary>
>
> - Listed files in `REQ-KVP-06-block-put`
> - Read `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md`
>
> </details>
>
> - Ran `xmllint --noout 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg; rsvg-convert -o /tmp/async-block-put-sequence-v2.png 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg`
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+10 -11)
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+1 -1)
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+1 -0)
>
> <details><summary>Explored a search</summary>
>
> - Searched for `^## |^### 2\.4|class LLMEngine|未满尾 block|最后一个 prompt chunk|post_process|pre_sampler_sync` in `REQ-KVP-06-block-put.md`
>
> </details>
>
> - Ran `xmllint --noout 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg && rsvg-convert -o /tmp/async-block-put-sequence-v3.png 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg`
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/设计串讲备忘.md` (+40 -0)
>
> </details>
>
> Viewed an image
>
> <details><summary>Edited a file, ran a command</summary>
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg` (+4 -4)
> - Ran `xmllint --noout 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg && rsvg-convert -o /tmp/async-block-put-sequence-v4.png 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg`
>
> </details>
>
> Viewed an image
>
> <details><summary>Edited a file, explored a search, a list, ran a command</summary>
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg` (+1 -1)
>
> <details><summary>Explored a search, a list</summary>
>
> - Searched for `^## |^### 2\.4|^class LLMEngine|^class KVPoolWorker|设计串讲备忘` in `REQ-KVP-06-block-put.md`
> - Listed files in `REQ-KVP-06-block-put`
>
> </details>
>
> - Ran `xmllint --noout 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg && rsvg-convert -o /tmp/async-block-put-sequence-final.png 设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg`
>
> </details>
</details>

已按要求完成修改：

1. **2.2 时序图已重画**

   - 改为标准时序图；
   - 使用垂直生命线表示各组件；
   - 主线程调用使用实线；
   - worker Put 和 completion 使用紫色虚线；
   - 去除跨泳道曲线，避免流程线穿过其他步骤框；
   - 已通过 `xmllint` 和 SVG 渲染检查。

2. **Prefill 描述已统一**

   - 普通 Prefill：本次 prompt 全部执行完成后提交完整 block；
   - Chunk Prefill：只有最后一个 prompt chunk 执行完成后提交完整 block；
   - 未满 block 不单独提交 Store；
   - Decode 后续填满该 block 后再提交。

3. **3.1 已补充真实主流程和新增逻辑**

   已明确：

   ```text
   LLMEngine.step()
     -> Scheduler.schedule()
     -> Worker.execute_model_step()
          -> _post_process_step(prev_step_state)
               -> 等待 sample_end_evt
               -> LLMEngine.post_process()
          -> 当前轮 forward/sample
          -> pre_sampler_sync()
   ```

   并区分了：

   - 已有的 `EngineCallbacks`、`post_process`、`pre_sampler_sync` 调用；
   - 新增的 `pe_prepare_save_kvcache`、`pe_do_save_kvcache` Put 逻辑；
   - `send_processed_prompts` 保持原有 PD/Prompt 交接职责。

4. **3.2 Decode 接入已补充**

   明确了：

   - Decode 现有保存入口是 `pre_sampler_sync()`；
   - `do_save_kvcache()` 处理上一轮已提交 KV；
   - 新增完整 block 计算、pin、PutTask 提交和 `next_unsaved_block` 更新；
   - 提交发生在 `finalize_seq()` 之前。

5. **伪代码职责已拆分**

   - 4.1 只保留 `LLMEngine`；
   - 4.2 只保留 `KVPoolWorker`；
   - 删除了重复的 `class LLMEngine`；
   - 明确由 Engine 负责 pin 和 completion 后 unpin，worker 不直接修改 `Sequence` 或 `BlockManager`。

6. **章节调整**

   - 删除原第 5 章；
   - 测试用例调整为第 5 章；
   - 删除原第 7 章；
   - 原第 8 章内容已移动到：
     [设计串讲备忘.md](./设计文档/REQ-KVP-06-block-put/设计串讲备忘.md)

更新后的文件：

- [REQ-KVP-06-block-put.md](./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md)
- [async-block-put-sequence.svg](./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg)

> 1. 请重新检查2.2的时序图，里面还是有逻辑和线不太对的位置，就比如复制block哪里有两条线输出，而且没有箭头，线的连接逻辑也不清晰奇怪。请重新检查这个时序图的逻辑
> 2. 3.1和3.2的主流程太详细了，其主要是说明哪些是已有的主流程，哪些是新增的提交异步任务的钩子函数就行
> 3. 4.1还是只说只准备最后一个 prompt chunk 的完整 block。应该是说明还有普通prefill的情况，同时4.1里也没有关于prefill和decode对应save kv cache提交异步put的伪代码
> 4. 4.2的store的相关get操作也是基于上游已有store的已有设计来重新修改伪代码

<details><summary>14 previous messages</summary>

> 我会把 2.2 改成单向、编号清晰的时序：每一步只有一个发送箭头和一个明确接收者，复制 block ID 改成 Engine 的自操作，不再画成两条输出线。正文会压缩 3.1/3.2，只保留已有主流程和新增钩子；4.1 增加普通 Prefill、Chunk Prefill 和 Decode 的保存伪代码，4.2 改用 `mooncake_store.md` 定义的 `put(items)`/分段 buffer 语义。
>
> <details><summary>Created a file, edited a file, deleted a file, explored a file</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md`
>
> </details>
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+9 -27)
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+26 -3)
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+9 -1)
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+2 -0)
>
> - Deleted `./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg` (+0 -83)
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg` (+85 -0)
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我会先检查当前设计文档和最新 SVG，重点核对章节、伪代码边界与 Store 接口的一致性，然后重新渲染时序图确认箭头和连接没有重叠。
>
> <details><summary>Explored 2 files, 2 searches, ran a command</summary>
>
> <details><summary>Explored 2 files, 2 searches</summary>
>
> - Read `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md`
> - Read `./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg`
> - Searched for `StoreItem|batch_put_from_multi_buffers|def put|put\(` in `mooncake_store.md`
> - Searched for `2\.4|## 6|## 7|## 8|只准备最后|prompt chunk|StoreItem|Store\.put|class LLMEngine|class KVPoolWorker` in `REQ-KVP-06-block-put.md`
>
> </details>
>
> - Ran `xmllint --noout '设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg' && rsvg-convert -o /tmp/async-block-put-sequence-final.png '设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg'`
>
> </details>
>
> Viewed an image
>
> SVG 已通过 XML 校验并渲染检查，当前时序是单向提交、后台写入、下一步轮询完成再 unpin。接下来我会做一次文档语义复核，特别是普通 Prefill 与 Chunk Prefill 的触发条件，以及 `mooncake_store.md` 中 `put(items)` 的参数约定。
>
> <details><summary>Explored a file, 2 searches, ran a command</summary>
>
> - Ran `sed -n '210,270p' '设计文档/mooncake_store.md'; printf '\n--- git diff/stat ---\n'; git diff --stat -- '设计文档/REQ-KVP-06-block-put'; git diff -- '设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md' | sed -n '1,260p'`
>
> <details><summary>Explored a file, 2 searches</summary>
>
> - Searched for `class .*Item|StoreItem|PutItem|segments` in `mooncake_store.md`
> - Read `./设计文档/mooncake_store.md`
> - Searched for `^#|^##|^###` in `设计串讲备忘.md`
>
> </details>
>
> </details>
>
> 复核发现上游文档把对象类型称为 `PutItem`，而当前草稿写成了 `StoreItem`；同时 `mooncake_store.put()` 对外返回布尔结果，底层 `0/-705` 只属于批量 API。我要把这两处接口描述改准，并补充普通 Prefill 与 Chunk Prefill 的钩子时序说明。
>
> <details><summary>Edited 2 files, explored a file, ran a command</summary>
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+7 -7)
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg` (+5 -1)
>
> <details><summary>Explored a file</summary>
>
> - Read `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md`
>
> </details>
>
> - Ran `xmllint --noout '设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg' && rsvg-convert -o /tmp/async-block-put-sequence-final.png '设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg'`
>
> </details>
>
> Viewed an image
>
> <details><summary>Explored a search, a list</summary>
>
> - Listed files in `REQ-KVP-06-block-put`
> - Searched for `^## |^### |StoreItem|PutItem|只准备最后|## 6|## 7|## 8|2\.4` in `REQ-KVP-06-block-put`
>
> </details>
>
> 又发现一处时序表述需要统一：Prefill 的 block 选择和 pin 发生在模型执行前，Store Put 要等模型执行完成后提交；原文“模型执行完成后检查完整 block”容易把两个钩子混成一个时点。我会把触发条件改成“执行前确定范围并 pin，执行后提交”，与 4.1 的伪代码一致。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+1 -1)
> - Wrote `./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md` (+1 -1)
>
> </details>
</details>

已完成本轮修改：

- 重新整理 2.2 时序图：
  - 明确 `LLMEngine -> Scheduler` 继续调度其他请求；
  - 明确 `KVPoolWorker` 后台执行 Put；
  - completion 由 Engine 下一轮轮询，再执行 `unpin_blocks`；
  - 删除歧义连接，所有流程线均有方向箭头。
- 简化 3.1、3.2，仅保留 DeepInfer 现有主流程和新增异步 Put 钩子。
- 修正 4.1：
  - 普通 Prefill；
  - Chunk Prefill 最后 chunk；
  - Decode 形成完整 block；
  - 明确 Prefill 执行前确定并 pin，执行完成后提交 Put。
- 修正 4.2：
  - 使用上游定义的 `PutItem`；
  - 对齐 `MooncakeStoreClient.put(items)`；
  - 说明底层 `0/-705` 结果归一逻辑。
- 保持章节结构和串讲备忘文档不变。

文件已更新：

- [REQ-KVP-06-block-put.md](./设计文档/REQ-KVP-06-block-put/REQ-KVP-06-block-put.md)
- [async-block-put-sequence.svg](./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg)

SVG 已通过 `xmllint` 校验并成功渲染检查。

> 2.2的时序图还是有问题，submit-put的引入线起始位置没有连上任何，所以先不要着急修改，请重新确认svg里时序图的逻辑是否正确，然后再统一修改

<details><summary>4 previous messages</summary>

> 我先核对图中的调用方向和连接点，不立即改图。当前确认有两个问题：`submit_put(task)` 的水平线从 Engine 车道右边缘的空白处开始，确实没有连接到前一个步骤；另外图上 `pin_blocks` 与 `submit_put` 看起来像并行调用，没有体现“先 pin、再提交”的顺序。正确逻辑应是 Engine 构造快照后先调用 `BlockManager.pin_blocks()`，pin 完成后再由 Engine 调用 `KVPoolWorker.submit_put()`，随后 worker 异步处理。
>
> 因此统一修改点很明确：保留参与者和时序层次，改成从“复制 block_ids / 构造 PutTask”框底部向下，再水平指向 `submit_put(task)`；这样线有明确起点，且位于 `pin_blocks` 框之后，表达“pin 完成后再入队”。同时保留 `Scheduler` 继续调度和 Engine 轮询 completion 的路径，不改变其他流程。
>
> <details><summary>Edited a file, explored a search, ran a command</summary>
>
> - Wrote `./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg` (+1 -1)
>
> <details><summary>Explored a search</summary>
>
> - Searched for `M134 274|M220 298|submit_put|pin_blocks|Scheduler.schedule` in `async-block-put-sequence.svg`
>
> </details>
>
> - Ran `xmllint --noout '设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg' && rsvg-convert -o /tmp/async-block-put-sequence-final.png '设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg'`
>
> </details>
>
> Viewed an image
</details>

已重新核对并修正 2.2 时序图。

确认的问题是：

- 原 `submit_put(task)` 的连线从 Engine 车道空白位置开始，未连接到任何步骤。
- 图形上没有清楚表达 `pin_blocks()` 完成后才提交 PutTask。

现在已修改为：

1. `LLMEngine` 完成 `block_ids` 快照和 `PutTask` 构造；
2. 调用 `BlockManager.pin_blocks()`；
3. 从构造框底部向下，再水平连接到 `KVPoolWorker.submit_put(task)`；
4. worker 异步等待 event 并调用 Store；
5. Engine 后续 step 继续调度其他请求；
6. completion 返回后执行 `unpin_blocks()`。

SVG 已重新通过 `xmllint` 校验并渲染确认，文件为：

[async-block-put-sequence.svg](./设计文档/REQ-KVP-06-block-put/img/async-block-put-sequence.svg)

> 很好，请在/Users/l/L/Code/TuringProject/D/设计文档下生成当前会话的session文件或者其他文件，我想把当前会话的记录迁移到另一个电脑的codex上继续使用，请生成相应的文件和使用方法，

<details><summary>2 previous messages</summary>

> 我会在 `设计文档` 下生成一份可迁移的会话交接文件，包含当前需求背景、已确认结论、已修改文件、关键代码位置、未完成事项和另一台电脑上的继续使用方法。不会写入聊天平台专用的内部 session 状态，只生成 Codex 可直接读取的文本交接记录。
>
> <details><summary>Created a file, ran a command</summary>
>
> - Wrote `./设计文档/codex-session-handoff-REQ-KVP-异步Put.md` (+166 -0)
> - Ran `wc -l '设计文档/codex-session-handoff-REQ-KVP-异步Put.md'; sed -n '1,40p' '设计文档/codex-session-handoff-REQ-KVP-异步Put.md'; tail -25 '设计文档/codex-session-handoff-REQ-KVP-异步Put.md'`
>
> </details>
</details>

已生成会话交接文件：

[codex-session-handoff-REQ-KVP-异步Put.md](./设计文档/codex-session-handoff-REQ-KVP-异步Put.md)

文件包含：

- 当前需求背景；
- 已确认的 Prefill、Chunk Prefill、Decode 和 HBM 生命周期结论；
- DeepInfer 代码接入位置；
- Mooncake Store 接口约定；
- 已生成文档和 SVG 文件；
- 当前文档结构；
- 另一台电脑上的 Codex 恢复提示词和使用方法。

将该文件及相关仓库同步到另一台电脑后，让新的 Codex 会话先读取该文件即可继续。

> /Users/l/L/Code/TuringProject/D/设计文档增加remote是git@github.com:tuliang1024/kv_cache.git，并提交/Users/l/L/Code/TuringProject/D/设计文档里的所有文件到对应的remote

<details><summary>6 previous messages</summary>

> 我会先确认 `设计文档` 是否是独立 Git 仓库、现有 remote 和工作区状态，然后添加指定 remote，提交该目录全部文件并推送。若已有未提交内容，我会保留并一并纳入本次提交。
>
> <details><summary>Explored a search, 2 lists, ran a command</summary>
>
> - Ran `pwd; git -C '设计文档' rev-parse --show-toplevel; git -C '设计文档' remote -v; git -C '设计文档' status --short --branch`
>
> <details><summary>Explored a search, 2 lists</summary>
>
> - Listed files in `设计文档`
> - Listed files in `设计文档`
> - Searched for `.git` in `设计文档`
>
> </details>
>
> </details>
>
> `设计文档` 当前不是 Git 仓库，因此需要先在该目录初始化仓库。接下来我会先读取远端分支信息，避免把本地提交推到错误的分支；然后再初始化、添加全部文件、提交并推送。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `git ls-remote --symref git@github.com:tuliang1024/kv_cache.git HEAD refs/heads/main refs/heads/master`
> - Ran `git init -b main`
> - Ran `git remote add origin git@github.com:tuliang1024/kv_cache.git && git remote -v`
> - Ran `git add --all && git status --short && git diff --cached --stat`
>
> </details>
>
> 远端为空或没有可读的默认分支，当前本地已初始化为 `main`。`git add --all` 已纳入 `设计文档` 下的 12 个文件，包括两套方案文档、SVG、Mooncake Store 设计、会话交接记录和草稿；现在提交并推送到 `origin/main`。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `git commit -m 'docs: add async KV cache get and put designs'`
> - Ran `git push -u origin main`
> - Ran `git status --short --branch; git log -1 --oneline; git remote -v`
>
> </details>
</details>

已完成：

- 在 `/Users/l/L/Code/TuringProject/D/设计文档` 初始化 Git 仓库；
- 添加 remote：
  `git@github.com:tuliang1024/kv_cache.git`
- 提交目录下全部 12 个文件；
- 提交记录：
  `f01c600 docs: add async KV cache get and put designs`
- 已推送到远端 `main` 分支；
- 当前工作区干净，并已跟踪 `origin/main`。

> session的文件里涉及的路径都给相对路径/Users/l/L/Code/TuringProject/D吧，然后修改后直接提交到remote

<details><summary>2 previous messages</summary>

> 我会把会话交接文件中的仓库路径统一改为相对于 `/Users/l/L/Code/TuringProject/D` 的路径，例如 `DeepInfer/...`、`vllm/...` 和 `设计文档/...`；迁移示例也改成 `<repo-root>/设计文档/...`。修改后提交并推送到现有 `origin/main`。
>
> <details><summary>Edited a file, explored 2 searches, ran 4 commands</summary>
>
> <details><summary>Explored a search</summary>
>
> - Searched for `/Users/l/L/Code/TuringProject/D|/path/to|设计文档/` in `codex-session-handoff-REQ-KVP-异步Put.md`
>
> </details>
>
> - Wrote `./设计文档/codex-session-handoff-REQ-KVP-异步Put.md` (+6 -6)
>
> - Ran `rg -n '/Users/|/path/to' '设计文档/codex-session-handoff-REQ-KVP-异步Put.md' || true; git diff -- 'codex-session-handoff-REQ-KVP-异步Put.md'; git status --short`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `/Users/|/path/to` in `codex-session-handoff-REQ-KVP-异步Put.md`
>
> </details>
>
> - Wrote `./设计文档/codex-session-handoff-REQ-KVP-异步Put.md` (+2 -2)
>
> - Ran `git add 'codex-session-handoff-REQ-KVP-异步Put.md' && git commit -m 'docs: make session handoff paths relative'`
>
> - Ran `git push origin main`
>
> - Ran `rg -n '/Users/|/path/to' 'codex-session-handoff-REQ-KVP-异步Put.md' || true; git status --short --branch; git log -2 --oneline`
>
> </details>
</details>

已完成并推送：

- 会话文件中的仓库路径已改为相对路径：
  - `vllm`
  - `vllm-ascend`
  - `DeepInfer`
  - `设计文档/...`
- 迁移示例中的绝对路径已改为 `<repo-root>/...`
- 提交记录：`5a519e7 docs: make session handoff paths relative`
- 已推送到 `origin/main`
- 当前工作区干净。

> 当前session应该还有/Users/l/L/Code/TuringProject/D/设计文档/REQ-KVP-04-async-kv-get相关的吧

<details><summary>2 previous messages</summary>

> 是的，当前会话还包含 `REQ-KVP-04-async-kv-get` 相关设计文档和 SVG，但之前生成的交接文件主要记录了 `REQ-KVP-06`，没有完整列出异步 KV Get 的文件和结论。我会补充这部分，确保迁移后的会话不会丢失前缀匹配异步 Get 的上下文，并提交推送。
>
> <details><summary>Edited a file, explored 3 files, a search, ran a command</summary>
>
> <details><summary>Explored 3 files, a search</summary>
>
> - Read `./设计文档/REQ-KVP-04-async-kv-get/REQ-KVP-04-async-kv-get.md`
> - Read `./设计文档/REQ-KVP-04-async-kv-get/设计串讲备忘.md`
> - Searched for `^##|REQ-KVP-04|REQ-KVP-06` in `codex-session-handoff-REQ-KVP-异步Put.md`
> - Read `./设计文档/codex-session-handoff-REQ-KVP-异步Put.md`
>
> </details>
>
> - Wrote `./设计文档/codex-session-handoff-REQ-KVP-异步Put.md` (+31 -2)
>
> - Wrote `./设计文档/codex-session-handoff-REQ-KVP-异步Put.md` (+13 -4)
>
> - Ran `rg -n '^#|^##|REQ-KVP-04|/Users/l|/path/to' '设计文档/codex-session-handoff-REQ-KVP-异步Put.md'; git diff --stat; git status --short`
>
> </details>
</details>
