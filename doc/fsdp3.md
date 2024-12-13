# FSDP1 -> FSDP2

## 为什么选择 FSDP2？

PyTorch 的全分片数据并行 (FSDP) API [`FullyShardedDataParallel`](https://pytorch.org/docs/stable/fsdp.html) 旨在提供高性能的 eager 模式实现，包括通信分桶和通信/计算重叠。它通过展平和连接一组参数来定义一个 `FlatParameter`，以表示一个通信桶。然而，这种 `FlatParameter` 使得对 `FlatParameter` 内的单个参数应用不同行为（例如参数冻结、参数类型转换等）变得复杂，影响了组合性，并且使内部实现复杂化，例如使状态字典逻辑代码行数达到数千行，并需要额外的通信。

考虑到这些限制，我们设计并实现了一个 FSDP 重写版本，去除了 `FlatParameter`。我们将这个重写版本称为 FSDP2，将原始版本称为 FSDP1。FSDP2 的目标与 FSDP1 相同，并支持更多用例，同时仍然在 eager 模式下追求良好的性能，使用了几种相同的技术。

与 FSDP1 相比：

- FSDP2 将分片参数表示为在 dim-0 上分片的 `DTensor`，允许轻松操作单个参数，实现无通信的分片状态字典，并简化元设备初始化流程。
- FSDP2 实现了一个改进的内存管理系统，通过避免 `recordStream` 实现了更低的确定性 GPU 内存，并且无需任何 CPU 同步。

未来，FSDP2 将提供一个扩展点来自定义 all-gather（例如用于 fp8 线性层的 fp8 all-gather），并改进对 `torch.compile` 的支持。

我们已经通过 torchtitan 验证了 FSDP2 的数值和性能（例如参见此 [PR](https://github.com/pytorch/torchtitan/pull/165)）。例如，在 8x H100 上运行的一些 Llama-7B 实验中，FSDP2 比 FSDP1 实现了更高的 MFU，峰值内存降低了 7%，并且匹配了相同的损失曲线。

有关动机、API 和系统设计的更多详细信息，请参阅 [此处](https://github.com/pytorch/pytorch/issues/114299)。在本 README 中，我们尝试提供更多面向用户的信息，并减少系统设计细节。

## FSDP1 <> FSDP2 API 差异

我们概述了 FSDP1 和 FSDP2 之间的一些 API 差异。总体而言，我们希望最小化 API 表面（包括参数数量），以避免出现庞大的 API。

```python
@contract(state_cls=FSDPState)
def fully_shard(
  module: nn.Module,
  *,
  mesh: Optional[DeviceMesh] = None,
  reshard_after_forward: Union[bool, int] = True,
  mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
  offload_policy: OffloadPolicy = OffloadPolicy(),
) -> nn.Module:  # 返回 `module` 以进行 `contract` 检查
```

| FSDP1                              | FSDP2                   |
| ---------------------------------- | ----------------------- |
| `module`                           | `module`                |
| `process_group`/`device_mesh`      | `mesh`                  |
| `sharding_strategy`                | `reshard_after_forward` |
| `cpu_offload`                      | `offload_policy`        |
| `auto_wrap_policy`                 | 移除                    |
| `backward_prefetch`                | 移除                    |
| `mixed_precision`                  | `mp_policy`             |
| `ignored_modules`/`ignored_states` | 尚未实现                |
| `param_init_fn`                    | 移除                    |
| `device_id`                        | 移除                    |
| `sync_module_states`               | 移除                    |
| `forward_prefetch`                 | 尚未实现                |
| `limit_all_gathers`                | 移除                    |
| `use_orig_params`                  | 移除                    |

- `fully_shard(module)` 类似于 `FullyShardedDataParallel(module)`，从 `module.parameters()` 构建一个通信桶，除了已经分配给嵌套 `fully_shard`/`FullyShardedDataParallel` 调用的参数。
  - `fully_shard(module)` 在 `module` 上添加一个 `FSDPState` 对象，可通过 `fully_shard.state(module)` 访问，而不是作为 `nn.Module` 包装器。这是通过 `@contract` 装饰器实现的。
  - 对应用了 FSDP2 的 `model` 调用 `model.named_parameters()` 将返回未更改的参数名称和分片的 `DTensor` 参数。这意味着优化器和梯度范数裁剪会看到 `DTensor`。
  - `fully_shard(module)` 对 `module` 执行动态类交换。例如，如果 `type(module)` 是 `Transformer`，则 FSDP2 构造一个新类 `FSDPTransformer`，该类继承自 `FSDPModule` 和 `Transformer`，并将 `module.__class__` 设置为 `FSDPTransformer`。这允许我们通过 `FSDPModule` 添加新方法并覆盖方法，而无需构造 `nn.Module` 包装器。
- FSDP1 的 `sharding_strategy` 和 `process_group`/`device_mesh` 映射到 FSDP2 的 `mesh` 和 `reshard_after_forward`。
  - `mesh` 应为 1D（用于 FSDP）或 2D（用于 HSDP）。对于 HSDP，我们假设在第 0 个网格维度上进行复制，在第 1 个网格维度上进行分片。如果 `mesh is None`，则 FSDP2 在默认进程组上初始化一个 1D 全局网格。
  - `reshard_after_forward=True` 或 `False` 决定参数是否在 forward 后重新分片（释放）。如果为 `True`，则在 backward 中重新 all-gather。这以额外的通信为代价节省内存。
  - （实验性）`reshard_after_forward: int` 意味着参数在 forward 后重新分片到较小的世界大小（例如 `reshard_after_forward=8` 可以表示节点内），以便 backward 中的 all-gather 在较小的世界大小上进行。
  - | FSDP1                                                       | FSDP2                                            | DeepSpeed  |
    | ----------------------------------------------------------- | ------------------------------------------------ | ---------- |
    | 1 `process_group` + `FULL_SHARD`                            | 1D `mesh` + `reshard_after_forward=True`         | ZeRO-3     |
    | 1 `process_group` + `SHARD_GRAD_OP`                         | 1D `mesh` + `reshard_after_forward=False`        | ZeRO-2     |
    | 2 `process_group`s/2D `device_mesh` + `HYBRID_SHARD`        | 2D `mesh` + `reshard_after_forward=True`         | MiCS       |
    | 2 `process_group`s/2D `device_mesh` + `_HYBRID_SHARD_ZERO2` | 2D `mesh` + `reshard_after_forward=False`        | -          |
    | -                                                           | 1D/2D `mesh` + `reshard_after_forward=8` (`int`) | ZeRO++ hpZ |
- FSDP2 将 `mixed_precision` 映射到 `mp_policy`，将 `cpu_offload` 映射到 `offload_policy`。
  - 对于 `mp_policy`，我们移除了 `buffer_dtype`，将 `cast_forward_inputs` 和 `cast_root_forward_inputs` 简化为仅 `cast_forward_inputs`，并添加了一个 `output_dtype`。
  - 对于 `offload_policy`，我们添加了一个 `pin_memory` 选项，以避免固定 CPU 内存。（此功能可能尚未上线。）
- FSDP2 移除了 `auto_wrap_policy`、`backward_prefetch`、`param_init_fn`、`device_id`、`sync_module_states`、`limit_all_gathers` 和 `use_orig_params`。
  - `auto_wrap_policy` 提供了基于策略给定的谓词调用 `FullyShardedDataParallel` 并将其分配回父模块的语法糖。FSDP2 不再是 `nn.Module` 包装器，因此无需将模块分配回父模块。我们更倾向于将此功能放在 `fully_shard` 之上，并且我们可能会在未来提供类似 `auto_wrap_policy` 的实用工具。
  - FSDP2 始终遵循 `backward_prefetch=BACKWARD_PRE`，因为没有选项可以正确重叠 backward 中的集合。`BACKWARD_POST` 在嵌套模块情况下可能会[错误地](https://github.com/pytorch/pytorch/issues/108190)预取。
  - FSDP2 支持新的元设备初始化流程，无需在分片之前立即将模块具体化到 GPU 上，从而无需 `param_init_fn`。有关更多详细信息，请参阅 [元设备初始化](#meta-device-initialization)。
  - FSDP2 始终将管理的参数/缓冲区移动到 `mesh` 对应的设备，从而无需 `device_id`。例如，如果 `mesh.device_type` 是 `"cuda"`，则 FSDP2 使用当前的 CUDA 设备。
  - FSDP2 使用新的内存管理系统，在保持通信/计算重叠的同时，实现了比 FSDP1 更低的确定性内存使用。该系统不需要任何 CPU 同步，因此无需 `limit_all_gathers`。
  - FSDP2 始终“使用原始参数”，因为没有更多的 `FlatParameter`，从而无需 `use_orig_params`。
- 如何在 FSDP2 中实现 `ignored_modules`/`ignored_states` 和 `forward_prefetch` 仍在讨论中。

| FSDP1                              | FSDP2                                        |
| ---------------------------------- | -------------------------------------------- |
| `model.state_dict()`: 完整状态字典 | `model.state_dict()`: 分片状态字典（无通信） |
| `optim.state_dict()`: 本地状态字典 | `optim.state_dict()`: 分片状态字典（无通信） |
| `summon_full_params()`             | 使用 `DTensor` API，如 `full_tensor()`       |
| `FSDP.clip_grad_norm_()`           | `nn.utils.clip_grad_norm_()`                 |
| `ShardedGradScaler`                | `amp.grad_scaler.GradScaler`                 |

## 元设备初始化

在 FSDP1 之前：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
with torch.device("meta"):
    model = Transformer()
policy = ModuleWrapPolicy({TransformerBlock})
# 在每个模块上调用 `reset_parameters()`
model = FSDP(model, auto_wrap_policy=policy)
# 在每个模块上调用 `param_init_fn`
def param_init_fn(module: nn.Module) -> None: ...
model = FSDP(model, auto_wrap_policy=policy, param_init_fn=param_init_fn)
```

在 FSDP2 之后：

```python
with torch.device("meta"):
    model = Transformer()
for module in model.modules():
    if isinstance(module, TransformerBlock):
        fully_shard(module)
fully_shard(model)
for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")
# 在 GPU 上分配缓冲区和分片参数
model.to_empty(device="cuda")
# 运行用户定义的初始化器
model.init_weights() # 或者 `model.apply(init_weights)`
```

FSDP1 要求在分片之前立即在 GPU 上具体化模块，因此需要 `reset_parameters` 或 `param_init_fn`。要正确执行此操作而不重新初始化任何张量需要小心处理，并且可能很麻烦。然而，FSDP2 允许在分片后将张量具体化到 GPU 上（利用 `DTensor` 和 `nn.Module._apply` 方法的新 `swap_tensors` 路径）。
