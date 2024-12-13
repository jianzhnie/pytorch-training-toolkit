# PyTorch 全分片数据并行（FSDP）API 介绍

作者：Yanli Zhao, Rohan Varma, Chien-Chin Huang, Shen Li, Min Xu, Alban Desmaison

最近的研究表明，大规模模型训练将有助于提高模型质量。在过去三年中，模型规模从拥有 1.1 亿参数的 [BERT](https://arxiv.org/abs/1810.04805) 增长到了拥有一万亿参数的 [Megatron-2](https://arxiv.org/abs/2104.04473)，增长了 10,000 倍。然而，训练大型 AI 模型并非易事——除了需要大量的计算资源外，软件工程的复杂性也是一个挑战。

除了需要大量的计算和工程资源外，大多数像这样的扩展方法还会引入额外的通信成本，并要求工程师仔细评估内存使用和计算效率之间的权衡。例如，典型的数据并行训练需要在每个 GPU 上维护模型的冗余副本，而模型并行训练会引入额外的通信成本，以在不同 worker（GPU）之间移动激活值。最近的方法，如 DeepSpeed ZeRO 和 FairScale 的全分片数据并行（Fully Sharded Data Parallel, FSDP），通过将模型的参数、梯度和优化器状态分片到数据并行worker上，打破了这一限制，同时仍然保持了数据并行的简单性。

全分片数据并行（Fully Sharded Data Parallel, FSDP）将 AI 模型的参数分片到数据并行 worker上，并可以选择将部分训练计算卸载到 CPU 上。顾名思义，FSDP 是一种数据并行训练算法。尽管参数被分片到不同的 [GPU](https://engineering.fb.com/2018/03/20/ml-applications/the-next-step-in-facebook-s-ai-hardware-infrastructure/) 上，但每个微批次数据的计算仍然在每个 GPU worker本地进行。这种概念上的简单性使得 FSDP 更容易理解和适用于更广泛的使用场景（与层内并行和流水线并行相比）。

- 与优化器状态+梯度分片数据并行方法相比，FSDP 将模型参数、梯度和优化器状态更均匀地分片到 GPU 上来提高内存效率
- 并通过在训练期间分解通信并将其与前向和后向传播重叠来提高计算效率，实现更好的性能。

FSDP 产生的结果与标准分布式数据并行（DDP）训练相同，并且提供了一个易于使用的接口，作为 PyTorch 的 DistributedDataParallel 模块的即插即用替换。通过 FSDP，现在可以更高效地使用更少的 GPU 训练规模大得多的模型。

在 AWS 上的 PyTorch FSDP 扩展测试表明，它可以扩展到训练拥有 1 万亿参数的`dense`模型。在我们的实验中，GPT 1T 模型在 A100 GPU 上实现了每秒 84 万亿次浮点运算（TFLOPS），而 GPT 175B 模型在 A100 GPU 上实现了每秒 159 万亿次浮点运算（TFLOPS）。与 FairScale 的原始实现相比，启用 CPU offload 时，原生 FSDP 实现显著改善了模型初始化时间。

### FSDP 的工作原理

在标准的 DDP 训练中，每个worker处理一个单独的批次，并使用 [all-reduce 操作](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce) 在所有worker之间对梯度进行求和。虽然 DDP 已经变得非常流行，但它占用了比实际需求更多的 GPU 内存，因为模型权重和优化器状态在所有 DDP worker之间是重复的。

减少重复的一种方法是应用一种称为全参数分片的过程，其中只有本地计算所需的模型参数、梯度和优化器的子集是可用的。微软推广了这种实现方法，称为 ZeRO-3。

解锁全参数分片的关键见解是，我们可以将 DDP 中的 [all-reduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce) 操作分解为单独的 [reduce-scatter](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reducescatter) 和 [all-gather](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather) 操作：

![全分片数据并行图](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-graph-2a.png?w=1024)

all-reduce 作为 reduce-scatter 和 all-gather 的组合。标准的 all-reduce 操作可以分解为两个独立的阶段：reduce-scatter 和 all-gather。

- 在 reduce-scatter 阶段，梯度根据每个 GPU 的排名索引在 GPU 之间以相等的块进行求和。
- 在 all-gather 阶段，每个 GPU 上可用的聚合梯度的分片部分被提供给所有 GPU。

我们可以重新排列 reduce-scatter 和 all-gather，使得每个 DDP worker只需要存储单个分片的参数和优化器状态。下图展示了标准 DDP 训练（顶部）和 FSDP 训练（底部）的对比：

![全分片数据并行图](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-Graph-2.png?w=907)

> 标准数据并行训练与全分片数据并行训练的对比。在标准数据并行训练方法中，每个 GPU 上都有一个模型的副本，并且仅对数据的分片进行一系列前向和后向传播。在这些本地计算之后，每个本地进程的参数和优化器与其它 GPU 共享，以计算全局权重更新。在 FSDP 中，每个 GPU 上只有一个模型的分片。然后，在本地，所有权重通过 all-gather 步骤从其它 GPU 收集，以计算前向传播。在前向传播之后，再次执行权重收集以进行后向传播。在后向传播之后，本地梯度通过 reduce-scatter 步骤在 GPU 之间进行平均和分片，从而使每个 GPU 能够更新其本地权重分片。

为了最大化内存效率，我们可以在每层的前向传播之后丢弃完整权重，为后续层节省内存。这可以通过将 FSDP 包装器应用于网络中的每一层来实现（使用 reshard_after_forward=True）。

FSDP 是一种数据并行训练策略，但与传统的数据并行不同，传统数据并行在每个 GPU 上维护模型的参数、梯度和优化器状态的副本，而 FSDP 将所有这些状态分片到数据并行worker上，并可以选择将分片的模型参数offload 到 CPU 上。

下图展示了 FSDP 在 2 个数据并行进程中的工作流程：

![img](https://pytorch.org/assets/images/fsdp_workflow.png)

图 1. FSDP 工作流程

通常，模型层以嵌套方式包装在 FSDP 中，因此只有单个 FSDP 实例中的层需要在正向或反向计算期间将完整参数收集到单个设备上。收集的完整参数将在计算后立即释放，释放的内存可用于下一层的计算。通过这种方式，可以节省峰值 GPU 内存，从而使训练能够扩展到使用更大的模型规模或更大的批量大小。为了进一步最大化内存效率，FSDP 可以在实例不活跃于计算时将参数、梯度和优化器状态offload 到 CPU。

### 在 PyTorch 中使用 FSDP

在 PyTorch 中，有两种方法可以将模型包装在 FSDP 中。自动包装是 DDP 的即插即用替换；手动包装需要对模型定义代码进行最小的更改，并能够探索复杂的分片策略。

#### 自动包装

模型层应以嵌套方式包装在 FSDP 中，以节省峰值内存并启用通信和计算重叠。最简单的方法是自动包装，它可以作为 DDP 的即插即用替换，而无需更改其余代码。

`fsdp_auto_wrap_policy` 参数允许指定一个可调用函数，以递归方式将层包装在 FSDP 中。PyTorch FSDP 提供的 `default_auto_wrap_policy` 函数递归地包装参数数量大于 1 亿层的层。您可以根据需要提供自己的包装策略。自定义包装策略的示例显示在 [FSDP API 文档](https://pytorch.org/docs/stable/fsdp.html) 中。

此外，可以选择配置 `cpu_offload`，以在计算中不使用这些参数时将包装的参数offload 到 CPU。这可以进一步提高内存效率，但代价是主机和设备之间的数据传输开销。

下面的示例展示了如何使用自动包装来包装 FSDP。

```python
from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)
from torch.distributed.fsdp.wrap import (
   default_auto_wrap_policy,
)
import torch.nn as nn
 
class model(nn.Module):
   def __init__(self):
       super().__init__()
       self.layer1 = nn.Linear(8, 4)
       self.layer2 = nn.Linear(4, 16)
       self.layer3 = nn.Linear(16, 4)
 
model = DistributedDataParallel(model())
fsdp_model = FullyShardedDataParallel(
   model(),
   fsdp_auto_wrap_policy=default_auto_wrap_policy,
   cpu_offload=CPUOffload(offload_params=True),
)
```

#### 手动包装

手动包装可用于通过有选择地对模型的某些部分应用 `wrap` 来探索复杂的分片策略。总体设置可以通过 `enable_wrap()` 上下文管理器传递。

```python
from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)
from torch.distributed.fsdp.wrap import (
   enable_wrap,
   wrap,
)
import torch.nn as nn
from typing import Dict
 
 
class model(nn.Module):
   def __init__(self):
       super().__init__()
       self.layer1 = wrap(nn.Linear(8, 4))
       self.layer2 = nn.Linear(4, 16)
       self.layer3 = wrap(nn.Linear(16, 4))
 
wrapper_kwargs = Dict(cpu_offload=CPUOffload(offload_params=True))
with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
   fsdp_model = wrap(model())
```

使用上述两种方法之一将模型包装在 FSDP 中后，模型可以以类似于本地训练的方式进行训练，如下所示：

```python
optim = torch.optim.Adam(fsdp_model.parameters(), lr=0.0001)
for sample, label in next_batch():
  out = fsdp_model(input)
  loss = criterion(out, label)
  loss.backward()
  optim.step()
```

### 基准测试结果

我们在 AWS 集群上使用 PyTorch FSDP 对 175B 和 1T GPT 模型进行了广泛的扩展测试。每个集群节点是配备 8 个 [NVIDIA A100-SXM4-40GB](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) GPU 的实例，节点间通过 AWS Elastic Fabric Adapter (EFA) 连接，网络带宽为 400 Gbps。

GPT 模型使用 [minGPT](https://github.com/karpathy/minGPT) 实现。随机生成的输入数据集用于基准测试目的。所有实验均使用 50K 词汇量、fp16 精度和 [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) 优化器运行。

| 模型     | 层数 | 隐藏大小 | 注意力头 | 模型大小（十亿参数） |
| -------- | ---- | -------- | -------- | -------------------- |
| GPT 175B | 96   | 12288    | 96       | 175                  |
| GPT 1T   | 128  | 25600    | 160      | 1008                 |

除了在实验中使用带有参数 CPU offload 的 FSDP 外，还在测试中应用了 PyTorch 的 [激活检查点功能](https://pytorch.org/docs/stable/checkpoint.html)。

在 128 个 GPU 上，GPT 175B 模型的最大每 GPU 吞吐量为 159 万亿次浮点运算/秒（占 NVIDIA A100 峰值理论性能的 51%，即每 GPU 312 万亿次浮点运算/秒），批量大小为 20，序列长度为 512；进一步增加 GPU 数量会导致每 GPU 吞吐量下降，因为节点间的通信量增加。

对于 GPT 1T 模型，在 128 个 GPU 上，批量大小为 4，序列长度为 2048 时，最大每 GPU 吞吐量为 84 万亿次浮点运算/秒（占峰值吞吐量的 27%）。然而，进一步增加 GPU 数量对每 GPU 吞吐量的影响不大，因为我们观察到 1T 模型训练中的最大瓶颈不是来自通信，而是来自 CUDA 缓存分配器在峰值 GPU 内存接近极限时的缓慢。使用具有更大内存容量的 A100 80G GPU 将主要解决此问题，并有助于扩展批量大小以实现更大的吞吐量。

![img](https://pytorch.org/assets/images/175b_throught.png)

![img](https://pytorch.org/assets/images/1t_thought.png)
