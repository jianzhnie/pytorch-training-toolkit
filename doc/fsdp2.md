# PyTorch 全分片数据并行（FSDP）API 介绍

作者：Yanli Zhao, Rohan Varma, Chien-Chin Huang, Shen Li, Min Xu, Alban Desmaison

最近的研究表明，大规模模型训练将有助于提高模型质量。在过去三年中，模型规模从拥有 1.1 亿参数的 [BERT](https://arxiv.org/abs/1810.04805) 增长到了拥有一万亿参数的 [Megatron-2](https://arxiv.org/abs/2104.04473)，增长了 10,000 倍。然而，训练大型 AI 模型并非易事——除了需要大量的计算资源外，软件工程的复杂性也是一个挑战。PyTorch 一直在致力于构建工具和基础设施，以使其变得更加容易。

PyTorch 的分布式数据并行（Distributed Data Parallel, DDP）是可扩展深度学习的基石，因其鲁棒性和简单性而广受欢迎。然而，它要求模型能够适应单个 GPU。最近的方法，如 DeepSpeed ZeRO 和 FairScale 的全分片数据并行（Fully Sharded Data Parallel, FSDP），通过将模型的参数、梯度和优化器状态分片到数据并行工作器上，打破了这一限制，同时仍然保持了数据并行的简单性。

在 AWS 上的 PyTorch FSDP 扩展测试表明，它可以扩展到训练拥有 1 万亿参数的密集模型。在我们的实验中，GPT 1T 模型在 A100 GPU 上实现了每秒 84 万亿次浮点运算（TFLOPS），而 GPT 175B 模型在 A100 GPU 上实现了每秒 159 万亿次浮点运算（TFLOPS）。与 FairScale 的原始实现相比，启用 CPU 卸载时，原生 FSDP 实现显著改善了模型初始化时间。

在未来的 PyTorch 版本中，我们将允许用户在 DDP、ZeRO-1、ZeRO-2 和 FSDP 数据并行之间无缝切换，以便用户可以通过统一的 API 使用简单的配置来训练不同规模的模型。

### FSDP 的工作原理

FSDP 是一种数据并行训练类型，但与传统的数据并行不同，传统数据并行在每个 GPU 上维护模型的参数、梯度和优化器状态的副本，而 FSDP 将所有这些状态分片到数据并行工作器上，并可以选择将分片的模型参数卸载到 CPU 上。

下图展示了 FSDP 在 2 个数据并行进程中的工作流程：

![img](https://pytorch.org/assets/images/fsdp_workflow.png)

图 1. FSDP 工作流程

通常，模型层以嵌套方式包装在 FSDP 中，因此只有单个 FSDP 实例中的层需要在正向或反向计算期间将完整参数收集到单个设备上。收集的完整参数将在计算后立即释放，释放的内存可用于下一层的计算。通过这种方式，可以节省峰值 GPU 内存，从而使训练能够扩展到使用更大的模型规模或更大的批量大小。为了进一步最大化内存效率，FSDP 可以在实例不活跃于计算时将参数、梯度和优化器状态卸载到 CPU。

### 在 PyTorch 中使用 FSDP

在 PyTorch 中，有两种方法可以将模型包装在 FSDP 中。自动包装是 DDP 的即插即用替换；手动包装需要对模型定义代码进行最小的更改，并能够探索复杂的分片策略。

#### 自动包装

模型层应以嵌套方式包装在 FSDP 中，以节省峰值内存并启用通信和计算重叠。最简单的方法是自动包装，它可以作为 DDP 的即插即用替换，而无需更改其余代码。

`fsdp_auto_wrap_policy` 参数允许指定一个可调用函数，以递归方式将层包装在 FSDP 中。PyTorch FSDP 提供的 `default_auto_wrap_policy` 函数递归地包装参数数量大于 1 亿层的层。您可以根据需要提供自己的包装策略。自定义包装策略的示例显示在 [FSDP API 文档](https://pytorch.org/docs/stable/fsdp.html) 中。

此外，可以选择配置 `cpu_offload`，以在计算中不使用这些参数时将包装的参数卸载到 CPU。这可以进一步提高内存效率，但代价是主机和设备之间的数据传输开销。

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

除了在实验中使用带有参数 CPU 卸载的 FSDP 外，还在测试中应用了 PyTorch 的 [激活检查点功能](https://pytorch.org/docs/stable/checkpoint.html)。

在 128 个 GPU 上，GPT 175B 模型的最大每 GPU 吞吐量为 159 万亿次浮点运算/秒（占 NVIDIA A100 峰值理论性能的 51%，即每 GPU 312 万亿次浮点运算/秒），批量大小为 20，序列长度为 512；进一步增加 GPU 数量会导致每 GPU 吞吐量下降，因为节点间的通信量增加。

对于 GPT 1T 模型，在 128 个 GPU 上，批量大小为 4，序列长度为 2048 时，最大每 GPU 吞吐量为 84 万亿次浮点运算/秒（占峰值吞吐量的 27%）。然而，进一步增加 GPU 数量对每 GPU 吞吐量的影响不大，因为我们观察到 1T 模型训练中的最大瓶颈不是来自通信，而是来自 CUDA 缓存分配器在峰值 GPU 内存接近极限时的缓慢。使用具有更大内存容量的 A100 80G GPU 将主要解决此问题，并有助于扩展批量大小以实现更大的吞吐量。

![img](https://pytorch.org/assets/images/175b_throught.png)

![img](https://pytorch.org/assets/images/1t_thought.png)

### 未来工作

在下一个 beta 版本中，我们计划添加高效的分布式模型/状态检查点 API、支持大模型实例化的元设备支持，以及 FSDP 计算和通信中的混合精度支持。我们还计划在新 API 中更轻松地在 [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)、[ZeRO1, ZeRO2](https://arxiv.org/abs/1910.02054) 和 FSDP 数据并行之间切换。为了进一步提高 FSDP 性能，我们还计划减少内存碎片化并改进通信效率。

### FSDP 的两个版本的简史

[FairScale FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/) 于 2021 年初作为 FairScale 库的一部分发布。随后，我们开始将 FairScale FSDP 上游到 PyTorch 1.11 中，使其达到生产就绪状态。我们选择性地上游并重构了 FairScale FSDP 的关键功能，重新设计了用户接口，并进行了性能改进。

在不久的将来，FairScale FSDP 将继续留在 FairScale 仓库中，供研究项目使用，而通用且广泛采用的功能将逐步上游到 PyTorch 并相应地进行强化。

同时，PyTorch FSDP 将更多地关注生产就绪性和长期支持。这包括与生态系统的更好集成以及在性能、可用性、可靠性、可调试性和可组合性方面的改进。

### 致谢

我们要感谢 FairScale FSDP 的作者：Myle Ott, Sam Shleifer, Min Xu, Priya Goyal, Quentin Duval, Vittorio Caggiano, Tingting Markstrum, Anjali Sridhar。感谢 Microsoft DeepSpeed ZeRO 团队开发并推广了分片数据并行技术。感谢 Pavel Belevich, Jessica Choi, Sisil Mehta 在不同集群上使用 PyTorch FSDP 进行实验。感谢 Geeta Chauhan, Mahesh Yadav, Pritam Damania, Dmytro Dzhulgakov 对这一工作的支持以及富有洞察力的讨论。
```