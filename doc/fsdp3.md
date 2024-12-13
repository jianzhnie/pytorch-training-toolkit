# 全分片数据并行：使用更少的 GPU 加速 AI 训练

![全分片数据并行英雄图](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-Hero-FINAL-1.png)

作者：[Myle Ott](https://engineering.fb.com/author/myle-ott/)、[Sam Shleifer](https://engineering.fb.com/author/sam-shleifer/)、[Min Xu](https://engineering.fb.com/author/min-xu/)、[Priya Goyal](https://engineering.fb.com/author/priya-goyal/)、[Quentin Duval](https://engineering.fb.com/author/quentin-duval/)、[Vittorio Caggiano](https://engineering.fb.com/author/vittorio-caggiano/)

大规模训练 AI 模型并非易事。除了需要大量的计算能力和资源外，训练非常大的模型背后还存在相当大的工程复杂性。在 Facebook AI Research (FAIR) 工程团队中，我们一直在构建工具和基础设施，以使训练大型 AI 模型变得更加容易。我们在诸如[层内模型并行](https://github.com/pytorch/fairseq/blob/master/examples/megatron_11b/README.md)、[流水线模型并行](https://fairscale.readthedocs.io/en/latest/deep_dive/pipeline_parallelism.html)、[优化器状态+梯度分片](https://github.com/facebookresearch/fairscale#optimizer-state-sharding-zero)和[专家混合](https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py)等领域的工作，只是我们为使任何任务的先进 AI 模型训练更加高效所做工作的一部分。

全分片数据并行（Fully Sharded Data Parallel, FSDP）是我们最新推出的工具。它将 AI 模型的参数分片到数据并行工作器上，并可以选择将部分训练计算卸载到 CPU 上。顾名思义，FSDP 是一种数据并行训练算法。尽管参数被分片到不同的 [GPU](https://engineering.fb.com/2018/03/20/ml-applications/the-next-step-in-facebook-s-ai-hardware-infrastructure/) 上，但每个微批次数据的计算仍然在每个 GPU 工作器本地进行。这种概念上的简单性使得 FSDP 更容易理解和适用于更广泛的使用场景（与层内并行和流水线并行相比）。与优化器状态+梯度分片数据并行方法相比，FSDP 更均匀地分片参数，并通过在训练期间重叠通信和计算来实现更好的性能。

通过 FSDP，现在可以更高效地使用更少的 GPU 训练规模大得多的模型。FSDP 已在 [FairScale 库](https://github.com/facebookresearch/fairscale) 中实现，并允许工程师和开发者通过简单的 API 扩展和优化其模型的训练。在 Facebook，FSDP 已集成并测试用于训练我们的一些 [NLP](https://github.com/pytorch/fairseq) 和 [视觉](https://github.com/facebookresearch/vissl) 模型。

## 大规模训练的高计算成本

[NLP 研究](https://arxiv.org/pdf/2001.08361.pdf) 是我们可以看到高效利用计算资源进行 AI 训练重要性的一个特定领域。去年，OpenAI 宣布他们已经训练了 [GPT-3](https://neurips.cc/virtual/2020/public/poster_1457c0d6bfcb4967418bfb8ac142f64a.html)，这是有史以来最大的神经语言模型，拥有 1750 亿个参数。据 [估计](https://lambdalabs.com/blog/demystifying-gpt-3/)，训练 GPT-3 大约需要 355 个 GPU 年，或相当于 1000 个 GPU 连续工作四个多月。

除了需要大量的计算和工程资源外，大多数像这样的扩展方法还会引入额外的通信成本，并要求工程师仔细评估内存使用和计算效率之间的权衡。例如，典型的数据并行训练需要在每个 GPU 上维护模型的冗余副本，而模型并行训练会引入额外的通信成本，以在不同工作器（GPU）之间移动激活值。

相比之下，FSDP 几乎没有权衡。它通过将模型参数、梯度和优化器状态分片到 GPU 上来提高内存效率，并通过分解通信并将其与前向和后向传播重叠来提高计算效率。FSDP 产生的结果与标准分布式数据并行（DDP）训练相同，并且提供了一个易于使用的接口，作为 PyTorch 的 DistributedDataParallel 模块的即插即用替换。我们的早期测试表明，FSDP 可以实现扩展到数万亿参数的训练。

## FSDP 的工作原理

在标准的 DDP 训练中，每个工作器处理一个单独的批次，并使用 [all-reduce 操作](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce) 在所有工作器之间对梯度进行求和。虽然 DDP 已经变得非常流行，但它占用了比实际需求更多的 GPU 内存，因为模型权重和优化器状态在所有 DDP 工作器之间是重复的。

减少重复的一种方法是应用一种称为全参数分片的过程，其中只有本地计算所需的模型参数、梯度和优化器的子集是可用的。微软已经推广了这种实现方法，称为 ZeRO-3。

解锁全参数分片的关键见解是，我们可以将 DDP 中的 [all-reduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce) 操作分解为单独的 [reduce-scatter](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reducescatter) 和 [all-gather](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather) 操作：

![全分片数据并行图](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-graph-2a.png?w=1024)

all-reduce 作为 reduce-scatter 和 all-gather 的组合。标准的 all-reduce 操作可以分解为两个独立的阶段：reduce-scatter 和 all-gather。在 reduce-scatter 阶段，梯度根据每个 GPU 的排名索引在 GPU 之间以相等的块进行求和。在 all-gather 阶段，每个 GPU 上可用的聚合梯度的分片部分被提供给所有 GPU（有关这些操作的详细信息，请参见此处）。

我们可以重新排列 reduce-scatter 和 all-gather，使得每个 DDP 工作器只需要存储单个分片的参数和优化器状态。下图展示了标准 DDP 训练（顶部）和 FSDP 训练（底部）的对比：

![全分片数据并行图](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-Graph-2.png?w=907)

标准数据并行训练与全分片数据并行训练的对比。在标准数据并行训练方法中，每个 GPU 上都有一个模型的副本，并且仅对数据的分片进行一系列前向和后向传播。在这些本地计算之后，每个本地进程的参数和优化器与其它 GPU 共享，以计算全局权重更新。在 FSDP 中，每个 GPU 上只有一个模型的分片。然后，在本地，所有权重通过 all-gather 步骤从其它 GPU 收集，以计算前向传播。在前向传播之后，再次执行权重收集以进行后向传播。在后向传播之后，本地梯度通过 reduce-scatter 步骤在 GPU 之间进行平均和分片，从而使每个 GPU 能够更新其本地权重分片。

为了最大化内存效率，我们可以在每层的前向传播之后丢弃完整权重，为后续层节省内存。这可以通过将 FSDP 包装器应用于网络中的每一层来实现（使用 reshard_after_forward=True）。

伪代码如下：

```python
FSDP 前向传播：
    for layer_i in layers:
        all-gather layer_i 的完整权重
        前向传播 layer_i
        丢弃 layer_i 的完整权重

FSDP 后向传播：
    for layer_i in layers:
        all-gather layer_i 的完整权重
        后向传播 layer_i
        丢弃 layer_i 的完整权重
        reduce-scatter layer_i 的梯度
```

## 如何使用 FSDP

在大规模 AI 研究中，有几种使用 FSDP 的方法。目前，我们提供了四种解决方案以适应不同的需求。

### 1. 在语言模型中使用 FSDP

对于语言模型，FSDP 在 [*fairseq* 框架](https://github.com/pytorch/fairseq) 中通过以下新参数得到支持：

- `--ddp-backend=fully_sharded`：通过 FSDP 启用全分片
- `--cpu-offload`：将优化器状态和 FP32 模型副本卸载到 CPU（结合 `--optimizer=cpu_adam`）
- `--no-reshard-after-forward`：提高大型模型（1B+ 参数）的训练速度，类似于 ZeRO 阶段 2
- 其他常用选项（`--fp16`、`--update-freq`、`--checkpoint-activations`、`--offload-activations` 等）继续正常工作

请参阅 [fairseq 教程](https://github.com/pytorch/fairseq/tree/master/examples/fully_sharded_data_parallel)，了解如何使用 FSDP 在八个 GPU 上训练 13B 参数的模型，或在单个 GPU 上使用 FSDP + CPU 卸载进行训练。

### 2. 在计算机视觉模型中使用 FSDP

对于计算机视觉模型，FSDP 在 [VISSL](https://github.com/facebookresearch/vissl) 中得到支持，并在 RegNets 架构上进行了测试。像 BatchNorm 和 ReLU 这样的层可以无缝处理并测试收敛性。

使用以下选项启用 FSDP：

- `config.MODEL.FSDP_CONFIG.AUTO_SETUP_FSDP=True`
- `config.MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE=pytorch`
- `config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch`

请参阅 [此部分](https://github.com/facebookresearch/vissl/blob/40441123a6f7098500676ca8800025c1f02e28b3/vissl/config/defaults.yaml#L498-L513) 的 yaml 配置，了解在 VISSL 中配置 FSDP 的更多选项。

### 3. 从 PyTorch Lightning 中使用 FSDP

为了更轻松地与更多通用用例集成，FSDP 作为测试版功能由 PyTorch Lightning 支持。[本教程](https://pytorch-lightning.readthedocs.io/en/latest/advanced/advanced_gpu.html#fully-sharded-training) 包含了一个详细的示例，展示了如何使用 FSDP 插件与 PyTorch Lightning 结合。在较高层次上，添加 `plugins='fsdp'` 即可激活它。

```python
model = MyModel()
trainer = Trainer(gpus=4, plugins='fsdp', precision=16)
trainer.fit(model)

trainer.test()
trainer.predict()
```

### 4. 直接从 FairScale 库中使用 FSDP

FSDP 的主要开发库，也是您可以找到最新更新的地方，是 [FairScale](https://fairscale.readthedocs.io/en/latest/deep_dive/oss_sdp_fsdp.html)。您可以直接使用 FairScale 中的 FSDP，只需将 `DDP(my_module)` 替换为以下示例：

```python
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
...
sharded_module = DDP(my_module)FSDP(my_module)
optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
for sample, label in dataload.next_batch:
  out = sharded_module(x=sample, y=3, z=torch.Tensor([1]))
  loss = criterion(out, label)
  loss.backward()
  optim.step()
```

FairScale 中的 FSDP 库公开了大规模训练的许多重要方面的低级选项。以下是一些在使用 FSDP 时需要考虑的重要领域。

1. **模型包装**：为了最小化瞬时 GPU 内存需求，用户需要以嵌套方式包装模型。这引入了额外的复杂性。[auto_wrap](https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/wrap/auto_wrap.py) 工具对于注释现有 PyTorch 模型代码以进行嵌套包装非常有用。
2. **模型初始化**：与 DDP 不同，FSDP **不会**自动在 GPU 工作器之间同步模型权重。这意味着必须小心进行模型初始化，以确保所有 GPU 工作器具有相同的初始权重。
3. **优化器设置**：由于分片和包装，FSDP 仅支持某些类型的优化器和优化器设置。特别是，如果一个模块被 FSDP 包装并且其参数被展平为一个张量，用户不能在该模块的不同参数组中使用不同的超参数。
4. **混合精度**：FSDP 支持带有 FP16 主权重的先进混合精度训练，以及梯度上的 FP16 reduce 和 scatter。模型的某些部分可能只有在使用全精度时才能收敛。在这些情况下，需要额外的包装以选择性地在全精度下运行模型的某些部分。
5. **状态检查点和推理**：当模型规模较大时，保存和加载模型状态可能会变得具有挑战性。FSDP 支持几种方法来实现这一任务，但这绝非易事。
6. 最后，FSDP 通常与 **激活检查点** 功能（如 FairScale 中的 [checkpoint_wrapper](https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/checkpoint/checkpoint_activations.py)）一起使用。用户可能需要仔细调整激活检查点策略，以使大型模型适应有限的 GPU 内存空间。

## 下一步

FSDP 是开源的，早期用户已经尝试并为其做出了贡献。我们认为它可以使整个研究社区受益，我们期待与大家一起努力使其变得更好。特别是，以下是一些重要的领域。

1. **使 FSDP 更加通用**：到目前为止，FSDP 已在带有 SGD 和 Adam 优化器的 NLP 和视觉模型上使用。随着新的模型和优化器的出现，FSDP 需要继续支持它们。作为一种纯粹的数据并行训练方案，FSDP 在支持广泛的 AI 算法方面具有最大的潜力。
2. **使 FSDP 自动调优**：用户今天可以通过 FSDP 调整许多旋钮以进行扩展和性能优化。我们期待开发算法，以自动调优 GPU 内存使用和训练性能。
3. 除了训练之外，更 **可扩展的推理** 和模型服务是 FSDP 可能需要支持的一个重要用例。
4. 最后但同样重要的是，重构并继续 **模块化 FSDP** 及其核心组件与新功能同样重要。

## 尝试并贡献！

FSDP 目前可以直接从 [FairScale 库](https://github.com/facebookresearch/fairscale) 中获得。

感谢您一直以来的支持。请在您的研究或生产工作中尝试 FSDP。我们很乐意听取您的反馈，并且一如既往，欢迎提交拉取请求！
```