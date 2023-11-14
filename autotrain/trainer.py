import os
import random
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    worker_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    rt /= worker_size
    return rt


def init_distribute_fn(args):
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    if args.seed is not None:
        print(f"Using seed = {args.seed}")
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)
        torch.backends.cudnn.deterministic = True


class TorchTrainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        grad_acc_steps: int = 1,
        static_loss_scale: float = 1.0,
        auto_mixed_percision: bool = False,
        torch_jit_script: bool = False,
        use_cuda: bool = True,
    ) -> None:
        def xform(m: nn.Module) -> nn.Module:
            if use_cuda:
                m = m.cuda()
            return m

        self.model = xform(model)
        if torch_jit_script:
            self.model = torch.jit.script(self.model)
        self.loss_fn = xform(loss_fn) if loss_fn is not None else None
        self.optimizer = optimizer
        self.grad_acc_steps = grad_acc_steps
        self.auto_mixed_persision = auto_mixed_percision
        self.divide_factor = 1
        self.is_distributed = False
        self._fwd_bwd = None
        self._forward = None
        self.steps_since_update = 0
        self.grad_scaler = GradScaler(
            init_scale=static_loss_scale,
            growth_factor=2,
            backoff_factor=0.5,
            growth_interval=100,
            enabled=auto_mixed_percision,
        )

    def distributed(self, gpu_id) -> None:
        self.is_distributed = True
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id)
        torch.cuda.current_stream().wait_stream(stream)

    def forward_fn(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(), autocast(enabled=self.auto_mixed_persision):
            output = self.model(inputs)
            loss = None if self.loss_fn is None else self.loss_fn(output, targets)

        return output if loss is None else loss, output

    def backward_fn(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, loss = self.forward_fn(inputs, targets)
        self.grad_scaler.scale(loss).backward()
        return output, loss

    def forward_backward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        with autocast(enabled=self.auto_mixed_persision):
            loss = self.loss_fn(self.model(inputs), targets)
            loss /= self.divide_factor

        self.grad_scaler.scale(loss).backward()
        return loss

    def train_mode(self) -> None:
        self.model.train()
        if self.loss_fn is not None:
            self.loss_fn.train()

    def eval_mode(self) -> None:
        self.model.eval()
        if self.loss_fn is not None:
            self.loss_fn.eval()

    def train_step(self, inputs, targets) -> torch.Tensor:
        inputs = inputs.cuda()
        targets = targets.cuda()
        loss = self.forward_backward(inputs, targets)
        self.steps_since_update += 1

        if self.steps_since_update == self.grad_acc_steps:
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps_since_update = 0

        torch.cuda.synchronize()
        return loss

    def validation_steps(self) -> Dict[str, Callable]:
        val_step_dict: Dict[str, Callable] = {"val": self.forward_fn}
        return val_step_dict

    def train_epoch(
        self,
        train_loader,
    ):
        data_iter = enumerate(train_loader)

        for i, (inputs, targets) in data_iter:
            loss = self.train_step(inputs, targets)
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    reduced_loss = reduce_tensor(loss.detach())
                else:
                    reduced_loss = loss.detach()
                print(reduced_loss)
        return reduced_loss

    def validate_epoch(self, infer_fn, val_loader, with_loss=True) -> None:
        data_iter = enumerate(val_loader)

        for i, (inputs, targets) in data_iter:
            if with_loss:
                loss, output = infer_fn(inputs, targets)
            else:
                output = infer_fn(inputs)

            with torch.no_grad():
                if torch.distributed.is_initialized():
                    if with_loss:
                        reduced_loss = reduce_tensor(loss.detach())
                else:
                    if with_loss:
                        reduced_loss = loss.detach()
        torch.cuda.synchronize()
        return reduced_loss

    def train(self, train_loader, val_loader, epochs) -> None:
        for epoch in range(epochs):
            self.train_mode()
            reduced_loss = self.train_epoch(train_loader)
            self.eval_mode()
            reduced_loss = self.validate_epoch(self.forward_fn, val_loader)
