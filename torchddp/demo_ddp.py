import os
import sys
import tempfile
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    """
    A simple toy neural network model for demonstration of distributed training.

    The model consists of two linear layers with a ReLU activation in between.
    
    Attributes:
        net1 (nn.Linear): First linear layer with 10 input and 10 output features
        relu (nn.ReLU): ReLU activation function
        net2 (nn.Linear): Second linear layer with 10 input and 5 output features
    """
    def __init__(self) -> None:
        """
        Initialize the ToyModel with two linear layers and ReLU activation.
        """
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 10)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 5)
        """
        return self.net2(self.relu(self.net1(x)))


def setup(rank: int, world_size: int) -> None:
    """
    Set up the distributed environment for training.

    Args:
        rank (int): Rank of the current process
        world_size (int): Total number of processes
    """
    # Set environment variables for process group initialization
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group using NCCL backend
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """
    Clean up the distributed environment by destroying the process group.
    """
    dist.destroy_process_group()


def demo_basic(rank: int, world_size: int) -> None:
    """
    Demonstrate basic Distributed Data Parallel (DDP) training.

    Args:
        rank (int): Rank of the current process
        world_size (int): Total number of processes
    """
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # Create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Set up loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Perform a training step
    optimizer.zero_grad()
    inputs = torch.randn(20, 10)
    outputs = ddp_model(inputs)
    labels = torch.randn(20, 5).to(rank)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


def run_demo(demo_fn: Callable, world_size: int) -> None:
    """
    Run a distributed demo function across multiple processes.

    Args:
        demo_fn (Callable): The demo function to run
        world_size (int): Total number of processes to spawn
    """
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def demo_checkpoint(rank: int, world_size: int) -> None:
    """
    Demonstrate Distributed Data Parallel (DDP) training with model checkpointing.

    Args:
        rank (int): Rank of the current process
        world_size (int): Total number of processes
    """
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Create a temporary checkpoint path
    CHECKPOINT_PATH = os.path.join(tempfile.gettempdir(), "model.checkpoint")
    
    # Save checkpoint from rank 0 process
    if rank == 0:
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Synchronize processes before loading
    dist.barrier()
    
    # Load checkpoint with proper device mapping
    map_location = {'cuda:%d' % 0: f'cuda:{rank}'}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True))

    # Perform a training step
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    inputs = torch.randn(20, 10)
    outputs = ddp_model(inputs)
    labels = torch.randn(20, 5).to(rank)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    # Remove checkpoint file from rank 0
    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    print(f"Finished running DDP checkpoint example on rank {rank}.")


class ToyMpModel(nn.Module):
    """
    A toy model demonstrating model parallelism across two devices.

    Attributes:
        dev0 (int): First device for the first layer
        dev1 (int): Second device for the second layer
        net1 (nn.Linear): First linear layer on dev0
        relu (nn.ReLU): ReLU activation function
        net2 (nn.Linear): Second linear layer on dev1
    """
    def __init__(self, dev0: int, dev1: int) -> None:
        """
        Initialize the model with two devices.

        Args:
            dev0 (int): First device ID
            dev1 (int): Second device ID
        """
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with model parallelism.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)


def demo_model_parallel(rank: int, world_size: int) -> None:
    """
    Demonstrate Distributed Data Parallel (DDP) with model parallelism.

    Args:
        rank (int): Rank of the current process
        world_size (int): Total number of processes
    """
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # Setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    # Perform a training step
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
    print(f"Finished running DDP with model parallel example on rank {rank}.")


def main() -> None:
    """
    Main function to run distributed training demonstrations.
    Checks GPU availability and runs different distributed training scenarios.
    """
    # Check GPU availability
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    # Run basic DDP example
    world_size = n_gpus
    run_demo(demo_basic, world_size)

    # Run checkpoint example
    run_demo(demo_checkpoint, world_size)

    # Run model parallel example
    world_size = n_gpus // 2
    run_demo(demo_model_parallel, world_size)


if __name__ == "__main__":
    main()