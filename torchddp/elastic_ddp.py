import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    """
    A simple neural network model for demonstrating Distributed Data Parallel (DDP) training.

    The model consists of two linear layers with a ReLU activation between them.
    
    Attributes:
        net1 (nn.Linear): First linear layer (10 input, 10 output features)
        relu (nn.ReLU): ReLU activation function
        net2 (nn.Linear): Second linear layer (10 input, 5 output features)
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


def demo_basic() -> None:
    """
    Demonstrate basic Distributed Data Parallel (DDP) training.

    This function:
    1. Sets the current CUDA device
    2. Initializes the process group
    3. Creates a model and wraps it with DDP
    4. Performs a simple training step
    5. Cleans up the process group
    """
    # Set the current CUDA device based on LOCAL_RANK environment variable
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Initialize the distributed process group
    dist.init_process_group(backend="nccl")
    
    # Get the current process rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        print(f"Start running basic DDP example on rank {rank} (World Size: {world_size}).")

        # Determine the device ID for this process
        device_id = rank % torch.cuda.device_count()

        # Create model and move it to the appropriate GPU
        model = ToyModel().to(device_id)
        
        # Wrap the model with Distributed Data Parallel
        ddp_model = DDP(model, device_ids=[device_id])

        # Set up loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        # Perform a mock training step
        optimizer.zero_grad()
        
        # Generate random input and label tensors
        inputs = torch.randn(20, 10).to(device_id)
        labels = torch.randn(20, 5).to(device_id)

        # Forward pass, loss calculation, and optimization
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Loss for rank {rank}: {loss.item():.4f}")

    except Exception as e:
        print(f"Error in DDP training on rank {rank}: {e}")
    finally:
        # Ensure process group is always destroyed
        dist.destroy_process_group()
        print(f"Finished running basic DDP example on rank {rank}.")


def main() -> None:
    """
    Main entry point for running the Distributed Data Parallel example.
    
    Checks for CUDA availability and provides guidance for running the script.
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA-capable GPUs.")

    # Provide guidance on how to run the distributed script
    print("To run this script in a distributed manner, use:")
    print("torchrun --nproc_per_node=<num_gpus> script_name.py")
    demo_basic()

if __name__ == "__main__":
    main()