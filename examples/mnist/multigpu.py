import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# Import your custom dataset
class MyTrainDataset(Dataset):
    """A custom PyTorch dataset for generating random training data.

    This dataset creates a specified number of random data points with
    20-dimensional input features and 1-dimensional targets.

    Attributes:
        size (int): Number of data points in the dataset.
        data (List[Tuple[torch.Tensor, torch.Tensor]]): List of (input, target) pairs.
    """

    def __init__(self, size: int):
        """Initialize the dataset with a specified number of data points.

        Args:
            size (int): Number of random data points to generate.
        """
        self.size = size
        # Generate random input tensors (20 dimensions) and target tensors (1 dimension)
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (torch.rand(20), torch.rand(1)) for _ in range(size)
        ]

    def __len__(self) -> int:
        """Return the total number of data points in the dataset.

        Returns:
            int: Number of data points.
        """
        return self.size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single data point by its index.

        Args:
            index (int): Index of the data point to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing input and target tensors.

        Raises:
            IndexError: If the index is out of bounds.
        """
        return self.data[index]


def ddp_setup(rank: int, world_size: int) -> None:
    """Set up the distributed environment for training.

    Args:
        rank (int): Unique identifier of each process.
        world_size (int): Total number of processes.
    """
    # Set environment variables for process communication
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Set the current device for the process
    torch.cuda.set_device(rank)

    # Initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    """Distributed Data Parallel (DDP) Trainer for PyTorch models.

    Handles distributed training across multiple GPUs, including batch
    processing, epoch management, and checkpointing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        """Initialize the Trainer with distributed training components.

        Args:
            model (torch.nn.Module): Neural network model to train.
            train_data (DataLoader): Distributed data loader.
            optimizer (torch.optim.Optimizer): Optimization algorithm.
            gpu_id (int): GPU device ID for this process.
            save_every (int): Frequency of saving checkpoints.
        """
        # Move model to the specific GPU
        self.model = model.to(gpu_id)

        # Wrap model with DistributedDataParallel
        self.model = DDP(self.model, device_ids=[gpu_id])

        self.gpu_id = gpu_id
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        """Process a single batch of training data.

        Args:
            source (torch.Tensor): Input data tensor.
            targets (torch.Tensor): Target labels tensor.

        Returns:
            float: Computed loss value for the batch.
        """
        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(source)

        # Compute loss
        loss = F.cross_entropy(output, targets)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _run_epoch(self, epoch: int) -> None:
        """Run a single training epoch.

        Args:
            epoch (int): Current epoch number.
        """
        # Determine batch size
        b_sz = len(next(iter(self.train_data))[0])

        # Print epoch information
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )

        # Set epoch for distributed sampler to ensure proper shuffling
        self.train_data.sampler.set_epoch(epoch)

        # Iterate through batches
        for source, targets in self.train_data:
            # Move data to GPU
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)

            # Run batch training
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch: int, path: Optional[str] = None) -> None:
        """Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
            path (Optional[str], optional): Custom path to save checkpoint.
        """
        # Only save checkpoint from the primary process
        if self.gpu_id == 0:
            # Use provided path or create default path
            checkpoint_path = path or f"checkpoint_epoch_{epoch}.pt"

            # Save model state dictionary
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.module.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                checkpoint_path,
            )

            print(f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}")

    def train(self, max_epochs: int) -> None:
        """Main training loop.

        Args:
            max_epochs (int): Total number of training epochs.
        """
        for epoch in range(max_epochs):
            # Run epoch
            self._run_epoch(epoch)

            # Save checkpoint periodically
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs() -> Tuple[Dataset, torch.nn.Module, torch.optim.Optimizer]:
    """Load training objects including dataset, model, and optimizer.

    Returns:
        Tuple containing training dataset, model, and optimizer.
    """
    # Load dataset (replace with your specific dataset)
    train_set = MyTrainDataset(20480)

    # Create model (replace with your model architecture)
    model = torch.nn.Linear(20, 1)

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """Prepare distributed DataLoader for training.

    Args:
        dataset (Dataset): Input dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: Configured distributed data loader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def main(
    rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int
) -> None:
    """Main training execution function for distributed training.

    Args:
        rank (int): Unique identifier of the current process.
        world_size (int): Total number of processes.
        save_every (int): Checkpoint saving frequency.
        total_epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
    """
    # Setup distributed environment
    ddp_setup(rank, world_size)

    # Load training objects
    dataset, model, optimizer = load_train_objs()

    # Prepare distributed data loader
    train_data = prepare_dataloader(dataset, batch_size)

    # Create trainer and start training
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)

    # Cleanup distributed environment
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Distributed Training Job")
    parser.add_argument(
        "--total_epochs", type=int, default=2, help="Total epochs to train the model"
    )
    parser.add_argument(
        "--save_every", type=int, default=1000, help="How often to save a snapshot"
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Determine number of available GPUs
    world_size = torch.cuda.device_count()

    # Launch distributed training
    mp.spawn(
        main,
        args=(world_size, args.save_every, args.total_epochs, args.batch_size),
        nprocs=world_size,
    )
