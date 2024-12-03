import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
# Import your custom dataset
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


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
        self.data: List[Tuple[torch.Tensor,
                              torch.Tensor]] = [(torch.rand(20), torch.rand(1))
                                                for _ in range(size)]

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


def ddp_setup() -> None:
    """Set up the distributed environment for training.

    Uses environment variables for configuration:
    - LOCAL_RANK: Specifies the local GPU device

    Initializes the process group using NCCL backend.
    """
    try:
        # Set the current device for the process
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)

        # Initialize the process group
        init_process_group(backend='nccl')
    except Exception as e:
        print(f'Error setting up distributed environment: {e}')
        raise


class Trainer:
    """Distributed Data Parallel (DDP) Trainer with snapshot resume capability.

    Handles distributed training across multiple GPUs, including batch
    processing, epoch management, and checkpointing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        """Initialize the Trainer with distributed training components.

        Args:
            model (torch.nn.Module): Neural network model to train.
            train_data (DataLoader): Distributed data loader.
            optimizer (torch.optim.Optimizer): Optimization algorithm.
            save_every (int): Frequency of saving checkpoints.
            snapshot_path (str): Path to save/load training snapshots.
        """
        # Determine GPU device
        self.gpu_id = int(os.environ.get('LOCAL_RANK', 0))

        # Move model to the specific GPU
        self.model = model.to(self.gpu_id)

        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.snapshot_path = snapshot_path

        # Track epochs for resume capability
        self.epochs_run = 0

        # Load snapshot if exists
        if os.path.exists(snapshot_path):
            print('Loading snapshot')
            self._load_snapshot(snapshot_path)

        # Wrap model with DistributedDataParallel
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path: str) -> None:
        """Load a training snapshot to resume training.

        Args:
            snapshot_path (str): Path to the snapshot file.

        Raises:
            FileNotFoundError: If snapshot file doesn't exist.
            RuntimeError: If there are issues loading the snapshot.
        """
        try:
            # Determine device location
            loc = f'cuda:{self.gpu_id}'

            # Load snapshot
            snapshot = torch.load(snapshot_path, map_location=loc)

            # Restore model state
            self.model.load_state_dict(snapshot['MODEL_STATE'])

            # Restore epoch count
            self.epochs_run = snapshot['EPOCHS_RUN']

            print(
                f'Resuming training from snapshot at Epoch {self.epochs_run}')
        except FileNotFoundError:
            print(f'Snapshot file not found: {snapshot_path}')
            raise
        except Exception as e:
            print(f'Error loading snapshot: {e}')
            raise

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
            f'[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}'
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

    def _save_snapshot(self, epoch: int) -> None:
        """Save training snapshot.

        Args:
            epoch (int): Current epoch number.
        """
        try:
            # Create snapshot dictionary
            snapshot = {
                'MODEL_STATE': self.model.module.state_dict(),
                'EPOCHS_RUN': epoch,
            }

            # Save snapshot
            torch.save(snapshot, self.snapshot_path)

            print(
                f'Epoch {epoch} | Training snapshot saved at {self.snapshot_path}'
            )
        except Exception as e:
            print(f'Error saving snapshot: {e}')

    def train(self, max_epochs: int) -> None:
        """Main training loop.

        Args:
            max_epochs (int): Total number of training epochs.
        """
        for epoch in range(self.epochs_run, max_epochs):
            # Run epoch
            self._run_epoch(epoch)

            # Save checkpoint periodically
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs(
) -> Tuple[Dataset, torch.nn.Module, torch.optim.Optimizer]:
    """Load training objects including dataset, model, and optimizer.

    Returns:
        Tuple containing training dataset, model, and optimizer.
    """
    # Load dataset (replace with your specific dataset)
    train_set = MyTrainDataset(2048)

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
    save_every: int,
    total_epochs: int,
    batch_size: int,
    snapshot_path: str = 'snapshot.pt',
) -> None:
    """Main training execution function for distributed training.

    Args:
        save_every (int): Checkpoint saving frequency.
        total_epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        snapshot_path (str, optional): Path to save/load training snapshots.
    """
    try:
        # Setup distributed environment
        ddp_setup()

        # Load training objects
        dataset, model, optimizer = load_train_objs()

        # Prepare distributed data loader
        train_data = prepare_dataloader(dataset, batch_size)

        # Create trainer and start training
        trainer = Trainer(model, train_data, optimizer, save_every,
                          snapshot_path)
        trainer.train(total_epochs)
    except Exception as e:
        print(f'Training failed: {e}')
    finally:
        # Cleanup distributed environment
        destroy_process_group()


if __name__ == '__main__':
    import argparse

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Distributed Training Job')
    parser.add_argument('--total_epochs',
                        type=int,
                        default=2,
                        help='Total epochs to train the model')
    parser.add_argument('--save_every',
                        type=int,
                        default=1000,
                        help='How often to save a snapshot')
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='Input batch size on each device (default: 32)',
    )

    # Parse arguments
    args = parser.parse_args()

    # Run main training function
    main(args.save_every, args.total_epochs, args.batch_size)
