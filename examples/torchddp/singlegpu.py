from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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


class Trainer:
    """A trainer class for distributed machine learning training with PyTorch.

    Handles model training, checkpointing, and GPU management.

    Attributes:
        gpu_id (int): The GPU device ID for training.
        model (torch.nn.Module): The neural network model to be trained.
        train_data (DataLoader): DataLoader containing training data.
        optimizer (torch.optim.Optimizer): Optimization algorithm for model training.
        save_every (int): Frequency of saving model checkpoints.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        """Initialize the Trainer with model, data, and training parameters.

        Args:
            model (torch.nn.Module): Neural network model to train.
            train_data (DataLoader): Iterable training data.
            optimizer (torch.optim.Optimizer): Optimization algorithm.
            gpu_id (int): GPU device ID for training.
            save_every (int): Checkpoint saving frequency.
        """
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        """Run a single batch of training data.

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

        print(
            f'[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}'
        )

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
                Defaults to None (uses default path).
        """
        # Use provided path or default
        checkpoint_path = path or f'checkpoint_epoch_{epoch}.pt'

        # Save model state dictionary
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

        print(
            f'Epoch {epoch} | Training checkpoint saved at {checkpoint_path}')

    def train(self, max_epochs: int) -> None:
        """Main training loop.

        Args:
            max_epochs (int): Total number of training epochs.
        """
        for epoch in range(max_epochs):
            # Run epoch
            self._run_epoch(epoch)

            # Save checkpoint periodically
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs(
) -> Tuple[Dataset, torch.nn.Module, torch.optim.Optimizer]:
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
    """Prepare DataLoader for training.

    Args:
        dataset (Dataset): Input dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: Configured data loader.
    """
    return DataLoader(dataset,
                      batch_size=batch_size,
                      pin_memory=True,
                      shuffle=True)


def main(device: int, total_epochs: int, save_every: int,
         batch_size: int) -> None:
    """Main training execution function.

    Args:
        device (int): GPU device ID.
        total_epochs (int): Number of training epochs.
        save_every (int): Checkpoint saving frequency.
        batch_size (int): Number of samples per batch.
    """
    # Load training objects
    dataset, model, optimizer = load_train_objs()

    # Prepare data loader
    train_data = prepare_dataloader(dataset, batch_size)

    # Create trainer and start training
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)


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

    # Set device (shorthand for cuda:0)
    device = 0

    # Run main training function
    main(device, args.total_epochs, args.save_every, args.batch_size)
