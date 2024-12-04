import argparse
import logging
import os
import sys
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lenet import Net
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

sys.path.append(os.getcwd())
from scaletorch.utils.dist_utils import (ddp_cleanup, ddp_setup,
                                         system_diagnostic)

# Configure global logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class DistributedTrainer:
    """A distributed trainer class for PyTorch model training using
    DistributedDataParallel."""

    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Initialize the Distributed Trainer.

        Args:
            args (argparse.Namespace): Command-line arguments
            rank (int): Local rank of the current process
            world_size (int): Total number of distributed processes
            model (nn.Module): Neural network model
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Test data loader
            optimizer (torch.optim.Optimizer): Optimization algorithm
            scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
        """
        self.args = args

        # 获取当前进程的 rank
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Setup device
        self.device = self._get_device()

        # Wrap model with DistributedDataParallel
        self.model = model.to(self.device)
        if dist.is_initialized():
            model = DDP(model,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Centralized logging configuration
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Configure logging with rank-aware settings."""
        if self.rank != 0:
            logging.disable(logging.INFO)

    def _get_device(self) -> torch.device:
        """Safely determine the compute device with comprehensive fallback.

        Returns:
            torch.device: Configured device for computation
        """
        if torch.cuda.is_available():
            return torch.device(f'cuda:{self.local_rank}')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        """Process a single training batch in distributed setting.

        Args:
            source (torch.Tensor): Input data tensor
            targets (torch.Tensor): Target labels tensor

        Returns:
            float: Computed loss value for the batch
        """
        # Reset gradients
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(source)

        # Compute loss
        loss = F.nll_loss(output, targets)

        # Backward pass and parameter update
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_epoch(self, epoch: int) -> float:
        """Train the model for one epoch in distributed setting.

        Args:
            epoch (int): Current epoch number

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()

        # Set epoch for distributed sampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)

            # Process batch and accumulate loss
            batch_loss = self.run_batch(data, target)
            total_loss += batch_loss

            # Log progress at specified intervals
            if self.rank == 0 and batch_idx % self.args.log_interval == 0:
                logger.info(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                    f'({100.0 * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {batch_loss:.6f}'
                )

                # Stop after first batch if dry run
                if self.args.dry_run:
                    break

        return total_loss / len(self.train_loader)

    def test(self) -> Dict[str, float]:
        """Evaluate the model on test dataset in distributed setting.

        Returns:
            Dict[str, float]: Dictionary containing test loss and accuracy
        """
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # Accumulate test loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()

                # Count correct predictions
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Reduce metrics across all processes
        metrics = torch.tensor([test_loss, correct], device=self.device)
        dist.all_reduce(metrics)

        test_loss = metrics[0].item() / len(self.test_loader.dataset)
        correct = metrics[1].item()
        accuracy = 100.0 * correct / len(self.test_loader.dataset)

        if self.rank == 0:
            logger.info(
                f'\nTest set: Average loss: {test_loss:.4f}, '
                f'Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.0f}%)\n'
            )

        return {'loss': test_loss, 'accuracy': accuracy}

    def train(self) -> None:
        """Execute complete distributed model training process."""
        try:
            for epoch in range(1, self.args.epochs + 1):
                # Train for one epoch
                epoch_loss = self.run_epoch(epoch)

                # Log epoch loss on primary process (optional)
                if self.rank == 0:
                    logger.info(f'Epoch {epoch} Loss: {epoch_loss:.4f}')

                # Synchronize all processes
                dist.barrier()

                # Perform testing on primary process
                test_metrics = self.test()
                if self.rank == 0:
                    logger.info(f'Epoch {epoch}, Eval Metrics: {test_metrics}')

                # Step learning rate scheduler
                self.scheduler.step()

            # Optional: Save trained model on primary process
            if self.rank == 0 and self.args.save_model:
                self.save_checkpoint(self.args.epochs)

        except Exception as e:
            logger.error(f'Training failed: {e}')
            raise

    def save_checkpoint(self,
                        epoch: int,
                        path: Optional[str] = None) -> Optional[str]:
        """Save model checkpoint with training state.

        Args:
            epoch (int): Current epoch number
            path (Optional[str], optional): Custom checkpoint path

        Returns:
            Optional[str]: Path where checkpoint was saved
        """
        if self.rank != 0:
            return None

        # Use provided path or generate default
        checkpoint_path = path or f'checkpoint_epoch_{epoch}.pt'

        # Save comprehensive checkpoint
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            },
            checkpoint_path,
        )

        logger.info(f'Epoch {epoch} | Checkpoint saved at {checkpoint_path}')
        return checkpoint_path


def prepare_data(args: argparse.Namespace) -> tuple:
    """Prepare distributed datasets and data loaders.

    Args:
        args (argparse.Namespace): Command-line arguments

    Returns:
        tuple: Containing train_loader and test_loader
    """
    # Prepare data transformations
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    # Load MNIST datasets
    train_dataset = datasets.MNIST(root=args.data_path,
                                   train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root=args.data_path,
                                  train=False,
                                  download=True,
                                  transform=transform)

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for distributed training.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Distributed PyTorch MNIST Training')

    # Training configuration
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Training batch size')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        help='Test batch size')
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        help='Learning rate decay')

    # Utility arguments
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Quick training check')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--log-interval',
                        type=int,
                        default=100,
                        help='Logging frequency')
    parser.add_argument('--save-model',
                        action='store_true',
                        help='Save trained model')
    parser.add_argument('--data-path',
                        type=str,
                        default='./data',
                        help='Dataset download path')

    return parser.parse_args()


def main() -> None:
    """Main function to launch distributed training."""
    # Parse arguments
    # 在脚本开始处调用
    system_diagnostic()

    args = parse_arguments()
    torch.backends.cudnn.benchmark = True

    # Set random seed
    torch.manual_seed(args.seed)

    # Determine number of GPUs
    # Distributed setup validation
    if torch.cuda.device_count() < 2:
        logger.error('Distributed training requires multiple GPUs')
        sys.exit(1)

    # Provide guidance on how to run the distributed script
    logger.info('To run this script in a distributed manner, use:')
    logger.info('torchrun --nproc_per_node=<num_gpus> script_name.py')

    # Setup distributed environment
    try:
        ddp_setup()
        # Prepare data loaders
        train_loader, test_loader = prepare_data(args)

        # Initialize model
        model = Net()

        # Setup optimizer and scheduler
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        # Create distributed trainer
        trainer = DistributedTrainer(
            args=args,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        # Start training
        trainer.train()
    except Exception as e:
        logger.error(f'Training failed: {e}')
    finally:
        ddp_cleanup()


if __name__ == '__main__':
    main()
