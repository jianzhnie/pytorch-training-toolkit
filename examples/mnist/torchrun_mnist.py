import argparse
import logging
import os
import sys
from typing import Dict, Optional, Tuple

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

# Append current working directory to system path
sys.path.append(os.getcwd())

# Import distributed utilities
from scaletorch.utils.torch_dist import ddp_cleanup, ddp_setup, get_system_info

# Configure global logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class DistributedTrainer:
    """A comprehensive distributed trainer class for PyTorch model training
    using DistributedDataParallel (DDP) strategy.

    This class encapsulates the entire training workflow, including:
    - Distributed setup
    - Device management
    - Model training
    - Validation
    - Checkpointing
    """

    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Initialize the Distributed Trainer with all necessary components.

        Args:
            args (argparse.Namespace): Parsed command-line arguments
            model (nn.Module): Neural network model to be trained
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Validation/test data loader
            optimizer (torch.optim.Optimizer): Optimization algorithm
            scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
        """
        self.args = args

        # Determine process rank and local rank
        self.rank: int = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank: int = int(os.environ.get('LOCAL_RANK', 0))

        # Setup device
        self.device: torch.device = self._get_device()

        # Wrap model with DistributedDataParallel
        self.model = model.to(self.device)
        if dist.is_initialized():
            self.model = DDP(model, device_ids=[self.local_rank])

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Configure logging
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Configure logging to minimize redundant log entries across
        distributed processes.

        Only the primary process (rank 0) will log informational messages.
        """
        if self.rank != 0:
            logging.disable(logging.INFO)

    def _get_device(self) -> torch.device:
        """Intelligently select the computation device with comprehensive
        fallback strategy.

        Returns:
            torch.device: Optimal device for computation
        """
        if torch.cuda.is_available():
            return torch.device(f'cuda:{self.local_rank}')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        """Process a single training batch in a distributed environment.

        Args:
            source (torch.Tensor): Input data tensor
            targets (torch.Tensor): Target labels tensor

        Returns:
            float: Computed loss value for the batch
        """
        # Zero out previous gradients
        self.optimizer.zero_grad()

        # Compute model output
        output = self.model(source)

        # Compute negative log-likelihood loss
        loss = F.nll_loss(output, targets)

        # Backpropagate and update parameters
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_epoch(self, epoch: int) -> float:
        """Train the model for a single epoch in a distributed setting.

        Args:
            epoch (int): Current training epoch number

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()

        # Set epoch for distributed sampler to ensure proper shuffling
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to selected device
            data, target = data.to(self.device), target.to(self.device)

            # Process batch and accumulate loss
            batch_loss = self.run_batch(data, target)
            total_loss += batch_loss

            # Periodic logging for primary process
            if self.rank == 0 and batch_idx % self.args.log_interval == 0:
                logger.info(
                    f'Train Epoch: {epoch} '
                    f'[{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                    f'({100.0 * batch_idx / len(self.train_loader):.0f}%)]\t'
                    f'Loss: {batch_loss:.6f}')

                # Optional: Early stopping for dry run
                if self.args.dry_run:
                    break

        return total_loss / len(self.train_loader)

    def test(self) -> Dict[str, float]:
        """Evaluate model performance on test dataset across distributed
        processes.

        Returns:
            Dict[str, float]: Evaluation metrics including loss and accuracy
        """
        self.model.eval()
        test_loss, correct = 0.0, 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # Accumulate test loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()

                # Count correct predictions
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Aggregate metrics across all processes
        metrics = torch.tensor([test_loss, correct], device=self.device)
        dist.all_reduce(metrics)

        # Compute final metrics
        test_loss = metrics[0].item() / len(self.test_loader.dataset)
        correct = metrics[1].item()
        accuracy = 100.0 * correct / len(self.test_loader.dataset)

        # Log results on primary process
        if self.rank == 0:
            logger.info(f'\nTest set: Average loss: {test_loss:.4f}, '
                        f'Accuracy: {correct}/{len(self.test_loader.dataset)} '
                        f'({accuracy:.0f}%)\n')

        return {'loss': test_loss, 'accuracy': accuracy}

    def train(self) -> None:
        """Execute the complete distributed training workflow.

        Manages epoch training, testing, and optional model checkpointing.
        """
        try:
            for epoch in range(1, self.args.epochs + 1):
                # Train for one epoch
                epoch_loss = self.run_epoch(epoch)

                # Log epoch loss on primary process
                if self.rank == 0:
                    logger.info(f'Epoch {epoch} Loss: {epoch_loss:.4f}')

                # Synchronize processes
                dist.barrier()

                # Perform testing
                test_metrics = self.test()

                if self.rank == 0:
                    logger.info(f'Epoch {epoch}, Eval Metrics: {test_metrics}')

                # Update learning rate
                self.scheduler.step()

            # Optional model saving
            if self.rank == 0 and self.args.save_model:
                self.save_checkpoint(self.args.epochs)

        except Exception as e:
            logger.error(f'Training failed: {e}')
            raise

    def save_checkpoint(self,
                        epoch: int,
                        path: Optional[str] = None) -> Optional[str]:
        """Save a comprehensive model checkpoint.

        Args:
            epoch (int): Current training epoch
            path (Optional[str], optional): Custom checkpoint path

        Returns:
            Optional[str]: Path where checkpoint was saved, or None
        """
        if self.rank != 0:
            return None

        checkpoint_path = path or f'checkpoint_epoch_{epoch}.pt'

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


def prepare_data(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Prepare distributed datasets and data loaders for training.

    Args:
        args (argparse.Namespace): Parsed command-line arguments

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing data loaders
    """
    # Data normalization parameters for MNIST
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

    # Create data loaders with enhanced configuration
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
    """Parse command-line arguments for distributed training configuration.

    Returns:
        argparse.Namespace: Parsed and validated training arguments
    """
    parser = argparse.ArgumentParser(
        description='Distributed PyTorch MNIST Training')

    # Training hyperparameters
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
                        help='Learning rate decay factor')

    # Utility arguments
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Quick training verification')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--log-interval',
                        type=int,
                        default=100,
                        help='Batch logging frequency')
    parser.add_argument('--save-model',
                        action='store_true',
                        help='Save trained model checkpoint')
    parser.add_argument('--data-path',
                        type=str,
                        default='./data',
                        help='Dataset download directory')

    return parser.parse_args()


def main() -> None:
    """Main entry point for distributed training workflow.

    Manages distributed setup, training initialization, and execution.
    """
    # Initial system diagnostic
    get_system_info()

    # Parse command-line arguments
    args = parse_arguments()

    # Enable cuDNN benchmark for performance optimization
    torch.backends.cudnn.benchmark = True

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Validate GPU availability
    if torch.cuda.device_count() < 2:
        logger.error('Distributed training requires multiple GPUs')
        sys.exit(1)

    # Provide distributed launch guidance
    logger.info('Distributed launch command:')
    logger.info('torchrun --nproc_per_node=<num_gpus> script_name.py')

    try:
        # Setup distributed environment
        ddp_setup()

        # Prepare data loaders
        train_loader, test_loader = prepare_data(args)

        # Initialize model (assuming Net is imported from lenet)
        model = Net()

        # Setup optimizer and learning rate scheduler
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
        # Cleanup distributed resources
        ddp_cleanup()


if __name__ == '__main__':
    main()
