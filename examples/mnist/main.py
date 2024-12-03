import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .lenet import Net


class Trainer:

    def __init__(
        self,
        args,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: Optional[int, torch.device] = None,
    ) -> None:
        """Initialize the Trainer with model, data, and training parameters.

        Args:
            model (torch.nn.Module): Neural network model to train.
            train_data (DataLoader): Iterable training data.
            optimizer (torch.optim.Optimizer): Optimization algorithm.
            device (int): GPU device ID for training.
            save_every (int): Checkpoint saving frequency.
        """
        self.args = args
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

    def run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
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
        loss = F.nll_loss(output, targets)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss

    def run_epoch(self, epoch) -> None:
        loss = self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.run_batch(data, target)
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100.0 * batch_idx / len(self.train_loader),
                    loss.item(),
                ))
                if self.args.dry_run:
                    break

    def test(self) -> None:
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(
                    output, target,
                    reduction='sum').item()  # sum up batch loss
                pred = output.argmax(
                    dim=1,
                    keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(
                  test_loss,
                  correct,
                  len(self.test_loader.dataset),
                  100.0 * correct / len(self.test_loader.dataset),
              ))

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            self.run_epoch(epoch)
            self.test()
            self.scheduler.step()

        if self.args.save_model:
            torch.save(self.model.state_dict(), 'mnist_cnn.pt')

    def save_checkpoint(self, epoch: int, path: Optional[str] = None) -> None:
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


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)',
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        metavar='N',
        help='input batch size for testing (default: 1000)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=14,
        metavar='N',
        help='number of epochs to train (default: 14)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1.0,
        metavar='LR',
        help='learning rate (default: 1.0)',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.7,
        metavar='M',
        help='Learning rate step gamma (default: 0.7)',
    )
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument(
        '--no-mps',
        action='store_true',
        default=False,
        help='disables macOS GPU training',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='quickly check a single pass',
    )
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status',
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        default=False,
        help='For Saving the current Model',
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device('cuda')
    elif use_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])
    dataset1 = datasets.MNIST(root=args.data_path,
                              train=True,
                              download=True,
                              transform=transform)
    dataset2 = datasets.MNIST(root=args.data_path,
                              train=False,
                              download=True,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    trainer = Trainer(args, model, train_loader, test_loader, optimizer,
                      scheduler, device)
    trainer.train()


if __name__ == '__main__':
    main()
