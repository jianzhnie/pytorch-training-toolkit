import io
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import boto3
import fsspec
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


@dataclass
class TrainerConfig:
    """Configuration class for distributed training parameters.

    Attributes:
        max_epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per batch.
        data_loader_workers (int): Number of subprocesses for data loading.
        grad_norm_clip (float): Gradient norm clipping threshold.
        snapshot_path (Optional[str]): Path to save/load training snapshots.
        save_every (int): Frequency of saving model snapshots (epochs).
        use_amp (bool): Whether to use Automatic Mixed Precision (AMP).
    """

    max_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    data_loader_workers: int = 0
    grad_norm_clip: float = 1.0
    snapshot_path: Optional[str] = 'snapshot.pt'
    save_every: int = 1
    use_amp: bool = False


@dataclass
class Snapshot:
    """Represents a training snapshot for model resumption.

    Attributes:
        model_state (OrderedDict): Serialized model state dictionary.
        optimizer_state (Dict): Serialized optimizer state dictionary.
        finished_epoch (int): Last completed training epoch.
    """

    model_state: OrderedDict[str, torch.Tensor]
    optimizer_state: Dict[str, Any]
    finished_epoch: int


class Trainer:
    """A flexible distributed training utility for PyTorch models.

    Supports features like:
    - Distributed Data Parallel (DDP) training
    - Automatic Mixed Precision (AMP)
    - Snapshot-based training resumption
    - S3 snapshot storage
    """

    def __init__(
        self,
        trainer_config: TrainerConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
    ) -> None:
        """Initialize the distributed trainer.

        Args:
            trainer_config (TrainerConfig): Training configuration.
            model (torch.nn.Module): Model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            train_dataset (Dataset): Training dataset.
            test_dataset (Optional[Dataset], optional): Validation dataset.

        Raises:
            RuntimeError: If distributed environment is not properly set up.
        """
        # Validate distributed environment
        if not all(key in os.environ for key in ['LOCAL_RANK', 'RANK']):
            raise RuntimeError(
                'Distributed environment not initialized. Use torchrun.')

        self.config = trainer_config

        # Set distributed training attributes
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.global_rank = int(os.environ['RANK'])

        # Prepare datasets and loaders
        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = (self._prepare_dataloader(test_dataset)
                            if test_dataset else None)

        # Initialize training state
        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer

        # Configure AMP if enabled
        self.scaler = torch.amp.GradScaler() if self.config.use_amp else None

        # Load snapshot if available
        self._load_snapshot()

        # Wrap model with Distributed Data Parallel
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _prepare_dataloader(
            self, dataset: Optional[Dataset]) -> Optional[DataLoader]:
        """Prepare a distributed dataloader.

        Args:
            dataset (Optional[Dataset]): Input dataset.

        Returns:
            Optional[DataLoader]: Configured DataLoader or None.
        """
        if not dataset:
            return None

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size or 1,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset),
        )

    def _load_snapshot(self) -> None:
        """Load training snapshot from file or S3.

        Handles resuming training from a previous checkpoint.
        """
        try:
            with fsspec.open(self.config.snapshot_path) as f:
                snapshot_data = torch.load(f, map_location='cpu')
        except FileNotFoundError:
            print('No snapshot found. Starting training from scratch.')
            return

        try:
            snapshot = Snapshot(**snapshot_data)
            self.model.load_state_dict(snapshot.model_state)
            self.optimizer.load_state_dict(snapshot.optimizer_state)
            self.epochs_run = snapshot.finished_epoch
            print(f'Resumed training from snapshot at Epoch {self.epochs_run}')
        except Exception as e:
            print(f'Error loading snapshot: {e}. Starting from scratch.')

    def _run_batch(self,
                   source: torch.Tensor,
                   targets: torch.Tensor,
                   train: bool = True) -> float:
        """Process a single batch of data.

        Args:
            source (torch.Tensor): Input data tensor.
            targets (torch.Tensor): Target data tensor.
            train (bool, optional): Whether in training mode. Defaults to True.

        Returns:
            float: Batch loss value.
        """
        with (
                torch.set_grad_enabled(train),
                torch.amp.autocast(
                    device_type='cuda',
                    dtype=torch.float16,
                    enabled=bool(self.config.use_amp),
                ),
        ):
            _, loss = self.model(source, targets)

        if train:
            self.optimizer.zero_grad(set_to_none=True)

            # Handle gradient scaling based on AMP configuration
            if self.config.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config.grad_norm_clip)
                self.optimizer.step()

        return loss.item()

    def _run_epoch(self,
                   epoch: int,
                   dataloader: DataLoader,
                   train: bool = True) -> None:
        """Run a single training or evaluation epoch.

        Args:
            epoch (int): Current epoch number.
            dataloader (DataLoader): Dataset loader.
            train (bool, optional): Training mode flag. Defaults to True.
        """
        dataloader.sampler.set_epoch(epoch)

        for iter, (source, targets) in enumerate(dataloader):
            step_type = 'Train' if train else 'Eval'
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)

            batch_loss = self._run_batch(source, targets, train)

            # Periodic logging
            if iter % 100 == 0:
                print(f'[GPU{self.global_rank}] '
                      f'Epoch {epoch} | Iter {iter} | '
                      f'{step_type} Loss {batch_loss:.5f}')

    def _save_snapshot(self, epoch: int) -> None:
        """Save training snapshot.

        Args:
            epoch (int): Current training epoch.
        """
        # Extract raw model state
        raw_model = self.model.module if hasattr(self.model,
                                                 'module') else self.model

        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
        )

        # Serialize and save snapshot
        snapshot_dict = asdict(snapshot)
        try:
            if self.config.snapshot_path.startswith('s3://'):
                self._upload_to_s3(snapshot_dict, self.config.snapshot_path)
            else:
                torch.save(snapshot_dict, self.config.snapshot_path)
            print(f'Snapshot saved at epoch {epoch}')
        except Exception as e:
            print(f'Failed to save snapshot: {e}')

    def _upload_to_s3(self, obj: Dict[str, Any], dst: str) -> None:
        """Upload snapshot to S3.

        Args:
            obj (Dict[str, Any]): Snapshot dictionary to upload.
            dst (str): S3 destination URL.
        """
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer.seek(0)

        parsed_url = urlparse(dst, allow_fragments=False)
        boto3.client('s3').upload_fileobj(buffer, parsed_url.netloc,
                                          parsed_url.path.lstrip('/'))

    def train(self) -> None:
        """Main training loop.

        Runs epochs, handles model saving, and optional validation.
        """
        for epoch in range(self.epochs_run, self.config.max_epochs or 0):
            epoch += 1

            # Training epoch
            self._run_epoch(epoch, self.train_loader, train=True)

            # Periodic snapshot saving (only on primary process)
            if self.local_rank == 0 and epoch % self.config.save_every == 0:
                self._save_snapshot(epoch)

            # Optional evaluation
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)


def main() -> None:
    """Example usage of the Trainer class."""
    # Example configuration and initialization
    config = TrainerConfig(max_epochs=10, batch_size=32, use_amp=True)

    # Placeholder model and dataset (replace with your actual implementations)
    model = torch.nn.Module()
    optimizer = torch.optim.Adam(model.parameters())
    train_dataset = torch.utils.data.Dataset()

    trainer = Trainer(config, model, optimizer, train_dataset)
    trainer.train()


if __name__ == '__main__':
    main()
