from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import fsspec
import torch
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    """Configuration class for character-level dataset parameters.

    Attributes:
        path (str): Path to the input text file.
        block_size (int): Length of input sequences for model training.
        train_split (float, optional): Proportion of data for training (not implemented).
        truncate (float, optional): Fraction of data to use. Defaults to 1.0 (full dataset).
        encoding (str, optional): File encoding. Defaults to 'utf-8'.
    """

    path: Optional[str] = None
    block_size: Optional[int] = None
    train_split: Optional[float] = None
    truncate: float = 1.0
    encoding: str = 'utf-8'


class CharDataset(Dataset):
    """A PyTorch Dataset for character-level text processing.

    This dataset reads a text file, tokenizes it at the character level,
    and prepares sequences for language model training.

    Attributes:
        stoi (Dict[str, int]): Character to index mapping.
        itos (Dict[int, str]): Index to character mapping.
        block_size (int): Length of input sequences.
        vocab_size (int): Number of unique characters in the dataset.
        data (str): Processed text data.
    """

    def __init__(self, data_cfg: DataConfig) -> None:
        """Initialize the CharDataset.

        Args:
            data_cfg (DataConfig): Configuration parameters for the dataset.

        Raises:
            ValueError: If required configuration parameters are missing.
            IOError: If the file cannot be read.
        """
        # Validate input configuration
        if not data_cfg.path:
            raise ValueError('Data path must be provided')
        if not data_cfg.block_size:
            raise ValueError('Block size must be specified')

        try:
            # Read and process the text file
            with fsspec.open(data_cfg.path, 'r',
                             encoding=data_cfg.encoding) as file:
                data = file.read()
        except Exception as e:
            raise IOError(f'Error reading file {data_cfg.path}: {e}')

        # Truncate data if specified
        data = data[:int(len(data) * data_cfg.truncate)]

        # Create character vocabulary
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f'Data has {data_size} characters, {vocab_size} unique.')

        # Create character-index mappings
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}

        # Set dataset attributes
        self.block_size: int = data_cfg.block_size
        self.vocab_size: int = vocab_size
        self.data: str = data

    def __len__(self) -> int:
        """Calculate the number of possible sequence starting positions.

        Returns:
            int: Number of sequences in the dataset.
        """
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a training sequence and its corresponding target.

        Args:
            idx (int): Starting index of the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Input sequence of character indices
            - Target sequence of character indices (shifted by one)
        """
        # Extract a chunk of characters
        chunk = self.data[idx:idx + self.block_size + 1]

        # Encode characters to indices
        try:
            dix = [self.stoi[s] for s in chunk]
        except KeyError as e:
            raise ValueError(f'Unknown character encountered: {e}')

        # Create input and target tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y


# Example usage
def main() -> None:
    """Demonstrate dataset creation and usage."""
    config = DataConfig(
        path=
        '/home/robin/work_dir/hpc/pytorch-training-toolkit/mingpt/data/input.txt',
        block_size=64,
    )

    try:
        dataset = CharDataset(config)
        print(f'Dataset created with {len(dataset)} sequences')

        # Demonstrate accessing a sample
        sample_x, sample_y = dataset[0]
        print('Sample input sequence:', sample_x)
        print('Sample target sequence:', sample_y)
    except Exception as e:
        print(f'Error creating dataset: {e}')


if __name__ == '__main__':
    main()
