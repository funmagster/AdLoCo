"""
Utility functions for ensuring reproducibility, handling training logs,
and splitting datasets into subsets.
"""

import torch
import numpy as np
import random
import os


def set_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility across libraries commonly used in machine learning.

    Args:
        seed (int, optional): Integer seed value. Defaults to 42.

    Returns:
        None

    Notes:
        - Sets seeds for `random`, `numpy`, and `torch`.
        - Ensures deterministic behavior in CUDA operations by disabling CuDNN auto-tuning.
    """
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"All seeds set to: {seed}")


def find_step_loss_file(training_csv_path):
    """
    Find the path to the step loss log file associated with a training log.

    Args:
        training_csv_path (str): Path to the main training log CSV file.

    Returns:
        str | None: Full path to `step_loss_log.csv` if it exists in the same directory
        as the training log. Returns `None` if the file is not found.

    Example:
        >>> find_step_loss_file("logs/training_log.csv")
        "logs/step_loss_log.csv"  # if exists
    """
    training_dir = os.path.dirname(training_csv_path)
    step_loss_path = os.path.join(training_dir, "step_loss_log.csv")
    if os.path.exists(step_loss_path):
        return step_loss_path
    return None

def data_split(dataset, splits):
    """
    Splits a dataset into multiple random subsets of specified sizes.

    Args:
        dataset (Sequence): The input dataset (e.g., list, tuple) to be split.
        splits (Iterable[int]): A sequence of integers specifying the sizes 
            of each subset to create. The sum of these values must not 
            exceed the length of the dataset.

    Returns:
        list[list]: A list of subsets, each containing the specified 
        number of randomly shuffled elements from the dataset.

    Raises:
        ValueError: If the sum of `splits` exceeds the length of `dataset`.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> data_split(data, [2, 3])
        [[3, 1], [4, 2, 5]]  # Order may vary due to shuffling
    """
    if sum(splits) > len(dataset):
        raise ValueError("Sum of split sizes exceeds dataset length")
    
    # Make a shuffled copy so we don't mutate the original dataset
    shuffled = list(dataset)[:]
    random.shuffle(shuffled)
    
    result = []
    start = 0
    for size in splits:
        result.append(shuffled[start:start+size])
        start += size
    return result
