"""
This module defines strategies for determining training parameters such as
the batch size for the DataLoader and the number of gradient accumulation steps
required to achieve a desired effective batch size in accordance to the SwitchMode policy. 

Strategies:
    - AbstractSwitchMode: Abstract base class for all switching strategies.
    - ThresholdSwitchMode: Switches between direct training and gradient accumulation
      based on a threshold relative to the maximum batch size.
    - OnlyGradientAccumulationStrategy: Always uses gradient accumulation regardless
      of effective batch size.
    - OnlyDataLoaderStrategy: Always uses direct DataLoader batch size without
      gradient accumulation.
"""
import math
from abc import ABC, abstractmethod


class AbstractSwitchMode(ABC):
    """
    Abstract base class for defining training strategies.

    A switch mode determines:
        - The actual batch size used in the DataLoader.
        - The number of gradient accumulation steps required to achieve the
          desired "effective" batch size.
    """

    @abstractmethod
    def decide_training_params(self, effective_batch_size: int, max_batch_size: int):
        """
        Determine training parameters based on the desired effective batch size.

        Args:
            effective_batch_size (int): The desired "effective" batch size.
            max_batch_size (int): The maximum batch size that fits into memory.

        Returns:
            tuple[int, int]: A tuple containing:
                - data_loader_batch_size (int): The actual batch size for the DataLoader.
                - grad_accumulation_steps (int): Number of gradient accumulation steps.
        """
        pass

    @property
    def name(self) -> str:
        """
        Returns the name of the strategy.

        Returns:
            str: The class name of the strategy.
        """
        return self.__class__.__name__


class ThresholdSwitchMode(AbstractSwitchMode):
    """
    Strategy that switches to gradient accumulation when the requested
    effective batch size exceeds `2 * max_batch_size`.

    - If `effective_batch_size > 2 * max_batch_size`:
        Use gradient accumulation.
    - Otherwise:
        Use direct training without accumulation.
    """

    def decide_training_params(self, effective_batch_size: int, max_batch_size: int):
        """
        Determine parameters according to the threshold-based strategy.

        Args:
            effective_batch_size (int): Desired effective batch size.
            max_batch_size (int): Maximum batch size that fits into memory.

        Returns:
            tuple[int, int]:
                - data_loader_batch_size (int): Batch size for the DataLoader.
                - grad_accumulation_steps (int): Gradient accumulation steps.
        """
        data_loader_batch_size = min(effective_batch_size, max_batch_size)
        if effective_batch_size > 2 * max_batch_size:
            # With grad. accum.
            grad_accumulation_steps = math.ceil(effective_batch_size / max_batch_size)
            return data_loader_batch_size, grad_accumulation_steps
        else:
            # Without grad. accum.
            grad_accumulation_steps = 1
            return data_loader_batch_size, grad_accumulation_steps


class OnlyGradientAccumulationStrategy(AbstractSwitchMode):
    """
    Strategy that always uses gradient accumulation.

    This ensures that the effective batch size is achieved entirely
    through gradient accumulation steps.
    """

    def decide_training_params(self, effective_batch_size: int, max_batch_size: int):
        """
        Always apply gradient accumulation to reach the desired effective batch size.

        Args:
            effective_batch_size (int): Desired effective batch size.
            max_batch_size (int): Maximum batch size that fits into memory.

        Returns:
            tuple[int, int]:
                - data_loader_batch_size (int): Batch size for the DataLoader.
                - grad_accumulation_steps (int): Gradient accumulation steps.
        """
        grad_accumulation_steps = math.ceil(effective_batch_size / max_batch_size)
        data_loader_batch_size = min(effective_batch_size, max_batch_size)
        return data_loader_batch_size, grad_accumulation_steps


class OnlyDataLoaderStrategy(AbstractSwitchMode):
    """
    Strategy that always uses the DataLoader batch size directly,
    without applying gradient accumulation.
    """

    def decide_training_params(self, effective_batch_size: int, max_batch_size: int):
        """
        Always use a single step without gradient accumulation.

        Args:
            effective_batch_size (int): Desired effective batch size.
            max_batch_size (int): Maximum batch size that fits into memory.

        Returns:
            tuple[int, int]:
                - data_loader_batch_size (int): Batch size for the DataLoader.
                - grad_accumulation_steps (int): Gradient accumulation steps (always 1).
        """
        grad_accumulation_steps = 1
        data_loader_batch_size = min(effective_batch_size, max_batch_size)
        return data_loader_batch_size, grad_accumulation_steps
