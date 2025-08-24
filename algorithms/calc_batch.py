"""
Batching strategies for adaptive training.

This module defines an abstract interface (`AbstractBatching`) and multiple
concrete strategies for dynamically adjusting the batch size based on gradient
statistics during training.

Strategies:
    - NormTestBatching: Updates batch size using variance of gradients relative to mean gradient norm.
    - AugmentedInnerProductTestBatching: Uses variance in inner products and orthogonality discrepancies.
    - ConstantBatching: Keeps the batch size constant but still logs statistics.

These methods allow for adaptive control of variance in stochastic gradient
estimation, improving training efficiency.
"""

import torch
import math
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Dict


class AbstractBatching(ABC):
    """
    Abstract base class for batching strategies.

    Provides an interface for updating the batch size based on gradient statistics,
    and includes common checks/calculations.

    Args:
        trainer_id (int): Unique identifier of the trainer.
        current_batch_size (int): Current batch size.
        config (Dict[str, Any]): Configuration dictionary.
        logger (logging.Logger): Logger instance for reporting.
    """
    def __init__(self, trainer_id: int, current_batch_size: int, config: Dict[str, Any], logger: logging.Logger):
        self.trainer_id = trainer_id
        self.current_batch_size = current_batch_size
        self.config = config
        self.logger = logger

    @abstractmethod
    def update_batch_size(self, all_last_gradients: List[torch.Tensor]) -> int:
        """
        Update the batch size given the most recent gradients.

        Args:
            all_last_gradients (List[torch.Tensor]): List of last gradients from different trainers.

        Returns:
            int: The updated batch size (may be unchanged depending on strategy).
        """
        ...

    def _common_checks_and_calculations(self, all_last_gradients: List[torch.Tensor]):
        """
        Perform common checks and compute shared statistics for batching strategies.

        Args:
            all_last_gradients (List[torch.Tensor]): List of last gradients.

        Returns:
            Tuple:
                - stacked_batch_grads (torch.Tensor or None): Stacked gradients if valid, else None.
                - g_kH (torch.Tensor or None): Mean gradient.
                - g_norm_sq (float or None): Squared norm of the mean gradient.
                - M (int): Number of gradients.
        """
        M = len(all_last_gradients)
        if M <= 1:
            self.logger.warning(
                f"Trainer {self.trainer_id}: Not enough gradients ({M}) to update batch_size. Return current.")
            return None, None, None, M

        stacked_batch_grads = torch.stack(all_last_gradients)
        g_kH = stacked_batch_grads.mean(dim=0)
        g_norm_sq = torch.norm(g_kH) ** 2

        if g_norm_sq < 1e-12:
            self.logger.warning(
                f"Trainer {self.trainer_id}: g_norm_sq is too small ({g_norm_sq:.2e}), won't update batch_size."
            )
            return None, None, None, M

        return stacked_batch_grads, g_kH, g_norm_sq, M


class NormTestBatching(AbstractBatching):
    """
    Batch size adaptation using the norm test strategy.

    Computes gradient variance and adjusts batch size based on ratio
    of variance to squared norm.
    """
    def __init__(self, trainer_id: int, current_batch_size: int, config: dict, logger):
        super().__init__(trainer_id, current_batch_size, config, logger)
        self.eta = self.config["ETA"]
        self.current_batch_size = self.config["initial_batch_size"]
        self.variance = None
        self.T_k = None
        self.sum_sq_diffs = None

    def update_batch_size(self, all_last_gradients: List[torch.Tensor]) -> int:
        """
        Update batch size using the norm test criterion.

        Args:
            all_last_gradients (List[torch.Tensor]): List of recent gradients.

        Returns:
            int: The updated batch size.
        """
        stacked_batch_grads, g_kH, g_norm_sq, M = self._common_checks_and_calculations(all_last_gradients)
        if stacked_batch_grads is None:
            return self.current_batch_size

        sum_sq_diffs = sum(torch.norm(grad - g_kH) ** 2 for grad in all_last_gradients)
        variance = (self.current_batch_size / (M - 1)) * sum_sq_diffs

        # Проверка на деление на ноль, если g_norm_sq очень мал
        if g_norm_sq.item() == 0:
            self.logger.warning(
                f"Trainer {self.trainer_id}: g_norm_sq = 0, can't calculate T_k. Return current batch_size."
            )
            return self.current_batch_size

        T_k = math.ceil(variance / (M * self.eta ** 2 * g_norm_sq.item()))
        new_batch_size = max(T_k, self.current_batch_size)

        self.logger.info(f"Trainer {self.trainer_id}: Обновление batch_size (NormTest). "
                         f"Дисперсия: {variance:.4f}, T_k: {T_k}, "
                         f"Старый b: {self.current_batch_size}, Новый b: {new_batch_size}")

        self.current_batch_size = new_batch_size
        self.variance = variance.item()
        self.T_k = T_k
        self.sum_sq_diffs = sum_sq_diffs.item()

        return self.current_batch_size


class AugmentedInnerProductTestBatching(AbstractBatching):
    """
    Batch size adaptation using the augmented inner product test strategy.

    Splits variance contributions into inner product variance and orthogonality
    discrepancy variance, adapting batch size accordingly.
    """
    def __init__(self, trainer_id: int, current_batch_size: int, config: dict, logger):
        super().__init__(trainer_id, current_batch_size, config, logger)
        self.vartheta = self.config["vartheta"]
        self.nu = self.config["nu"]

        self.current_batch_size = self.config["initial_batch_size"]
        self.inner_variance = None
        self.ortho_variance = None
        self.T_k = None
        self.T_ortho = None
        self.T_ip = None

    def update_batch_size(self, all_last_gradients: List[torch.Tensor]) -> int:
        """
        Update batch size using inner product and orthogonality variance.

        Args:
            all_last_gradients (List[torch.Tensor]): List of recent gradients.

        Returns:
            int: The updated batch size.
        """
        stacked_batch_grads, g_kH, g_norm_sq, M = self._common_checks_and_calculations(all_last_gradients)
        if stacked_batch_grads is None:
            return self.current_batch_size

        if g_norm_sq.item() == 0:
            self.logger.warning(
                f"Trainer {self.trainer_id}: g_norm_sq = 0, can't calculate T_ip/T_ortho. Return current batch_size."
            )
            return self.current_batch_size

        inner_prods = torch.stack([torch.dot(single_batch_grad, g_kH) for single_batch_grad in stacked_batch_grads])
        inner_prods_variance = torch.var(inner_prods, unbiased=True).item()

        T_ip = inner_prods_variance / ((self.nu ** 2) * (g_norm_sq.item() ** 2))
        orthogonality_discrepancy = stacked_batch_grads - (inner_prods / g_norm_sq.item()).unsqueeze(
            1) * g_kH.unsqueeze(0)

        orthogonality_discrepancy_variance = torch.var(orthogonality_discrepancy, unbiased=True).item()

        T_ortho = orthogonality_discrepancy_variance / ((self.nu ** 2) * g_norm_sq.item())
        T = max(T_ip, T_ortho)

        new_batch_size = max(math.ceil(T), self.current_batch_size)

        self.logger.info(f"Trainer {self.trainer_id}: Updating batch_size (AugmentedInnerProductTest). "
                         f"Inner variance: {inner_prods_variance:.4f}, Ortho variance: {orthogonality_discrepancy_variance:.4f}, T_k: {T:.4f}, "
                         f"Старый b: {self.current_batch_size}, New b: {new_batch_size}")

        self.current_batch_size = new_batch_size
        self.inner_variance = inner_prods_variance
        self.ortho_variance = orthogonality_discrepancy_variance
        self.T_k = T
        self.T_ortho = T_ortho
        self.T_ip = T_ip

        return self.current_batch_size


class ConstantBatching(AbstractBatching):
    """
    Constant batch size strategy.

    Keeps batch size fixed but still computes statistics for logging.
    """
    def __init__(self, trainer_id: int, current_batch_size: int, config: dict, logger):
        super().__init__(trainer_id, current_batch_size, config, logger)
        self.eta = self.config['ETA']
        self.variance = None
        self.T_k = None
        self.sum_sq_diffs = None

    def update_batch_size(self, all_last_gradients: List[torch.Tensor]) -> int:
        """
        Return the current batch size (no update).

        Args:
            all_last_gradients (List[torch.Tensor]): List of recent gradients.

        Returns:
            int: The unchanged batch size.
        """
        stacked_batch_grads, g_kH, g_norm_sq, M = self._common_checks_and_calculations(all_last_gradients)
        if stacked_batch_grads is None:
            return self.current_batch_size

        sum_sq_diffs = sum(torch.norm(grad - g_kH) ** 2 for grad in all_last_gradients)
        variance = (self.current_batch_size / (M - 1)) * sum_sq_diffs

        T_k = 0
        if g_norm_sq.item() > 0:
            T_k = math.ceil(variance / (M * self.eta ** 2 * g_norm_sq.item()))

        self.logger.info(f"Trainer {self.trainer_id}: Without updating batch_size {self.current_batch_size} (Constante). "
                         f"Variance: {variance:.4f}, T_k: {T_k}, ")

        self.variance = variance.item()
        self.T_k = T_k
        self.sum_sq_diffs = sum_sq_diffs.item()
        return self.current_batch_size
