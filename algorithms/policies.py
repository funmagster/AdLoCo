"""
This module defines abstract and concrete policies for merging multiple trainers during
the training process. The policies specify how and when trainers should be merged
based on their batch demand. The merging process involves performing a weighted
average of model parameters and cleaning up unused trainers.

Classes:
    AbstractPolicy: Abstract base class for all merge policies.
    MergeBestWithWorstPolicy: Merges the trainer with the highest batch demand with
        the one with the lowest demand.
    MergeWorstWithWorstPolicy: Merges the two trainers with the lowest batch demand.
"""
from abc import ABC, abstractmethod
from typing import List
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cluster import Trainer

class AbstractPolicy(ABC):
    """
    Abstract base class for defining trainer merge policies.

    Args:
        merge_frequency (int): Frequency of merging (every N outer steps).
            Must be a positive integer.

    Raises:
        ValueError: If `merge_frequency` is not a positive integer.
    """
    def __init__(self, merge_frequency: int = 2):
        """
        Initialise policy
        :param merge_frequency: How often to request merge (every N steps)
        """
        if merge_frequency <= 0:
            raise ValueError("Frequency should be positive")
        self.merge_frequency = merge_frequency

    def should_merge(self, outer_step: int, trainers) -> bool:
        """
        Determines whether a merge should be performed at the current outer step.

        Args:
            outer_step (int): Current outer training step.
            trainers (List[Trainer]): List of active trainers.

        Returns:
            bool: True if merge should be performed, False otherwise.
        """
        if len(trainers) <= 1:
            return False
        return outer_step % self.merge_frequency == 0

    @abstractmethod
    def execute_merge(self, trainers, logger) -> None:
        """
        Executes the merge logic for trainers.

        This method should modify the `trainers` list in-place.

        Args:
            trainers (List[Trainer]): List of all active trainers.
            logger (logging.Logger): Logger for logging progress and results.
        """
        pass

    def _perform_weighted_average_and_cleanup(self, merging_trainer, victim_trainer, trainers, logger):
        """
        Performs weighted averaging of model parameters and cleans up the victim trainer.

        Args:
            merging_trainer (Trainer): The trainer that will absorb the victim trainer.
            victim_trainer (Trainer): The trainer that will be merged and removed.
            trainers (List[Trainer]): List of active trainers (modified in place).
            logger (logging.Logger): Logger for progress and debug information.

        Side Effects:
            - Updates weights of `merging_trainer` with a weighted average.
            - Removes `victim_trainer` from the `trainers` list.
            - Cleans up model and optimizer of `victim_trainer`.
        """
        logger.info(f"Merging Trainer {victim_trainer.trainer_id} (demand: {victim_trainer.get_batch_demand()}) "
                    f"into Trainer {merging_trainer.trainer_id} (demand: {merging_trainer.get_batch_demand()})")

        b_merge = merging_trainer.get_batch_demand()
        b_victim = victim_trainer.get_batch_demand()
        total_b = b_merge + b_victim

        # Defense not to defend by zero
        w_merge, w_victim = (1.0, 0.0) if total_b == 0 else (b_merge / total_b, b_victim / total_b)

        merging_trainer.model.cpu()
        victim_trainer.model.cpu()

        merging_model_dict = merging_trainer.model.state_dict()
        victim_model_dict = victim_trainer.model.state_dict()

        with torch.no_grad():
            for key in merging_model_dict.keys():
                merging_model_dict[key] = merging_model_dict[key].cpu() * w_merge + victim_model_dict[key].cpu() * w_victim

        merging_trainer.model.load_state_dict(merging_model_dict)
        logger.info(f"Model Trainer {merging_trainer.trainer_id} updated with averaged weights.")

        # Resources clean-up
        victim_trainer.model.cpu()
        if hasattr(victim_trainer, 'outer_optimizer') and victim_trainer.outer_optimizer:
            del victim_trainer.outer_optimizer
        del victim_trainer.model
        victim_trainer.nodes.clear()
        trainers.remove(victim_trainer)
        del victim_trainer

        # Cleaning up dictionaries for efficiency
        del merging_model_dict, victim_model_dict


class MergeBestWithWorstPolicy(AbstractPolicy):
    """
    Merge policy that merges the trainer with the highest batch demand ("best")
    with the trainer with the lowest batch demand ("worst").
    """
    def execute_merge(self, trainers: List['Trainer'], logger) -> None:
        """
        Executes the "best with worst" merge strategy.

        Args:
            trainers (List[Trainer]): List of active trainers (modified in place).
            logger (logging.Logger): Logger for progress and debug information.
        """
        logger.info("\n--- Merging by policy 'Best with worst by demand' ---")
        trainers.sort(key=lambda t: t.get_batch_demand(), reverse=True)

        merging_trainer = trainers[0]
        victim_trainer = trainers[-1]

        self._perform_weighted_average_and_cleanup(merging_trainer, victim_trainer, trainers, logger)

class MergeWorstWithWorstPolicy(AbstractPolicy):
    """
    Merge policy that merges the two trainers with the lowest batch demand.
    """
    def execute_merge(self, trainers: List['Trainer'], logger) -> None:
        """
        Executes the "worst with worst" merge strategy.

        Args:
            trainers (List[Trainer]): List of active trainers (modified in place).
            logger (logging.Logger): Logger for progress and debug information.
        """
        logger.info("\n--- Merging by policy 'Worst with worst by demand' ---")
        trainers.sort(key=lambda t: t.get_batch_demand())

        merging_trainer = trainers[1]
        victim_trainer = trainers[0]

        self._perform_weighted_average_and_cleanup(merging_trainer, victim_trainer, trainers, logger)
