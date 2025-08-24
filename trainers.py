"""
trainer.py

This module implements abstract and concrete trainer classes for AdLoCo trainers.

Classes:
    AbstractTrainer: Base class defining the interface and shared logic for distributed trainers.
    LocalSGDTrainer: Concrete trainer implementing the LocalSGD algorithm.
    DiLoCoTrainer: Concrete trainer implementing the DiLoCo algorithm with outer optimization.

Dependencies:
    - PyTorch (torch, torch.nn, torch.optim)
    - Custom batching strategies (algorithms.calc_batch)
    - Evaluation fabric (fabrics.eval_tool)
    - Utility functions (utils.utils)
    - Logging utilities (training_logs.log_processor)
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim import AdamW, SGD
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

from algorithms.calc_batch import AbstractBatching, NormTestBatching, AugmentedInnerProductTestBatching, \
    ConstantBatching
from fabrics.eval_tool import EvalFabric

from utils.utils import set_seeds, data_split
from training_logs.log_processor import process_and_log_metrics
from torch.utils.data import Subset


class AbstractTrainer(ABC):
    """Abstract base class for distributed/federated trainers.

    Handles dataset sharding, node assignment, model initialization,
    batching strategies, and orchestration of training/evaluation rounds.

    Attributes:
        trainer_id (str): Identifier for this trainer instance.
        model_fabric(ModelFabric): Object responsible for creating model instances.
        model (nn.Module): The main model under training.
        eval_fabric (EvalFabric): Evaluation tool for model metrics.
        config (dict): Training configuration dictionary.
        logger (Logger): Logger for recording training/evaluation events.
        dataset (Dataset): Dataset to be distributed across nodes.
        training_logger (Logger): Logger for structured training metrics.
        nodes (list): List of assigned nodes.
        total_batches_processed (int): Running total of batches processed.
        current_batch_size (int): Current per-node batch size.
        batching_strategy (AbstractBatching): Strategy for dynamic batch sizing.
        shard_length (int): Size of data shards per node.
        sync_number (int): Synchronization counter.
    """
    def __init__(self, trainer_id, model_fabric, config, logger, dataset, training_logger):
        """Initialize a new AbstractTrainer.

        Args:
            trainer_id (str): Unique trainer identifier.
            model_fabric: Factory for model creation and manipulation.
            config (dict): Configuration parameters for training.
            logger: Logging utility.
            dataset (Dataset): Dataset to train on.
            training_logger: Structured metrics logger.
        """
        self.trainer_id = trainer_id
        self.model_fabric = model_fabric
        self.model = model_fabric.create_model()
        self.eval_fabric = EvalFabric(config, self.model)
        self.config = config
        self.logger = logger
        self.dataset = dataset
        self.training_logger = training_logger

        if self.config['need_to_reinit_output_layers']:
            self._reinitialize_output_layers(seed=self.config["training_seed"])

        self.num_inner_steps = config["num_inner_steps"]
        self.lr_inner = config["lr_inner"]
        self.distributed_algo = config["distributed_algo"]

        self.nodes = []
        self.total_batches_processed = 0
        self.current_batch_size = config["initial_batch_size"]
        self.batching_strategy: AbstractBatching = self._initialize_batching_strategy()

        self.sync_number = 0

        self.shard_length = int(len(dataset) // (self.config["nodes_per_gpu"] * len(self.config["devices"])))

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Returns the name of the distributed algorithm"""
        pass

    @abstractmethod
    def _create_inner_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Creates an optimizer for inner training on the nodes

        Args:
            model (nn.Module): Local model copy for a node.

        Returns:
            torch.optim.Optimizer: Optimizer for node training.
        """
        pass

    @abstractmethod
    def _outer_step(self, new_params_model: dict, main_model_state_dict: dict, num_successful_nodes: int):
        """Aggregates results from the nodes and updates the main model

        Args:
            new_params_model (dict): Aggregated parameter updates from nodes.
            main_model_state_dict (dict): Current state dict of the main model.
            num_successful_nodes (int): Number of nodes that completed training without errors.
        """
        pass

    def _reinitialize_output_layers(self, seed=None):
        """Reinitialize model output layers with Xavier uniform initialization.

        Args:
            seed (int, optional): Random seed for reproducibility.
        """
        output_layer_names = self.model_fabric.get_output_layer_names()
        if seed:
            set_seeds(seed)
        for layer_name in output_layer_names:
            layer = getattr(self.model, layer_name, None)
            if layer:
                if hasattr(layer, 'modules'):
                    for module in layer.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.xavier_uniform_(module.weight)
                            if module.bias is not None:
                                nn.init.constant_(module.bias, 0)
                elif isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def _get_primary_device(self) -> torch.device:
        """Determines the primary device for the main model (GPU if available, otherwise CPU).

        Returns:
            torch.device: CUDA device if available, otherwise CPU.
        """
        if torch.cuda.is_available():
            return torch.device(self.config["devices"][0])
        return torch.device("cpu")

    def _initialize_batching_strategy(self) -> AbstractBatching:
        """Initialize batching strategy based on config."""
        sampling_method = self.config.get("sampling_method", "norm_test")
        if sampling_method == "norm_test":
            return NormTestBatching(self.trainer_id, self.current_batch_size, self.config, self.logger)
        elif sampling_method == "augmented_inner_product_test":
            return AugmentedInnerProductTestBatching(self.trainer_id, self.current_batch_size, self.config, self.logger)
        else:
            return ConstantBatching(self.trainer_id, self.current_batch_size, self.config, self.logger)

    def get_batch_demand(self) -> int:
        """Get the total batch demand for current round."""
        return len(self.nodes) * self.current_batch_size if self.nodes else self.current_batch_size

    def assign_nodes(self, nodes):
        """Assign nodes for distributed training.

        Args:
            nodes (list): List of nodes participating in training.
        """
        self.nodes = nodes

    def _calculate_initial_metrics(self):
        """Calculate initial metrics (loss, evaluation score) before training starts."""
        self.logger.info("\n" + "-" * 40)
        self.logger.info(f"TRAINER: {self.trainer_id} started work with step zero to calculate initial metrics.")
        if not self.nodes or self.dataset is None:
            return

        # Fix the seed for reproducibility of the split at step zero
        set_seeds(self.config["training_seed"])
        num_nodes_assigned = len(self.nodes)
        shard_lengths = [self.shard_length] * num_nodes_assigned
        for i in range(len(self.dataset) % num_nodes_assigned):
            shard_lengths[i] += 1
        self.logger.info(f"Shard lengths: {shard_lengths}")
        shards = data_split(self.dataset, shard_lengths)
        for i, node in enumerate(self.nodes):
            node.assign_data(shards[0])
            max_shard_batch_size = len(node.data_shard_dataset)
            node.max_batch_size = min(max_shard_batch_size, node.max_batch_size)

        self.model.to("cpu")
        main_model_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        new_params_model = {k: torch.zeros_like(v) for k, v in main_model_state_dict.items()}

        # Compute initial loss and metrics on each node
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._run_node_training,
                    node, main_model_state_dict,
                    new_params_model, shuffle=False
                ): node
                for node in self.nodes
            }

            total_loss = 0
            num_nodes_with_data = 0

            for future in as_completed(futures):
                try:
                    _, _, avg_loss, _ = future.result()
                    total_loss += avg_loss
                    num_nodes_with_data += 1
                except Exception as exc:
                    self.logger.error(f"Error while calculating initial metrics on node: {exc}", exc_info=True)

        if num_nodes_with_data > 0:
            avg_loss = total_loss / num_nodes_with_data

            self.model.to(self.eval_fabric.device)
            metric = self.eval_fabric.run_evaluation()
            self.model.to("cpu")

            self.logger.info(
                f"Trainer {self.trainer_id}: Initial loss: {avg_loss:.4f}, Initial metric: {metric:.4f}")
            self.log_train(round_samples=0, avg_round_loss=avg_loss, metric=metric, outer_step=0)

        if self.config['single_seed'] and self.config['need_to_reinit_output_layers']:
            self._reinitialize_output_layers()

        process_and_log_metrics(self.config)

    def _run_node_training(self, node, main_model_state_dict, new_params_model, shuffle=True):
        """Run inner training on a single node.

        Args:
            node: Training node instance.
            main_model_state_dict (dict): State dict of main model.
            new_params_model (dict): Placeholder for parameter updates.
            shuffle (bool): Whether to shuffle data.

        Returns:
            tuple: (batches, samples, avg_loss, last_gradient)
        """
        local_model = self.model_fabric.create_model()
        local_model.load_state_dict(main_model_state_dict)

        local_optimizer = self._create_inner_optimizer(local_model)

        return node.train_inner_steps(
            local_model, local_optimizer, None,
            self.num_inner_steps, self.current_batch_size, new_params_model, shuffle=shuffle
        )

    def train_one_round(self, current_outer_step: int) -> int:
        """Run one round of distributed training.

        Args:
            current_outer_step (int): Current outer step index.

        Returns:
            int: Number of batches processed.
        """
        self.logger.info(f"TRAINER: {self.trainer_id} started working at step {current_outer_step}")
        if not self.nodes or self.dataset is None:
            return 0

        # --- Step 1: Data distribution (without fixed seed) ---
        num_nodes_assigned = len(self.nodes)
        shard_lengths = [self.shard_length] * num_nodes_assigned
        for i in range(len(self.dataset) % num_nodes_assigned):
            shard_lengths[i] += 1

        # Splitting without fixed seed
        shards = data_split(self.dataset, shard_lengths)

        for i, node in enumerate(self.nodes):
            node.assign_data(shards[i])
            max_shard_batch_size = len(node.data_shard_dataset)
            node.max_batch_size = min(max_shard_batch_size, node.max_batch_size)

        # --- Step 2: Preparation for training on nodes ---
        self.model.to("cpu")
        main_model_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        new_params_model = {k: torch.zeros_like(v) for k, v in main_model_state_dict.items()}

        round_batches, round_samples, round_loss, all_last_gradients = 0, 0, 0, []
        num_successful_nodes = 0

        # --- Step 3: Parallel training on nodes ---
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._run_node_training, node, main_model_state_dict, new_params_model): node
                for node in self.nodes
            }
            for future in as_completed(futures):
                try:
                    batches, samples, avg_loss, last_grad = future.result()
                    if batches > 0:
                        num_successful_nodes += 1
                        round_batches += batches
                        round_loss += avg_loss
                        round_samples += samples
                        if last_grad is not None:
                            all_last_gradients.append(last_grad)
                except Exception as exc:
                    self.logger.error(f"Node generated an exception: {exc}", exc_info=True)

        self.sync_number += 1
        if num_successful_nodes == 0:
            self.logger.warning(f"Trainer {self.trainer_id}: No node processed data.")
            return 0

        # --- Step 4: Aggregation, update, and evaluation ---
        primary_device = self._get_primary_device()
        self.model.to(primary_device)
        self._outer_step(new_params_model, main_model_state_dict, num_successful_nodes)

        self.model.to(self.eval_fabric.device)
        metric_after = self.eval_fabric.run_evaluation()

        del main_model_state_dict, new_params_model
        gc.collect()

        # --- Step 5: Update batch size and logging ---
        if all_last_gradients:
            self.current_batch_size = self.batching_strategy.update_batch_size(all_last_gradients)

        self.total_batches_processed += round_batches
        avg_round_loss = round_loss / num_successful_nodes

        self.log_train(round_samples, avg_round_loss, metric_after, outer_step=self.sync_number)

        self.logger.info(
            f"Trainer {self.trainer_id}: outer_step {current_outer_step}, avg loss: {avg_round_loss:.4f}, new demand: {self.current_batch_size}")

        return round_batches

    def log_train(self, round_samples, avg_round_loss, metric, outer_step):
        """Log training metrics for the current round."""
        log_data = {
            "sampling_method": self.config.get("sampling_method", "none"),
            "trainer_id": self.trainer_id,
            "round_batches": self.get_batch_demand(),
            "round_samples": round_samples,
            "train_loss": avg_round_loss,
            "current_batch_size": self.current_batch_size,
            "metric": metric,
            "outer_step_nubmer": outer_step,
        }
        if isinstance(self.batching_strategy, NormTestBatching):
            log_data.update({
                "variance": self.batching_strategy.variance,
                "T_k": self.batching_strategy.T_k,
                "sum_sq_diffs": self.batching_strategy.sum_sq_diffs,
            })
        elif isinstance(self.batching_strategy, AugmentedInnerProductTestBatching):
            log_data.update({
                "inner_variance": self.batching_strategy.inner_variance,
                "ortho_variance": self.batching_strategy.ortho_variance,
                "T_k": self.batching_strategy.T_k,
                "T_ortho": self.batching_strategy.T_ortho,
                "T_ip": self.batching_strategy.T_ip,
            })
        self.training_logger.log(log_data)

    def save_model_weights(self, filepath: str):
        """Save model weights and metadata checkpoint to file.

        Args:
            filepath (str): Path to save checkpoint.
        """

        # Moving model to CPU to save memory
        self.model.to('cpu')

        # Creating a checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'trainer_id': self.trainer_id,
            'current_batch_size': self.current_batch_size,
            'total_batches_processed': self.total_batches_processed,
            'num_inner_steps': self.num_inner_steps,
            'nodes_count': len(self.nodes),
            'model_config': self.model_fabric.config if hasattr(self.model_fabric, 'config') else None,
            'device': str(self.get_model_device()),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'batch_demand': self.get_batch_demand(),
            'lr_inner': self.lr_inner,
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Trainer {self.trainer_id}: Веса модели сохранены в {filepath}")


class LocalSGDTrainer(AbstractTrainer):
    """Trainer implementing the LocalSGD algorithm.

    Uses SGD as the optimizer for local updates and averages parameters across nodes.
    """
    @property
    def algorithm_name(self) -> str:
        return "LocalSGD"

    def _create_inner_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create SGD optimizer for node-local training."""
        return SGD(model.parameters(), lr=self.lr_inner)

    def _outer_step(self, new_params_model: dict, main_model_state_dict: dict, num_successful_nodes: int):
        """Aggregate parameters across nodes and update main model."""
        aggregated_model = {
            key: new_params_model[key] / num_successful_nodes
            for key in new_params_model.keys()
        }

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_model:
                    param.copy_(aggregated_model[name].to(param.device))

        del aggregated_model
        gc.collect()


class DiLoCoTrainer(AbstractTrainer):
    """Trainer implementing the DiLoCo algorithm.

    Uses AdamW for local updates and SGD for outer updates with gradient-based aggregation.
    """
    def __init__(self, trainer_id, model_fabric, config, logger, dataset, training_logger):
        super().__init__(trainer_id, model_fabric, config, logger, dataset, training_logger)
        self.lr_outer = config["lr_outer"]
        self.outer_optimizer = SGD(self.model.parameters(), lr=self.lr_outer)

    @property
    def algorithm_name(self) -> str:
        return "DiLoCo"

    def _create_inner_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create AdamW optimizer for node-local training."""
        return AdamW(model.parameters(), lr=self.lr_inner)

    def _outer_step(self, new_params_model: dict, main_model_state_dict: dict, num_successful_nodes: int):
        "Aggregate deltas and apply outer optimization step using SGD."""
        aggregated_deltas = {
            key: (new_params_model[key] / num_successful_nodes) - main_model_state_dict[key]
            for key in new_params_model.keys()
        }

        self.outer_optimizer.zero_grad()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_deltas:
                    param.grad = -aggregated_deltas[name].to(param.device)
        self.outer_optimizer.step()

        del aggregated_deltas
        gc.collect()
