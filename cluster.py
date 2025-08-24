"""
This module provides classes and logic for distributed training using multiple
nodes (TrainerNode) managed by a central controller (ClusterController).
It supports adaptive batch sizing, node-to-trainer assignment, distributed
algorithms (LocalSGD, DiLoCo), and memory/logging utilities.

Core Components:
----------------
- TrainerNode: A training worker tied to a device (e.g., GPU). Handles local
  training, gradient accumulation, and logging.
- ClusterController: Orchestrates all nodes and trainers, manages distributed
  algorithms, performs merging when required, and coordinates training rounds.

Features:
---------
- Adaptive batch size adjustment (via switch mode strategies).
- Support for LocalSGD and DiLoCo distributed algorithms.
- Flexible node allocation policies (best-with-worst, worst-with-worst).
- Logging for losses, metrics, and GPU memory usage.
- Checkpointing and loading of model weights/metadata.
"""


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import math
import gc
import json
import os
import datetime

from training_logs.log_processor import process_and_log_metrics
from training_logs.memory_monitor import *
from utils.max_batch_calculator import find_max_batch_size_binsearch
from algorithms.policies import *
from trainers import AbstractTrainer, LocalSGDTrainer, DiLoCoTrainer
from algorithms.switch_mode import *
from algorithms.switch_mode import AbstractSwitchMode


class TrainerNode:
    """
    Represents a training node bound to a specific device.

    A node trains on its local data shard, applying gradient accumulation,
    optimizer updates, and step-level logging. It adapts effective batch size
    and gradient accumulation steps based on the chosen switch mode.

    Attributes:
        node_id (int): Unique ID of the node.
        device (torch.device): Device (CPU/GPU) where training runs.
        max_batch_size (int): Maximum batch size this node can handle.
        logger: Logger for status messages.
        training_logger: Logger for training-level events.
        step_loss_logger: Logger for per-step loss values.
        switch_mode (AbstractSwitchMode): Strategy for deciding batch size and accumulation.
        data_shard_dataset (TensorDataset): Dataset assigned to this node.
        last_avg_loss (float): Last average training loss recorded.
        grad_accumulation_step (int): Number of gradient accumulation steps.
        current_outer_step (int): Current outer step index.
        current_trainer_id (int): ID of the trainer this node is assigned to.
    """
    
    def __init__(self, node_id: int, device: torch.device, max_batch_size: int, logger, training_logger,
                 step_loss_logger, switch_mode: AbstractSwitchMode) -> None:
        """
        Initialize a training node.

        Args:
            node_id (int): Unique identifier for the node.
            device (torch.device): Device assigned to this node.
            max_batch_size (int): Maximum batch size the node can handle.
            logger: Logger for node-level messages.
            training_logger: Logger for training-level messages.
            step_loss_logger: Logger for step-level losses.
            switch_mode (AbstractSwitchMode): Strategy to determine training parameters.
        """
        
        self.node_id = node_id
        self.device = device
        self.data_shard_dataset = None
        self.last_avg_loss = float('inf')
        self.max_batch_size = max_batch_size
        self.logger = logger
        self.training_logger = training_logger
        self.step_loss_logger = step_loss_logger
        self.current_outer_step = None
        self.current_trainer_id = None

        self.grad_accumulation_step = 1  # Default value
        self.switch_mode = switch_mode

    def assign_data(self, data_shard_dataset: TensorDataset):
        """
        Assign a dataset shard to this node.

        Args:
            data_shard_dataset (TensorDataset): The dataset shard.
        """
        self.data_shard_dataset = data_shard_dataset

    def set_training_context(self, outer_step: int, trainer_id: int):
        """
        Set the current training context for the node.

        Args:
            outer_step (int): Current outer step index.
            trainer_id (int): ID of the trainer this node is assigned to.
        """
        self.current_outer_step = outer_step
        self.current_trainer_id = trainer_id

    def train_inner_steps(self, model: nn.Module, inner_optimizer: torch.optim.Optimizer,
                          inner_scheduler, num_inner_steps: int, batch_size: int,
                          new_params_model: dict, shuffle: bool):
        """
        Perform inner training steps on the assigned dataset shard.

        Args:
            model (nn.Module): Model to train.
            inner_optimizer (torch.optim.Optimizer): Optimizer for updates.
            inner_scheduler: Learning rate scheduler (optional).
            num_inner_steps (int): Number of inner steps to perform.
            batch_size (int): Effective batch size requested.
            new_params_model (dict): Dictionary to accumulate updated parameters.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            tuple:
                - batches_processed (int): Number of optimizer steps performed.
                - samples_processed (int): Number of samples processed.
                - avg_loss (float): Average training loss.
                - last_gradient_vector (torch.Tensor or None): Flattened gradient vector from last step.
        """
        torch.cuda.empty_cache()

        # Decide actual loader batch size and accumulation steps
        data_loader_batch_size, self.grad_accumulation_step = self.switch_mode.decide_training_params(
            effective_batch_size=batch_size,
            max_batch_size=self.max_batch_size
        )

        self.logger.info(
            f"Node {self.node_id}: Effective batch={batch_size}, "
            f"Loader batch={data_loader_batch_size}, Accum steps={self.grad_accumulation_step}, Data shard length: {len(self.data_shard_dataset)}"
        )

        # Check if there is enough data for a single batch in the dataloader
        if self.data_shard_dataset is None or len(self.data_shard_dataset) < data_loader_batch_size:
            self.logger.warning(
                f"Data loader batch size ({data_loader_batch_size}) is bigger than data shard "
                f"({len(self.data_shard_dataset)}). Skipping training."
            )
            return 0, 0, 0.0, None

        model.to(self.device)
        model.train()

        data_loader = DataLoader(self.data_shard_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

        total_loss = 0
        batches_processed = 0
        samples_proccesed = 0
        last_gradient_vector = None

        if not data_loader:
            self.logger.warning(f"Dataloader is None")
            return 0, 0, 0.0, None

        # Cleaning up the gradients
        inner_optimizer.zero_grad(set_to_none=True)

        # Extracting learning rate
        current_lr = inner_optimizer.param_groups[0]['lr'] if inner_optimizer.param_groups else None

        num_batches_in_loader = len(data_loader)
        for i, batch in enumerate(data_loader):
            if i >= num_inner_steps:
                break

            input_ids, attention_mask, labels = None, None, None

            # --- Forward pass ---
            if hasattr(model, 'classifier'):
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            elif hasattr(model, 'lm_head'):
                input_ids, attention_mask = [b.to(self.device) for b in batch]
                shift_input_ids = input_ids[..., :-1].contiguous()
                shift_attention_mask = attention_mask[..., :-1].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                outputs = model(input_ids=shift_input_ids, attention_mask=shift_attention_mask, labels=shift_labels)
                loss = outputs.loss
                del shift_labels, shift_attention_mask, shift_input_ids
            else:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            # Saving unnormalized loss for logging purposes
            unnormalized_loss = loss.item()

            # Logging loss
            if (self.step_loss_logger is not None and
                    self.current_outer_step is not None and
                    self.current_trainer_id is not None):
                self.step_loss_logger.log_step_loss(
                    outer_step=self.current_outer_step,
                    trainer_id=self.current_trainer_id,
                    node_id=self.node_id,
                    inner_step=i,
                    batch_idx=i,
                    loss=unnormalized_loss,
                    batch_size=len(input_ids),
                    learning_rate=current_lr,
                    epoch_within_round=0
                )

            # Normalizing loss
            if self.grad_accumulation_step > 1:
                loss = loss / self.grad_accumulation_step

            loss.backward()

            # The weight update block is executed when enough gradients have been accumulated
            # The condition (i + 1) % ... correctly accounts for 0-based indexing
            # Also check whether this is the final step within the outer_step
            # or within the specified num_inner_steps
            is_last_step = (i == num_inner_steps - 1) or (i == num_batches_in_loader - 1)
            if (i + 1) % self.grad_accumulation_step == 0 or is_last_step:

                # Gradient norm calculation before optimization step is done
                # on every step, not only in the end
                last_gradient_vector = None
                with torch.no_grad():
                    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
                    if grads:
                        last_gradient_vector = torch.cat(grads).cpu()

                inner_optimizer.step()
                if inner_scheduler:
                    inner_scheduler.step()

                # Cleaning up gradients after step
                inner_optimizer.zero_grad(set_to_none=True)

                total_loss += unnormalized_loss
                batches_processed += 1

            samples_proccesed += batch_size
            del input_ids, attention_mask, labels, outputs, loss

            if i % 3 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()

        if batches_processed == 0:
            self.logger.warning(f"Processed batches is 0")
            return 0, 0, 0.0, None

        # Average loss by the number of successfull steps 
        avg_loss = total_loss / batches_processed
        self.last_avg_loss = avg_loss

        with torch.no_grad():
            for name, param in model.named_parameters():
                new_params_model[name] += param.cpu()

        return batches_processed, samples_proccesed, avg_loss, last_gradient_vector


class ClusterController:
    """
    Manages a cluster of TrainerNodes and orchestrates distributed training.

    Functionality:
    - Initialize nodes across devices.
    - Create trainers based on distributed algorithm (LocalSGD, DiLoCo).
    - Distribute nodes to trainers based on demand.
    - Coordinate outer training steps across trainers.
    - Perform merging (DiLoCo) based on policy.
    - Manage GPU memory and logging.
    - Save/load model weights and metadata.

    Attributes:
        config (dict): Training configuration dictionary.
        devices (list[str]): List of devices for training.
        logger: Logger for cluster messages.
        memory_monitor: Memory monitor utility.
        training_logger: Logger for training metrics.
        step_loss_logger: Logger for per-step losses.
        nodes (list[TrainerNode]): All nodes in the cluster.
        trainers (list[AbstractTrainer]): Active trainers in the cluster.
        merging_policy (AbstractPolicy): Policy for merging trainers.
        switch_mode (AbstractSwitchMode): Strategy for adjusting batch sizing.
    """
    def __init__(
            self, config, logger, dataset,
            model_fabric,
            memory_monitor, training_logger, step_loss_logger
    ):
        """
        Initialize the cluster controller and create training nodes.

        Args:
            config (dict): Training configuration dictionary.
            logger: Logger for status and debug messages.
            dataset: Training dataset (preprocessed and sharded if needed).
            model_fabric: Model wrapper for creating trainer models.
            memory_monitor: Memory monitoring utility for GPU devices.
            training_logger: Logger for global training/outer-step metrics.
            step_loss_logger: Logger for step-level losses.
        """
        self.config = config
        self.devices = config["devices"]

        self.logger = logger
        self.memory_monitor = memory_monitor
        self.step_loss_logger = step_loss_logger
        self.training_logger = training_logger

        nodes_per_gpu = config["nodes_per_gpu"]
        self.merging_policy = self._initialize_merging_policy()
        self.switch_mode = self._initialize_switch_mode_strategy()
        self.nodes: list[TrainerNode] = []

        node_id_counter = 0
        for device in self.devices:
            torch_device = torch.device(device)
            all_device_memory = torch.cuda.get_device_properties(torch_device).total_memory
            print("-" * 10 + f"Device {device} ({all_device_memory / 1e9:.2f} GB)" + "-" * 10)
            max_bs = find_max_batch_size_binsearch(
                model_fabric=model_fabric,
                device=torch_device,
                dataset=dataset,
                padding_percentage=self.config["padding_percentage"],
                logger=self.logger,
                nodes_per_gpu=nodes_per_gpu
            )
            for _ in range(nodes_per_gpu):
                self.nodes.append(
                    TrainerNode(
                        node_id_counter, torch_device, max_bs,
                        logger, self.training_logger, self.step_loss_logger,
                        switch_mode=OnlyGradientAccumulationStrategy()
                    )
                )
                node_id_counter += 1
            print("-" * 40)
            print()

        self.trainers: list[AbstractTrainer] = []
        algo_name = config.get("distributed_algo")
        if algo_name == "LocalSGD":
            TrainerClass = LocalSGDTrainer
        elif algo_name == "DiLoCo":
            TrainerClass = DiLoCoTrainer
        else:
            raise ValueError(f"Unknown distributed algo: {algo_name}")

        for i in range(config["num_init_trainers"]):
            self.trainers.append(
                TrainerClass(
                    trainer_id=i, model_fabric=model_fabric,
                    config=config, logger=logger, dataset=dataset,
                    training_logger=self.training_logger
                )
            )

        self.logger.info(
            f"\nClusterController created. Total nodes: {len(self.nodes)}. Trainers at the start: {len(self.trainers)}"
        )

    def _initialize_merging_policy(self) -> AbstractPolicy:
        """
        Initialize and return the model merging policy used during distributed training.

        The merging policy defines how model weights are combined across trainers
        after synchronization steps. The specific policy is selected from the
        configuration.
    
        Supported policies:
            - "best_with_worst": Pairs the best-performing model(s) with the
              worst-performing ones to balance performance.
            - "worst_with_worst": Merges models with the lowest performance
              metrics to stabilize training.
    
        Configuration keys used:
            - merging_policy (str): Name of the merging strategy.
                Defaults to "worst_with_worst".
            - merge_frequency (int): Number of outer steps between merge
                operations. Defaults to 2.
    
        Returns:
            AbstractPolicy: An initialized merging policy object implementing
            the chosen strategy.
    
        Raises:
            ValueError: If the configuration specifies an unknown merging policy.
        """
        
        policy_name = self.config.get("merging_policy", "worst_with_worst")
        merge_frequency = self.config.get("merge_frequency", 2)

        if policy_name == "best_with_worst":
            return MergeBestWithWorstPolicy(merge_frequency=merge_frequency)
        elif policy_name == "worst_with_worst":
            return MergeWorstWithWorstPolicy(merge_frequency=merge_frequency)
        else:
            raise ValueError(f"Неизвестная политика объединения: {policy_name}")

    def _initialize_switch_mode_strategy(self) -> AbstractSwitchMode:
        """
        Initialize and return the training mode switching strategy.
    
        The switch mode strategy controls how training alternates between
        gradient accumulation and data loading approaches. The chosen strategy
        is determined from the configuration.
    
        Supported strategies:
            - "threshold_switch_mode": Switches based on performance thresholds.
            - "only_gradient_accumulation": Uses only gradient accumulation mode.
            - "only_data_loader": Uses only data loader mode.
    
        Configuration keys used:
            - switch_mode (str): Name of the strategy. Defaults to
              "threshold_switch_mode".
    
        Returns:
            AbstractSwitchMode: An initialized switch mode strategy instance.
    
        Raises:
            ValueError: If the configuration specifies an unknown switch mode.
        """
        switch_mode_name = self.config.get("switch_mode", "threshold_switch_mode")
        if switch_mode_name == "threshold_switch_mode":
            return ThresholdSwitchMode()
        elif switch_mode_name == "only_gradient_accumulation":
            return OnlyGradientAccumulationStrategy()
        elif switch_mode_name == "only_data_loader":
            return OnlyDataLoaderStrategy()
        else:
            raise ValueError(f"Неизвестный режим обучения: {switch_mode_name}")

    def _distribute_nodes(self):
        """
        Distribute available compute nodes among trainers.
    
        If no batch demand is reported, nodes are divided evenly across trainers.
        Otherwise, nodes are allocated proportionally to each trainer's reported
        batch demand, ensuring fair distribution. Adjustments are made to correct
        rounding differences.
    
        Logging:
            - Reports total batch demand across trainers.
            - Logs the number of nodes allocated to each trainer.
    
        Returns:
            None
        """
        self.logger.info("\n--- Nodes redistribution ---")
        if not self.trainers:
            return

        total_demand = sum(t.get_batch_demand() for t in self.trainers)
        self.logger.info(f"Total batch demand: {total_demand}")

        if total_demand == 0:
            num_nodes_per_trainer = len(self.nodes) // len(self.trainers) if self.trainers else 0
            node_idx = 0
            for i, trainer in enumerate(self.trainers):
                end_idx = node_idx + num_nodes_per_trainer
                assigned_nodes = self.nodes[node_idx:] if i == len(self.trainers) - 1 else self.nodes[node_idx:end_idx]
                trainer.assign_nodes(assigned_nodes)
                node_idx = end_idx
        else:
            node_assignments = {}
            nodes_to_assign = len(self.nodes)

            sorted_trainers = sorted(self.trainers, key=lambda t: t.get_batch_demand(), reverse=True)

            for trainer in sorted_trainers:
                proportion = trainer.get_batch_demand() / total_demand
                num_nodes = max(1, round(proportion * nodes_to_assign))
                node_assignments[trainer.trainer_id] = int(num_nodes)

            allocated_sum = sum(node_assignments.values())
            delta = nodes_to_assign - allocated_sum
            for i in range(abs(delta)):
                trainer_to_adjust = sorted_trainers[i % len(sorted_trainers)]
                node_assignments[trainer_to_adjust.trainer_id] += np.sign(delta)

            temp_node_pool = list(self.nodes)
            for trainer in sorted_trainers[::-1]:
                num = min(node_assignments.get(trainer.trainer_id, 0), len(temp_node_pool))
                assigned_nodes = [temp_node_pool.pop(0) for _ in range(int(num))]
                trainer.assign_nodes(assigned_nodes)

        for trainer in self.trainers:
            self.logger.info(
                f"Trainer {trainer.trainer_id} got {len(trainer.nodes)} nodes: {[n.node_id for n in trainer.nodes]}"
            )

    def _switch_mode(self):
        """
        Apply the current switch mode strategy to all nodes.
    
        Ensures that each node in the cluster follows the same
        training mode as defined by the switch mode strategy.
    
        Returns:
            None
        """
        for node in self.nodes:
            node.switch_mode = self.switch_mode

    def train(self, num_outer_steps: int, save_weights_every: int):
        """
        Run the main distributed training loop.
    
        The training process consists of multiple outer steps. During each step:
            - Nodes are distributed across trainers.
            - Trainers execute training rounds in parallel.
            - Metrics and memory usage are logged.
            - Merging policies are applied (if configured, e.g., for DiLoCo).
            - Model weights are periodically saved.
    
        Args:
            num_outer_steps (int): Total number of outer steps to run.
            save_weights_every (int): Frequency (in outer steps) at which
                model weights are saved.
    
        Returns:
            List[torch.nn.Module]: List of trained models from all trainers.
    
        Raises:
            Exception: Propagates trainer execution errors with logging.
        """
        self._distribute_nodes()

        if len(self.trainers) == 1:
            self._switch_mode()

        for trainer in self.trainers:
            for node in trainer.nodes:
                node.set_training_context(0, trainer.trainer_id)
            trainer._calculate_initial_metrics()

        self.training_logger.flush()
        # Don't reset step_loss_logger buffer after every step
        process_and_log_metrics(
                self.config,
        )

        memory_checkpoint(self.memory_monitor, "начало_обучения", print_summary=True)
        for outer_step in range(1, num_outer_steps + 1):
            self._empty_cache()
            self.logger.info(f"\n{'=' * 20} OUTER_STEP {outer_step} {'=' * 20}")

            reset_and_track_peak_memory(self.memory_monitor)
            memory_checkpoint(self.memory_monitor, "до_outer_step", outer_step, print_summary=True,
                              extra_info=f"Начало outer_step {outer_step}")

            self._distribute_nodes()

            # Set training context for all nodes
            for trainer in self.trainers:
                for node in trainer.nodes:
                    node.set_training_context(outer_step, trainer.trainer_id)

            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(trainer.train_one_round, outer_step): ind
                    for ind, trainer in enumerate(self.trainers)
                }

                for future in as_completed(futures):
                    ind = futures[future]
                    try:
                        _ = future.result()
                        self.logger.info(f"TRAINER: {ind} finished successfuly")
                    except Exception as exc:
                        self.logger.error(f"TRAINER {ind} encountered an exception: {exc}", exc_info=True)

            self.training_logger.flush()
            # Reset step_loss_logger buffer after outer_step
            self.step_loss_logger.flush()

            memory_checkpoint(self.memory_monitor, "пик_outer_step", outer_step, print_summary=False)

            # --- Check for merge ---
            # Check if trainers exist and we use an algorithm, that requirest merging
            if self.trainers and self.trainers[0].algorithm_name == "DiLoCo":
                if self.merging_policy.should_merge(outer_step, self.trainers):
                    self.logger.info(f"\n!!! Step {outer_step}: STARTING TRAINER MERGER ACCORDING TO POLICY !!!")
                    self.merging_policy.execute_merge(self.trainers, self.logger)
                    self._empty_cache()
                    self._distribute_nodes()
                    self.logger.info(f"Merging finished successfully. Number of active trainers: {len(self.trainers)}")

                    if len(self.trainers) == 1:
                        self._switch_mode()
                        self.logger.info(f"Only one trainer left. Switching mode into {self.switch_mode.name}")

            if outer_step % save_weights_every == 0 and outer_step != num_outer_steps:
                self.save_all_models(outer_step)

            memory_checkpoint(self.memory_monitor, f"после_outer_step_{outer_step}", outer_step,
                              print_summary=True,
                              extra_info=f"Конец outer_step {outer_step}")

            self._empty_cache()

            process_and_log_metrics(
                self.config,
            )
            if outer_step % 10 == 0:
                self.training_logger.flush()
                self.training_logger.plot_training_logs()
                finalize_memory_analysis(self.memory_monitor, save_detailed=True)

        self.save_all_models("final")
        memory_checkpoint(self.memory_monitor, "конец_обучения", print_summary=True)
        finalize_memory_analysis(self.memory_monitor, save_detailed=True)
        self.training_logger.flush()

        # Final loss logging and plot processing
        self.step_loss_logger.flush()
        self.step_loss_logger.plot_loss_trends()

        # Print loss over step statistics
        loss_stats = self.step_loss_logger.get_loss_statistics()
        self.logger.info(f"\n=== Loss over steps statistics ===")
        for key, value in loss_stats.items():
            self.logger.info(f"{key}: {value}")

        self.training_logger.plot_training_logs()

        self.logger.info("\nTraining finished!")
        return [trainer.model for trainer in self.trainers]

    def save_all_models(self, outer_step_or_label):
        """
        Save model weights for all trainers and record metadata.
    
        Each trainer's model weights are saved with a unique filename containing:
            - Trainer ID
            - Outer step or label
            - Timestamp
    
        A metadata JSON file is also generated to summarize the saved state.
    
        Args:
            outer_step_or_label (Union[int, str]): Current outer step number or
                a descriptive label (e.g., "final").
    
        Returns:
            None
    
        Raises:
            Exception: If saving weights or metadata fails, errors are logged
            but execution continues.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"\n--- Saving model weights (outer_step {outer_step_or_label}) ---")

        for trainer in self.trainers:
            # Creating a file with a timestamp and trainer information

            filename = f"trainer_{trainer.trainer_id}_outer_step_{outer_step_or_label}_{timestamp}.pth"

            filepath = os.path.join(self.config["model_weights_dir_full_path"], filename)

            try:
                trainer.save_model_weights(filepath)
            except Exception as e:
                self.logger.info(f"Error while saving weights {trainer.trainer_id}: {e}")

        metadata = {
            'outer_step': outer_step_or_label,
            'timestamp': timestamp,
            'num_trainers': len(self.trainers),
            'trainer_ids': [t.trainer_id for t in self.trainers],
            'total_nodes': len(self.nodes),
            'devices': [str(d) for d in self.devices],
        }

        metadata_filename = f"training_metadata_outer_step_{outer_step_or_label}_{timestamp}.json"
        metadata_filepath = os.path.join(self.config["model_weights_dir_full_path"], metadata_filename)

        try:
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Metadata saved into {metadata_filepath}")
        except Exception as e:
            self.logger.info(f"Encountered an error while trying to save metadata: {e}")

    def load_models_from_weights(self, outer_step_or_label, timestamp: str = None):
        """
        Load model weights for trainers from saved files.
    
        This function locates and loads weight files corresponding to the given
        outer step (and optionally a specific timestamp). Trainers are matched
        to weight files by their IDs.
    
        Args:
            outer_step_or_label (Union[int, str]): Outer step or label used to
                identify the saved weights.
            timestamp (str, optional): If provided, restricts loading to files
                matching this timestamp.
    
        Returns:
            None
    
        Logs:
            - Reports trainers whose weights were successfully loaded.
            - Warns if no weights are found or trainer IDs cannot be matched.
        """
        self.logger.info(f"\n--- Loading model weights (outer_step {outer_step_or_label}) ---")

        # Looking for model weights
        pattern = f"trainer_*_outer_step_{outer_step_or_label}_"
        weight_files = [f for f in os.listdir(self.config["model_weights_dir_full_path"]) if
                        f.startswith(pattern.replace('*', '')) and f.endswith('.pth')]

        if timestamp:
            weight_files = [f for f in weight_files if timestamp in f]

        if not weight_files:
            self.logger.info(f"Не найдены файлы весов для outer_step {outer_step_or_label}")
            return

        # Sort by creation date
        weight_files.sort()

        loaded_trainers = []
        for weight_file in weight_files:
            # Extract trainer_id from file name
            parts = weight_file.split('_')
            if len(parts) >= 2:
                try:
                    trainer_id = int(parts[1])

                    # Finding the corresponding trainer
                    trainer = next((t for t in self.trainers if t.trainer_id == trainer_id), None)
                    if trainer:
                        filepath = os.path.join(self.config["model_weights_dir_full_path"], weight_file)
                        trainer.load_model_weights(filepath)
                        loaded_trainers.append(trainer_id)
                    else:
                        self.logger.info(f"Unable to find trainer with ID {trainer_id}")

                except ValueError:
                    self.logger.info(f"Couldn't extract trainer_id from file {weight_file}")

        self.logger.info(f"Loaded weights for trainers: {loaded_trainers}")

    def _empty_cache(self):
        """
        Clear Python and GPU memory caches.
    
        Runs garbage collection and releases cached GPU memory on all devices
        to prevent memory buildup between outer steps.
    
        Returns:
            None
        """
        gc.collect()
        for device in self.devices:
            if torch.cuda.is_available():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
