"""
main.py

This script serves as the entry point for distributed/federated model training.
It loads configurations, sets up logging and monitoring, initializes datasets,
models, and the cluster controller, then orchestrates training and evaluation.

Workflow:
    1. Load configuration and set random seeds.
    2. Prepare directories for logs, reports, and model weights.
    3. Configure logging and monitoring tools.
    4. Initialize model fabric, dataset loader, and cluster controller.
    5. Train models across multiple nodes/devices.
    6. Evaluate the final trained model(s) and log metrics.
"""


import os
import yaml
import warnings
import datetime
from tqdm import tqdm
import threading
import logging
import json
import torch.multiprocessing as mp

from cluster import ClusterController
from utils.utils import set_seeds
from fabrics.models import ModelFabric
from fabrics.dataset_loader import DatasetLoader
from fabrics.eval_tool import EvalFabric

import logging
from training_logs.log_processor import process_and_log_metrics
from training_logs.memory_monitor import *
from training_logs.training_monitoring import TrainingLogger
from training_logs.step_loss_logger import StepLossLogger


def run(config=None):
    """Main entry point for training and evaluation.

    Orchestrates the entire workflow: config loading, logging setup,
    monitoring initialization, model/dataset preparation, cluster
    training, and final evaluation.

    Args:
        config (dict, optional): Configuration dictionary. If None,
            the function attempts to load `config.yaml` from the current directory.

    Returns:
        None
    """
    if not hasattr(tqdm, '_lock'):
        tqdm._lock = threading.Lock()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 1. Load and prepare config, set seeds
    if config is None:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

    gpu_ids = config["CUDA_VISIBLE_DEVICES"].split(',')
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    devices = [f"cuda:{i}" for i in range(len(gpu_ids))]
    config["devices"] = devices
    config["lr_inner"] = float(config["lr_inner"])
    config["lr_outer"] = float(config["lr_outer"])
    set_seeds(config['training_seed'])

    # 2. Create folders for reports and logs
    current_date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_dir_path = os.path.join(os.getcwd(), config["log_dir"])
    os.makedirs(log_dir_path, exist_ok=True)

    current_log_dir = os.path.join(log_dir_path, f"{config['model_type']}_{config['dataset']}_{current_date}")
    os.makedirs(current_log_dir)
    config['log_dir_path'] = current_log_dir

    other_dirs = [
        ("memory_report_dir_full_path", config["memory_report_dir"]),
        ("model_training_report_dir_full_path", config["model_training_report_dir"]),
        ("model_weights_dir_full_path", config["model_weights_dir"])
    ]
    for report_dir_name, report_dir_path_part in other_dirs:
        report_dir_full_path = os.path.join(current_log_dir, report_dir_path_part)
        os.makedirs(report_dir_full_path, exist_ok=True)
        config[report_dir_name] = report_dir_full_path

    logger_filepath = os.path.join(current_log_dir, config["log_file"])
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logger_filepath),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    memory_monitor = MemoryMonitor(devices, config["memory_report_dir_full_path"], logger=logger)
    step_loss_logger = StepLossLogger(config)
    training_logger = TrainingLogger(config)
    training_logger.set_logger(logger)

    # 3. Initialize key components
    model_fabric = ModelFabric(config)
    dataset_loader = DatasetLoader(config)
    full_dataset = dataset_loader.load_training_dataset()

    # Logging configs for reproducibility
    config_str = json.dumps(config, indent=2, default=lambda o: str(o))
    logger.info(config_str)

    cluster = ClusterController(
        config=config, logger=logger, dataset=full_dataset,
        model_fabric=model_fabric,
        memory_monitor=memory_monitor, training_logger=training_logger, step_loss_logger=step_loss_logger
    )

    # 4. Start model training
    final_models = cluster.train(
        num_outer_steps=config['num_outer_steps'],
        save_weights_every=config["save_weights_every"]
    )
    print(f"\nTraining complete. Obtained {len(final_models)} final models.")

    # 5. Model evaluation
    print("Running evaluation metric")
    eval_fabric = EvalFabric(config, final_models[0])
    eval_fabric.run_evaluation()

    # ClearML reports are kept for safety purposes, as a second measure
    process_and_log_metrics(config)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    mp.set_start_method('spawn', force=True)
    run()
