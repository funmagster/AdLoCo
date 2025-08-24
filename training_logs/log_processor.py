"""
This module handles the processing and logging of training metrics to ClearML.

Functions:
    - process_and_log_metrics(config: dict) -> Tuple[float, int]
"""
import pandas as pd
from clearml import Task
import os
from typing import Tuple

STEP_COLUMN_NAME = 'count_outer_step'


def process_and_log_metrics(config: dict) -> Tuple[float, int]:
    """
    Processes training metrics from a CSV file and logs them to ClearML.

    Args:
        config (dict): Configuration dictionary containing at least:
            - model_training_report_dir_full_path (str): Path to directory containing 'training_log.csv'.
            - project_name (str): ClearML project name.
            - task_name (str): ClearML task name.

    Returns:
        Tuple[float, int]: 
            - final_aggregated_metric (float or None): Aggregated value of the 'metric' column after processing all steps.
            - total_communication_rounds (int): Number of unique outer training steps found in the CSV.

    Behavior:
        - Reads 'training_log.csv' from the provided directory.
        - Initializes a ClearML task for logging.
        - Logs trainer-specific and global metrics for each outer step.
        - Aggregates metrics according to predefined rules:
            - 'count_outer_step', 'round_batches', 'round_samples', 'current_batch_size', 'sum_sq_diffs' are summed.
            - 'train_loss', 'variance', 'T_k', 'metric' are averaged.
        - Connects configuration and results to ClearML.
        - Handles missing files or missing required columns gracefully.
        - Returns final aggregated metric and total number of communication rounds.
    """
    csv_path = os.path.join(config["model_training_report_dir_full_path"], "training_log.csv")
    task = Task.init(project_name=config["project_name"], task_name=config["task_name"])
    logger = task.get_logger()

    configuration_dict = {}
    if config is not None:
        configuration_dict.update(config)

    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: file not found in {csv_path}. Check the path. ")
        return None, 0

    df.columns = df.columns.str.strip()
    print(df.columns)

    if STEP_COLUMN_NAME not in df.columns:
        print(f"Error: Column '{STEP_COLUMN_NAME}' not found in the csv file.")
        return None, 0

    metrics_to_log = [
        col for col in df.columns
        if col not in ['timestamp', 'trainer_id', "sampling_method", STEP_COLUMN_NAME]
    ]

    aggregation_rules = {
        'count_outer_step': 'sum',
        'round_batches': 'sum',
        'round_samples': 'sum',
        'train_loss': 'mean',
        'current_batch_size': 'sum',
        'variance': 'mean',
        'T_k': 'mean',
        'sum_sq_diffs': 'sum',
        'metric': 'mean'
    }
    task.connect(aggregation_rules, name='Aggregation Rules')

    steps = sorted(df[STEP_COLUMN_NAME].unique())

    total_communication_rounds = len(df)
    final_aggregated_metric = None

    print(f"Found {total_communication_rounds} unique communication rounds. Begin processings...")

    for step in steps:
        step_df = df[df[STEP_COLUMN_NAME] == step]

        for _, row in step_df.iterrows():
            trainer_id = int(row['trainer_id'])
            for metric_name in metrics_to_log:
                if metric_name in row and pd.notna(row[metric_name]):
                    logger.report_scalar(
                        title=f"trainer_{trainer_id}/{metric_name}",
                        series=metric_name,
                        value=row[metric_name],
                        iteration=int(step)
                    )

        if not step_df.empty:
            valid_agg_keys = [key for key in aggregation_rules.keys() if key in step_df.columns]
            aggregated_metrics = step_df[valid_agg_keys].agg({
                k: aggregation_rules[k] for k in valid_agg_keys
            })

            for metric_name, value in aggregated_metrics.items():
                logger.report_scalar(
                    title=f"global_metric/{metric_name}",
                    series=metric_name,
                    value=value,
                    iteration=int(step)
                )

            if 'metric' in aggregated_metrics:
                final_aggregated_metric = aggregated_metrics['metric']

        print(f"Round {step} processed and logged.")

    result_dict = {
        'final_aggregated_metric': final_aggregated_metric,
        'total_communication_rounds': total_communication_rounds
    }

    task.connect(configuration_dict, name='Configuration')
    task.connect(result_dict, name='Results')

    print("\nProcessing finished. All data was sent to ClearML.")
    print(f"  - Final aggregated metric: {final_aggregated_metric}")
    print(f"  - Total communication rounds: {total_communication_rounds}")
    print(f"ClearML link: {task.get_output_log_web_page()}")

    task.close()

    return final_aggregated_metric, total_communication_rounds
