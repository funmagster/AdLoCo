"""
Module for logging training metrics and visualizing them.
All data is stored in a CSV file, which can later be used for plotting
the progression of metrics during training.

Classes:
    TrainingLogger: Logger for recording training metrics and generating plots.
"""

import os
import datetime
import csv
import pandas as pd
import matplotlib.pyplot as plt
import glob


class TrainingLogger:
    """
    Class for logging the training process and analyzing metrics.

    Attributes:
        log_dir (str): Directory for saving logs and plots.
        config (dict): Experiment configuration dictionary.
        log_file_path (str): Path to the CSV log file.
        fieldnames (list[str]): CSV column headers.
        count_outer_steps (dict): Counter of outer steps per trainer.
        logger_instance: Optional external logger instance.
        log_data_buffer (list[dict]): Buffer for accumulating log entries before flushing to disk.
    """

    def __init__(self, config: dict):
        """
        Initialize a TrainingLogger instance.

        Args:
            config (dict): Configuration dictionary with key
                "model_training_report_dir_full_path" for the logging directory
                and other experiment parameters.
        """
        self.log_dir = config['model_training_report_dir_full_path']
        self.config = config
        self.log_file_path = os.path.join(
            self.log_dir, f"training_log.csv"
        )

        self.fieldnames = [
            "timestamp",
            "sampling_method",
            "count_outer_step",
            "trainer_id",
            "round_batches",
            "round_samples",
            "train_loss",
            "current_batch_size",
            "variance",
            "inner_variance",
            "ortho_variance",
            "T_k",
            "T_ortho",
            "T_ip",
            "sum_sq_diffs",
            "metric",
            "outer_step_nubmer",
        ]

        self.count_outer_steps = {}
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.log_file_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
        self.logger_instance = None
        self.log_data_buffer = []

    def set_logger(self, logger_instance):
        """
        Set an external logger (e.g., MLFlow, WandB).

        Args:
            logger_instance: An external logger instance.
        """
        self.logger_instance = logger_instance

    def log(self, data: dict):
        """
        Log a single training step.

        Args:
            data (dict): Dictionary containing log fields.
                Must include "trainer_id". Other fields should match self.fieldnames.

        Notes:
            - Automatically assigns a timestamp.
            - Tracks outer step counts per trainer.
            - Buffers data until the buffer size reaches 100, then flushes to disk.
        """
        data["timestamp"] = datetime.datetime.now().isoformat()

        if data["trainer_id"] not in self.count_outer_steps:
            self.count_outer_steps[data["trainer_id"]] = 0
        else:
            self.count_outer_steps[data["trainer_id"]] += 1
        data["count_outer_step"] = self.count_outer_steps[data["trainer_id"]]

        self.log_data_buffer.append(data)
        if len(self.log_data_buffer) >= 100:
            self.flush()

    def flush(self):
        """
        Flush buffered log data to the CSV file.

        Returns:
            None
        """
        if self.log_data_buffer:
            with open(self.log_file_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                for entry in self.log_data_buffer:
                    row = {field: entry.get(field, None) for field in self.fieldnames}
                    writer.writerow(row)
            self.log_data_buffer.clear()

    def _get_plot_info_text(self):
        """
        Generate metadata text for plot annotations.

        Returns:
            str: Experiment metadata including model, dataset, learning rates,
            and number of steps.
        """
        return (
            f"Model: {self.config.get('model_type', 'N/A')}\n"
            f"Dataset: {self.config.get('dataset', 'N/A')} (Size: {self.config.get('data_sample_size', 'N/A')})\n"
            f"LR (inner): {self.config.get('lr_inner', 'N/A')}\n"
            f"LR (outer): {self.config.get('lr_outer', 'N/A')}\n"
            f"Steps (outer): {self.config.get('num_outer_steps', 'N/A')}\n"
            f"Steps (inner): {self.config.get('num_inner_steps', 'N/A')}"
        )

    def plot_variable_trend(self, variable_name: str, df: pd.DataFrame, log_file_name_prefix: str):
        """
        Plot the trend of a variable across outer steps.

        Args:
            variable_name (str): Name of the variable to plot.
            df (pd.DataFrame): DataFrame containing log data.
            log_file_name_prefix (str): Prefix for the saved plot file.

        Returns:
            None
        """
        plot_df = df.dropna(subset=[variable_name])
        if not plot_df.empty:
            fig, (ax1, ax_info) = plt.subplots(2, 1, figsize=(12, 9),
                                               gridspec_kw={'height_ratios': [0.8, 0.2]},
                                               constrained_layout=True)

            for trainer_id in plot_df['trainer_id'].unique():
                trainer_df = plot_df[plot_df['trainer_id'] == trainer_id]
                ax1.plot(trainer_df['count_outer_step'], trainer_df[variable_name], label=f'Тренер {trainer_id}')

            ax1.set_xlabel('Внешний шаг', fontsize=14)
            ax1.set_ylabel(variable_name.replace('_', ' ').title(), fontsize=14)
            ax1.set_title(f'{variable_name.replace("_", " ").title()} по внешним шагам', fontsize=16)
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax1.legend(loc='best', fontsize=12, frameon=True)

            info_text = self._get_plot_info_text()
            ax_info.text(0.0, 1.0, info_text, transform=ax_info.transAxes,
                         fontsize=10, verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

            ax_info.axis('off')

            fig.savefig(os.path.join(self.log_dir, f"{log_file_name_prefix}_{variable_name}_outer_steps.png"), dpi=300)
            plt.close(fig)
            print(f"Сохранен график: {log_file_name_prefix}_{variable_name}_outer_steps.png")
        else:
            print(f"Нет данных для построения графика {variable_name.replace('_', ' ').title()}")

    def plot_training_logs(self):
        """
        Generate plots for training logs.

        - Flushes pending log data.
        - Loads the latest log file.
        - Plots variables depending on the sampling method used.

        Returns:
            None
        """
        self.flush()
        csv_files = glob.glob(os.path.join(self.log_dir, "training_log.csv"))
        latest_file = max(csv_files, key=os.path.getctime)
        df = pd.read_csv(latest_file)

        if df.empty:
            print("Файл логов пуст. Нечего отображать.")
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        log_file_name_prefix = os.path.splitext(os.path.basename(latest_file))[0]


        if df.iloc[0]['sampling_method'] == "norm_test":
            variables_to_plot = [
            "train_loss", "round_batches", "round_samples", "current_batch_size", "variance",
            "T_k", "sum_sq_diffs", "metric", "outer_step_nubmer"
            ]
        elif df.iloc[0]['sampling_method'] == "augmented_inner_product_test":
            variables_to_plot = [
                "train_loss", "round_batches", "round_samples", "current_batch_size", "inner_variance",
                "ortho_variance", "T_ortho", "T_ip", "T_k", "metric", "outer_step_nubmer"
            ]
        else:
            variables_to_plot = [
                "train_loss", "round_batches", "round_samples", "current_batch_size", "metric", "outer_step_nubmer"
            ]
        for variable_name in variables_to_plot:
            self.plot_variable_trend(variable_name=variable_name, df=df, log_file_name_prefix=log_file_name_prefix)

        if 'T_k' in df.columns and 'current_batch_size' in df.columns:
            ratio_df = df.dropna(subset=['T_k', 'current_batch_size'])
            if not ratio_df.empty:
                fig, (ax1, ax_info) = plt.subplots(2, 1, figsize=(12, 9),
                                                   gridspec_kw={'height_ratios': [0.8, 0.2]},
                                                   constrained_layout=True)

                ratio_df['T_k_ratio_to_current_bs'] = ratio_df['T_k'] / ratio_df['current_batch_size']
                for trainer_id in ratio_df['trainer_id'].unique():
                    trainer_ratio_df = ratio_df[ratio_df['trainer_id'] == trainer_id]
                    ax1.plot(trainer_ratio_df['count_outer_step'], trainer_ratio_df['T_k_ratio_to_current_bs'],
                             label=f'Тренер {trainer_id} T_k / Текущий размер батча')

                ax1.set_xlabel('Внешний шаг', fontsize=14)
                ax1.set_ylabel('Соотношение T_k / Текущий размер батча', fontsize=14)
                ax1.set_title('Соотношение предлагаемого размера батча (T_k) к текущему размеру батча', fontsize=16)
                ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax1.legend(loc='best', fontsize=12, frameon=True)

                info_text = self._get_plot_info_text()
                ax_info.text(0.0, 1.0, info_text, transform=ax_info.transAxes,
                             fontsize=10, verticalalignment='top', horizontalalignment='left',
                             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

                ax_info.axis('off')

                fig.savefig(os.path.join(self.log_dir, f"{log_file_name_prefix}_Tk_current_bs_ratio_outer_steps.png"),
                            dpi=300)
                plt.close(fig)
                print(f"Сохранен график: {log_file_name_prefix}_Tk_current_bs_ratio_outer_steps.png")
            else:
                print("Нет данных для построения графика T_k / Текущий размер батча")
        else:
            print(
                "Отсутствуют необходимые столбцы ('T_k' или 'current_batch_size') для построения графика соотношения T_k к текущему размеру батча.")
