"""
Provides the StepLossLogger class for logging training loss at each step into a CSV file,
reading step losses, generating statistics, and plotting trends.
"""
import os
import csv
import pandas as pd
import datetime
from typing import Dict, Any


class StepLossLogger:
    """
    Logger for step-wise training loss to CSV files with optional analysis and plotting.
    """

    def __init__(self, config: dict):
        """
        Initialize the logger, create directory and CSV file.

        Args:
            config (dict): Configuration dictionary containing:
                - "model_training_report_dir_full_path": directory to save logs
        """
        self.log_dir = config["model_training_report_dir_full_path"]
        self.config = config
        self.log_file_path = os.path.join(self.log_dir, "step_loss_log.csv")

        self.fieldnames = [
            "timestamp",
            "outer_step",
            "trainer_id",
            "node_id",
            "inner_step",
            "batch_idx",
            "loss",
            "batch_size",
            "learning_rate",
            "epoch_within_round"
        ]

        # Create folder if needed
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize CSV file with header
        with open(self.log_file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

        self.log_data_buffer = []
        self.buffer_size = 50   # Buffer entries for performance

    def log_step_loss(self, outer_step: int, trainer_id: int, node_id: int,
                      inner_step: int, batch_idx: int, loss: float,
                      batch_size: int, learning_rate: float = None,
                      epoch_within_round: int = None):
        
        """
        Log loss for a single training step.

        Args:
            outer_step: Outer step index
            trainer_id: Trainer ID
            node_id: Node ID
            inner_step: Inner step index
            batch_idx: Batch index within inner step
            loss: Step loss value
            batch_size: Batch size
            learning_rate: Optional learning rate
            epoch_within_round: Optional epoch within round
        """
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "outer_step": outer_step,
            "trainer_id": trainer_id,
            "node_id": node_id,
            "inner_step": inner_step,
            "batch_idx": batch_idx,
            "loss": loss,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epoch_within_round": epoch_within_round
        }

        self.log_data_buffer.append(log_entry)

        # Сбрасываем буфер если он заполнен
        if len(self.log_data_buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Write buffered log entries to the CSV file."""
        if self.log_data_buffer:
            with open(self.log_file_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                for entry in self.log_data_buffer:
                    # Заполняем пропущенные поля значением None
                    row = {field: entry.get(field, None) for field in self.fieldnames}
                    writer.writerow(row)
            self.log_data_buffer.clear()

    def read_step_losses(self) -> pd.DataFrame:
        """
        Read step loss CSV into a DataFrame.

        Returns:
            pd.DataFrame: Step loss data, with 'timestamp' converted to datetime
        """
        try:
            df = pd.read_csv(self.log_file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except FileNotFoundError:
            return pd.DataFrame(columns=self.fieldnames)

    def get_loss_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics for logged step losses.

        Returns:
            Dict[str, Any]: Dictionary of loss statistics
        """
        df = self.read_step_losses()

        if df.empty:
            return {"message": "Нет данных о лоссах"}

        stats = {
            "total_steps": len(df),
            "avg_loss": df['loss'].mean(),
            "min_loss": df['loss'].min(),
            "max_loss": df['loss'].max(),
            "std_loss": df['loss'].std(),
            "unique_trainers": df['trainer_id'].nunique(),
            "unique_nodes": df['node_id'].nunique(),
            "unique_outer_steps": df['outer_step'].nunique(),
            "loss_trend": self._calculate_loss_trend(df)
        }

        return stats

    def _calculate_loss_trend(self, df: pd.DataFrame) -> str:
        """
        Estimate the trend of loss (increasing, decreasing, stable).

        Args:
            df: Step loss DataFrame

        Returns:
            str: Description of trend
        """
        if len(df) < 2:
            return "not enough data"

        # Group by outer_step and take average loss
        avg_loss_by_outer_step = df.groupby('outer_step')['loss'].mean()

        if len(avg_loss_by_outer_step) < 2:
            return "not enough data"

        # Linear regression for trend
        x = range(len(avg_loss_by_outer_step))
        y = avg_loss_by_outer_step.values

        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        if abs(slope) < 0.01:
            return "stable"
        elif slope < 0:
            return "increasing"
        else:
            return "decreasing"

    def plot_loss_trends(self):
        """Generate plots for loss trends over training steps."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            df = self.read_step_losses()
            if df.empty:
                print("Not enough data for loss plot")
                return

            # Plot 1: Loss by outer steps for every trainer
            fig, axes = plt.subplots(1, 1, figsize=(15, 12))
            fig.suptitle('Loss analysis by steps', fontsize=16)

            # Plot 2: Loss by inner steps for every trainer
            ax1 = axes[0, 0]
            for trainer_id in df['trainer_id'].unique():
                trainer_df = df[df['trainer_id'] == trainer_id]
                ax1.plot(trainer_df.index, trainer_df['loss'],
                         label=f'Тренер {trainer_id}', alpha=0.7)
            ax1.set_xlabel('Outer step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Лосс over outer steps')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save the plot
            plot_path = os.path.join(self.log_dir, "step_loss_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Loss plot saved into: {plot_path}")

        except ImportError:
            print("Matplotlib not installed, plots were not created")
        except Exception as e:
            print(f"Error while creating plot: {e}")
