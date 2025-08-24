"""
This module provides a MemoryMonitor class to track GPU/CPU memory usage during training, 
log statistics, and generate visualizations and summary tables.

Functions:
    - memory_checkpoint: record memory at a checkpoint and optionally print summary.
    - reset_and_track_peak_memory: reset peak memory stats on all devices.
    - finalize_memory_analysis: save logs, plot, and optionally perform detailed analysis.
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
import os
import datetime


class MemoryMonitor:
    """
    Monitors memory usage across devices (GPU/CPU), tracks per-step memory statistics,
    and provides methods for logging, plotting, and detailed analysis.

    Attributes:
        devices (List[torch.device]): List of devices to track.
        memory_stats (List[dict]): List of memory snapshots with timestamp, stage, and device stats.
        outer_step_memory_data (dict): Memory statistics indexed by outer_step and stage.
        path_to_save (str): Directory to save logs and plots.
        logger: Logger object for printing summaries and info.
    """
    
    def __init__(self, devices: List[torch.device], path_to_save: str, logger):
        self.devices = devices
        self.memory_stats = []
        self.outer_step_memory_data = {}
        self.path_to_save = path_to_save
        self.logger = logger

    def record_memory_state(self, stage: str, outer_step: int = None, extra_info: str = ""):
        """
        Records memory usage on all devices at a given stage.

        Args:
            stage (str): Current stage label.
            outer_step (int, optional): Outer step index (if applicable).
            extra_info (str, optional): Additional information to store.
        """
        timestamp = datetime.datetime.now()
        memory_snapshot = {}

        for device in self.devices:
            if isinstance(device, str):
                device = torch.device(device)

            if torch.cuda.is_available() and device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                max_allocated = torch.cuda.max_memory_allocated(device)
                total = torch.cuda.get_device_properties(device).total_memory

                memory_snapshot[str(device)] = {
                    'allocated_mb': allocated / (1024 ** 2),
                    'reserved_mb': reserved / (1024 ** 2),
                    'max_allocated_mb': max_allocated / (1024 ** 2),
                    'total_mb': total / (1024 ** 2),
                    'allocated_gb': allocated / (1024 ** 3),
                    'reserved_gb': reserved / (1024 ** 3),
                    'max_allocated_gb': max_allocated / (1024 ** 3),
                    'total_gb': total / (1024 ** 3)
                }
            else:
                # Fallback for CPU or unavailable CUDA
                memory_snapshot[str(device)] = {
                    'allocated_mb': 0.0,
                    'reserved_mb': 0.0,
                    'max_allocated_mb': 0.0,
                    'total_mb': 0.0,
                    'allocated_gb': 0.0,
                    'reserved_gb': 0.0,
                    'max_allocated_gb': 0.0,
                    'total_gb': 0.0
                }

        record = {
            'timestamp': timestamp,
            'stage': stage,
            'outer_step': outer_step,
            'extra_info': extra_info,
            'memory_snapshot': memory_snapshot
        }

        self.memory_stats.append(record)

        if outer_step is not None:
            if outer_step not in self.outer_step_memory_data:
                self.outer_step_memory_data[outer_step] = {}
            self.outer_step_memory_data[outer_step][stage] = memory_snapshot

    def reset_peak_memory_stats(self):
        """Resets peak memory statistics on all GPU devices."""
        for device in self.devices:
            if isinstance(device, str):
                device = torch.device(device)

            if torch.cuda.is_available() and device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)

    def get_peak_memory_usage(self, device: torch.device) -> float:
        """
        Returns the peak memory usage in GB for a device.

        Args:
            device (torch.device): Device to query.

        Returns:
            float: Peak memory in GB.
        """
        if isinstance(device, str):
            device = torch.device(device)

        if torch.cuda.is_available() and device.type == 'cuda':
            return torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        return 0.0

    def print_memory_summary(self, stage: str):
        """Prints a summary of memory usage for all devices at a given stage."""
        self.logger.info(f"\n=== MEMORY SUMMARY: {stage} ===")
        for device in self.devices:
            if isinstance(device, str):
                device = torch.device(device)

            if torch.cuda.is_available() and device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
                self.logger.info(f"GPU {device}: Занято {allocated:.2f}GB | Зарезервировано {reserved:.2f}GB | "
                                 f"Пик {max_allocated:.2f}GB | Всего {total:.2f}GB")
            else:
                self.logger.info(f"Устройство {device}: CUDA недоступна или устройство не GPU")
        self.logger.info("=" * 70)

    def plot_memory_usage_by_outer_step(self):
        """Plots memory usage over outer steps for each device and stage."""
        if not self.outer_step_memory_data:
            self.logger.info("No data available for memory plotting")
            return

        outer_steps = sorted(self.outer_step_memory_data.keys())

        num_devices = len(self.devices)
        fig, axes = plt.subplots(num_devices, 1, figsize=(12, 6 * num_devices))
        if num_devices == 1:
            axes = [axes]

        for device_idx, device in enumerate(self.devices):
            device_str = str(device)
            ax = axes[device_idx]

            # Собираем данные для каждого этапа
            stages = ['до_outer_step', 'пик_outer_step', 'после_outer_step', 'после_merge']
            stage_colors = {'до_outer_step': 'blue', 'пик_outer_step': 'red', 'после_outer_step': 'green',
                            'после_merge': 'orange'}

            for stage in stages:
                memory_values = []
                outer_step_values = []

                for outer_step in outer_steps:
                    if stage in self.outer_step_memory_data[outer_step] and device_str in \
                            self.outer_step_memory_data[outer_step][stage]:
                        memory_gb = self.outer_step_memory_data[outer_step][stage][device_str]['allocated_gb']
                        memory_values.append(memory_gb)
                        outer_step_values.append(outer_step)

                if memory_values:
                    ax.plot(outer_step_values, memory_values, 'o-', label=stage, color=stage_colors.get(stage, 'black'))

            ax.set_xlabel('Outer_steps')
            ax.set_ylabel('Memory usage (GB)')
            ax.set_title(f'VRAM usage on device {device}')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

        path_to_save = os.path.join(self.path_to_save, "memory_usage_by_outer_step.png")
        plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
        self.logger.info(f"График сохранен: {path_to_save}")

        plt.show()

    def save_memory_log(self, filepath: str = None):
        """Saves memory logs to a text file."""
        if filepath is None:
            filepath = os.path.join(
                self.path_to_save, f"memory_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=== ЛОГ ИСПОЛЬЗОВАНИЯ ВИДЕОПАМЯТИ ===\n\n")

            for record in self.memory_stats:
                f.write(f"Время: {record['timestamp']}\n")
                f.write(f"Этап: {record['stage']}\n")
                if record['outer_step'] is not None:
                    f.write(f"Outer_stepа: {record['outer_step']}\n")
                if record['extra_info']:
                    f.write(f"Доп. инфо: {record['extra_info']}\n")

                for device, stats in record['memory_snapshot'].items():
                    f.write(f"  {device}: {stats['allocated_gb']:.2f}GB занято, "
                            f"{stats['max_allocated_gb']:.2f}GB пик, "
                            f"{stats['total_gb']:.2f}GB всего\n")
                f.write("\n")

        self.logger.info(f"Лог памяти сохранен: {filepath}")

    def create_detailed_memory_analysis(self):
        """Generates detailed analysis with multiple plots for memory usage."""

        if not self.outer_step_memory_data:
            self.logger.info("Нет данных для анализа")
            return

        # Preparing data for analysis
        outer_steps = sorted(self.outer_step_memory_data.keys())
        devices = self.devices

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed VRAM usage analysis', fontsize=16, fontweight='bold')

        # Plot 1: Outer steps dynamics for each GPU
        ax1 = axes[0, 0]
        stages = ['до_outer_step', 'пик_outer_step', 'после_outer_step', 'после_merge']
        stage_colors = {'до_outer_step': 'blue', 'пик_outer_step': 'red', 'после_outer_step': 'green',
                        'после_merge': 'orange'}

        for device in devices:
            device_str = str(device)
            for stage in stages:
                memory_values = []
                outer_step_values = []

                for outer_step in outer_steps:
                    if (stage in self.outer_step_memory_data[outer_step] and
                            device_str in self.outer_step_memory_data[outer_step][stage]):
                        memory_gb = self.outer_step_memory_data[outer_step][stage][device_str]['allocated_gb']
                        memory_values.append(memory_gb)
                        outer_step_values.append(outer_step)

                if memory_values:
                    label = f"{device} - {stage}"
                    ax1.plot(outer_step_values, memory_values, 'o-', label=label,
                             color=stage_colors.get(stage, 'black'), alpha=0.7)

        ax1.set_xlabel('Outer_steps')
        ax1.set_ylabel('Memory usage (GB)')
        ax1.set_title(f'VRAM usage on device {device}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Peak memory usage
        ax2 = axes[0, 1]
        for device in devices:
            device_str = str(device)
            peak_values = []
            outer_step_values = []

            for outer_step in outer_steps:
                if ('пик_outer_step' in self.outer_step_memory_data[outer_step] and
                        device_str in self.outer_step_memory_data[outer_step]['пик_outer_step']):
                    peak_gb = self.outer_step_memory_data[outer_step]['пик_outer_step'][device_str]['max_allocated_gb']
                    peak_values.append(peak_gb)
                    outer_step_values.append(outer_step)

            if peak_values:
                ax2.plot(outer_step_values, peak_values, 's-', label=f"{device} (пик)", linewidth=2)

        ax2.set_xlabel('Outer_steps')
        ax2.set_ylabel('Memory usage (GB)')
        ax2.set_title(f'VRAM usage on device {device}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Difference between peak and base levels
        ax3 = axes[1, 0]
        for device in devices:
            device_str = str(device)
            memory_diff = []
            outer_step_values = []

            for outer_step in outer_steps:
                outer_step_data = self.outer_step_memory_data[outer_step]
                if ('до_outer_step' in outer_step_data and 'пик_outer_step' in outer_step_data and
                        device_str in outer_step_data['до_outer_step'] and device_str in outer_step_data[
                            'пик_outer_step']):
                    before_gb = outer_step_data['до_outer_step'][device_str]['allocated_gb']
                    peak_gb = outer_step_data['пик_outer_step'][device_str]['max_allocated_gb']
                    diff = peak_gb - before_gb

                    memory_diff.append(diff)
                    outer_step_values.append(outer_step)

            if memory_diff:
                ax3.bar([f"Эп.{e}" for e in outer_step_values], memory_diff,
                        label=f"{device}", alpha=0.7)

        ax3.set_xlabel('Outer_steps')
        ax3.set_ylabel('Memory usage (GB)')
        ax3.set_title(f'VRAM usage on device {device}')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Memory clean-up effectiveness
        ax4 = axes[1, 1]
        for device in devices:
            device_str = str(device)
            cleanup_efficiency = []
            outer_step_values = []

            for outer_step in outer_steps:
                outer_step_data = self.outer_step_memory_data[outer_step]
                if ('пик_outer_step' in outer_step_data and 'после_outer_step' in outer_step_data and
                        device_str in outer_step_data['пик_outer_step'] and device_str in outer_step_data[
                            'после_outer_step']):

                    peak_gb = outer_step_data['пик_outer_step'][device_str]['allocated_gb']
                    after_gb = outer_step_data['после_outer_step'][device_str]['allocated_gb']

                    if peak_gb > 0:
                        efficiency = (peak_gb - after_gb) / peak_gb * 100
                        cleanup_efficiency.append(efficiency)
                        outer_step_values.append(outer_step)

            if cleanup_efficiency:
                ax4.plot(outer_step_values, cleanup_efficiency, 'o-', label=f"{device}")

        ax4.set_xlabel('Outer_steps')
        ax4.set_ylabel('Memory usage (GB)')
        ax4.set_title(f'VRAM usage on device {device}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)

        plt.tight_layout()

        path_to_save = os.path.join(self.path_to_save, "detailed_memory_analysis.png")
        plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
        self.logger.info(f"Analysis saved: {path_to_save}")
        plt.show()

    def create_memory_statistics_table(self):
        """Creates a table (DataFrame) summarizing memory usage statistics."""
        
        if not self.outer_step_memory_data:
            self.logger.info("No data to create table")
            return None

        outer_steps = sorted(self.outer_step_memory_data.keys())
        devices = self.devices

        stats_data = []

        for outer_step in outer_steps:
            outer_step_data = self.outer_step_memory_data[outer_step]

            for device in devices:
                device_str = str(device)
                row = {'Outer_stepа': outer_step, 'Устройство': device_str}

                for stage in ['до_outer_step', 'пик_outer_step', 'после_outer_step', 'после_merge']:
                    if stage in outer_step_data and device_str in outer_step_data[stage]:
                        stats = outer_step_data[stage][device_str]
                        row[f'{stage}_allocated_gb'] = round(stats['allocated_gb'], 2)
                        row[f'{stage}_max_allocated_gb'] = round(stats.get('max_allocated_gb', 0), 2)
                    else:
                        row[f'{stage}_allocated_gb'] = None
                        row[f'{stage}_max_allocated_gb'] = None

                stats_data.append(row)

        df = pd.DataFrame(stats_data)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'memory_statistics_{timestamp}.csv'
        path_to_save = os.path.join(self.path_to_save, filename)
        df.to_csv(path_to_save, index=False, encoding='utf-8')
        self.logger.info(f"Статистика памяти сохранена: {path_to_save}")

        return df

    def enhanced_analysis(self):
        """Performs full detailed memory analysis and returns statistics table."""
        self.create_detailed_memory_analysis()
        return self.create_memory_statistics_table()


def memory_checkpoint(monitor: MemoryMonitor, stage: str, outer_step: int = None,
                      print_summary: bool = True, extra_info: str = ""):
    """
    Records a memory checkpoint and optionally prints a summary.

    Args:
        monitor: MemoryMonitor instance.
        stage: Stage label.
        outer_step: Optional outer step index.
        print_summary: Whether to print summary to logger.
        extra_info: Additional info for the record.
    """
    monitor.record_memory_state(stage, outer_step, extra_info)

    if print_summary:
        monitor.print_memory_summary(stage)


def reset_and_track_peak_memory(monitor: MemoryMonitor):
    """
    Resets peak memory statistics to start fresh tracking.

    Args:
        monitor: MemoryMonitor instance
    """
    monitor.reset_peak_memory_stats()


def finalize_memory_analysis(monitor: MemoryMonitor, save_detailed: bool = True):
    """
    Finalize memory analysis - logs and plot 

    Args:
        monitor: MemoryMonitor instance
        save_detailed: whether to create detailed analysis
    """
    monitor.save_memory_log()
    monitor.plot_memory_usage_by_outer_step()

    if save_detailed:
        monitor.enhanced_analysis()
