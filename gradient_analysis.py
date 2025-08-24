"""
Provides functions to compute full and batch gradients for a model, analyze deviations, 
compute kurtosis, and plot the distribution of batch gradient deviations from the full gradient.

Functions:
    - calculate_full_gradient(model, data_loader, device) -> torch.Tensor
    - calculate_batch_gradients(model, data_loader, device) -> Generator[torch.Tensor, None, None]
    - main(config)
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from tqdm import tqdm
from scipy.stats import kurtosis

from models import ModelFabric
from data import DataFabric
import warnings

import datetime


def calculate_full_gradient(model, data_loader, device):
    """
    Calculates and returns a single full gradient vector across the entire dataset
    """
    model.train()
    model.to(device)
    model.zero_grad()

    total_samples = 0

    # Accumulate gradients across all batches
    for batch in tqdm(data_loader, desc="Calculating full gradient"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        total_samples += input_ids.size(0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

    # Collect all gradients into a single tensor
    with torch.no_grad():
        full_grads_list = []
        for p in model.parameters():
            if p.grad is not None:
                full_grads_list.append((p.grad / total_samples).view(-1).cpu().clone())

        if not full_grads_list:
            return None
        return torch.cat(full_grads_list)


def calculate_batch_gradients(model, data_loader, device):
    """
    Generator function that yields the gradient for each individual batch.
    Progress tracking is added with tqdm.
    """
    model.train()
    model.to(device)

    for batch in tqdm(data_loader, desc="Calculating batch gradients"):
        model.zero_grad()  # Zero out gradients for each new batch
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        with torch.no_grad():
            grads = [p.grad.view(-1).cpu().clone() for p in model.parameters() if p.grad is not None]
            if grads:
                yield torch.cat(grads)


def main(config):
    device = torch.device(config["cuda"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    print(f"Using DataFabric to prepare dataset: {config['dataset_name']}")
    try:
        data_fabric = DataFabric(
            config['model_name'], config['model_config'],
            config['data_sample_size'], config['seed']
        )
        if config['dataset_name'] == 'imdb':
            full_dataset = data_fabric.prepare_imdb_dataset()
        else:
            raise ValueError(f"Unknown dataset name: {config['dataset_name']}")

        print(f"Dataset '{config['dataset_name']}' loaded successfully. Size: {len(full_dataset)} samples.")
    except Exception as e:
        print(f"Error while loading dataset: {e}")
        return

    # 2. Load model from checkpoint
    print(f"Loading model from checkpoint: {config['checkpoint_path']}")
    try:
        checkpoint = torch.load(str(config['checkpoint_path']), map_location='cpu', weights_only=True)

        model_fabric = ModelFabric(config['model_name'], config['model_config'])
        model = model_fabric.create_model()
        model.load_state_dict(checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("Model loaded successfully from checkpoint.")
    except Exception as e:
        print(f"Error loading model from checkpoint {config['checkpoint_path']}: {e}")
        return

    full_loader = DataLoader(full_dataset, batch_size=config['batch_size'])

    # 3. Compute full gradient
    print("Computing full gradient across the dataset...")
    full_gradient = calculate_full_gradient(model, full_loader, device)
    if full_gradient is None:
        print("Failed to compute full gradient. Check the model and data.")
        return
    print("Full gradient computed successfully.")
    full_gradient = full_gradient.cpu()

    # 4. Compute deviations for each batch
    print("Computing deviations for each batch...")
    deviations = []
    batch_loader_for_deviation = DataLoader(full_dataset, batch_size=config['batch_size'], shuffle=True)

    for batch_gradient in calculate_batch_gradients(model, batch_loader_for_deviation, device):
        metric = torch.norm(batch_gradient - full_gradient).item()
        deviations.append(metric)

    if not deviations:
        print("Failed to compute batch gradient deviations. Check the data and model.")
        return

    deviations_np = np.array(deviations)

    # 5. Compute sample excess kurtosis
    excess_kurtosis_val = kurtosis(deviations_np, fisher=True)

    # 6. Plot and save distribution
    print("Plotting distribution chart...")
    plt.figure(figsize=(12, 7))

    plt.hist(deviations_np, bins=50, density=True, alpha=0.75, color='#0072BC', edgecolor='black')

    mean_deviation = np.mean(deviations_np)
    median_deviation = np.median(deviations_np)

    plt.axvline(mean_deviation, color='red', linestyle='--', linewidth=2,
                label=f'Mean deviation: {mean_deviation:.4f}')
    plt.axvline(median_deviation, color='green', linestyle=':', linewidth=2,
                label=f'Median deviation: {median_deviation:.4f}')

    plt.title('Distribution of batch gradient deviations from full gradient', fontsize=16)
    plt.xlabel('Deviation $||\\nabla F_B(x) - \\nabla F(x)||$', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    info_text = (
        f"Dataset name: {script_settings['dataset_name']}\n"
        f"Model name: {script_settings['model_name']}\n"
        f"Number of outer_steps: {script_settings['outer_steps']}\n"
        f"Batch Size: {script_settings['batch_size']}\n"
        f"Data Sample Size: {script_settings['data_sample_size']}\n"
        f"Sample excess kurtosis: {excess_kurtosis_val:.4f}"
    )

    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    plt.legend(fontsize=12)
    plt.tight_layout()

    current_date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_path = os.path.join(config['output_dir'], f"gradient_distribution_{current_date}.png")
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to: {output_path}")
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    with open('config_analyze_gradient.yaml', 'r') as file:
        config = yaml.safe_load(file)
    script_settings = config.get('script_settings', {})
    os.makedirs(script_settings['output_dir'], exist_ok=True)

    main(script_settings)