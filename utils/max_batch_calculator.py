"""
Module for automatically determining the maximum feasible batch size on a CUDA device
using binary search, subject to memory constraints.

Functions:
    find_max_batch_size_binsearch: Finds the largest batch size that fits within 
    the GPU memory budget for a given model and dataset.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm


def find_max_batch_size_binsearch(
        model_fabric,
        dataset: TensorDataset,
        device: torch.device,
        logger,
        nodes_per_gpu: int,
        max_batch_size: int = 4096,
        padding_percentage: float = 0.85,
) -> int:
    """
    Find the maximum batch size that fits within the available GPU memory using binary search.

    Args:
        model_fabric: Object with a `create_model()` method that instantiates the model.
        dataset (TensorDataset): Dataset containing input tensors (input_ids, attention_mask, [labels]).
        device (torch.device): CUDA device on which the search is performed. Must be 'cuda'.
        logger: Logger instance for recording progress and results.
        nodes_per_gpu (int): Number of parallel nodes sharing the GPU memory.
        max_batch_size (int, optional): Maximum batch size upper bound for the search. Defaults to 4096.
        padding_percentage (float, optional): Fraction of available memory to use (safety margin).
            Defaults to 0.85 (85%).

    Returns:
        int: The largest batch size that fits into GPU memory under the defined budget.

    Raises:
        ValueError: If the device is not CUDA, the dataset is empty, 
            or no feasible batch size could be found.
        RuntimeError: If other runtime errors occur during model execution (besides OOM).
    """
    if device.type != 'cuda':
        raise ValueError("Batch size search is only supported for CUDA devices.")
    if not dataset or len(dataset) == 0:
        raise ValueError("Provided dataset is empty.")

    # 1. Compute the effective memory budget
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    available_memory = total_memory - reserved_memory
    memory_budget = available_memory * padding_percentage
    memory_budget_for_node = memory_budget / nodes_per_gpu

    logger.info(
        f"Per-node memory budget with margin ({padding_percentage * 100}%): "
        f"{memory_budget_for_node / 1e9:.2f} GB."
    )

    # 2. Create model and optimizer
    model = model_fabric.create_model()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # 3. Initialize binary search boundaries
    low = 1
    high = min(len(dataset), max_batch_size)
    optimal_bs = 0

    if len(dataset) < high:
        logger.warning(
            f"Dataset size ({len(dataset)}) is smaller than the maximum "
            f"batch size upper bound ({max_batch_size}). Search boundary adjusted."
        )

    pbar = tqdm(total=high.bit_length(), desc="Searching optimal batch_size", dynamic_ncols=True)
    pbar.update(1)
    while low <= high:
        bs = low + (high - low) // 2
        if bs == 0:
            break

        pbar.set_postfix(bs=bs, low=low, high=high, found_bs=optimal_bs)

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            loader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=True)
            batch = next(iter(loader))

            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device) if len(batch) > 2 else input_ids

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # 4. Check peak memory usage against the budget
            peak_memory = torch.cuda.max_memory_allocated(device)

            if peak_memory <= memory_budget_for_node:
                optimal_bs = bs
                low = bs + 1
                # trend = "↑ (peak <= budget)"
            else:
                high = bs - 1
                # trend = "↓ (peak > budget)"

            del input_ids, attention_mask, labels, outputs, loss, batch, loader

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high = bs - 1
                # logger.info(f"Batch size {bs}. ↓ (OOM)")
            else:
                raise e
        finally:
            torch.cuda.empty_cache()
            pbar.update(1)

    pbar.close()
    del model, optimizer
    torch.cuda.empty_cache()

    if optimal_bs == 0:
        error_msg = "Failed to find any batch size that fits into the memory budget."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Maximum feasible batch size per node: {optimal_bs}")
    return optimal_bs
