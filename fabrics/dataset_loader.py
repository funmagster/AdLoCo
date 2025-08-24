"""
This module provides classes and utilities for loading and processing datasets
for training and evaluation. Currently supports IMDB for classification tasks 
and C4 for language modeling tasks. Includes tokenization and conversion to 
PyTorch tensors.

Classes:
    - TokenizerFactory: Factory class to create tokenizers for different model types.
    - BaseDatasetProcessor: Abstract base class for dataset processing.
    - IMDBProcessor: Processor for the IMDB dataset.
    - C4Processor: Processor for the C4 dataset.
    - DatasetLoader: High-level class for dataset loading based on configuration.
"""
import os
from typing import Optional, Dict
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
from transformers import AutoTokenizer, BertTokenizer
from torch.utils.data import TensorDataset
from tqdm import tqdm


class TokenizerFactory:
    """Factory for creating tokenizers based on the model type."""
    
    @staticmethod
    def create_tokenizer(model_type: str, model_config: Dict):
        """
        Create a tokenizer for the specified model type.

        Args:
            model_type (str): Model type, e.g., 'bert' or 'microllama'.
            model_config (Dict): Configuration dictionary with at least 'name' key.

        Returns:
            transformers.PreTrainedTokenizer: Initialized tokenizer.

        Raises:
            NotImplementedError: If the model type is unsupported.
        """
        if model_type == 'bert':
            return BertTokenizer.from_pretrained(model_config['name'])
        elif model_type == 'microllama':
            tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        else:
            raise NotImplementedError(f"Model type {model_type} is not supported.")


class BaseDatasetProcessor:
    """Abstract base class for dataset processing and tokenization."""
    
    def __init__(self, tokenizer, seed: int = 42):
        """
        Args:
            tokenizer: Tokenizer object from the transformers library.
            seed (int): Random seed for reproducibility.
        """
        self.tokenizer = tokenizer
        self.seed = seed

    def _tokenize(self, examples, text_column: str, max_length: int = 512):
        """
        Tokenize a batch of examples.

        Args:
            examples (Dict): A batch of examples containing text.
            text_column (str): Column name in examples containing text.
            max_length (int): Maximum sequence length.

        Returns:
            Dict: Tokenized outputs containing input_ids and attention_mask.
        """
        return self.tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )

    def load_training_data(self, data_sample_size: int) -> TensorDataset:
        """
        Abstract method for loading training data.

        Args:
            data_sample_size (int): Number of training examples to load.

        Returns:
            TensorDataset: PyTorch dataset with tokenized input.
        """
        raise NotImplementedError

    def load_evaluation_data(self, val_size: int) -> Dataset:
        """
        Abstract method for loading evaluation data.

        Args:
            val_size (int): Number of validation examples to load.

        Returns:
            Dataset: HuggingFace dataset for evaluation.
        """
        raise NotImplementedError


class IMDBProcessor(BaseDatasetProcessor):
    """Processor for the IMDB dataset (binary sentiment classification)."""
    
    def load_training_data(self, data_sample_size: int) -> TensorDataset:
        """
        Load and tokenize IMDB training data.

        Args:
            data_sample_size (int): Number of training examples to load.

        Returns:
            TensorDataset: PyTorch dataset with input_ids, attention_mask, and labels.
        """
        print("Loading and preparing IMDB dataset...")
        full_dataset = load_dataset("imdb", split="train").shuffle(seed=self.seed)
        sampled_dataset = full_dataset.select(range(data_sample_size))

        tokenized_dataset = sampled_dataset.map(
            lambda examples: self._tokenize(examples, text_column="text", max_length=128),
            batched=True
        )
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        return TensorDataset(
            tokenized_dataset['input_ids'],
            tokenized_dataset['attention_mask'],
            tokenized_dataset['label']
        )

    def load_evaluation_data(self, samples_per_class: int = 500) -> Dataset:
        """
        Load and balance IMDB evaluation data.

        Args:
            samples_per_class (int): Number of samples to select per class.

        Returns:
            Dataset: Balanced evaluation dataset.
        """
        dataset = load_dataset("imdb", split="test").shuffle(seed=self.seed)

        subset_zeros = dataset.filter(lambda example: example['label'] == 0).select(range(samples_per_class))
        subset_ones = dataset.filter(lambda example: example['label'] == 1).select(range(samples_per_class))

        balanced_dataset = concatenate_datasets([subset_zeros, subset_ones])
        return balanced_dataset.shuffle(seed=self.seed)


class C4Processor(BaseDatasetProcessor):
    """Processor for the C4 dataset (large-scale language modeling)."""

    def __init__(self, tokenizer, seed: int = 42, config: Optional[Dict] = None):
        """
        Args:
            tokenizer: Tokenizer object from the transformers library.
            seed (int): Random seed for reproducibility.
            config (Optional[Dict]): Configuration dictionary for the C4 dataset.
        """
        super().__init__(tokenizer, seed)
        self.config = config or {}
        self.tokenized_path = self.config.get("c4_tokenized_path")

    def load_training_data(self, data_sample_size: int) -> TensorDataset:
        """
        Load or tokenize C4 training data.

        Args:
            data_sample_size (int): Number of training examples to load.

        Returns:
            TensorDataset: PyTorch dataset with input_ids and attention_mask.
        """
        if self.tokenized_path:
            return self._load_from_disk(self.tokenized_path, data_sample_size)

        dataset = self._load_raw_c4_data(data_sample_size, split="train")
        tokenized_dataset = dataset.map(
            lambda examples: self._tokenize(examples, text_column="text"),
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing C4 text",
        )
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        return TensorDataset(
            tokenized_dataset['input_ids'],
            tokenized_dataset['attention_mask']
        )

    def load_evaluation_data(self, val_size: int) -> Dataset:
        """
        Load or tokenize C4 evaluation data.

        Args:
            val_size (int): Number of validation examples to load.

        Returns:
            Dataset: Tokenized evaluation dataset.
        """
        if self.tokenized_path and os.path.exists(self._get_eval_path()):
            return self._load_from_disk(self._get_eval_path(), val_size, split="eval")
        else:
            return self._load_raw_c4_data(val_size, split="validation")

    def _load_raw_c4_data(self, size: int, split: str) -> Dataset:
        """
        Load raw C4 dataset examples.

        Args:
            size (int): Number of examples to load.
            split (str): Dataset split ('train' or 'validation').

        Returns:
            Dataset: Loaded HuggingFace dataset.
        """
        language = self.config.get("c4_language", "en")
        streaming = self.config.get("c4_streaming", False)

        if streaming:
            dataset = load_dataset("c4", language, split=split, streaming=True, trust_remote_code=True)
            examples = [ex for i, ex in
                        tqdm(zip(range(size), dataset), total=size, desc=f"Loading {split} examples from C4")]
            return Dataset.from_list(examples)
        else:
            return load_dataset("c4", language, split=f"{split}[:{size}]", trust_remote_code=True)

    def _load_from_disk(self, path: str, size: int, split: str = "train") -> Dataset:
        """
        Load tokenized dataset from disk.

        Args:
            path (str): Path to the tokenized dataset.
            size (int): Number of examples to load.
            split (str): Split type ('train' or 'eval').

        Returns:
            Union[TensorDataset, Dataset]: PyTorch TensorDataset for training or HuggingFace Dataset for evaluation.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the dataset does not contain enough samples.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenized dataset not found: {path}")

        dataset = load_from_disk(path)
        if len(dataset) > size:
            print(f"Truncating dataset to {size} examples...")
            dataset = dataset.select(range(size))
        else:
            raise ValueError("Not enough data.")
        if split == "eval":
            return dataset

        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        return TensorDataset(
            dataset['input_ids'],
            dataset['attention_mask']
        )

    def _get_eval_path(self):
        """
        Construct evaluation path for the tokenized dataset.

        Returns:
            str: Path to the tokenized evaluation dataset.
        """
        return self.tokenized_path.replace("_tokenized", "_eval_tokenized")


class DatasetLoader:
    """
    High-level class for loading datasets according to configuration.

    Handles tokenizer initialization and selects the appropriate processor
    based on dataset type.
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config (dict): Configuration dictionary with keys:
                - dataset (str): Name of dataset ('imdb' or 'c4').
                - training_seed (int): Random seed.
                - data_sample_size (int): Number of training samples.
                - val_size (int): Number of validation samples.
                - model_type (str): Type of model ('bert', 'microllama', etc.).
                - model_config (dict): Model-specific configuration.
        """
        self.config = config
        self.seed = config['training_seed']
        self.dataset_name = config['dataset']
        self.data_sample_size = config['data_sample_size']
        self.val_size = config['val_size']

        self.tokenizer = TokenizerFactory.create_tokenizer(config['model_type'], config['model_config'])
        self.processor = self._get_processor()

    def _get_processor(self) -> BaseDatasetProcessor:
        """
        Select processor class based on dataset name.

        Returns:
            BaseDatasetProcessor: Processor instance for the dataset.

        Raises:
            ValueError: If the dataset type is not supported.
        """
        if self.dataset_name == 'imdb':
            return IMDBProcessor(self.tokenizer, self.seed)
        elif self.dataset_name == 'c4':
            return C4Processor(self.tokenizer, self.seed, self.config)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def load_training_dataset(self):
        """
        Load training dataset using the selected processor.

        Returns:
            TensorDataset: Tokenized PyTorch training dataset.
        """
        return self.processor.load_training_data(self.data_sample_size)

    def load_evaluation_dataset(self):
        """
        Load evaluation dataset using the selected processor.

        Returns:
            Dataset: Tokenized HuggingFace evaluation dataset.
        """
        return self.processor.load_evaluation_data(self.val_size)
