"""
Provides classes for evaluating models on different datasets. Supported tasks:
- IMDB: text classification (accuracy)
- C4: language modeling (perplexity)

Classes:
    - BaseEvaluator: Abstract base class for evaluators.
    - IMDBEvaluator: Evaluator for the IMDB classification task.
    - C4Evaluator: Evaluator for the C4 language modeling task.
    - EvalFabric: High-level wrapper for dataset-specific evaluation.
"""


import os
import torch
from abc import ABC, abstractmethod
from datasets import Dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm
from typing import Any

from fabrics.dataset_loader import DatasetLoader

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


class BaseEvaluator(ABC):
    """Abstract base class for dataset evaluators."""
    
    def __init__(self, model: Any, tokenizer: Any, dataset: Dataset, device: str):
        """
        Args:
            model (Any): HuggingFace model instance.
            tokenizer (Any): Tokenizer associated with the model.
            dataset (Dataset): Evaluation dataset.
            device (str): Device used for evaluation ('cuda' or 'cpu').
        """
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        self.model.to(self.device)

    @abstractmethod
    def run_evaluation(self) -> float:
        """
        Run evaluation on the dataset.

        Returns:
            float: Main evaluation metric (accuracy for classification, perplexity for language modeling).
        """
        raise NotImplementedError("Method 'run_evaluation' must be implemented in subclasses.")


class IMDBEvaluator(BaseEvaluator):
    """Evaluator for the IMDB sentiment classification dataset."""

    def run_evaluation(self) -> float:
        """
        Evaluate the model on the IMDB dataset using accuracy and a classification report.

        Returns:
            float: Accuracy score.
        """
        
        classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

        texts = self.dataset["text"]
        true_labels_int = self.dataset["label"]
        true_labels_str = [self.model.config.id2label[label] for label in true_labels_int]

        predictions = []
        for out in tqdm(classifier(texts, batch_size=12, truncation=True), total=len(texts), desc="Evaluating IMDB"):
            predictions.append(out)

        predicted_labels_str = [pred['label'] for pred in predictions]

        accuracy = accuracy_score(true_labels_str, predicted_labels_str)
        report = classification_report(
            true_labels_str,
            predicted_labels_str,
            target_names=list(self.model.config.id2label.values())
        )

        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification report:")
        print(report)
        return accuracy


class C4Evaluator(BaseEvaluator):
    """Evaluator for the C4 language modeling dataset."""
    
    def run_evaluation(self) -> float:
        """
        Evaluate the model on the C4 dataset using perplexity.

        Returns:
            float: Perplexity score.
        """
        self.model.eval()

        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for example in tqdm(self.dataset, desc="Evaluating perplexity"):
                inputs = self.tokenizer(
                    example['text'],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )

                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)

                if input_ids.size(1) < 2:
                    continue

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

                num_tokens = attention_mask.sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                del input_ids, attention_mask, outputs, loss, num_tokens, inputs

        if total_tokens == 0:
            print("No tokens could be processed.")
            return float('inf')

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        print(f"\nAverage validation loss: {avg_loss:.4f}")
        print(f"Validation perplexity: {perplexity:.4f}")
        return perplexity


class EvalFabric:
    """High-level wrapper that selects the appropriate evaluator based on the dataset."""

    def __init__(self, config: dict, model: Any):
        """
        Args:
            config (dict): Configuration dictionary containing dataset name and device info.
            model (Any): Model to be evaluated.
        """
        self.config = config
        self.model = model
        self.dataset_name = config["dataset"]

        device_config = config.get('devices', [None])[0]
        self.device = device_config if device_config else ("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Load dataset and tokenizer
        self.dataset_loader = DatasetLoader(config)
        self.dataset = self.dataset_loader.load_evaluation_dataset()
        self.tokenizer = self.dataset_loader.tokenizer

        # 3. Create evaluator
        self.evaluator = self._create_evaluator()

    def run_evaluation(self) -> float:
        """
        Run evaluation using the selected evaluator.

        Returns:
            float: Evaluation metric (accuracy or perplexity).

        Raises:
            ValueError: If evaluator was not created successfully.
        """
        if not self.evaluator:
            raise ValueError("Evaluator was not created. Check the configuration.")
        return self.evaluator.run_evaluation()

    def _create_evaluator(self) -> BaseEvaluator:
        """
        Instantiate the evaluator corresponding to the dataset.

        Returns:
            BaseEvaluator: Evaluator instance.

        Raises:
            ValueError: If the dataset type is not supported.
        """
        evaluator_map = {
            'imdb': IMDBEvaluator,
            'c4': C4Evaluator
        }

        evaluator_class = evaluator_map.get(self.dataset_name)

        if not evaluator_class:
            raise ValueError(f"Dataset type '{self.dataset_name}' is not supported.")

        return evaluator_class(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            device=self.device
        )
