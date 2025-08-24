"""
Provides a factory class to create HuggingFace models based on configuration.

Classes:
    - ModelFabric: Factory for creating models (BERT or MicroLLaMA) and retrieving output layer names.
"""
import torch

from transformers import BertForSequenceClassification, AutoConfig
from transformers import LlamaForCausalLM, LlamaConfig


class ModelFabric:
    """Factory class to create models and retrieve output layer names."""
    def __init__(self, config):
        """
        Initialize the model fabric.

        Args:
            config (dict): Configuration dictionary containing:
                - model_type (str): Type of model ('bert' or 'microllama').
                - model_config (dict): Dictionary of model-specific configuration including:
                    - name (str): Pretrained model name or path.
                    - pretrained (bool): Whether to load pretrained weights.
                    - num_labels (int, optional): Number of labels (for classification tasks).
        """
        self.model_type = config['model_type']
        self.model_config = config['model_config']

    def create_model(self):
        """
        Create a model instance based on the configuration.

        Returns:
            torch.nn.Module: Instantiated model (BertForSequenceClassification or LlamaForCausalLM).

        Raises:
            NotImplementedError: If the specified model_type is not supported.
        """
        if self.model_type == 'bert':
            if self.model_config['pretrained']:
                model = BertForSequenceClassification.from_pretrained(
                    self.model_config['name'], num_labels=self.model_config['num_labels'],
                    torch_dtype=torch.float32, local_files_only=True,
                )
            else:
                config = AutoConfig.from_pretrained(
                    self.model_config['name'],
                    num_labels=self.model_config['num_labels'], 
                    local_files_only=True,
                )
                model = BertForSequenceClassification(config)
        elif self.model_type == 'microllama':
            if self.model_config.get('pretrained', True):
                model = LlamaForCausalLM.from_pretrained(
                    self.model_config['name'],
                    torch_dtype=torch.float32,
                    local_files_only=True,
                )
            else:
                config = LlamaConfig.from_pretrained(self.model_config['name'], local_files_only=True)
                model = LlamaForCausalLM(config)
        else:
            raise NotImplementedError(f"Model type {self.model_type} is not supported")

        return model

    def get_output_layer_names(self):
        """
        Get the names of the output layers for the model type.

        Returns:
            list[str]: List of output layer names, e.g. ['classifier'] for BERT, ['lm_head'] for MicroLLaMA.
        """
        if self.model_type == 'bert':
            return ['classifier']
        elif self.model_type == 'microllama':
            return ['lm_head']
        else:
            return []
