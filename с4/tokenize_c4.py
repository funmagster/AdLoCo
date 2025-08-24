"""
This script provides utilities for loading, tokenizing, and saving subsets of the
C4 (Colossal Clean Crawled Corpus) dataset using Hugging Face's `transformers`
and `datasets` libraries.

Features:
---------
- Load the C4 dataset in streaming or direct mode.
- Extract a training subset and optionally an evaluation split.
- Tokenize text using a Hugging Face tokenizer.
- Save tokenized datasets to disk for later use.
- Store metadata (e.g., model, tokenizer config, dataset size) in YAML format.

Usage:
------
Run from the command line:

    python tokenize_c4.py --model_name keeeeenw/MicroLlama --language en \
        --data_sample_size 2000 --max_length 512 --output_dir ./tokenized_c4 \
        --create_eval_split --eval_size 200
"""
import os
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import yaml

class C4Tokenizer:
    """
    A helper class for tokenizing the C4 dataset using a Hugging Face tokenizer.
    
    This class handles dataset loading, tokenization, splitting into training/evaluation,
    and saving the results to disk along with metadata.
    """
    def __init__(self, model_name, max_length=512):
        """
        Initialize the tokenizer.

        Args:
            model_name (str): Name of the pretrained model for tokenization.
            max_length (int): Maximum sequence length for tokenization. Defaults to 512.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        
    def tokenize_function(self, examples):
        """
        Tokenize input text examples.

        Args:
            examples (dict): A dictionary containing the "text" field.

        Returns:
            dict: Tokenized representation of the text.
        """
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )
    
    def tokenize_c4_dataset(self, language="en", data_sample_size=1000, streaming=True, 
                           output_dir="./tokenized_c4", create_eval_split=True, eval_size=None):
        """
        Tokenize and save a subset of the C4 dataset.

        Args:
            language (str): Language of the dataset to load (default: "en").
            data_sample_size (int): Number of training samples to extract (default: 1000).
            streaming (bool): Whether to use streaming mode for large datasets (default: True).
            output_dir (str): Directory where the tokenized dataset will be saved.
            create_eval_split (bool): Whether to create an evaluation dataset split.
            eval_size (int): Size of the evaluation dataset. If None, defaults to 10% of training size.

        Returns:
            str: Path to the saved tokenized training dataset.
        """
        print(f"Loading C4 dataset ({language})...")
        print(f"Training sample size: {data_sample_size}")
        
        if eval_size is None:
            eval_size = min(100, data_sample_size // 10)  # 10% или минимум 100
        
        if create_eval_split:
            print(f"Evaluation sample size: {eval_size}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if dataset was already tokenized
        tokenized_path = os.path.join(output_dir, f"c4_{language}_{data_sample_size}_tokenized")
        eval_tokenized_path = os.path.join(output_dir, f"c4_{language}_{eval_size}_eval_tokenized")
        
        if os.path.exists(tokenized_path) and (not create_eval_split or os.path.exists(eval_tokenized_path)):
            print(f"Tokenized datasets already exist:")
            print(f"  Training: {tokenized_path}")
            if create_eval_split:
                print(f"  Evaluation: {eval_tokenized_path}")
            return tokenized_path
        
        if streaming:
            # We use streaming for large datasets
            dataset = load_dataset(
                "c4",
                language,
                split="train",
                streaming=True
            )
            
            # Collecting data
            total_needed = data_sample_size + (eval_size if create_eval_split else 0)
            collected_examples = []
            print(f"Сбор данных (всего нужно: {total_needed})...")
            
            for example in tqdm(dataset, desc="Loading samples from C4", total=total_needed):
                collected_examples.append(example)
                
                # Check when we've collected enough
                if len(collected_examples) >= total_needed:
                    break
            
            train_examples = collected_examples[:data_sample_size]
            eval_examples = collected_examples[data_sample_size:data_sample_size + eval_size] if create_eval_split else []
            
            train_dataset = Dataset.from_list(train_examples)
            eval_dataset = Dataset.from_list(eval_examples) if create_eval_split else None
            
        else:
            # Direct loading
            print("Using direct loading")
            total_needed = data_sample_size + (eval_size if create_eval_split else 0)
            full_dataset = load_dataset(
                "c4",
                language,
                split=f"train[:{total_needed}]",
                streaming=False
            )
            
            # Splitting into training and validation 
            train_dataset = full_dataset.select(range(data_sample_size))
            eval_dataset = full_dataset.select(range(data_sample_size, data_sample_size + eval_size)) if create_eval_split else None
        
        print(f"Training dataset size: {len(train_dataset)} samples")
        if create_eval_split and eval_dataset:
            print(f"Evaluation dataset size: {len(eval_dataset)} samples")
        
        # Tokenizing the training dataset
        print("Tokenizing the training dataset...")
        tokenized_train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Токенизация C4 текста (обучение)",
        )
        
        # Saving the training dataset
        print(f"Сохранение обучающего датасета в {tokenized_path}...")
        tokenized_train_dataset.save_to_disk(tokenized_path)
        
        # Tokenizing the validation dataset
        if create_eval_split and eval_dataset:
            print("Tokenizing and saving the validation dataset...")
            tokenized_eval_dataset = eval_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Токенизация C4 текста (evaluation)",
            )
            
            # Saving the validation dataset
            print(f"Сохранение evaluation датасета в {eval_tokenized_path}...")
            tokenized_eval_dataset.save_to_disk(eval_tokenized_path)
        
        # Saving training dataset metadata
        metadata = {
            "language": language,
            "data_sample_size": data_sample_size,
            "max_length": self.max_length,
            "model_name": self.tokenizer.name_or_path,
            "create_eval_split": create_eval_split,
            "eval_size": eval_size if create_eval_split else None,
            "tokenizer_config": {
                "pad_token": self.tokenizer.pad_token,
                "eos_token": self.tokenizer.eos_token,
                "vocab_size": self.tokenizer.vocab_size
            }
        }
        
        metadata_path = os.path.join(output_dir, f"c4_{language}_{data_sample_size}_metadata.yaml")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
        
        # Saving evaluation dataset metadata
        if create_eval_split:
            eval_metadata = metadata.copy()
            eval_metadata["data_sample_size"] = eval_size
            eval_metadata["dataset_type"] = "evaluation"
            
            eval_metadata_path = os.path.join(output_dir, f"c4_{language}_{eval_size}_eval_metadata.yaml")
            with open(eval_metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(eval_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Tokenization finished!")
        print(f"Training dataset saved into {tokenized_path}")
        if create_eval_split:
            print(f"Evaluation dataset saved into {eval_tokenized_path}")
        return tokenized_path

def main():
    parser = argparse.ArgumentParser(description="Токенизация C4 датасета")
    parser.add_argument("--model_name", type=str, default="keeeeenw/MicroLlama",
                        help="Имя модели для токенизатора")
    parser.add_argument("--language", type=str, default="en",
                        help="Язык для загрузки C4 (по умолчанию 'en')")
    parser.add_argument("--data_sample_size", type=int, default=1000,
                        help="Размер выборки (по умолчанию 1000)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Максимальная длина последовательности (по умолчанию 512)")
    parser.add_argument("--output_dir", type=str, default="./tokenized_c4",
                        help="Директория для сохранения токенизированного датасета")
    parser.add_argument("--streaming", action="store_true", default=True,
                        help="Использовать streaming режим")
    parser.add_argument("--no_streaming", action="store_true",
                        help="Отключить streaming режим")
    parser.add_argument("--create_eval_split", action="store_true",
                        help="Создать evaluation датасет")
    parser.add_argument("--eval_size", type=int, default=None,
                        help="Размер evaluation датасета (по умолчанию 10% от основного)")
    
    args = parser.parse_args()
    
    # Setting up streaming
    if args.no_streaming:
        streaming = False
    else:
        streaming = args.streaming
    
    # If creating an evaluation split, use 10% for default
    eval_size = args.eval_size
    if args.create_eval_split and eval_size is None:
        eval_size = max(100, int(args.data_sample_size * 0.1))  # Minimum 100 samples
    
    # Creating a tokenizer
    tokenizer = C4Tokenizer(args.model_name, args.max_length)
    
    # Tokenizing the dataset
    tokenized_path = tokenizer.tokenize_c4_dataset(
        language=args.language,
        data_sample_size=args.data_sample_size,
        streaming=streaming,
        output_dir=args.output_dir,
        create_eval_split=args.create_eval_split,
        eval_size=eval_size
    )
    
    print(f"Tokenized dataset saved into: {tokenized_path}")
    if args.create_eval_split:
        eval_path = tokenized_path.replace("_tokenized", "_eval_tokenized")
        print(f"Evaluation dataset saved into: {eval_path}")
    print(f"PUT THIS INTO CONFIG AS THE DATASET PATH: {tokenized_path}")


if __name__ == "__main__":
    main()
