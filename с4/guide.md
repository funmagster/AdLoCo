# Instructions for Using C4 Dataset Tokenization

## Overview

The C4 dataset can now be pre-tokenized into a separate file, which significantly speeds up the training process.  
The system supports two modes:

1. **Pre-tokenization** – tokenize the dataset in advance and save it to disk  
2. **Using a pre-tokenized dataset** – load an already prepared dataset  

---

## Step 1: Pre-tokenization

Use the `tokenize_c4.py` script to create a tokenized version of the dataset.

### Basic usage:
```bash
python tokenize_c4.py --data_sample_size 2048 --language en --create_eval_split --output_dir /home/shared/mit/
```

### Advanced options:
```bash
python tokenize_c4.py \
  --model_name "keeeeenw/MicroLlama" \
  --language en \
  --data_sample_size 2048 \
  --max_length 512 \
  --output_dir "./tokenized_c4" \
  --streaming
  --create_eval_split
```

### Parameters
- --model_name: Model name for the tokenizer (default: "keeeeenw/MicroLlama")
- --language: C4 dataset language (default: "en")
- --data_sample_size: Sample size (default: 1000)
- --max_length: Maximum sequence length (default: 512)
- --output_dir: Directory to save the dataset (default: "./tokenized_c4")
- --streaming: Use streaming mode (default: True)
- --no_streaming: Disable streaming mode
- --create_eval_split: Create an evaluation dataset

## Step 2: Configuration setup

Update your config.yaml file to use the tokenized dataset:

```yaml
# --- C4 DATASET SETTINGS ---
c4_language: "en"
c4_streaming: true
# Set the path to the tokenized dataset, which will be shown by tokenize_c4.py at the end of execution
c4_tokenized_path: "/home/shared/mit/c4_en_1000_tokenized"
```

### Config parameters
- c4_language: Dataset language
- c4_streaming: Whether to use streaming tokenization (if no pre-tokenized dataset is found)
- c4_tokenized_path: Path to the pre-tokenized dataset (if null, on-the-fly tokenization will be used)

## Step 3: Training 

Run training as usual:

```bash
python run.py
```

The system will automatically:
	1.	Check if the tokenized dataset exists at the specified path
	2.	If found – load it
	3.	If not found – perform tokenization on the fly

## Benefits of Pre-tokenization

1.	Faster startup – no need to re-tokenize data each run
2.	Memory efficiency – only required data is loaded
3.	Reusability – one tokenized dataset can be reused multiple times
4.	Compatibility check – the system will warn if the tokenizer is incompatible

## Example Workflow

	1.	Tokenize the dataset for experiments:
```bash
python tokenize_c4.py --data_sample_size 1000 --language en --create_eval_split --output_dir /home/shared/mit/
```

2. Update `config.yaml`:
```yaml
dataset: "c4"
data_sample_size: 1000
c4_tokenized_path: "./home/shared/mit/c4_en_1000_tokenized"
```

3. Run training
```bash
python run.py
```

4. For larger experiments, tokenize a bigger dataset:
```bash
python tokenize_c4.py --data_sample_size 10000 --language en --create_eval_split --output_dir /home/shared/mit/
```

5. Update config for the larger dataset:
```yaml
data_sample_size: 10000
c4_tokenized_path: "./home/shared/mit/c4_en_1000_tokenized"
```

## File Structure After Tokenization

After tokenization, the following files will be created:
```
tokenized_c4/
├── c4_en_1000_tokenized/      # Tokenized dataset
│   ├── dataset_info.json
│   ├── state.json
│   └── data-00000-of-00001.arrow
└── c4_en_1000_metadata.yaml   # Metadata
```

## Compatibility

- The system checks tokenizer compatibility with the model
- A warning will be shown if incompatible
- Metadata is saved for each tokenized dataset

## Troubleshooting

1.	Error: “Tokenized dataset not found
- Check the path in c4_tokenized_path
- Ensure tokenization finished successfully
2.	Tokenizer compatibility warning
- Re-tokenize with the correct model
- Or ignore the warning if you are sure about compatibility
3.	Out of memory during tokenization
- Reduce --data_sample_size
- Use --streaming mode
