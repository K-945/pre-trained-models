---
dataset_info:
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 21498947
    num_examples: 10000
  - name: validation
    num_bytes: 21659922
    num_examples: 10000
  - name: test
    num_bytes: 21607334
    num_examples: 10000
  download_size: 39991477
  dataset_size: 64766203
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# Dataset Card for Small C4 Dataset (10k Train, 10k Validation, 10k Test)

## Dataset Summary

The **Small C4 Dataset** is a reduced version of the original [C4 dataset](https://huggingface.co/datasets/allenai/c4) (Colossal Clean Crawled Corpus), designed to facilitate lightweight experimentation and model training without the need to process the full C4 dataset. This dataset includes:
- **10,000 examples** for training,
- **10,000 examples** for validation, and
- **10,000 examples** for testing.

Each example consists of a single text passage, sourced from the English subset of the original C4 corpus.

## Dataset Details

- **Source**: [allenai/c4](https://huggingface.co/datasets/allenai/c4)
- **Subset Language**: English
- **Streaming Enabled**: Yes (streaming=True used to sample without downloading the entire dataset)
- **Sampling Method**:
  - **Training Set**: First 10,000 examples from the `train` split of C4.
  - **Validation Set**: First 10,000 examples from the `validation` split of C4.
  - **Test Set**: The next 10,000 examples from the `validation` split (after the validation set).
- **Dataset Size**: 30,000 examples in total.

## Dataset Creation

The dataset was created using Hugging Face’s `datasets` library with streaming enabled to handle the large size of the original C4 dataset efficiently. A subset of examples was sampled in parallel for each of the train, validation, and test splits.

## Usage

This dataset is suitable for lightweight model training, testing, and experimentation, particularly useful when:
- **Computational resources** are limited,
- **Prototyping** models before scaling to the full C4 dataset, or
- **Evaluating** model performance on a smaller, representative sample of the full corpus.

## Example Usage

```python
from datasets import load_dataset

# Load the small C4 dataset
dataset = load_dataset("brando/small-c4-dataset")

# Access train, validation, and test splits
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

# Example: Display a random training example
print(train_data[0])
```


License
This dataset inherits the licensing of the original C4 dataset.

Citation
If you use this dataset in your work, please cite the original C4 dataset or my ultimate utils repo:

```bibtex
@misc{miranda2021ultimateutils,
    title={Ultimate Utils - the Ultimate Utils Library for Machine Learning and Artificial Intelligence},
    author={Brando Miranda},
    year={2021},
    url={https://github.com/brando90/ultimate-utils},
    note={Available at: \url{https://www.ideals.illinois.edu/handle/2142/112797}},
    abstract={Ultimate Utils is a comprehensive library providing utility functions and tools to facilitate efficient machine learning and AI research, including efficient tensor manipulations and gradient handling with methods such as `detach()` for creating gradient-free tensors.}
}
```

Script that created it

```python
import os
from huggingface_hub import login
from datasets import Dataset, DatasetDict, load_dataset
from concurrent.futures import ThreadPoolExecutor

# Function to load the Hugging Face API token from a file
def load_token(file_path: str) -> str:
    """Load API token from a specified file path."""
    with open(os.path.expanduser(file_path)) as f:
        return f.read().strip()

# Function to log in to Hugging Face using a token
def login_to_huggingface(token: str) -> None:
    """Authenticate with Hugging Face Hub."""
    login(token=token)
    print("Login successful")

# Function to sample a specific number of examples from a dataset split
def sample_from_split(split_name: str, num_samples: int) -> list:
    """Sample a specified number of examples from a dataset split."""
    c4_split = load_dataset("allenai/c4", "en", split=split_name, streaming=True)
    samples = []
    for i, example in enumerate(c4_split):
        if i >= num_samples:
            break
        samples.append(example["text"])
    return samples

# Main function to create a smaller C4 dataset with three subsets and upload it
def main() -> None:
    # Step 1: Load token and log in
    key_file_path: str = "/lfs/skampere1/0/brando9/keys/brandos_hf_token.txt"
    token: str = load_token(key_file_path)
    login_to_huggingface(token)

    # Step 2: Define sampling parameters
    num_samples = 10000

    # Step 3: Sample subsets concurrently
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_train = executor.submit(sample_from_split, "train", num_samples)
        future_val = executor.submit(sample_from_split, "validation", num_samples)
        future_test = executor.submit(sample_from_split, "validation", num_samples * 2)

        train_samples = future_train.result()
        val_samples = future_val.result()
        test_samples = future_test.result()[num_samples:]  # Second 10k from validation for test

    # Step 4: Create DatasetDict
    small_c4_dataset = DatasetDict({
        "train": Dataset.from_dict({"text": train_samples}),
        "validation": Dataset.from_dict({"text": val_samples}),
        "test": Dataset.from_dict({"text": test_samples})
    })

    # Step 5: Upload to Hugging Face Hub
    dataset_name_c4: str = "brando/small-c4-dataset"
    small_c4_dataset.push_to_hub(dataset_name_c4)
    print(f"Small C4 dataset uploaded to https://huggingface.co/datasets/{dataset_name_c4}")

# Run the main function
if __name__ == "__main__":
    main()

```