"""
Create HuggingFaceTB/CoT_Reasoning_Bushcraft_Survival from mattwesney/CoT_Reasoning_Bushcraft_Survival.

Keeps only `id` and a new `messages` column built from the `question` column:
  [{"content": <question>, "role": "user"}]
"""

from datasets import load_dataset

# Load the source dataset
ds = load_dataset("mattwesney/CoT_Reasoning_Bushcraft_Survival", split="train")

# Build the messages column and keep only id + messages
def transform(example):
    return {
        "id": example["id"],
        "messages": [{"content": example["question"], "role": "user"}],
    }

ds = ds.map(transform, remove_columns=ds.column_names)

# Push to Hub
ds.push_to_hub("HuggingFaceTB/CoT_Reasoning_Bushcraft_Survival", split="train")
print(f"Done – uploaded {len(ds)} rows.")
