from huggingface_hub import hf_hub_download
import os

files = [
    "data/tasks/sql_create_context/train.json",
    "data/tasks/samsum/train.json",
    "data/safety_bench/hex_phi/harmful_behaviors.json",
    "data/safety_bench/hex_phi/benign_behaviors.json"
]

save_root = "finetuning_buckets/datasets/data"

for file_path in files:
    try:
        local_path = hf_hub_download(
            repo_id="Unispac/shallow-vs-deep-safety-alignment-dataset",
            filename=file_path,
            repo_type="dataset",
        )
        target_path = os.path.join(save_root, file_path.replace("data/", ""))
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        os.system(f"cp {local_path} {target_path}")
        print(f"✅ Saved: {target_path}")
    except Exception as e:
        print(f"❌ Failed to download {file_path}: {e}")
