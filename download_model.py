# Download the model from Hugging Face Hub

from huggingface_hub import snapshot_download
import os


target_dir = "ckpts/Llama-3-8B-Lexi-Uncensored"
os.makedirs(target_dir, exist_ok=True)
snapshot_download(
    repo_id="Orenguteng/Llama-3-8B-Lexi-Uncensored",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)


