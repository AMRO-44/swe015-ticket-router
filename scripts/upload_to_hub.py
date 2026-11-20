"""Upload models/ to a Hugging Face Hub model repo (run once after training).

Usage:  huggingface-cli login            (or set HF_TOKEN)
        HF_REPO_ID=<user>/swe015-ticket-router python scripts/upload_to_hub.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from huggingface_hub import HfApi  # noqa: E402
from src.config import HF_REPO_ID, MODELS_DIR  # noqa: E402

if __name__ == "__main__":
    if HF_REPO_ID.startswith("YOUR_HF_USERNAME"):
        sys.exit("Set HF_REPO_ID (env var or src/config.py) to <your-username>/swe015-ticket-router first.")
    api = HfApi()
    api.create_repo(HF_REPO_ID, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(MODELS_DIR), repo_id=HF_REPO_ID, repo_type="model",
                      ignore_patterns=["teacher_int8.pt", ".gitkeep"],
                      commit_message="SWE015 ticket router weights")
    print(f"uploaded -> https://huggingface.co/{HF_REPO_ID}")
