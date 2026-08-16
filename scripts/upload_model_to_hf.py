"""
scripts/upload_model_to_hf.py
==============================
Upload fine-tuned model artifacts to Hugging Face Hub.

Why this is needed:
-------------------
Model weights (e.g. pytorch_model.bin / model.safetensors) are >300MB and
cannot be committed to GitHub. This script uploads your fine-tuned model
from `models/trocr_finetuned/` directly to Hugging Face Hub so your live
Streamlit Community Cloud app can download it on demand.

How to get your Hugging Face Access Token:
------------------------------------------
1. Log in to https://huggingface.co
2. Go to Settings -> Access Tokens (https://huggingface.co/settings/tokens)
3. Click "New token" -> Role: "Write" -> Name: "streamlit-upload"
4. Copy the token (starts with 'hf_...')

Usage
-----
    # Interactive mode (prompts for token and repo name):
    python scripts/upload_model_to_hf.py

    # CLI arguments:
    python scripts/upload_model_to_hf.py \
        --repo-id "mohaedafham2004/trocr-prescription-finetuned" \
        --folder "models/trocr_finetuned" \
        --token "hf_YourTokenHere" \
        --private
"""

import argparse
import os
import sys
from pathlib import Path

# Project root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
except ImportError:
    print(
        "❌ huggingface_hub is not installed.\n"
        "Run: pip install huggingface_hub",
        file=sys.stderr,
    )
    sys.exit(1)


def upload_model(
    folder_path: str,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
):
    folder = Path(folder_path).resolve()
    if not folder.exists():
        print(f"❌ Error: Model folder does not exist: {folder}", file=sys.stderr)
        print("   Did you download and extract the fine-tuned model from Colab?", file=sys.stderr)
        sys.exit(1)

    # Check for expected model files
    files = [f.name for f in folder.iterdir() if f.is_file()]
    if not files:
        print(f"❌ Error: Model folder is empty: {folder}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  Hugging Face Model Uploader")
    print(f"{'='*65}")
    print(f"  Source folder : {folder}")
    print(f"  Target repo   : {repo_id}")
    print(f"  Visibility    : {'Private' if private else 'Public'}")
    print(f"  Files found   : {len(files)} ({', '.join(files[:6])}{'...' if len(files) > 6 else ''})")
    print(f"{'='*65}\n")

    api = HfApi(token=token)

    try:
        # Create repo if it doesn't already exist
        print(f"Creating / verifying repo '{repo_id}' on Hugging Face Hub...")
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model",
        )
        print("✅ Repository verified on Hugging Face Hub.")

        # Upload the entire folder
        print(f"Uploading files from '{folder}' to '{repo_id}' ...")
        print("This may take a minute depending on your internet connection.")
        upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            token=token,
            repo_type="model",
            commit_message="Upload fine-tuned TrOCR prescription model weights",
        )

        print(f"\n{'='*65}")
        print(f"  🎉 SUCCESS! Model uploaded to:")
        print(f"  https://huggingface.co/{repo_id}")
        print(f"{'='*65}")
        print("\nNext Steps for Streamlit Deployment:")
        print(f"1. In config.yaml, set:")
        print(f'   trocr_use_pretrained: true')
        print(f'   trocr_model_name: "{repo_id}"')
        if private:
            print("2. Since this repo is PRIVATE, add your token in Streamlit Community Cloud:")
            print("   App Settings -> Secrets -> add:")
            print(f'   HF_TOKEN = "{token or "hf_..."}"')
        print(f"{'='*65}\n")

    except Exception as e:
        print(f"\n❌ Upload failed: {e}", file=sys.stderr)
        print("\nTroubleshooting tips:")
        print(" - Verify your token has 'Write' permissions at https://huggingface.co/settings/tokens")
        print(" - Verify your repo-id format is 'username/repo-name'")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload fine-tuned model to Hugging Face Hub.")
    parser.add_argument(
        "--folder",
        "-f",
        default="models/trocr_finetuned",
        help="Local path to fine-tuned model directory (default: models/trocr_finetuned)",
    )
    parser.add_argument(
        "--repo-id",
        "-r",
        default=None,
        help="Target Hugging Face repo ID (e.g., 'mohaedafham2004/trocr-prescription-finetuned')",
    )
    parser.add_argument(
        "--token",
        "-t",
        default=None,
        help="Hugging Face Write Token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as Private (default is Public)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("\n🔑 Hugging Face Authentication")
        print("Get a write token from: https://huggingface.co/settings/tokens")
        token = input("Enter your Hugging Face write token (hf_...): ").strip()
        if not token:
            print("❌ Error: A token is required to upload.", file=sys.stderr)
            sys.exit(1)

    repo_id = args.repo_id
    if not repo_id:
        default_repo = "mohaedafham2004/trocr-prescription-finetuned"
        user_input = input(f"Enter target repo ID [{default_repo}]: ").strip()
        repo_id = user_input if user_input else default_repo

    upload_model(
        folder_path=args.folder,
        repo_id=repo_id,
        token=token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
