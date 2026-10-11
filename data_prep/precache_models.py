"""
Internet-Heavy Task Runner:
Pre-caches all base model tokenizers and weights from HuggingFace Hub locally
so you can work completely offline tomorrow without waiting on any model downloads.

Models Cached:
1. DistilBERT (Intent + NER): distilbert-base-uncased (~260MB)
2. Sentence-Transformers (Schema Linker): sentence-transformers/all-mpnet-base-v2 (~420MB)
3. Qwen2.5-Coder Tokenizer + Config: Qwen/Qwen2.5-Coder-7B-Instruct (~15MB)
4. Fallback Coder: Qwen/Qwen2.5-Coder-1.5B-Instruct tokenizer + config (~15MB)
"""

import os
import sys
import os
from huggingface_hub import snapshot_download

REPOS_TO_PRECACHE = [
    ("distilbert-base-uncased", "model"),
    ("sentence-transformers/all-mpnet-base-v2", "model"),
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "model"),
]

def main():
    print("=" * 70)
    print("PRE-CACHING HUGGINGFACE MODELS VIA SNAPSHOT DOWNLOAD")
    print("=" * 70)

    for repo_id, repo_type in REPOS_TO_PRECACHE:
        print(f"\n---> Fetching {repo_id}...")
        try:
            path = snapshot_download(repo_id=repo_id, repo_type=repo_type)
            print(f"     [SUCCESS] Cached to: {path}")
        except Exception as e:
            print(f"     [FAILED] {repo_id}: {e}")

    print("\n" + "=" * 70)
    print("ALL INTERNET-HEAVY DOWNLOADS COMPLETE! [READY FOR TOMORROW]")
    print("=" * 70)

if __name__ == "__main__":
    main()
