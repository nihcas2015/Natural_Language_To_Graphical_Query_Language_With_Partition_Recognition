"""
Downloader & Ingestion Script for Neo4j Text2Cypher (2024v1) Dataset.
Downloads the official enterprise Text2Cypher dataset from HuggingFace (neo4j/text2cypher-2024v1).
Converts parquet files into structured JSONL format for training the query generator.
"""

import os
import sys
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import json

DATASET_REPO = "neo4j/text2cypher-2024v1"
DEST_DIR = "d:/vscode/text_to_gql/data/neo4j_text2cypher"


def download_and_process_split(split_filename: str, output_name: str):
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"\n[1/3] Downloading {split_filename} from HuggingFace ({DATASET_REPO})...")
    
    local_parquet = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"data/{split_filename}",
        repo_type="dataset",
        local_dir=DEST_DIR
    )
    print(f"Downloaded to: {local_parquet}")

    print(f"[2/3] Reading Parquet data with pyarrow...")
    table = pq.read_table(local_parquet)
    df = table.to_pandas()
    print(f"Loaded {len(df):,} records with columns: {list(df.columns)}")

    out_file = os.path.join(DEST_DIR, f"{output_name}.jsonl")
    print(f"[3/3] Exporting to standardized JSONL: {out_file}...")

    with open(out_file, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            record = {
                "id": str(row.get("id", idx)),
                "nl_question": row.get("question", row.get("text", "")),
                "gold_cypher": row.get("cypher", row.get("query", "")),
                "schema": row.get("schema", ""),
                "source": "neo4j_text2cypher_2024v1"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Successfully processed {len(df):,} records into {out_file} ✅")


def main():
    print("=" * 70)
    print("NEO4J TEXT2CYPHER (2024v1) DOWNLOADER")
    print("=" * 70)
    
    # Download and process train split
    download_and_process_split("train-00000-of-00001.parquet", "train")
    
    # Download and process test split
    download_and_process_split("test-00000-of-00001.parquet", "test")
    
    print("\nAll Neo4j Text2Cypher datasets ready in data/neo4j_text2cypher/! 🚀")


if __name__ == "__main__":
    main()

