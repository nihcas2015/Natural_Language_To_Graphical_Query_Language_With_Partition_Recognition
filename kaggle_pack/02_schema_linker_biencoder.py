"""
KAGGLE NOTEBOOK 2: Partition-Aware Schema Linking Bi-Encoder
Target: Free GPU P100 or T4 on Kaggle
Dependencies: pip install sentence-transformers faiss-cpu datasets scikit-learn torch

Objective:
Fine-tunes a Sentence-Transformers bi-encoder (all-mpnet-base-v2) to map natural language questions
to schema elements (labels, properties, relationship types) AND their physical partition IDs.
Builds a persistent FAISS index for sub-millisecond retrieval.
"""

import json
import os
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
import numpy as np

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 32
EPOCHS = 5

def main():
    print("=== Training Partition-Aware Schema Linking Bi-Encoder ===")
    data_dir = "/kaggle/input/text2gql-partition-data"
    if not os.path.exists(data_dir):
        data_dir = "data/output"

    records = []
    for domain in ["movie", "nba", "company"]:
        fp = os.path.join(data_dir, f"{domain}_augmented.jsonl")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line))

    print(f"Loaded {len(records)} records for bi-encoder training.")

    # Construct positive training pairs: (NL question, schema element with partition metadata)
    train_examples = []
    schema_elements_catalog = set()

    for rec in records:
        q = rec["nl_question"]
        labels = rec.get("referenced_labels", [])
        partitions = rec.get("referenced_partitions", [0])
        part_str = f"partitions: {partitions}"

        for label in labels:
            pos_doc = f"Node label: {label} ({part_str})"
            schema_elements_catalog.add(pos_doc)
            train_examples.append(InputExample(texts=[q, pos_doc]))

    print(f"Constructed {len(train_examples)} training pairs. Catalog size: {len(schema_elements_catalog)}")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    model = SentenceTransformer(MODEL_NAME)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print("Starting fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=int(len(train_dataloader) * 0.1),
        show_progress_bar=True
    )

    output_dir = "/kaggle/working/schema_linker"
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    print(f"Model saved to {output_dir}")

    # Build FAISS Index
    try:
        import faiss
        print("Building FAISS index over schema catalog...")
        catalog_list = sorted(list(schema_elements_catalog))
        embeddings = model.encode(catalog_list, convert_to_numpy=True, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype("float32"))

        faiss.write_index(index, os.path.join(output_dir, "schema_index.faiss"))
        with open(os.path.join(output_dir, "catalog.json"), "w", encoding="utf-8") as f:
            json.dump(catalog_list, f, indent=2)
        print("FAISS index and catalog metadata successfully saved! ✅")
    except ImportError:
        print("FAISS not installed, saving numpy embeddings instead.")
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

    print("Bi-Encoder training pipeline complete! ✅")

if __name__ == "__main__":
    main()

