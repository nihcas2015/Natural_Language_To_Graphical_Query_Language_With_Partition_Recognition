"""
KAGGLE NOTEBOOK 1: Joint Intent Classification + NER Model
Target: Free GPU P100 or T4 on Kaggle
Dependencies: pip install transformers datasets seqeval accelerate scikit-learn torch

Objective:
Trains a multi-task DistilBERT model with:
1. Sequence Classification Head: Predicts intent in {lookup, path_traversal, aggregation, multi_hop_join}
2. Token Classification Head: Predicts BIO tags for ENTITY, ATTRIBUTE, and VALUE tokens.
"""

import json
import os
import re
from typing import List, Dict, Tuple, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, DistilBertModel, DistilBertPreTrainedModel,
    TrainingArguments, Trainer, DataCollatorWithPadding
)

# ----------------------------------------------------------------------
# 1. Configuration & Label Sets
# ----------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 8
LR = 2e-5

INTENT_LABELS = ["lookup", "path_traversal", "aggregation", "multi_hop_join"]
INTENT2ID = {lbl: i for i, lbl in enumerate(INTENT_LABELS)}
ID2INTENT = {i: lbl for lbl, i in INTENT2ID.items()}

NER_TAGS = ["O", "B-ENTITY", "I-ENTITY", "B-ATTRIBUTE", "I-ATTRIBUTE", "B-VALUE", "I-VALUE"]
TAG2ID = {tag: i for i, tag in enumerate(NER_TAGS)}
ID2TAG = {i: tag for tag, i in TAG2ID.items()}

# ----------------------------------------------------------------------
# 2. Rule-Based Heuristic Annotation (Phase 0 Data Augmentation)
# ----------------------------------------------------------------------
def derive_intent(cypher: str) -> str:
    cypher_upper = cypher.upper()
    if any(fn in cypher_upper for fn in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]):
        return "aggregation"
    if "*" in cypher and ("*1.." in cypher or "*2.." in cypher):
        return "path_traversal"
    hops = cypher.count("-[") + cypher.count("<-[")
    if hops >= 2 or cypher_upper.count("MATCH") > 1:
        return "multi_hop_join"
    return "lookup"


def derive_ner_tags(nl_question: str, cypher: str) -> List[Tuple[str, str]]:
    tokens = nl_question.split()
    tags = ["O"] * len(tokens)
    literals = re.findall(r"['\"]([^'\"]+)['\"]", cypher)
    for lit in literals:
        lit_words = lit.split()
        for i in range(len(tokens) - len(lit_words) + 1):
            if " ".join(tokens[i:i+len(lit_words)]).lower() == lit.lower():
                tags[i] = "B-VALUE"
                for j in range(1, len(lit_words)):
                    tags[i+j] = "I-VALUE"
                break
    return list(zip(tokens, tags))


# ----------------------------------------------------------------------
# 3. Custom Multi-Task Model
# ----------------------------------------------------------------------
class JointIntentNERModel(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.intent_classifier = nn.Linear(config.hidden_size, len(INTENT_LABELS))
        self.ner_classifier = nn.Linear(config.hidden_size, len(NER_TAGS))
        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        intent_labels=None,
        ner_labels=None
    ):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        cls_output = seq_output[:, 0, :]        # [batch, hidden]

        intent_logits = self.intent_classifier(cls_output)
        ner_logits = self.ner_classifier(seq_output)

        loss = None
        if intent_labels is not None and ner_labels is not None:
            l_intent = self.intent_loss_fn(intent_logits, intent_labels)
            l_ner = self.ner_loss_fn(ner_logits.view(-1, len(NER_TAGS)), ner_labels.view(-1))
            loss = l_intent + 2.0 * l_ner

        return {
            "loss": loss,
            "intent_logits": intent_logits,
            "ner_logits": ner_logits
        }


# ----------------------------------------------------------------------
# 4. Multi-Task PyTorch Dataset
# ----------------------------------------------------------------------
class MultiTaskQADataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_length: int = 128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        tokens, ner_tags = zip(*rec["ner_spans"])
        encoding = self.tokenizer(
            list(tokens),
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        word_ids = encoding.word_ids(batch_index=0)
        ner_label_ids = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                ner_label_ids.append(-100)
            elif wid != prev_word:
                tag = ner_tags[wid]
                ner_label_ids.append(TAG2ID.get(tag, 0))
            else:
                ner_label_ids.append(-100)
            prev_word = wid

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "intent_labels": torch.tensor(INTENT2ID[rec["intent"]], dtype=torch.long),
            "ner_labels": torch.tensor(ner_label_ids, dtype=torch.long)
        }


def main():
    print("=== Training Joint Intent + NER Model on Kaggle ===")
    data_dir = "/kaggle/input/text2gql-partition-data"
    if not os.path.exists(data_dir):
        data_dir = "data/output"  # local fallback

    records = []
    for domain in ["movie", "nba", "company"]:
        fp = os.path.join(data_dir, f"{domain}_augmented.jsonl")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    rec["intent"] = derive_intent(rec["gold_cypher"])
                    rec["ner_spans"] = derive_ner_tags(rec["nl_question"], rec["gold_cypher"])
                    records.append(rec)

    print(f"Loaded {len(records)} augmented records.")
    train_data, val_data = train_test_split(records, test_size=0.15, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = MultiTaskQADataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = MultiTaskQADataset(val_data, tokenizer, MAX_LENGTH)

    model = JointIntentNERModel.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir="/kaggle/working/intent_ner_model",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    print("Training finished! Saving model and tokenizer to /kaggle/working/intent_ner_model...")
    model.save_pretrained("/kaggle/working/intent_ner_model")
    tokenizer.save_pretrained("/kaggle/working/intent_ner_model")
    print("DONE ✅")


if __name__ == "__main__":
    main()

