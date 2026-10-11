"""
KAGGLE NOTEBOOK 3: Grammar-Constrained Text-to-Cypher Generation Model (LoRA)
Target: Free GPU P100 or T4x2 on Kaggle
Dependencies: pip install transformers peft bitsandbytes datasets accelerate trl torch

Objective:
Fine-tunes Qwen2.5-Coder-7B-Instruct (or 1.5B) using 4-bit QLoRA to generate openCypher
queries conditioned on schema elements and physical partition annotations:
Target format:
  // partition:1
  MATCH (m:Movie)-[:directedBy]->(p:Person) ...
"""

import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Use Qwen2.5-Coder-7B-Instruct on T4x2/P100 (loads into ~5.5GB VRAM in 4-bit)
# Fallback: Qwen/Qwen2.5-Coder-1.5B-Instruct
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 3
LR = 2e-4

def format_instruction_prompt(record: dict) -> str:
    labels = record.get("referenced_labels", [])
    partitions = record.get("referenced_partitions", [0])
    question = record["nl_question"]

    schema_lines = [f"- {l} (owning partitions: {partitions})" for l in labels]
    schema_context = "\n".join(schema_lines)

    system_prompt = (
        "You are an expert Graph Query generator for distributed graph databases. "
        "Generate a syntactically correct openCypher query for the user question. "
        "Annotate the physical partition for each pattern using `// partition:N` comments."
    )
    user_prompt = f"Schema Context:\n{schema_context}\n\nQuestion: {question}"

    # Partition annotated gold target
    gold = record["gold_cypher"]
    target = f"// partition:{partitions[0]}\n{gold}"

    return system_prompt, user_prompt, target


def main():
    print(f"=== Fine-Tuning {MODEL_ID} with QLoRA on Kaggle ===")
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

    print(f"Loaded {len(records)} QA training examples.")

    # 4-bit Quantization Config (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {MODEL_ID} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Prepare tokenized dataset
    def preprocess_function(examples):
        inputs = []
        targets = []
        for i in range(len(examples["nl_question"])):
            rec = {k: examples[k][i] for k in examples}
            sys_p, user_p, target_p = format_instruction_prompt(rec)
            messages = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p},
                {"role": "assistant", "content": target_p}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs.append(text)

        tokenized = tokenizer(inputs, max_length=MAX_SEQ_LENGTH, truncation=True, padding=True)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    raw_dataset = Dataset.from_list(records)
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=raw_dataset.column_names)

    output_dir = "/kaggle/working/text2cypher_lora"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=10,
        fp16=True,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )

    print("Beginning LoRA fine-tuning...")
    trainer.train()

    print(f"Saving LoRA adapter weights to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning complete! LoRA weights ready for download. ✅")


if __name__ == "__main__":
    main()

