# Kaggle Training Guide: Partition-Aware Text-to-GQL Models

This guide provides step-by-step instructions for fine-tuning the NLP/ML models for Partition-Aware Text-to-GQL on Kaggle's free GPU compute (30 hours/week of NVIDIA T4 or P100 GPUs).

---

## 1. Preparing the Dataset to Upload

You already have the augmented dataset generated in `d:/vscode/text_to_gql/data`.

### Files to pack into a single zip file:
Create a zip archive called `text2gql-partition-data.zip` containing:
1. `data/output/movie_augmented.jsonl`
2. `data/output/nba_augmented.jsonl`
3. `data/output/company_augmented.jsonl`
4. `data/output/summary_stats.json`
5. `data/maps/movie_k4.json`
6. `data/maps/nba_k4.json`
7. `data/maps/company_k4.json`

### Uploading to Kaggle:
1. Log in to [kaggle.com](https://www.kaggle.com).
2. On the left sidebar, click **Datasets** ➔ **New Dataset**.
3. Set the Dataset Title to: `text2gql-partition-data`.
4. Drag and drop the `.jsonl` and `.json` files (or your zip file).
5. Click **Create**.
6. Note the Kaggle dataset slug: `/kaggle/input/text2gql-partition-data`.

---

## 2. Setting Up Kaggle Notebooks

For each model, create a new Kaggle Notebook:
1. Click **+ New Notebook**.
2. Under **Notebook settings** (right panel):
   - **Accelerator**: Select **GPU T4 x2** (or **GPU P100**).
   - **Internet**: Toggle **ON** (needed for HuggingFace downloads).
3. In the right panel, click **+ Add Data**, search for your dataset `text2gql-partition-data`, and attach it.

---

## 3. Four Dedicated Notebooks

### Notebook 1: Joint Intent Classification + NER
- **File**: `kaggle_pack/01_intent_ner_distilbert.py`
- **Cell 1 (Dependencies)**:
  ```bash
  !pip install transformers datasets seqeval accelerate scikit-learn torch
  ```
- **Cell 2**: Paste the full code from `01_intent_ner_distilbert.py` and run.
- **Output Artifact**: `/kaggle/working/intent_ner_model/`
- **Runtime**: ~15 minutes on T4.

### Notebook 2: Partition-Aware Schema Linking Bi-Encoder
- **File**: `kaggle_pack/02_schema_linker_biencoder.py`
- **Cell 1 (Dependencies)**:
  ```bash
  !pip install sentence-transformers faiss-cpu datasets scikit-learn torch
  ```
- **Cell 2**: Paste the full code from `02_schema_linker_biencoder.py` and run.
- **Output Artifacts**:
  - `/kaggle/working/schema_linker/` (model weights)
  - `/kaggle/working/schema_linker/schema_index.faiss` (FAISS fast retrieval index)
- **Runtime**: ~10 minutes on T4.

### Notebook 3: Text-to-Cypher Generation Model (Qwen2.5-Coder + QLoRA)
- **File**: `kaggle_pack/03_text2cypher_lora_qwen.py`
- **Cell 1 (Dependencies)**:
  ```bash
  !pip install transformers peft bitsandbytes datasets accelerate trl torch
  ```
- **Cell 2**: Paste the full code from `03_text2cypher_lora_qwen.py` and run.
- **Output Artifact**: `/kaggle/working/text2cypher_lora/` (LoRA adapter weights)
- **Runtime**: ~45 minutes on T4 x2.

### Notebook 4: Cross-Partition Cost Estimator (LightGBM)
- **File**: `kaggle_pack/04_cost_estimator_lightgbm.py`
- **Cell 1 (Dependencies)**:
  ```bash
  !pip install lightgbm scikit-learn pandas joblib
  ```
- **Cell 2**: Paste the full code from `04_cost_estimator_lightgbm.py` and run.
- **Output Artifact**: `/kaggle/working/cost_estimator/lightgbm_cost_estimator.joblib`
- **Runtime**: ~2 minutes (CPU is sufficient).

---

## 4. Downloading the Trained Weights Locally

Once training finishes in each Kaggle notebook:
1. In the right panel of the notebook, expand **Output** ➔ `/kaggle/working`.
2. Click the three dots next to the folder/file and select **Download**.
3. Place the downloaded folders into your local project directory:
   - `/kaggle/working/intent_ner_model` ➔ `d:/vscode/text_to_gql/models/intent_ner/`
   - `/kaggle/working/schema_linker` ➔ `d:/vscode/text_to_gql/models/schema_linker/`
   - `/kaggle/working/text2cypher_lora` ➔ `d:/vscode/text_to_gql/models/text2cypher_lora/`
   - `/kaggle/working/cost_estimator` ➔ `d:/vscode/text_to_gql/models/cost_estimator/`

