# MetalMind

**MetalMind** is a domain-specific representation learning project aimed at training high-quality embeddings over metal commodity market text — starting with London Metal Exchange (LME) *Aluminum*. These embeddings are intended to support Retrieval-Augmented Generation (RAG) and semantic search workflows focused on metal markets.

---

## 📌 Project Status (As of Now)

✔️ Vocabulary model training completed for **64K vocab size** using **SentencePiece Unigram**
✔️ Systematic evaluation across multiple vocabulary candidates
✔️ Supporting scripts for calculating evaluation metrics and selecting optimal vocabulary
✔️ Tokenizer artifacts produced (vocab, model file) ready for downstream embedding training
✔️ Beginning stage of building a lightweight transformer encoder for embedding learning
✔️ Designed for eventual RAG pipelines over metal markets

> ❗ **Training data cannot be shared due to data source confidentiality policies.**

---

## 🧠 Tokenizer Training

A SentencePiece tokenizer was trained to support domain-specific tokenization over market text.

### Training command used:

```python
spm.SentencePieceTrainer.train(
    input=str(RAW_CORPUS_TXT),
    model_prefix=str(final_model_path),
    model_type="unigram",
    vocab_size=best_vocab,
    character_coverage=1.0,
    minloglevel=2,
)
```

**Note:**
`best_vocab` was chosen via an elbow/score analysis over multiple trained vocab sizes.

---

## 📊 Vocab Selection Results

You experimented with different vocab sizes and calculated evaluation metrics such as:

| Vocab Size | Unknown Ratio | Fragmentation | Compression | Vocab Utilization | Normalized Score |
| ---------- | ------------- | ------------- | ----------- | ----------------- | ---------------- |
| 110000     | ~0.0          | 1.146         | 0.189       | 104215            | 0.90             |
| 96000      | ~0.0          | 1.147         | 0.189       | 94938             | 0.886            |
| **64000**  | ~0.05         | 1.153         | 0.190       | 63934             | **0.827**        |
| 32000      | ~0.20         | 1.172         | 0.193       | 31993             | 0.698            |
| 16000      | ~0.48         | 1.212         | 0.199       | 15995             | 0.491            |
| 8000       | 1.0           | 1.290         | 0.212       | 7996              | 0.10             |

The **64K vocabulary model** stands out as a good balance of:

* low unknown token ratio
* reasonable fragmentation and compression
* strong overall *normalized score* based on your custom weighting

---

## 🔍 Score Calculation Method

You computed scores to compare vocab models using normalized ratios:

```python
df_scores["score"] = (
    FRAGMENTATION_RATE_WT * df_scores["fragmentation_ratio_n"]
    + COMPRESSION_RATIO_WT * df_scores["compression_ratio_n"]
    + VOCABULARY_UTILIZATION_WT * df_scores["vocabulary_utilization_n"]
    + UNK_TOKEN_RATIO_WT * df_scores["unknown_token_ratio_n"]
)

df_scores = df_scores.sort_values("score", ascending=False).reset_index(drop=True)
```

---

## 🪛 Automatic Elbow Finder

To select the optimal vocabulary size, an elbow detection approach was implemented:

```python
def find_elbow_from_df(df, x_col="vocab_size", y_col="score"):
    df_sorted = df.sort_values(x_col).reset_index(drop=True)

    x = df_sorted[x_col].values.astype(int)
    y = df_sorted[y_col].values.astype(float)

    # Normalize
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    distances = []
    for i in range(len(x_norm)):
        p = np.array([x_norm[i], y_norm[i]])
        distance = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(distance)

    elbow_index = np.argmax(distances)
    return int(df_sorted.loc[elbow_index+1, x_col])
```

This method helped confirm that 64K was a strong candidate.

---

## 📦 Embedding Model Development

At the moment you have:

* A tokenizer trained on metal market text
* Evaluation workflow for tokenizer configurations
* Embedding learning code in progress *focused initially on Aluminum*

This work will eventually support RAG systems where:

* documents are encoded into vectors
* efficient retrieval and semantic search is possible
* augmenting LLM responses with domain data becomes feasible

(General RAG workflows combine **retrieval + generation** to improve relevance over domain text) ([GitHub][1])

---

## 🛠 Structure Overview

```
metal-mind/
├── 📁 artifacts
│   ├── 📄 metal-mind-tokenizer.model
│   ├── 📄 metal-mind-tokenizer.vocab
│   └── 📄 tokenizer_ablation_results.csv
├── 📁 core
│   ├── 🐍 __init__.py
│   ├── 🐍 evaluate.py
│   └── 🐍 logging.py
├── 📁 development
│   └── 🐍 train-tokenizer.py
├── 📁 logs
├── ⚙️ .gitignore
├── 📄 LICENSE
├── 📝 README.md
├── ⚙️ pyproject.toml
└── 📄 requirements.txt
```

---

## 🧪 Usage

**1. Tokenizer Inference**

```python
from sentencepiece import SentencePieceProcessor

sp = SentencePieceProcessor()
sp.load("/artifacts/metal-mind-tokenizer.model")

tokens = sp.encode("LME Aluminum price moved up", out_type=str)
```

---

## 📌 Notes

* This README does **not** describe future features not yet implemented.
* Data cannot be shared publicly due to confidentiality.
* Metal focus currently on **LME Aluminum only**.

---

## 📜 License

This project is licensed under the **MIT License**.