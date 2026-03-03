import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import sentencepiece as spm
from typing import Final, List
from core.logging import LoggerFactory
from core.evaluate import EvaluateTokenize

logger = LoggerFactory().get_logger(__name__)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
RAW_CORPUS: Final[Path] = BASE_DIR / "dataset" / "lme-al-news-corpus.json"
RAW_CORPUS_TXT: Final[Path] = BASE_DIR / "dataset" / "lme-al-news-corpus.txt"
ARTIFACT_DIR: Final[Path] = BASE_DIR / "artifacts"
MODEL_PREFIX: Final[str] = "metal-mind-tokenizer"

VOCAB_SIZES: Final[List[int]] = [8000, 16000, 32000, 64000, 96000, 110000]

UNK_TOKEN_RATIO_WT: float = 0.1
FRAGMENTATION_RATE_WT: float = 0.4
COMPRESSION_RATIO_WT: float = 0.4
VOCABULARY_UTILIZATION_WT: float = 0.1

# Remove existing artifacts directory completely
if ARTIFACT_DIR.exists():
    shutil.rmtree(ARTIFACT_DIR)
    logger.info("Artifact directory removed")

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
logger.info("Artifact directory ready: %s", ARTIFACT_DIR)


def normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Safe min-max normalization."""
    min_val, max_val = series.min(), series.max()

    if max_val == min_val:
        return pd.Series([1.0] * len(series))

    if higher_is_better:
        return (series - min_val) / (max_val - min_val)
    else:
        return (max_val - series) / (max_val - min_val)


def find_elbow_from_df(df, x_col="vocab_size", y_col="score"):
    df_sorted = df.sort_values(x_col).reset_index(drop=True)

    x = df_sorted[x_col].values.astype(int)
    y = df_sorted[y_col].values.astype(float)

    # Normalize to [0,1]
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


if __name__ == "__main__":

    logger.info("==== Unigram Tokenizer Training Pipeline Started ====")

    logger.info("Loading corpus")
    df = pd.read_json(RAW_CORPUS)["article"].tolist()
    train_corpus = [article.lower() for article in df]

    RAW_CORPUS_TXT.write_text("\n".join(train_corpus), encoding="utf-8")

    results = []

    logger.info("Running vocabulary size ablation")

    for size in VOCAB_SIZES:
        model_prefix_path = ARTIFACT_DIR / f"{MODEL_PREFIX}_{size}"

        spm.SentencePieceTrainer.train(
            input=str(RAW_CORPUS_TXT),
            model_prefix=str(model_prefix_path),
            model_type="unigram",
            vocab_size=size,
            character_coverage=1.0,
            minloglevel=2,
        )

        logger.info(f"Tokenizer trained: vocab_size={size}")

        evaluator = EvaluateTokenize(
            model_path=str(model_prefix_path) + ".model", input_text_list=train_corpus
        )

        results.append(
            {
                "vocab_size": size,
                "unknown_token_ratio": evaluator.unk_token_ratio(),
                "fragmentation_ratio": evaluator.fragmentation_rate(),
                "compression_ratio": evaluator.compression_ratio(),
                "vocabulary_utilization": evaluator.vocab_utilization(),
            }
        )

    df_scores = pd.DataFrame(results)

    logger.info("Raw evaluation metrics:")
    logger.info(df_scores)

    logger.info("Normalizing metrics")

    df_scores["unknown_token_ratio_n"] = normalize(
        df_scores["unknown_token_ratio"], higher_is_better=False
    )

    df_scores["fragmentation_ratio_n"] = normalize(
        df_scores["fragmentation_ratio"], higher_is_better=False
    )

    df_scores["compression_ratio_n"] = normalize(
        df_scores["compression_ratio"], higher_is_better=False
    )

    df_scores["vocabulary_utilization_n"] = normalize(
        df_scores["vocabulary_utilization"], higher_is_better=True
    )

    df_scores["score"] = (
        FRAGMENTATION_RATE_WT * df_scores["fragmentation_ratio_n"]
        + COMPRESSION_RATIO_WT * df_scores["compression_ratio_n"]
        + VOCABULARY_UTILIZATION_WT * df_scores["vocabulary_utilization_n"]
        + UNK_TOKEN_RATIO_WT * df_scores["unknown_token_ratio_n"]
    )

    df_scores = df_scores.sort_values("score", ascending=False).reset_index(drop=True)

    logger.info("Ranked vocabulary sizes:")
    logger.info(df_scores[["vocab_size", "score"]])

    best_vocab = find_elbow_from_df(df_scores)
    logger.info(f"Selected vocabulary size: {best_vocab}")

    logger.info("Cleaning intermediate tokenizer artifacts")

    for size in VOCAB_SIZES:
        prefix = ARTIFACT_DIR / f"{MODEL_PREFIX}_{size}"

        model_file = prefix.with_suffix(".model")
        vocab_file = prefix.with_suffix(".vocab")

        if model_file.exists():
            model_file.unlink()

        if vocab_file.exists():
            vocab_file.unlink()

    logger.info("Intermediate artifacts removed")

    final_model_path = ARTIFACT_DIR / MODEL_PREFIX

    spm.SentencePieceTrainer.train(
        input=str(RAW_CORPUS_TXT),
        model_prefix=str(final_model_path),
        model_type="unigram",
        vocab_size=best_vocab,
        character_coverage=1.0,
        minloglevel=2,
    )

    logger.info("Final tokenizer trained")

    df_scores.to_csv(ARTIFACT_DIR / "tokenizer_ablation_results.csv", index=False)

    logger.info("Ablation results saved")
    logger.info("==== Tokenizer Training Pipeline Completed ====")
