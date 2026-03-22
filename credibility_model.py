
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


class TextStatsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        series = pd.Series(X).fillna("").astype(str)

        char_len = series.str.len().astype(float)
        word_count = series.str.split().str.len().astype(float)
        exclam_count = series.str.count("!").astype(float)
        question_count = series.str.count(r"\?").astype(float)

        token_lists = series.str.split()
        unique_ratio = []
        avg_word_len = []
        digit_ratio = []
        uppercase_ratio = []

        for text, tokens in zip(series.tolist(), token_lists.tolist()):
            if tokens:
                unique_ratio.append(len(set(tokens)) / len(tokens))
                avg_word_len.append(sum(len(t) for t in tokens) / len(tokens))
            else:
                unique_ratio.append(0.0)
                avg_word_len.append(0.0)

            if text:
                digit_ratio.append(sum(ch.isdigit() for ch in text) / len(text))
                letters = [ch for ch in text if ch.isalpha()]
                if letters:
                    uppercase_ratio.append(sum(ch.isupper() for ch in letters) / len(letters))
                else:
                    uppercase_ratio.append(0.0)
            else:
                digit_ratio.append(0.0)
                uppercase_ratio.append(0.0)

        arr = np.column_stack(
            [
                char_len.to_numpy(),
                word_count.to_numpy(),
                exclam_count.to_numpy(),
                question_count.to_numpy(),
                np.asarray(unique_ratio, dtype=float),
                np.asarray(avg_word_len, dtype=float),
                np.asarray(digit_ratio, dtype=float),
                np.asarray(uppercase_ratio, dtype=float),
            ]
        )
        return sparse.csr_matrix(arr)

try:
    setattr(sys.modules.get("__main__"), "TextStatsTransformer", TextStatsTransformer)
except Exception:
    pass


@dataclass
class CredibilityConfig:
    duplicate_weight: float = 0.30
    generic_weight: float = 0.22
    mismatch_weight: float = 0.22
    unverified_weight: float = 0.08
    repetitive_weight: float = 0.10
    very_short_weight: float = 0.12
    medium_threshold: float = 0.30
    high_threshold: float = 0.60
    suspicious_threshold: float = 0.45


class ReviewCredibilityScorer:

    def __init__(self, sentiment_bundle: dict, config: Optional[CredibilityConfig] = None):
        self.sentiment_bundle = sentiment_bundle
        self.sentiment_model = sentiment_bundle.get("model") or sentiment_bundle.get("pipeline")
        self.label_encoder = sentiment_bundle["label_encoder"]
        self.sentiment_label_names = list(sentiment_bundle["label_names"])
        self.config = config or CredibilityConfig()
        self.generic_patterns = [
            r"\bjust as described\b",
            r"\bfive stars\b",
            r"\bfour stars\b",
            r"\bgood\b",
            r"\bgreat\b",
            r"\bnice\b",
            r"\bworks great\b",
            r"\bworks well\b",
            r"\blove it\b",
            r"\bexcellent\b",
            r"\bperfect\b",
            r"\bawesome\b",
            r"\brecommended\b",
            r"\bso far so good\b",
            r"\bexactly what i needed\b",
        ]

    @classmethod
    def from_sentiment_bundle_path(cls, sentiment_bundle_path: str, config: Optional[CredibilityConfig] = None):
        bundle = joblib.load(sentiment_bundle_path)
        return cls(bundle, config=config)

    def _normalize_verified(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        return text in {"true", "1", "yes", "y"}

    def _rating_to_sentiment(self, overall) -> Optional[str]:
        try:
            rating = float(overall)
        except Exception:
            return None
        if rating <= 2:
            return "negative"
        if rating == 3:
            return "neutral"
        return "positive"

    def _token_stats(self, text: str):
        tokens = text.split()
        word_count = len(tokens)
        if word_count == 0:
            return {
                "word_count": 0,
                "char_count": 0,
                "unique_ratio": 0.0,
                "repetition_ratio": 0.0,
            }
        counts = Counter(tokens)
        repeated_tokens = sum(c for c in counts.values() if c > 1)
        return {
            "word_count": word_count,
            "char_count": len(text),
            "unique_ratio": len(set(tokens)) / word_count,
            "repetition_ratio": repeated_tokens / word_count,
        }

    def _generic_short_flag(self, text: str, word_count: int) -> bool:
        if word_count > 8:
            return False
        return any(re.search(pattern, text) for pattern in self.generic_patterns)

    def _duplicate_text_flags(self, clean_text_series: pd.Series) -> pd.Series:
        counts = clean_text_series.value_counts(dropna=False)
        return clean_text_series.map(counts).fillna(0).astype(int) > 1

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"clean_text", "category"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        work = df.copy()
        work["clean_text"] = work["clean_text"].fillna("").astype(str).str.strip()
        work["category"] = work["category"].fillna("Unknown").astype(str)
        if "text" not in work.columns:
            work["text"] = work["clean_text"]
        if "overall" not in work.columns:
            work["overall"] = np.nan
        if "verified" not in work.columns:
            work["verified"] = False

        pred_ids = self.sentiment_model.predict(work[["clean_text", "category"]])
        pred_sentiment = self.label_encoder.inverse_transform(pred_ids)
        work["predicted_sentiment"] = pred_sentiment

        work["duplicate_or_repeated_text"] = self._duplicate_text_flags(work["clean_text"])

        stats = work["clean_text"].apply(self._token_stats)
        work["word_count"] = stats.apply(lambda x: x["word_count"])
        work["char_count"] = stats.apply(lambda x: x["char_count"])
        work["unique_ratio"] = stats.apply(lambda x: x["unique_ratio"])
        work["repetition_ratio"] = stats.apply(lambda x: x["repetition_ratio"])

        work["very_short_text"] = work["word_count"] <= 3
        work["generic_short_text"] = [
            self._generic_short_flag(text, wc)
            for text, wc in zip(work["clean_text"].tolist(), work["word_count"].tolist())
        ]
        work["high_repetition"] = (work["repetition_ratio"] >= 0.40) | (work["unique_ratio"] <= 0.55)
        work["verified_bool"] = work["verified"].apply(self._normalize_verified)
        work["unverified_flag"] = ~work["verified_bool"]

        inferred_rating_sentiment = work["overall"].apply(self._rating_to_sentiment)
        work["rating_text_mismatch"] = [
            (rs is not None) and (
                (rs == "negative" and ps == "positive")
                or (rs == "positive" and ps == "negative")
                or (rs == "neutral" and ps in {"negative", "positive"})
            )
            for rs, ps in zip(inferred_rating_sentiment.tolist(), work["predicted_sentiment"].tolist())
        ]

        score = (
            work["duplicate_or_repeated_text"].astype(float) * self.config.duplicate_weight
            + work["generic_short_text"].astype(float) * self.config.generic_weight
            + work["rating_text_mismatch"].astype(float) * self.config.mismatch_weight
            + work["unverified_flag"].astype(float) * self.config.unverified_weight
            + work["high_repetition"].astype(float) * self.config.repetitive_weight
            + work["very_short_text"].astype(float) * self.config.very_short_weight
        )
        work["low_credibility_score"] = score.clip(0.0, 1.0).round(4)

        def score_to_label(value: float) -> str:
            if value >= self.config.high_threshold:
                return "high"
            if value >= self.config.medium_threshold:
                return "medium"
            return "low"

        work["low_credibility_label"] = work["low_credibility_score"].apply(score_to_label)
        work["suspicious_label"] = np.where(
            work["low_credibility_score"] >= self.config.suspicious_threshold,
            "suspicious",
            "not_suspicious",
        )

        reason_cols = [
            ("duplicate_or_repeated_text", "duplicate_or_repeated_text"),
            ("generic_short_text", "very_short_generic_text"),
            ("rating_text_mismatch", "rating_text_mismatch"),
            ("unverified_flag", "not_verified_purchase"),
            ("high_repetition", "high_token_repetition"),
            ("very_short_text", "very_short_text"),
        ]

        reasons: List[str] = []
        for _, row in work.iterrows():
            active = [name for col, name in reason_cols if bool(row[col])]
            reasons.append("; ".join(active) if active else "none")
        work["suspicious_reasons"] = reasons

        return work

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.predict_dataframe(df)
