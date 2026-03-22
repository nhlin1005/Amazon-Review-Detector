
import json
from pathlib import Path

import joblib

from credibility_model import CredibilityConfig, ReviewCredibilityScorer

SENTIMENT_BUNDLE_PATH = r"dataset\best_single_review_model_v3.joblib"
OUTPUT_MODEL_PATH = r"dataset\combined_review_credibility_model.joblib"


def main():
    sentiment_path = Path(SENTIMENT_BUNDLE_PATH)
    if not sentiment_path.exists():
        raise FileNotFoundError(f"Sentiment bundle not found: {sentiment_path}")

    config = CredibilityConfig(
        duplicate_weight=0.30,
        generic_weight=0.22,
        mismatch_weight=0.22,
        unverified_weight=0.08,
        repetitive_weight=0.10,
        very_short_weight=0.12,
        medium_threshold=0.30,
        high_threshold=0.60,
        suspicious_threshold=0.45,
    )

    scorer = ReviewCredibilityScorer.from_sentiment_bundle_path(
        str(sentiment_path),
        config=config,
    )

    output_bundle = {
        "model_type": "combined_sentiment_plus_credibility_scorer",
        "description": (
            "One packaged model object that combines the existing sentiment classifier "
            "with transparent suspiciousness / low-credibility scoring rules."
        ),
        "scorer": scorer,
        "config": config.__dict__,
        "source_sentiment_bundle": str(sentiment_path),
    }

    output_path = Path(OUTPUT_MODEL_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output_bundle, output_path)

    print(f"Saved combined credibility model to: {output_path}")
    print("Config:")
    print(json.dumps(config.__dict__, indent=2))


if __name__ == "__main__":
    main()
