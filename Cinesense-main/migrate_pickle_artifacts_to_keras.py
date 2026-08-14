"""One-shot migration: convert legacy pickle artifacts to paired Keras + version manifest.

Does not retrain. Loads existing model.pkl + tokenizer.pkl from the same directory,
writes sentiment_model.keras + tokenizer.json + model_version.json, and leaves the
legacy pickle files in place for rollback.
"""
from __future__ import annotations

import os
import pickle
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cinesense")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from review.ml_artifacts import (  # noqa: E402
    LEGACY_MODEL_PICKLE,
    LEGACY_TOKENIZER_PICKLE,
    default_artifact_dir,
    load_paired_artifacts,
    save_paired_artifacts,
)


def main() -> int:
    artifact_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cinesense",
        "review",
        "models",
    )
    model_pkl = os.path.join(artifact_dir, LEGACY_MODEL_PICKLE)
    tokenizer_pkl = os.path.join(artifact_dir, LEGACY_TOKENIZER_PICKLE)
    if not os.path.isfile(model_pkl) or not os.path.isfile(tokenizer_pkl):
        print(
            f"ERROR: expected legacy pair at {model_pkl} and {tokenizer_pkl}",
            file=sys.stderr,
        )
        return 1

    print(f"Loading legacy pickles from {artifact_dir}")
    with open(model_pkl, "rb") as handle:
        model = pickle.load(handle)
    with open(tokenizer_pkl, "rb") as handle:
        tokenizer = pickle.load(handle)

    saved = save_paired_artifacts(
        model,
        tokenizer,
        artifact_dir,
        also_write_legacy_pickles=False,
        extra={
            "source": "migrate_pickle_artifacts_to_keras.py",
            "migrated_from": [LEGACY_MODEL_PICKLE, LEGACY_TOKENIZER_PICKLE],
            "note": "Weights preserved from pickle; not a fresh training run.",
        },
    )
    print("Wrote paired artifacts:")
    for key in ("model", "tokenizer", "version"):
        print(f"  {saved[key]}")
    print(f"version_id={saved['version_id']}")

    # Verify the new pair loads and checksums match.
    loaded_model, loaded_tok, pad_sequences, manifest = load_paired_artifacts(artifact_dir)
    sample = "This movie was fantastic and thrilling."
    seq = pad_sequences(loaded_tok.texts_to_sequences([sample]), maxlen=200)
    pred = float(loaded_model.predict(seq, verbose=0)[0][0])
    print(f"Verification predict={pred:.6f} version_id={manifest['version_id']}")
    print(f"Legacy pickles retained at {artifact_dir} (not used by serving).")
    print(f"default_artifact_dir probe={default_artifact_dir(os.path.join(APP_DIR))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
