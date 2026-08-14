"""Paired sentiment-model artifacts: native Keras model + tokenizer + version manifest.

Deployable unit under review/models/:
  - sentiment_model.keras
  - tokenizer.json
  - model_version.json  (SHA-256 of the two files + training metadata)

The Keras model is never serialized with pickle.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

MODEL_FILENAME = 'sentiment_model.keras'
TOKENIZER_FILENAME = 'tokenizer.json'
VERSION_FILENAME = 'model_version.json'

# Legacy pickle artifacts (kept on disk during migration; not used for serving).
LEGACY_MODEL_PICKLE = 'model.pkl'
LEGACY_TOKENIZER_PICKLE = 'tokenizer.pkl'

NUM_WORDS = 5000
OOV_TOKEN = '<OOV>'
MAXLEN = 200


class ArtifactIntegrityError(Exception):
    """Raised when the model/tokenizer pair is missing or mismatched."""


def default_artifact_dir(base_dir: str | os.PathLike[str] | None = None) -> str:
    if base_dir is None:
        from django.conf import settings

        base_dir = settings.BASE_DIR
    return os.path.join(str(base_dir), 'review', 'models')


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_paths(artifact_dir: str) -> dict[str, str]:
    return {
        'model': os.path.join(artifact_dir, MODEL_FILENAME),
        'tokenizer': os.path.join(artifact_dir, TOKENIZER_FILENAME),
        'version': os.path.join(artifact_dir, VERSION_FILENAME),
    }


def build_version_manifest(
    artifact_dir: str,
    *,
    tensorflow_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = artifact_paths(artifact_dir)
    for key in ('model', 'tokenizer'):
        if not os.path.isfile(paths[key]):
            raise FileNotFoundError(f'Missing artifact for versioning: {paths[key]}')

    model_hash = sha256_file(paths['model'])
    tokenizer_hash = sha256_file(paths['tokenizer'])
    version_id = f'{model_hash[:12]}-{tokenizer_hash[:12]}'

    if tensorflow_version is None:
        try:
            import tensorflow as tf

            tensorflow_version = tf.__version__
        except Exception:  # pragma: no cover - optional at write time
            tensorflow_version = 'unknown'

    manifest: dict[str, Any] = {
        'version_id': version_id,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'artifacts': {
            'model': {
                'filename': MODEL_FILENAME,
                'sha256': model_hash,
            },
            'tokenizer': {
                'filename': TOKENIZER_FILENAME,
                'sha256': tokenizer_hash,
            },
        },
        'training': {
            'num_words': NUM_WORDS,
            'oov_token': OOV_TOKEN,
            'maxlen': MAXLEN,
            'framework': 'tensorflow',
            'tensorflow_version': tensorflow_version,
            'model_format': 'keras_v3',
            'tokenizer_format': 'keras_tokenizer_json',
        },
    }
    if extra:
        manifest['extra'] = extra
    return manifest


def write_version_manifest(artifact_dir: str, manifest: dict[str, Any] | None = None) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    paths = artifact_paths(artifact_dir)
    if manifest is None:
        manifest = build_version_manifest(artifact_dir)
    with open(paths['version'], 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return paths['version']


def read_version_manifest(artifact_dir: str) -> dict[str, Any]:
    paths = artifact_paths(artifact_dir)
    if not os.path.isfile(paths['version']):
        raise ArtifactIntegrityError(
            f'Missing {VERSION_FILENAME}; deploy model+tokenizer+manifest as one unit.'
        )
    with open(paths['version'], encoding='utf-8') as handle:
        return json.load(handle)


def verify_artifact_pair(
    artifact_dir: str,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ensure on-disk model/tokenizer match the version manifest checksums."""
    paths = artifact_paths(artifact_dir)
    if manifest is None:
        manifest = read_version_manifest(artifact_dir)

    missing = [label for label in ('model', 'tokenizer') if not os.path.isfile(paths[label])]
    if missing:
        raise FileNotFoundError(
            'Missing sentiment artifact(s): '
            + ', '.join(paths[label] for label in missing)
        )

    expected_artifacts = manifest.get('artifacts') or {}
    errors: list[str] = []
    for label in ('model', 'tokenizer'):
        expected = (expected_artifacts.get(label) or {}).get('sha256')
        if not expected:
            errors.append(f'manifest missing sha256 for {label}')
            continue
        actual = sha256_file(paths[label])
        if actual.lower() != str(expected).lower():
            errors.append(
                f'{label} checksum mismatch '
                f'(expected {expected[:16]}…, got {actual[:16]}…)'
            )

    if errors:
        version_id = manifest.get('version_id', 'unknown')
        raise ArtifactIntegrityError(
            'Incompatible model/tokenizer pair for version_id='
            f'{version_id}: ' + '; '.join(errors)
        )

    return manifest


def save_paired_artifacts(
    model,
    tokenizer,
    artifact_dir: str,
    *,
    also_write_legacy_pickles: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Persist .keras model + tokenizer.json + model_version.json as one unit."""
    import pickle

    os.makedirs(artifact_dir, exist_ok=True)
    paths = artifact_paths(artifact_dir)

    model.save(paths['model'])
    with open(paths['tokenizer'], 'w', encoding='utf-8') as handle:
        handle.write(tokenizer.to_json())

    manifest = build_version_manifest(artifact_dir, extra=extra)
    write_version_manifest(artifact_dir, manifest)

    if also_write_legacy_pickles:
        # Optional local copies only; serving no longer loads these.
        with open(os.path.join(artifact_dir, LEGACY_MODEL_PICKLE), 'wb') as handle:
            pickle.dump(model, handle)
        with open(os.path.join(artifact_dir, LEGACY_TOKENIZER_PICKLE), 'wb') as handle:
            pickle.dump(tokenizer, handle)

    logger.info(
        'Saved paired sentiment artifacts version_id=%s dir=%s',
        manifest['version_id'],
        artifact_dir,
    )
    return {
        'model': paths['model'],
        'tokenizer': paths['tokenizer'],
        'version': paths['version'],
        'version_id': manifest['version_id'],
    }


def load_paired_artifacts(artifact_dir: str | None = None):
    """Load and verify the deployable model/tokenizer pair.

    Returns (model, tokenizer, pad_sequences, manifest).
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    if artifact_dir is None:
        artifact_dir = default_artifact_dir()
    paths = artifact_paths(artifact_dir)

    manifest = verify_artifact_pair(artifact_dir)
    logger.info(
        'Loading sentiment artifacts version_id=%s from %s',
        manifest.get('version_id'),
        artifact_dir,
    )

    model = tf.keras.models.load_model(paths['model'])
    with open(paths['tokenizer'], encoding='utf-8') as handle:
        tokenizer = tokenizer_from_json(handle.read())

    return model, tokenizer, pad_sequences, manifest
