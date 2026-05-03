"""
api/startup.py — Downloads model artifacts from HF Hub at Space startup.

Only downloads if artifacts are not already present in the container,
so warm restarts (without full container teardown) skip re-downloading.

Required HF Space Secrets:
  HF_TOKEN           — read-access token for the private model repo
  ARTIFACTS_REPO_ID  — e.g. "gh0stFREAK/airsight-artifacts"
"""

from __future__ import annotations

import os
from pathlib import Path


def download_artifacts_if_needed(artifacts_dir: Path) -> None:
    """
    Downloads best_model.keras, scaler.pkl, and vectorstore/ from the
    private HF model repo into artifacts_dir — only if they don't exist.
    """
    model_path      = artifacts_dir / "best_model.keras"
    scaler_path     = artifacts_dir / "scaler.pkl"
    vectorstore_dir = artifacts_dir / "vectorstore"

    if model_path.exists() and scaler_path.exists() and vectorstore_dir.exists():
        print("  ✅  Artifacts already present — skipping download.")
        return

    hf_token = os.getenv("HF_TOKEN")
    repo_id  = os.getenv("ARTIFACTS_REPO_ID", "gh0stFREAK/airsight-artifacts")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN secret is not set. Cannot download model artifacts from "
            f"'{repo_id}'. Add HF_TOKEN to your HF Space Secrets."
        )

    print(f"  ⬇  Downloading artifacts from: {repo_id}")
    print("      First boot — this takes ~60 s for the 168 MB model …")

    try:
        from huggingface_hub import snapshot_download  # lazy import

        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            local_dir=str(artifacts_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md", ".gitattributes"],
        )
        print(f"  ✅  Artifacts downloaded to: {artifacts_dir}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download artifacts from '{repo_id}': {exc}"
        ) from exc
