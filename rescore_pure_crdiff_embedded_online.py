#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the embedded-image CR-Diff rescorer with Hugging Face online mode forced.

This wrapper clears offline environment variables before importing huggingface_hub,
pre-downloads ImageReward.pt into ImageReward's expected cache directory, then
runs rescore_pure_crdiff_embedded_v2.main(). No SDXL model is loaded.
"""
from __future__ import annotations

import os
from pathlib import Path


def force_online_mode() -> None:
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "DIFFUSERS_OFFLINE",
    ):
        os.environ.pop(name, None)

    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.HF_HUB_OFFLINE = False
    except Exception:
        pass

    try:
        import transformers.utils.hub as transformers_hub

        if hasattr(transformers_hub, "_is_offline_mode"):
            transformers_hub._is_offline_mode = False
    except Exception:
        pass


def ensure_imagereward_weights() -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir = Path.home() / ".cache" / "ImageReward"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="THUDM/ImageReward",
        filename="ImageReward.pt",
        local_dir=str(cache_dir),
    )
    resolved = Path(path).resolve()
    print(f"ImageReward weights ready: {resolved}", flush=True)
    return resolved


def main() -> None:
    force_online_mode()
    ensure_imagereward_weights()
    from rescore_pure_crdiff_embedded_v2 import main as score_main

    score_main()


if __name__ == "__main__":
    main()
