#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run embedded-image ImageReward rescoring under modern Transformers.

Clears offline flags, restores tokenizer APIs removed in Transformers 5,
pre-downloads ImageReward assets, then invokes the existing self-contained
HTML rescorer. No SDXL model is loaded or run.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List


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


def patch_tokenizer_compatibility() -> Dict[str, str]:
    """Restore tokenizer properties expected by ImageReward's bundled BLIP."""
    from transformers import BertTokenizer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    patched: Dict[str, str] = {}

    def additional_special_tokens_ids(self) -> List[int]:
        tokens: Any = []
        try:
            tokens = self.special_tokens_map.get("additional_special_tokens", [])
        except Exception:
            tokens = getattr(self, "_additional_special_tokens", []) or []

        if tokens is None:
            tokens = []
        if isinstance(tokens, str):
            tokens = [tokens]

        ids = self.convert_tokens_to_ids(list(tokens))
        if isinstance(ids, int):
            ids = [ids]
        return [int(value) for value in ids]

    for cls in (PreTrainedTokenizerBase, BertTokenizer):
        if not hasattr(cls, "additional_special_tokens_ids"):
            setattr(cls, "additional_special_tokens_ids", property(additional_special_tokens_ids))
            patched[f"{cls.__module__}.{cls.__name__}.additional_special_tokens_ids"] = (
                "local computed compatibility property"
            )

    return patched


def ensure_imagereward_assets() -> Dict[str, str]:
    from huggingface_hub import hf_hub_download

    image_reward_dir = Path.home() / ".cache" / "ImageReward"
    image_reward_dir.mkdir(parents=True, exist_ok=True)

    assets = {}
    for filename in ("ImageReward.pt", "med_config.json"):
        path = hf_hub_download(
            repo_id="THUDM/ImageReward",
            filename=filename,
            local_dir=str(image_reward_dir),
        )
        assets[filename] = str(Path(path).resolve())

    # Warm the tokenizer cache while online mode is definitely active.
    from transformers import BertTokenizer
    BertTokenizer.from_pretrained("bert-base-uncased")
    return assets


def main() -> None:
    force_online_mode()
    tokenizer_patch = patch_tokenizer_compatibility()
    assets = ensure_imagereward_assets()
    print(f"Tokenizer compatibility patch: {tokenizer_patch}", flush=True)
    print(f"ImageReward assets ready: {assets}", flush=True)

    from rescore_pure_crdiff_embedded_v2 import main as score_main
    score_main()


if __name__ == "__main__":
    main()
