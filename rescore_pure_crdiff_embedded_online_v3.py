#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score embedded CR-Diff report images with ImageReward on Transformers 5.

This wrapper clears offline flags, restores the removed modeling utility imports,
patches ImageReward's bundled BLIP tokenizer initializer to obtain the [ENC] ID
by direct token conversion, warms the required caches, and then invokes the
existing self-contained HTML rescorer. It never loads or executes SDXL.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


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


def ensure_imagereward_assets() -> Dict[str, str]:
    from huggingface_hub import hf_hub_download

    root = Path.home() / ".cache" / "ImageReward"
    root.mkdir(parents=True, exist_ok=True)
    assets: Dict[str, str] = {}
    for filename in ("ImageReward.pt", "med_config.json"):
        path = hf_hub_download(
            repo_id="THUDM/ImageReward",
            filename=filename,
            local_dir=str(root),
        )
        assets[filename] = str(Path(path).resolve())
    return assets


def patch_imagereward_blip_tokenizer() -> Dict[str, str]:
    """Replace BLIP's legacy tokenizer initializer with a TF5-safe version."""
    from transformers import BertTokenizer

    # These legacy utility names must exist before ImageReward's BLIP modules import.
    from rescore_pure_crdiff_embedded_v2 import patch_transformers_for_imagereward
    modeling_patch = patch_transformers_for_imagereward()

    import ImageReward.models.BLIP.blip as blip_module
    import ImageReward.models.BLIP.blip_pretrain as blip_pretrain_module

    def init_tokenizer_tf5():
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})

        enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
        if isinstance(enc_token_id, list):
            if not enc_token_id:
                raise RuntimeError("BertTokenizer returned no ID for [ENC]")
            enc_token_id = enc_token_id[0]
        enc_token_id = int(enc_token_id)

        if enc_token_id < 0 or enc_token_id == tokenizer.unk_token_id:
            raise RuntimeError(
                f"[ENC] was not registered correctly; token ID={enc_token_id}, "
                f"unk_token_id={tokenizer.unk_token_id}"
            )

        tokenizer.enc_token_id = enc_token_id
        return tokenizer

    # blip_pretrain imports init_tokenizer into its own module namespace, so both
    # references must be replaced before BLIP_Pretrain is instantiated.
    blip_module.init_tokenizer = init_tokenizer_tf5
    blip_pretrain_module.init_tokenizer = init_tokenizer_tf5

    result = dict(modeling_patch)
    result["ImageReward.models.BLIP.blip.init_tokenizer"] = (
        "direct [ENC] token-to-ID compatibility implementation"
    )
    result["ImageReward.models.BLIP.blip_pretrain.init_tokenizer"] = (
        "direct [ENC] token-to-ID compatibility implementation"
    )
    return result


def main() -> None:
    force_online_mode()
    assets = ensure_imagereward_assets()
    patch = patch_imagereward_blip_tokenizer()

    print(f"ImageReward assets ready: {assets}", flush=True)
    print(f"ImageReward compatibility patch: {patch}", flush=True)

    from rescore_pure_crdiff_embedded_v2 import main as score_main
    score_main()


if __name__ == "__main__":
    main()
