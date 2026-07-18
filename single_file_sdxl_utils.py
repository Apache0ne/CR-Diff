#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared loading helpers for complete single-file SDXL checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)


def resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(table)}")
    return table[name]


def set_offline_mode(local_files_only: bool) -> None:
    """Keep hub/transformers cached offline flags consistent with the CLI choice."""
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.HF_HUB_OFFLINE = bool(local_files_only)
    except Exception:
        pass

    try:
        import transformers.utils.hub as transformers_hub

        if hasattr(transformers_hub, "_is_offline_mode"):
            transformers_hub._is_offline_mode = bool(local_files_only)
    except Exception:
        pass


def configure_scheduler(pipe: StableDiffusionXLPipeline, preset: str) -> None:
    """Install a reproducible scheduler preset without changing model weights."""
    if preset == "checkpoint":
        return
    if preset == "dpmpp-sde-normal":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config,
            solver_order=2,
            algorithm_type="sde-dpmsolver++",
            solver_type="midpoint",
            lower_order_final=True,
            thresholding=False,
            use_karras_sigmas=False,
            use_exponential_sigmas=False,
            use_beta_sigmas=False,
            final_sigmas_type="zero",
            steps_offset=0,
        )
        return
    if preset == "ddim-trailing":
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        return
    if preset == "euler-trailing":
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        return
    raise ValueError(f"Unknown scheduler preset: {preset}")


def load_sdxl_pipeline(
    *,
    device: str,
    model_path: Optional[str] = None,
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    model_dtype: str = "float16",
    scheduler: str = "checkpoint",
    local_files_only: bool = False,
    model_config: Optional[str] = None,
    original_config_file: Optional[str] = None,
) -> StableDiffusionXLPipeline:
    """Load either a complete local SDXL file or a Diffusers repository."""
    set_offline_mode(bool(local_files_only))
    dtype = resolve_dtype(model_dtype)

    if model_path:
        path = Path(os.path.expandvars(os.path.expanduser(model_path))).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Single-file SDXL checkpoint not found: {path}")
        if path.suffix.lower() not in {".safetensors", ".ckpt"}:
            raise ValueError(f"Expected .safetensors or .ckpt, got: {path.suffix}")

        kwargs = {
            "torch_dtype": dtype,
            "local_files_only": bool(local_files_only),
        }
        if model_config:
            kwargs["config"] = model_config
        if original_config_file:
            kwargs["original_config_file"] = original_config_file
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            kwargs["token"] = token

        pipe = StableDiffusionXLPipeline.from_single_file(str(path), **kwargs)
        source = str(path)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=bool(local_files_only),
        )
        source = model_id

    missing = [
        name
        for name in ("unet", "text_encoder", "text_encoder_2", "vae")
        if getattr(pipe, name, None) is None
    ]
    if missing:
        raise RuntimeError(
            f"{source} did not produce a complete SDXL pipeline; missing: {', '.join(missing)}"
        )

    configure_scheduler(pipe, scheduler)
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)
