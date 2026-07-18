"""POA_inference.py — single-image Pruned Output Amplification inference.

POA mixes dense and pruned UNet outputs:
    D_w = (1 - w) * D_dense + w * D_pruned

This version supports complete local single-file SDXL checkpoints.
"""

import argparse
import copy
import json
import os

import torch
import torch.nn as nn
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline, FluxPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

from pruning_utils import apply_pruning_to_pipe
from single_file_sdxl_utils import load_sdxl_pipeline

MODEL_MAP = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}


def load_base_pipeline(
    model_name: str,
    *,
    device: str = "cuda:0",
    model_path: str | None = None,
    model_dtype: str = "float16",
    scheduler: str = "checkpoint",
    local_files_only: bool = False,
    model_config: str | None = None,
    original_config_file: str | None = None,
):
    if model_path or "sdxl" in model_name:
        return load_sdxl_pipeline(
            device=device,
            model_path=model_path,
            model_id=MODEL_MAP.get(model_name, model_name),
            model_dtype=model_dtype,
            scheduler=scheduler,
            local_files_only=local_files_only,
            model_config=model_config,
            original_config_file=original_config_file,
        )

    if "sd3" in model_name:
        pipe_class = StableDiffusion3Pipeline
    elif "flux" in model_name:
        pipe_class = FluxPipeline
    else:
        pipe_class = StableDiffusionPipeline

    return pipe_class.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        local_files_only=local_files_only,
    ).to(device)


class POAUnetWrapper(nn.Module):
    """D_w = (1-w) * dense + w * pruned."""

    def __init__(self, unet_good, unet_bad, guidance_weight: float):
        super().__init__()
        self.unet_good = unet_good
        self.unet_bad = unet_bad
        self.w = float(guidance_weight)
        self.unet_good.eval().requires_grad_(False)
        self.unet_bad.eval().requires_grad_(False)
        self.config = unet_good.config
        if hasattr(unet_good.config, "in_channels"):
            self.in_channels = unet_good.config.in_channels
        elif hasattr(unet_good, "in_channels"):
            self.in_channels = unet_good.in_channels
        if hasattr(unet_good, "add_embedding"):
            self.add_embedding = unet_good.add_embedding

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        return_dict = kwargs.get("return_dict", True)
        out_good = self.unet_good(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        out_bad = self.unet_bad(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        sample_good = out_good[0] if isinstance(out_good, tuple) else out_good.sample
        sample_bad = out_bad[0] if isinstance(out_bad, tuple) else out_bad.sample
        guided_sample = (1.0 - self.w) * sample_bad + self.w * sample_good
        if not return_dict:
            return (guided_sample,)
        return UNet2DConditionOutput(sample=guided_sample)


def build_poa_pipeline(config: dict, args, device: str, w: float):
    model_config = config.get("model_config", {})
    model_name = model_config.get("name", "sdxl")
    if "sd3" in model_name or "flux" in model_name:
        raise ValueError("POA inference currently supports UNet models only.")

    model_path = args.model_path or model_config.get("model_path")
    model_dtype = model_config.get("model_dtype", args.model_dtype)
    scheduler = model_config.get("scheduler", args.scheduler)

    base_pipe = load_base_pipeline(
        model_name,
        device=device,
        model_path=model_path,
        model_dtype=model_dtype,
        scheduler=scheduler,
        local_files_only=args.local_files_only,
        model_config=args.model_config,
        original_config_file=args.original_config_file,
    )

    class SimpleLogger:
        def info(self, msg):
            print(msg)

        def error(self, msg):
            print(msg)

    pruned_pipe = apply_pruning_to_pipe(base_pipe, config, SimpleLogger())
    if pruned_pipe is None:
        raise RuntimeError("Failed to apply pruning; check the config.")

    unet_bad = copy.deepcopy(base_pipe.unet)
    unet_good = pruned_pipe.unet
    base_pipe.unet = POAUnetWrapper(
        unet_good=unet_good,
        unet_bad=unet_bad,
        guidance_weight=w,
    ).to(device)
    del pruned_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return base_pipe


def main():
    parser = argparse.ArgumentParser(description="CR-Diff POA inference")
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--original_config_file", type=str, default=None)
    parser.add_argument("--model_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument(
        "--scheduler",
        choices=["checkpoint", "dpmpp-sde-normal", "ddim-trailing", "euler-trailing"],
        default="checkpoint",
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/sample.png")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--poa_weight", type=float, default=1.5)
    parser.add_argument("--no_prune", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    with open(args.config_json, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_config = config.get("model_config", {})
    model_name = model_config.get("name", "sdxl")
    width = int(model_config.get("width", 1024))
    height = int(model_config.get("height", 1024))
    model_path = args.model_path or model_config.get("model_path")
    model_dtype = model_config.get("model_dtype", args.model_dtype)
    scheduler = model_config.get("scheduler", args.scheduler)

    print(f"Model: {model_name}")
    print(f"Source: {model_path or MODEL_MAP.get(model_name, model_name)}")
    print(f"Resolution: {width}x{height}")
    print(f"Scheduler preset: {scheduler}")
    print(f"POA w={args.poa_weight}: {'disabled' if args.no_prune else 'enabled'}")

    if args.no_prune:
        pipe = load_base_pipeline(
            model_name,
            device=device,
            model_path=model_path,
            model_dtype=model_dtype,
            scheduler=scheduler,
            local_files_only=args.local_files_only,
            model_config=args.model_config,
            original_config_file=args.original_config_file,
        )
    else:
        pipe = build_poa_pipeline(config, args, device=device, w=args.poa_weight)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    image.save(args.output)
    print(f"Inference finished. Image saved to: {args.output}")


if __name__ == "__main__":
    main()
