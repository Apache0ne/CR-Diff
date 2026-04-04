"""
POA_inference.py — single-image POA (Pruned Output Amplification) inference.

Uses POA (w) to mix dense and pruned UNet outputs:
    D_w = (1 - w) * D_dense + w * D_pruned
Runs one forward pass for a single prompt and saves one image.
"""
import argparse
import json
import os
import copy

import torch
import torch.nn as nn
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    FluxPipeline,
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

from pruning_utils import apply_pruning_to_pipe


# Short model name -> HuggingFace model ID, used by load_base_pipeline.
# Currently only sdxl is listed; add more entries here if needed.
MODEL_MAP = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}


def load_base_pipeline(model_name: str, device: str = "cuda:0"):
    """
    Load the appropriate diffusers Pipeline for the given model_name.
    """
    model_id = MODEL_MAP.get(model_name, model_name)

    if "sd3" in model_name:
        pipe_class = StableDiffusion3Pipeline
    elif "flux" in model_name:
        pipe_class = FluxPipeline
    elif "sdxl" in model_name:
        pipe_class = StableDiffusionXLPipeline
    elif "sd" in model_name:
        pipe_class = StableDiffusionPipeline
    else:
        pipe_class = StableDiffusionPipeline

    pipe = pipe_class.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
    ).to(device)

    return pipe


class POAUnetWrapper(nn.Module):
    """
    POA U-Net wrapper:
        D_w = (1 - w) * D_bad + w * D_good
      - D_bad: dense / baseline U-Net
      - D_good: pruned U-Net
      - w: POA weight
    """

    def __init__(self, unet_good, unet_bad, guidance_weight: float):
        super().__init__()
        self.unet_good = unet_good
        self.unet_bad = unet_bad
        self.w = guidance_weight

        self.unet_good.eval()
        self.unet_bad.eval()
        self.unet_good.requires_grad_(False)
        self.unet_bad.requires_grad_(False)

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

        if isinstance(out_good, tuple):
            sample_good = out_good[0]
            sample_bad = out_bad[0]
        else:
            sample_good = out_good.sample
            sample_bad = out_bad.sample

        guided_sample = (1.0 - self.w) * sample_bad + self.w * sample_good

        if not return_dict:
            return (guided_sample,)

        return UNet2DConditionOutput(sample=guided_sample)


def build_poa_pipeline(config: dict, device: str, w: float):
    """
    Build a POA (w) inference pipeline from config.json:
      - load dense base model (unet_bad)
      - create pruned model (unet_good) according to config
      - wrap with POAUnetWrapper and return the final pipeline
    """
    model_config = config.get("model_config", {})
    model_name = model_config.get("name", "sdxl")

    if "sd3" in model_name or "flux" in model_name:
        raise ValueError("POA inference currently only supports UNet models (sd1.5/sdxl/sd2.1).")

    # 1) Load dense pipeline
    base_pipe = load_base_pipeline(model_name, device=device)

    # 2) Build pruned pipeline
    class SimpleLogger:
        def info(self, msg):
            print(msg)

        def error(self, msg):
            print(msg)

    logger = SimpleLogger()
    pruned_pipe = apply_pruning_to_pipe(base_pipe, config, logger)
    if pruned_pipe is None:
        raise RuntimeError("Failed to apply pruning; please check the config.")

    # 3) Extract unet_good / unet_bad and build wrapper
    unet_bad = copy.deepcopy(base_pipe.unet)
    unet_good = pruned_pipe.unet

    wrapped_pipe = base_pipe
    wrapped_pipe.unet = POAUnetWrapper(
        unet_good=unet_good,
        unet_bad=unet_bad,
        guidance_weight=w,
    ).to(device)

    # Free pruned_pipe to save memory
    del pruned_pipe
    torch.cuda.empty_cache()

    return wrapped_pipe


def main():
    parser = argparse.ArgumentParser(
        description="CR-Diff simple inference script: single-prompt image generation (optional pruning)."
    )
    parser.add_argument(
        "--config_json",
        type=str,
        required=True,
        help="Path to pruning config JSON (must contain model_config.name/width/height).",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sample.png",
        help="Output image path (default: outputs/sample.png).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (e.g. cuda:0 or cpu, default cuda:0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of diffusion steps (default 30).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (default 7.5).",
    )
    parser.add_argument(
        "--poa_weight",
        type=float,
        default=1.5,
        help="POA weight w (default 1.5).",
    )
    parser.add_argument(
        "--no_prune",
        action="store_true",
        help="Use dense model only (no POA), for baseline comparison.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU.")
        device = "cpu"
    else:
        device = args.device

    with open(args.config_json, "r") as f:
        config = json.load(f)

    model_config = config.get("model_config", {})
    model_name = model_config.get("name", "sdxl")
    width = model_config.get("width", 1024)
    height = model_config.get("height", 1024)

    print(f"Model: {model_name}")
    print(f"Resolution: {width}x{height}")
    print(f"Device: {device}")
    print(
        f"POA w={args.poa_weight}: "
        f"{'disabled (dense only)' if args.no_prune else 'enabled (dense + pruned)'}"
    )

    # Build pipeline
    if args.no_prune:
        pipe = load_base_pipeline(model_name, device=device)
    else:
        pipe = build_poa_pipeline(config, device=device, w=args.poa_weight)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    out = pipe(
        prompt=args.prompt,
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    image = out.images[0]

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    image.save(args.output)
    print(f"Inference finished. Image saved to: {args.output}")


if __name__ == "__main__":
    main()

