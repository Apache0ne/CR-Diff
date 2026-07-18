"""
unet_pruning_experiment.py: run UNet pruning experiments and generate demo images.
Supports --prune_method magnitude / ratio and complete single-file SDXL checkpoints.
"""

import argparse
import copy
import json
import logging
import os
import random
from importlib.metadata import version

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

from lib.prune_unet import check_sparsity, prune_by_ratio, prune_magnitude
from single_file_sdxl_utils import load_sdxl_pipeline, resolve_dtype

print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus:", torch.cuda.device_count())

MODEL_MAP = {
    "sd1.5": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd2.1": "stabilityai/stable-diffusion-2-1",
}


def setup_logging(log_file, logger_name):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    f_handler = logging.FileHandler(log_file, mode="w")
    c_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
    f_handler.setFormatter(formatter)
    c_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    logger.addHandler(c_handler)
    return logger


def load_pipeline(args, device):
    """Load the requested model while preserving a complete single-file SDXL checkpoint."""
    if args.model_path:
        if args.model_name not in (None, "sdxl"):
            raise ValueError("--model_path currently supports complete SDXL checkpoints only")
        args.model_name = "sdxl"
        return load_sdxl_pipeline(
            device=device,
            model_path=args.model_path,
            model_dtype=args.model_dtype,
            scheduler=args.scheduler,
            local_files_only=args.local_files_only,
            model_config=args.model_config,
            original_config_file=args.original_config_file,
        )

    model_id_or_path = MODEL_MAP.get(args.model_name, args.model_name)
    if args.model_name == "sdxl":
        return load_sdxl_pipeline(
            device=device,
            model_id=model_id_or_path,
            model_dtype=args.model_dtype,
            scheduler=args.scheduler,
            local_files_only=args.local_files_only,
        )

    return StableDiffusionPipeline.from_pretrained(
        model_id_or_path,
        torch_dtype=resolve_dtype(args.model_dtype),
        local_files_only=args.local_files_only,
    ).to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Run UNet pruning experiments and generate demo images."
    )
    parser.add_argument("--config_json", type=str, default=None, help="Optional BWP output config.json")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Complete local SDXL .safetensors/.ckpt containing UNet, both text encoders, and VAE.",
    )
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--original_config_file", type=str, default=None)
    parser.add_argument(
        "--model_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--scheduler",
        choices=["checkpoint", "dpmpp-sde-normal", "ddim-trailing", "euler-trailing"],
        default="checkpoint",
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_tag", type=str, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--base_save_path", type=str, default="outresults/unet_experiment/saves")
    parser.add_argument("--base_demo_path", type=str, default="outresults/unet_experiment/demos")
    parser.add_argument("--base_log_path", type=str, default="outresults/unet_experiment/logs")
    parser.add_argument("--prune_method", type=str, default=None, choices=["magnitude", "ratio"])
    parser.add_argument("--use_log_scale", action="store_true")
    parser.add_argument("--target_modules", type=str, nargs="+", default=["attn", "ff"])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Semicolon-separated experiment config strings (format depends on prune_method)",
    )
    args = parser.parse_args()

    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model_cfg = cfg.get("model_config", {})
        if args.model_name is None:
            args.model_name = model_cfg.get("name", "sdxl")
        if args.model_path is None:
            args.model_path = model_cfg.get("model_path")
        args.model_dtype = model_cfg.get("model_dtype", args.model_dtype)
        args.scheduler = model_cfg.get("scheduler", args.scheduler)
        args.width = int(model_cfg.get("width", args.width))
        args.height = int(model_cfg.get("height", args.height))

        if args.prune_method is None:
            args.prune_method = cfg.get("pruning_method", "ratio")

        if args.experiments is None:
            params = cfg.get("best_global_params_ratio", {})
            down = params.get("down_ratio")
            up = params.get("up_ratio")
            mid = params.get("mid_ratio")
            if down is None or up is None or mid is None:
                raise ValueError("config_json missing best_global_params_ratio.(down_ratio/up_ratio/mid_ratio)")
            args.experiments = f"{down} {up} {mid} 0 0 0"

        if args.prompt is None:
            args.prompt = "A cat holding a sign that says hello world"
        if args.prompt_tag is None:
            args.prompt_tag = "demo"

    if args.model_path and args.model_name is None:
        args.model_name = "sdxl"
    if not args.model_name or not args.prompt or not args.prompt_tag:
        raise ValueError("Missing required args: provide --prompt/--prompt_tag/--model_name, or use --config_json.")
    if not args.prune_method:
        args.prune_method = "ratio"
    if not args.experiments:
        raise ValueError("Missing --experiments (or use --config_json to auto-generate it).")

    os.makedirs(args.base_save_path, exist_ok=True)
    os.makedirs(args.base_demo_path, exist_ok=True)
    os.makedirs(args.base_log_path, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    print("=" * 30 + f"\nInitializing model for prompt '{args.prompt_tag}'...\n" + "=" * 30)
    try:
        base_pipe = load_pipeline(args, device)
        source = args.model_path or MODEL_MAP.get(args.model_name, args.model_name)
        print(f"Base model '{source}' loaded successfully.")
        print(f"Scheduler: {base_pipe.scheduler.__class__.__name__}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc

    experiment_configs = [exp.strip() for exp in args.experiments.split(";") if exp.strip()]
    print(f"Received {len(experiment_configs)} experiment configs. Starting...")

    for i, exp_config_str in enumerate(experiment_configs):
        logger = logging.getLogger(f"experiment_{i}")
        try:
            parts = exp_config_str.split()
            if len(parts) != 6:
                print(f"Skip malformed experiment config: '{exp_config_str}' (expected 6 values)")
                continue

            val1, val2, val3, _val4, _val5, _val6 = map(float, parts)
            config_name_parts = [args.prune_method]
            if args.prune_method == "magnitude":
                config_name_parts.append(f"thresh-d{val1}_u{val2}_m{val3}")
            else:
                config_name_parts.append(f"ratio-d{val1:.2f}_u{val2:.2f}_m{val3:.2f}")
            config_name_parts.append(f"step{args.num_inference_steps}")
            base_config_name = "_".join(config_name_parts)

            config_name = base_config_name
            counter = 2
            while os.path.exists(os.path.join(args.base_demo_path, f"{config_name}_seed{args.seed}.png")):
                config_name = f"{base_config_name}_v{counter}"
                counter += 1

            save_model_path = os.path.join(args.base_save_path, config_name)
            log_file_path = os.path.join(args.base_log_path, f"{config_name}.log")
            logger = setup_logging(log_file_path, f"experiment_{i}")
            logger.info("=" * 80 + f"\nStart experiment {args.prompt_tag} {i + 1}/{len(experiment_configs)}: {config_name}\n" + "=" * 80)

            pipe = copy.deepcopy(base_pipe)
            model = pipe.unet
            analysis_results = {}
            is_baseline_run = val1 == 0.0 and val2 == 0.0 and val3 == 0.0

            if is_baseline_run:
                logger.info("Baseline config: pruning skipped")
                analysis_results = {"status": f"Baseline run ({args.prune_method}), pruning skipped."}
            elif args.prune_method == "magnitude":
                model, analysis_results = prune_magnitude(
                    model,
                    {"down": val1, "up": val2, "mid": val3},
                    args.target_modules,
                )
            else:
                model, analysis_results = prune_by_ratio(
                    model,
                    {"down": val1, "up": val2, "mid": val3},
                    args.target_modules,
                )

            pipe.unet = model
            analysis_path = os.path.join(args.base_log_path, f"{config_name}_analysis.json")
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, indent=4)
            logger.info(f"Pruning analysis saved to: {analysis_path}")

            try:
                check_sparsity(pipe.unet, {"down": val1, "up": val2, "mid": val3}, logger)
            except Exception:
                logger.exception("Sparsity summary failed")

            if not args.no_save_model and not is_baseline_run:
                os.makedirs(save_model_path, exist_ok=True)
                pipe.save_pretrained(save_model_path)
                logger.info(f"Model saved to: {save_model_path}")
            else:
                logger.info("Model save skipped")

            generator = torch.Generator(device).manual_seed(args.seed)
            image = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            demo_image_path = os.path.join(args.base_demo_path, f"{config_name}_seed{args.seed}.png")
            image.save(demo_image_path)
            logger.info(f"Demo image saved to: {demo_image_path}")
        except Exception as exc:
            logger.error(f"Experiment '{exp_config_str}' failed: {exc}", exc_info=True)

    print(f"\nAll experiments for prompt '{args.prompt_tag}' finished.")


if __name__ == "__main__":
    main()
