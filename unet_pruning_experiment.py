"""
unet_pruning_experiment.py: run UNet pruning experiments and generate a single demo image.
Supports --prune_method: magnitude / ratio (BWP uses ratio).
BWP_trial.sh calls this script with a specific down/up/mid ratio.
"""
# Imports
import argparse
import os
import torch
import logging
import json
import copy
import random
import numpy as np
from importlib.metadata import version
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from lib.prune_unet import (
    prune_magnitude,
    check_sparsity,
    prune_by_ratio,
)

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

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
    f_handler = logging.FileHandler(log_file, mode='w')
    c_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    f_handler.setFormatter(formatter)
    c_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    logger.addHandler(c_handler)
    return logger

def main():
    parser = argparse.ArgumentParser(
        description="Run UNet pruning experiments and generate demo images."
    )
    # If --config_json is provided, model_config and best_global_params_ratio
    # will be loaded from config.json, so you do not need to pass
    # --model_name/--width/--height/--experiments manually.
    parser.add_argument("--config_json", type=str, default=None, help="Optional BWP output config.json")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_tag", type=str, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--base_save_path", type=str, default="outresults/unet_experiment/saves")
    parser.add_argument("--base_demo_path", type=str, default="outresults/unet_experiment/demos")
    parser.add_argument("--base_log_path", type=str, default="outresults/unet_experiment/logs")
    parser.add_argument("--prune_method", type=str, default=None, choices=['magnitude', 'ratio'])
    parser.add_argument("--use_log_scale", action='store_true')
    parser.add_argument("--target_modules", type=str, nargs='+', default=['attn', 'ff'])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--no-save-model", action='store_true')
    parser.add_argument(
        "--experiments", type=str, default=None,
        help="Semicolon-separated experiment config strings (format depends on prune_method)"
    )
    args = parser.parse_args()

    # Read config_json (optional) and fill defaults
    if args.config_json:
        with open(args.config_json, "r") as f:
            cfg = json.load(f)

        model_cfg = cfg.get("model_config", {})
        if args.model_name is None:
            args.model_name = model_cfg.get("name", "sdxl")

        args.width = int(model_cfg.get("width", args.width))
        args.height = int(model_cfg.get("height", args.height))

        if args.prune_method is None:
            args.prune_method = cfg.get("pruning_method", "ratio")

        if args.experiments is None:
            params = cfg.get("best_global_params_ratio", {})
            down = params.get("down_ratio", None)
            up = params.get("up_ratio", None)
            mid = params.get("mid_ratio", None)
            if down is None or up is None or mid is None:
                raise ValueError("config_json missing best_global_params_ratio.(down_ratio/up_ratio/mid_ratio)")
            args.experiments = f"{down} {up} {mid} 0 0 0"

        if args.prompt is None:
            args.prompt = "A cat holding a sign that says hello world"
        if args.prompt_tag is None:
            args.prompt_tag = "demo"

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
    
    print("="*30 + f"\nInitializing model for prompt '{args.prompt_tag}'...\n" + "="*30)
    model_id_or_path = MODEL_MAP.get(args.model_name, args.model_name)
    pipe_class = StableDiffusionXLPipeline if args.model_name == "sdxl" else StableDiffusionPipeline
    try:
        base_pipe = pipe_class.from_pretrained(model_id_or_path, torch_dtype=torch.float16).to(device)
        print(f"Base model '{model_id_or_path}' loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

    experiment_configs = [exp.strip() for exp in args.experiments.split(';') if exp.strip()]
    total_experiments = len(experiment_configs)
    
    print(f"Received {total_experiments} experiment configs from shell. Starting...")

    for i, exp_config_str in enumerate(experiment_configs):
        try:
            parts = exp_config_str.split()
            if len(parts) != 6:
                print(f"Skip malformed experiment config: '{exp_config_str}' (expected 6 values)")
                continue
            
            val1, val2, val3, val4, val5, val6 = map(float, parts)

            config_name_parts = [args.prune_method]
            if args.prune_method == 'magnitude':
                # val1-3 are thresholds
                config_name_parts.append(f"thresh-d{val1}_u{val2}_m{val3}")
            elif args.prune_method == 'ratio':
                # val1-3 are ratios
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

            logger.info("="*80 + f"\nStart experiment {args.prompt_tag} {i+1}/{total_experiments}: {config_name}\n" + "="*80)
            
            pipe = copy.deepcopy(base_pipe)
            model = pipe.unet
            analysis_results = {} # initialize

            pruning_config = {}
            if args.prune_method == 'magnitude':
                pruning_config = {'down': val1, 'up': val2, 'mid': val3}
            elif args.prune_method == 'ratio':
                ratio_config = {'down': val1, 'up': val2, 'mid': val3}

            is_baseline_run = False
            if args.prune_method == 'magnitude' and (val1 == 0.0 and val2 == 0.0 and val3 == 0.0):
                is_baseline_run = True
            elif args.prune_method == 'ratio' and (val1 == 0.0 and val2 == 0.0 and val3 == 0.0):
                is_baseline_run = True

            if is_baseline_run:
                logger.info("="*20 + f" Baseline config for {args.prune_method}, skip pruning " + "="*20)
                analysis_results = {"status": f"Baseline run ({args.prune_method}), pruning skipped."}
            else:
                logger.info("="*20 + f" Start pruning with method {args.prune_method} " + "="*20)
                if args.prune_method == 'magnitude':
                    model, analysis_results = prune_magnitude(model, pruning_config, args.target_modules)
                elif args.prune_method == 'ratio':
                    model, analysis_results = prune_by_ratio(model, ratio_config, args.target_modules)

                logger.info("="*20 + " Pruning finished " + "="*20)

            analysis_path = os.path.join(args.base_log_path, f"{config_name}_analysis.json")
            with open(analysis_path, 'w') as f:
                json.dump(analysis_results, f, indent=4)
            logger.info(f"Pruning analysis saved to: {analysis_path}")
            
            if not args.no_save_model:
                if is_baseline_run:
                    logger.info("Baseline run, skip saving model.")
                else:
                    os.makedirs(save_model_path, exist_ok=True)
                    pipe.save_pretrained(save_model_path)
                    logger.info(f"Model saved to: {save_model_path}")
            else:
                logger.info("Skip saving model because --no-save-model was set.")
            
            logger.info("="*20 + " Generating demo image " + "="*20)
            generator = torch.Generator(device).manual_seed(args.seed)
            image = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator
            ).images[0]
            demo_image_path = os.path.join(args.base_demo_path, f"{config_name}_seed{args.seed}.png")
            image.save(demo_image_path)
            logger.info(f"Demo image saved to: {demo_image_path}")
            logger.info(f"--- Experiment {config_name} finished ---")

        except Exception as e:
            logger.error(f"Experiment '{exp_config_str}' failed: {e}", exc_info=True)
            continue

    print(f"\nAll experiments for prompt '{args.prompt_tag}' finished.")

if __name__ == "__main__":
    main()