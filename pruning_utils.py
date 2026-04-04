import argparse
import copy
import json

from typing import Any, Dict


def apply_pruning_to_pipe(base_pipe, config: Dict[str, Any], logger):
    """
    Read pruning settings from config.json, clone the input pipeline, apply pruning,
    and return the pruned pipeline.
    """
    logger.info("=" * 80)
    logger.info("Applying pruning to model (for inference)...")
    logger.info("=" * 80)

    pipe = copy.deepcopy(base_pipe)

    # Auto-detect unet or transformer
    model = getattr(pipe, "transformer", getattr(pipe, "unet", None))
    if model is None:
        logger.error("No 'unet' or 'transformer' attribute found in pipeline.")
        return None
    logger.info(f"Core model: {model.__class__.__name__}")

    model_name = config.get("model_config", {}).get("name", "sd1.5")
    pruning_method = config.get("pruning_method", "ratio")

    analysis = {}

    # SD3 / Flux: shared_ratio (DiT)
    if "best_global_params_sd3" in config and ("sd3" in model_name or "flux" in model_name):
        logger.info(f"Detected SD3/Flux model ({model_name}), using 'shared_ratio' pruning.")
        try:
            from lib.prune_dit import prune_magnitude

            logger.info("Imported prune_magnitude from lib.prune_dit.")
        except ImportError:
            logger.error("Failed to import prune_magnitude from lib.prune_dit.")
            return None

        params = config.get("best_global_params_sd3", {})
        ratio = params.get("shared_ratio", 0.0)
        logger.info(f"Using shared_ratio from 'best_global_params_sd3': {ratio}")

        mock_args = argparse.Namespace(sparsity_ratio=ratio)
        target_modules = ["attn", "ff"]
        attn_modules = [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ]
        ff_modules = ["ff.net.0.proj", "ff.net.2"]
        target_names = []
        if "attn" in target_modules:
            target_names.extend(attn_modules)
        if "ff" in target_modules:
            target_names.extend(ff_modules)

        prune_magnitude(mock_args, model, target_names, device=pipe.device)

    # SD / SDXL: ratio (down/up/mid)
    elif pruning_method == "ratio":
        logger.info(f"Detected {model_name}. Using 'ratio' (down/up/mid) pruning.")
        try:
            from lib.prune_unet import prune_by_ratio

            logger.info("Imported prune_by_ratio from lib.prune_unet.")
        except ImportError:
            logger.error("Failed to import prune_by_ratio from lib.prune_unet.")
            return None

        params = config.get("best_global_params_ratio", {})
        logger.info(f"Pruning ratios: {json.dumps(params, indent=2)}")

        pruning_config = {
            "down": params.get("down_ratio", 0.5),
            "up": params.get("up_ratio", 0.5),
            "mid": params.get("mid_ratio", 0.5),
        }

        logger.info(f"Applying ratio-based pruning: {pruning_config}")
        pipe.unet, analysis = prune_by_ratio(
            model,
            pruning_config,
            target_modules=["attn", "ff"],
        )

    # SD / SDXL: magnitude
    elif pruning_method == "magnitude":
        logger.info(f"Detected {model_name}. Using 'magnitude' (down/up/mid) pruning.")
        try:
            from lib.prune_unet import prune_magnitude
        except ImportError:
            logger.error("Failed to import prune_magnitude from lib.prune_unet.")
            return None

        params = config.get("best_global_params_ratio", {})
        pruning_config = {
            "down": params.get("down_ratio", 0.5),
            "up": params.get("up_ratio", 0.5),
            "mid": params.get("mid_ratio", 0.5),
        }

        pipe.unet, analysis = prune_magnitude(
            model,
            pruning_config,
            target_modules=["attn", "ff"],
        )

    else:
        logger.error(f"Unknown pruning method: {pruning_method}")
        return None

    if analysis:
        logger.info("Pruning analysis:")
        logger.info(json.dumps(analysis, indent=2))

    logger.info("Pruning applied successfully (inference-ready).")
    return pipe

