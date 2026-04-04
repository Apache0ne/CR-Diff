#!/bin/bash
# Single UNet pruning + generation: use the BWP config.json and best_global_params_ratio to prune and generate one image.
cd "$(dirname "$0")"

OUT=outresults/unet_experiment
mkdir -p "$OUT/saves" "$OUT/demos" "$OUT/logs"

python unet_pruning_experiment.py \
  --config_json outresults/SA_Ratio/sdxl_512x512/config.json \
  --prompt "A cat holding a sign that says hello world" \
  --prompt_tag demo \
  --base_save_path "$OUT/saves" \
  --base_demo_path "$OUT/demos" \
  --base_log_path "$OUT/logs" \
  --no-save-model
