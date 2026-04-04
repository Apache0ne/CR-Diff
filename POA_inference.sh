#!/bin/bash
# POA single-image inference. Requires a config JSON produced by BWP; edit the paths and prompt below.
python POA_inference.py \
  --config_json outresults/SA_Ratio/sdxl_512x512/config.json \
  --prompt "A cat holding a sign that says hello world" \
  --output outputs/poa_sample.png
