#!/bin/bash
# BWP_trial.sh — one CR-Diff ratio-pruning trial.
set -euo pipefail

if [ "$#" -lt 10 ]; then
    echo "Error: at least 10 arguments are required"
    exit 1
fi

MODEL_NAME=$1
WIDTH=$2
HEIGHT=$3
PROMPT=$4
PROMPT_TAG=$5
DOWN_RATIO=$6
UP_RATIO=$7
MID_RATIO=$8
FINAL_SAVE_PATH=${9}
GPU_ID=${10}
INFERENCE_STEPS=${11:-50}

# Optional custom-checkpoint controls inherited from BWP_strategy.py's environment.
MODEL_PATH=${CRDIFF_MODEL_PATH:-}
SCHEDULER=${CRDIFF_SCHEDULER:-checkpoint}
MODEL_DTYPE=${CRDIFF_MODEL_DTYPE:-float16}
MODEL_CONFIG=${CRDIFF_MODEL_CONFIG:-}
ORIGINAL_CONFIG_FILE=${CRDIFF_ORIGINAL_CONFIG_FILE:-}

readonly BASE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
readonly RUN_ID="trial_$$_${GPU_ID}"
readonly WORK_DIR="${BASE_DIR}/outresults/optim_temp/${RUN_ID}"
trap 'rm -rf "${WORK_DIR}"' EXIT
mkdir -p "${WORK_DIR}"

SEED=24432
PRUNE_METHOD="ratio"
EXPERIMENTS_STR="${DOWN_RATIO} ${UP_RATIO} ${MID_RATIO} 0 0 0"

readonly BATCH_SCRIPT="${BASE_DIR}/unet_pruning_experiment.py"
readonly PROMPT_BASE_SAVE_PATH="${WORK_DIR}/saves"
readonly PROMPT_BASE_DEMO_PATH="${WORK_DIR}/demos"
readonly PROMPT_BASE_LOG_PATH="${WORK_DIR}/logs"
readonly IMAGE_DIR_RESIZED="${WORK_DIR}/demos_resized"
readonly METRIC_OUTPUT_DIR="${WORK_DIR}/metrics"
mkdir -p "${PROMPT_BASE_SAVE_PATH}" "${PROMPT_BASE_DEMO_PATH}" "${PROMPT_BASE_LOG_PATH}" "${IMAGE_DIR_RESIZED}" "${METRIC_OUTPUT_DIR}"

readonly RESIZE_SCRIPT="${BASE_DIR}/evaluation/resize_images_256.py"
readonly IMAGEREWARD_SCRIPT="${BASE_DIR}/evaluation/compute_imagereward.py"
readonly TARGET_RESOLUTION=256

EXTRA_MODEL_ARGS=(--scheduler "${SCHEDULER}" --model_dtype "${MODEL_DTYPE}")
if [[ -n "${MODEL_PATH}" ]]; then
    EXTRA_MODEL_ARGS+=(--model_path "${MODEL_PATH}" --model_name sdxl)
fi
if [[ -n "${MODEL_CONFIG}" ]]; then
    EXTRA_MODEL_ARGS+=(--model_config "${MODEL_CONFIG}")
fi
if [[ -n "${ORIGINAL_CONFIG_FILE}" ]]; then
    EXTRA_MODEL_ARGS+=(--original_config_file "${ORIGINAL_CONFIG_FILE}")
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python "${BATCH_SCRIPT}" \
    --model_name "${MODEL_NAME}" --prune_method "${PRUNE_METHOD}" \
    --height "${HEIGHT}" --width "${WIDTH}" --seed "${SEED}" \
    --prompt "${PROMPT}" --prompt_tag "${PROMPT_TAG}" \
    --base_save_path "${PROMPT_BASE_SAVE_PATH}" \
    --base_demo_path "${PROMPT_BASE_DEMO_PATH}" \
    --base_log_path "${PROMPT_BASE_LOG_PATH}" \
    --gpu_id 0 --num_inference_steps "${INFERENCE_STEPS}" \
    --experiments "${EXPERIMENTS_STR}" --no-save-model \
    "${EXTRA_MODEL_ARGS[@]}"

python "${RESIZE_SCRIPT}" --input_folder "${PROMPT_BASE_DEMO_PATH}" --output_folder "${IMAGE_DIR_RESIZED}" --target_res "${TARGET_RESOLUTION}" > /dev/null 2>&1

TEMP_CSV_PATH="${METRIC_OUTPUT_DIR}/temp_prompt.csv"
TEMP_OUTPUT_BASE="${METRIC_OUTPUT_DIR}/temp_results"
TEMP_DETAILS_CSV="${TEMP_OUTPUT_BASE}_details.csv"

if compgen -G "${IMAGE_DIR_RESIZED}/*.png" > /dev/null; then
    RESIZED_IMAGE_NAME=$(basename -- "$(find ${IMAGE_DIR_RESIZED} -name '*.png' | head -n 1)" .png)
    echo "prompt_name,caption" > "${TEMP_CSV_PATH}"
    echo "\"${RESIZED_IMAGE_NAME}\",\"${PROMPT}\"" >> "${TEMP_CSV_PATH}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python "${IMAGEREWARD_SCRIPT}" --image_dir "${IMAGE_DIR_RESIZED}" --csv_path "${TEMP_CSV_PATH}" --output_csv "${TEMP_OUTPUT_BASE}.csv" > /dev/null 2>&1

    if [[ -n "${FINAL_SAVE_PATH}" && -f "$(find ${PROMPT_BASE_DEMO_PATH} -name '*.png' | head -n 1)" ]]; then
        SOURCE_IMAGE_PATH=$(find "${PROMPT_BASE_DEMO_PATH}" -name '*.png' | head -n 1)
        cp "${SOURCE_IMAGE_PATH}" "${FINAL_SAVE_PATH}"
        echo "Saved artifact to ${FINAL_SAVE_PATH}"
    fi

    if [[ -f "${TEMP_DETAILS_CSV}" ]]; then
        SCORE=$(tail -n 1 "${TEMP_DETAILS_CSV}" | awk -F, '{print $NF}')
        echo "FINAL_SCORE:${SCORE}"
    else
        echo "Warning: ImageReward details file not found: ${TEMP_DETAILS_CSV}"
        echo "FINAL_SCORE:-2.0"
    fi
else
    echo "Warning: No PNG image found in ${IMAGE_DIR_RESIZED} for prompt tag ${PROMPT_TAG}"
    echo "FINAL_SCORE:-2.0"
fi

rm -rf "${WORK_DIR}"
