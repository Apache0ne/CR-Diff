# python BWP_strategy.py (BWP = Block-Wise Pruning Ratio Strategy)
import subprocess
import re
import os
import time
import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
import random
import math
import operator
import logging
import itertools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ITERATION_COUNT = 120
PATIENCE_FACTOR = 0.2

SEED_SEARCH_EXPERIMENTS = [
    # Format: "down_ratio up_ratio mid_ratio"
    "0.3 0.3 0.3",
]
PARAM_RANGES = {
    "down_ratio": (0.1, 0.8),
    "up_ratio":   (0.1, 0.8),
    "mid_ratio":  (0.1, 0.8),
}

MODEL_CONFIGS = [
    {"name": "sdxl", "width": 512, "height": 512},
]

REHEAT_UNLOCK_THRESHOLD = 0.8
MAX_REHEATS_AFTER_THRESHOLD = 3
MAX_RESTARTS = 2
LARGE_MUTATION_FACTOR = 3.0
INITIAL_TEMPERATURE = 0.1
COOLING_RATE = 0.95
INITIAL_NEIGHBOR_STEP_SIZE = 0.10 
MIN_NEIGHBOR_STEP_SIZE = 0.005
REHEAT_FACTOR = 1.5
SEED_SEARCH_STEPS = 30
EVAL_STEPS = 30
MAX_WORKERS = 1
AVAILABLE_GPUS = [0]
WORKER_SCRIPT_PATH = os.path.join(BASE_DIR, "BWP_trial.sh")
RESULTS_ROOT_DIR = os.path.join(BASE_DIR, "outresults/SA_Ratio")
PROMPTS_TO_TEST = {
    "hello_world": "A cat holding a sign that says hello world",
    "text_future_is_now": "A billboard in Times Square showing the text 'FUTURE IS NOW' in bold neon letters",
}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, prompt, tag, model_config):
        self.prompt = prompt
        self.tag = tag
        self.cache = {}
        self.model_config = model_config

    def run_experiment(self, params, steps, gpu_id, final_save_path=""):
        param_key = (
            f"{self.model_config['name']}_{self.model_config['width']}x{self.model_config['height']}_"
            f"{params['down_ratio']:.4f}_{params['up_ratio']:.4f}_{params['mid_ratio']:.4f}_{steps}"
        )

        if not final_save_path and param_key in self.cache:
            return self.cache[param_key], self.tag

        save_path = ""
        if final_save_path:
             os.makedirs(final_save_path, exist_ok=True)
             save_path = os.path.join(final_save_path, f"{self.tag}_d{params['down_ratio']:.2f}_u{params['up_ratio']:.2f}_m{params['mid_ratio']:.2f}.png")

        cmd = [
            WORKER_SCRIPT_PATH,
            self.model_config["name"], str(self.model_config["width"]), str(self.model_config["height"]),
            self.prompt, self.tag,
            str(params["down_ratio"]), str(params["up_ratio"]), str(params["mid_ratio"]),
            save_path, str(gpu_id), str(steps)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1200, encoding='utf-8')
            match = re.search(r"FINAL_SCORE:(-?\d+\.\d+)", result.stdout)
            if match:
                score = float(match.group(1))
                if not final_save_path:
                    self.cache[param_key] = score
                return score, self.tag
            else:
                logger.warning(f"  Warning: No FINAL_SCORE found for ({self.tag}) @ GPU {gpu_id}.")
                logger.warning(f"  STDOUT was: {result.stdout[-500:]}")
                return None, self.tag
        except subprocess.CalledProcessError as e:
            logger.error(f"  ERROR in experiment ({self.tag}) @ GPU {gpu_id}: Command returned non-zero exit status {e.returncode}.")
            logger.error(f"  --- FAILED COMMAND ---")
            logger.error(f"  {' '.join(e.cmd)}")
            logger.error(f"  --- STDERR (Error Message) ---")
            logger.error(e.stderr if e.stderr else "[No STDERR output]")
            logger.error(f"  --- STDOUT (Last 500 chars) ---")
            logger.error(e.stdout[-500:] if e.stdout else "[No STDOUT output]")
        except Exception as e:
            logger.error(f"  UNEXPECTED ERROR in experiment ({self.tag}) @ GPU {gpu_id}: {e}")

        return None, self.tag

def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds))) if seconds is not None else "N/A"

def parse_seed_experiments(experiments_list):
    return [
        {"down_ratio": float(p[0]), "up_ratio": float(p[1]), "mid_ratio": float(p[2])}
        for p in (exp.split() for exp in experiments_list)
    ]

def generate_neighbor(params, current_temperature, initial_temperature):
    param_to_change = random.choice(list(params.keys()))
    new_params = params.copy()

    dynamic_step_size = max(MIN_NEIGHBOR_STEP_SIZE, INITIAL_NEIGHBOR_STEP_SIZE * (current_temperature / initial_temperature))
    perturbation = random.uniform(-dynamic_step_size, dynamic_step_size)

    new_params[param_to_change] += perturbation

    for param_key, (min_val, max_val) in PARAM_RANGES.items():
        new_params[param_key] = np.clip(new_params[param_key], min_val, max_val)

    return new_params

def calculate_weighted_average(scores_dict):
    """
    Compute simple mean of all scores, treating failed runs (None) as -2.0.
    """
    if not scores_dict: 
        # Always return a float
        return -float('inf')
    
    scores_list = []
    for tag, score in scores_dict.items():
        if score is None:
            # Failed runs count as -2.0
            scores_list.append(-2.0)
        else:
            # Successful runs keep their original score
            scores_list.append(score)
    
    if not scores_list:
        # Safety check; should normally not happen
        return -float('inf')
        
    # Simple mean over all scores
    raw_avg = np.mean(scores_list)
    
    return raw_avg

def evaluate_configuration(params, all_runners, steps, gpu_pool_iter):
    scores_dict = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for runner in all_runners:
            try: gpu_id = next(gpu_pool_iter)
            except StopIteration:
                logger.error("GPU Pool exhausted!"); gpu_id = AVAILABLE_GPUS[0]
            futures.append(executor.submit(runner.run_experiment, params, steps, gpu_id, final_save_path=""))
        for future in futures:
            score, tag = future.result()
            scores_dict[tag] = score
            
    # For v3, "weighted_avg_score" is identical to "raw_avg_score"
    avg_score = calculate_weighted_average(scores_dict)

    logger.info(
        f"  Params: d_ratio={params['down_ratio']:.4f} u_ratio={params['up_ratio']:.4f} m_ratio={params['mid_ratio']:.4f} "
        f"-> AvgScore: {avg_score:.4f}"  # simplified log
    )
    return avg_score, scores_dict  # return avg_score

def main():
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    script_start_time = time.time()
    if not os.access(WORKER_SCRIPT_PATH, os.X_OK):
        try:
            os.chmod(WORKER_SCRIPT_PATH, 0o755); logger.info(f"Added execute permission to {WORKER_SCRIPT_PATH}")
        except Exception as e:
            logger.error(f"Failed to chmod {WORKER_SCRIPT_PATH}: {e}. Please add execute permission manually."); return
    os.makedirs(RESULTS_ROOT_DIR, exist_ok=True)
    
    # ★★★ v3: 更改日志文件名 ★★★
    log_file_handler = logging.FileHandler(os.path.join(RESULTS_ROOT_DIR, "global_sa_run_ratio_simple_mean.log"), mode='w')
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(log_file_handler)

    for model_config in MODEL_CONFIGS:
        model_config_str = f"{model_config['name']}_{model_config['height']}x{model_config['width']}"
        model_config_dir = os.path.join(RESULTS_ROOT_DIR, model_config_str)
        os.makedirs(model_config_dir, exist_ok=True)

        gpu_pool_iter = itertools.cycle(AVAILABLE_GPUS)
        logger.info("\n" + "="*80)
        logger.info(f"STARTING GLOBAL RATIO OPTIMIZATION (SIMPLE MEAN) for Model: {model_config_str}")
        logger.info(f"MAX_WORKERS: {MAX_WORKERS}, AVAILABLE_GPUS: {AVAILABLE_GPUS}")
        logger.info(f"Worker Script: {WORKER_SCRIPT_PATH}")
        logger.info("="*80)
        all_runners = [ExperimentRunner(prompt, tag, model_config) for tag, prompt in PROMPTS_TO_TEST.items()]
        patience = max(5, int(ITERATION_COUNT * PATIENCE_FACTOR))

        # Start from seeds; only save final best config
        baseline_params = {"down_ratio": 0.0, "up_ratio": 0.0, "mid_ratio": 0.0}
        logger.info("Evaluating global baseline (Dense Model)...")
        baseline_score = 0.0
        logger.info("\n" + "-"*80 + "\nRunning Seed Search...\n" + "-"*80)
        seed_params_list = parse_seed_experiments(SEED_SEARCH_EXPERIMENTS)
        seed_trials = []
        for seed_params in seed_params_list:
            logger.info(f"Evaluating Seed: {seed_params}")
            seed_score, _ = evaluate_configuration(seed_params, all_runners, SEED_SEARCH_STEPS, gpu_pool_iter)
            if seed_score > -float('inf'): seed_trials.append({"score": seed_score, "params": seed_params})

        if not seed_trials:
            logger.error("\n--- Seeding failed. Skipping model config. ---")
            continue

        best_seed_trial = max(seed_trials, key=lambda x: x['score'])
        current_solution, current_score = best_seed_trial['params'], best_seed_trial['score']
        overall_best_score, overall_best_solution = current_score, current_solution
        logger.info(f"Best Seed Score: {overall_best_score:.4f} with params: {overall_best_solution}")
        logger.info("\n" + "-"*80 + "\nStarting Simulated Annealing...\n" + "-"*80)
        temperature, no_improvement_count, reheat_count = INITIAL_TEMPERATURE, 0, 0
        reheat_unlocked = False
        restarts_count = 0

        # Main optimization loop
        for i in range(ITERATION_COUNT):
            logger.info(f"\n--- Iteration {i+1}/{ITERATION_COUNT} (Temp: {temperature:.4f}, Best: {overall_best_score:.4f}) ---")
            neighbor_solution = generate_neighbor(current_solution, temperature, INITIAL_TEMPERATURE)
            neighbor_score, _ = evaluate_configuration(neighbor_solution, all_runners, EVAL_STEPS, gpu_pool_iter)
            if neighbor_score <= -float('inf'): logger.warning("  Evaluation failed, skipping iteration."); continue
            delta_score = neighbor_score - current_score
            if delta_score > 0 or random.random() < math.exp(delta_score / temperature):
                log_prefix = "  Accepted new solution (Better):" if delta_score > 0 else "  Accepted new solution (Annealing):"
                logger.info(f"{log_prefix} {neighbor_score:.4f}")
                current_solution, current_score = neighbor_solution, neighbor_score
                if current_score > overall_best_score:
                    logger.info(f"  *** NEW GLOBAL BEST: {current_score:.4f} ***")
                    
                    # Keep assignment order as (score, solution)
                    overall_best_score, overall_best_solution = current_score, current_solution
                    
                    no_improvement_count = 0
                    if not reheat_unlocked and overall_best_score > REHEAT_UNLOCK_THRESHOLD:
                        reheat_unlocked = True; logger.info(f"  *** Re-heating Unlocked ***")
                else: no_improvement_count += 1
            else:
                logger.info(f"  Rejected solution: {neighbor_score:.4f}")
                no_improvement_count += 1

            temperature *= COOLING_RATE
            
            if no_improvement_count >= patience:
                if reheat_unlocked and reheat_count < MAX_REHEATS_AFTER_THRESHOLD:
                    reheat_count += 1; temperature *= REHEAT_FACTOR; no_improvement_count = 0
                    logger.info(f"  --- PATIENCE LIMIT: Re-heating ({reheat_count}/{MAX_REHEATS_AFTER_THRESHOLD}) ---")
                elif restarts_count < MAX_RESTARTS:
                    restarts_count += 1; no_improvement_count = 0
                    logger.info(f"  --- PATIENCE LIMIT: Restarting ({restarts_count}/{MAX_RESTARTS}) ---")
                    new_start_solution = generate_neighbor(overall_best_solution, INITIAL_TEMPERATURE * LARGE_MUTATION_FACTOR, INITIAL_TEMPERATURE)
                    logger.info(f"  Restarting with params: {new_start_solution}")
                    new_start_score, _ = evaluate_configuration(new_start_solution, all_runners, SEED_SEARCH_STEPS, gpu_pool_iter)
                    if new_start_score <= -float('inf'): logger.warning("  Restart evaluation failed."); restarts_count -= 1; continue
                    current_solution, current_score = new_start_solution, new_start_score; temperature = INITIAL_TEMPERATURE
                    if new_start_score > overall_best_score:
                         logger.info(f"  *** NEW GLOBAL BEST from restart: {new_start_score:.4f} ***")
                         
                         # Keep assignment order as (score, solution)
                         overall_best_score, overall_best_solution = new_start_score, new_start_solution
                else:
                    logger.info("\n--- PATIENCE LIMIT: Max restarts reached. Stopping SA. ---")
                    break

        logger.info("\n" + "="*80 + f"\nGLOBAL RATIO OPTIMIZATION (SIMPLE MEAN) FINISHED for {model_config_str}")
        logger.info(f"Best Avg Score: {overall_best_score:.4f}")
        logger.info(f"Best Global Params: {overall_best_solution}")

        # Save final best config only
        best_config_path = os.path.join(model_config_dir, 'config.json')
        with open(best_config_path, 'w') as f:
            json.dump({
                "model_config": model_config,
                "best_avg_score": overall_best_score,
                "best_global_params_ratio": overall_best_solution
            }, f, indent=4)
        logger.info(f"Best global config saved to: {best_config_path}")

    total_time = time.time() - script_start_time
    logger.info(f"\nTotal execution time: {format_duration(total_time)}")

if __name__ == "__main__":
    main()