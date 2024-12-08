import os
import subprocess
from pathlib import Path
import sys 
import time
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from sequential_attention.experiments import hyperparams_sa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_experiment(dataset, base_dir="/Users/amber/Desktop/ece695_proj/result", algo="sa"):
    env = os.environ.copy()
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
    env['PYTHONPATH'] = f"{src_path}:{env.get('PYTHONPATH', '')}"
    
    cmd = [
        "conda", "run", "-n", "sa",
        "python", "-m", "sequential_attention.experiments.run",
        f"--data_name={dataset}",
        f"--algo={algo}",
        f"--model_dir={os.path.join(base_dir, algo, dataset)}",
        f"--batch_size={hyperparams_sa.BATCH[dataset]}",
        f"--num_epochs_select={hyperparams_sa.EPOCHS[dataset]}",
        f"--num_epochs_fit={hyperparams_sa.EPOCHS_FIT[dataset]}",
        f"--learning_rate={hyperparams_sa.LEARNING_RATE[dataset]}",
        f"--decay_steps={hyperparams_sa.DECAY_STEPS[dataset]}",
        f"--decay_rate={hyperparams_sa.DECAY_RATE[dataset]}",
        f"--hidden_layers={67}", 
    ]
    
    print(f"Running {algo} on {dataset}...")
    return subprocess.run(cmd, check=True, env=env).returncode == 0

def main():
    datasets = ["mice", "mnist", "fashion", "isolet", "coil", "activity"]
    algos = ["sa", "lly"] # ["sa", "lly", "seql", "gl", "omp"]
    results = {"successful": [], "failed": []}
    
    for algo in algos:
        for dataset in datasets:
            
            start_time = time.time()
            
            exp_id = f"{dataset}_{algo}"
            if run_experiment(dataset, algo=algo):
                results["successful"].append(exp_id)
            else:
                results["failed"].append(exp_id)
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Duration for running {algo} on {dataset} is: {duration:.6f} seconds")
    
    print("\nResults:")
    print(f"Successful: {', '.join(results['successful'])}")
    print(f"Failed: {', '.join(results['failed'])}")

if __name__ == "__main__":
    main()
    