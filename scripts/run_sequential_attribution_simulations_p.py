import os
import subprocess
import sys
import time
import logging
import errno
from pathlib import Path
import numpy as np 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from sequential_attention.experiments import hyperparams_sa
from data_utils import generate_data

def setup_directories(base_dir):
    """Create necessary directories if they don't exist."""
    if not os.path.isdir(base_dir):
        try:
            os.makedirs(base_dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(base_dir):
                pass
            else:
                raise

def run_experiment(X, y, dataset_id, base_dir="/Users/amber/Desktop/ece695_proj/result", algo="sa"):
    """Run experiment for a specific algorithm on synthetic dataset."""
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{src_path}:{env.get('PYTHONPATH', '')}"
    
    # Save synthetic data to temporary files
    dataset_dir = os.path.join(base_dir, algo, dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)
    
    data_path = os.path.join(dataset_dir, 'data.npz')
    np.savez(data_path, X=X, y=y)
    
    cmd = [
        "conda", "run", "-n", "sa",
        "python", "-m", "sequential_attention.experiments.run",
        f"--data_name=synthetic",
        f"--data_path={data_path}",
        f"--algo={algo}",
        f"--model_dir={os.path.join(base_dir, algo, dataset_id)}",
        "--batch_size=32", 
        "--num_epochs_select=250",
        "--num_epochs_fit=250",
        "--learning_rate=0.001",
        "--decay_steps=1000",
        "--decay_rate=0.9",
    ]
    
    logger.info(f"Running {algo} on synthetic dataset {dataset_id}...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        success = subprocess.run(cmd, check=True, env=env).returncode == 0
        if success:
            logger.info(f"Successfully completed {algo} on {dataset_id}")
        else:
            logger.error(f"Failed to run {algo} on {dataset_id}")
        return success
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {algo} on {dataset_id}: {e}")
        return False

def main():
    start_time = time.time()
    
    # Define synthetic data settings
    settings = [
        (100, 500, 10),
        (100, 500, 20),
        (100, 500, 50),
        (1000, 3000, 100),
        (1000, 3000, 200),
    ]
    
    # Define algorithms to run
    algos = ["sa", "lly", "seql", "gl", "omp"]
    
    # Setup results directory
    base_dir = '/Users/amber/Desktop/ece695_proj/result'
    # setup_directories(base_dir)
    
    results = {"successful": [], "failed": []}
    
    for n, p, k in settings:
        dataset_id = f"n{n}_p{p}_k{k}"
        logger.info(f"\nGenerating synthetic dataset: {dataset_id}")
        
        try:
            # Generate synthetic data
            X, y, beta = generate_data(
                n=n,
                p=p,
                k=k,
                response_type='categorical',
                rho=0.5,
                seed=31415
            )
            
            # Run each algorithm on the generated dataset
            for algo in algos:
                exp_id = f"{dataset_id}_{algo}"
                if run_experiment(X, y, dataset_id, algo=algo, base_dir=base_dir):
                    results["successful"].append(exp_id)
                else:
                    results["failed"].append(exp_id)
                
        except Exception as e:
            logger.error(f"Error in dataset generation/processing: {str(e)}")
            continue
    
    # Print summary
    logger.info("\nExperiment Summary:")
    logger.info(f"Successful experiments: {', '.join(results['successful'])}")
    if results["failed"]:
        logger.info(f"Failed experiments: {', '.join(results['failed'])}")
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Total duration: {duration:.2f} seconds")

if __name__ == "__main__":
    main()