import sys 
import time
import logging
import os 

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.hsic_lassonet_utils import run_single_simulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start_time = time.time()

settings = [
    (100, 5, 2),
    (1000, 5, 2),
    (10000, 5, 2),
    (100, 20, 5),
    (1000, 20, 5),
    (10000, 20, 5),
    (100, 100, 10),
    (1000, 100, 10),
    (10000, 100, 10)
]

result_dir = '/Users/amber/Desktop/ece695_proj/result/hsic_lasso/'

for n, p, k in settings:
    for response_type in ['continuous', 'binary']:
        print(f"#########################################")
        print(f"################ n={n}, p={p}, k={k}, type={response_type} ################")
        print(f"#########################################")

        try:
            logger.info(f"Starting simulation: n={n}, p={p}, k={k}, type={response_type}")
            run_single_simulation(
                n=n, 
                p=p, 
                k=k, 
                response_type=response_type, 
                result_dir=result_dir, 
                n_epochs=1000,
                seed=31415,)
            logger.info("Simulation completed successfully")
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            continue

end_time = time.time()
duration = end_time - start_time
logger.info(f"Duration for running hsic_lasso: {duration:.6f} seconds")

