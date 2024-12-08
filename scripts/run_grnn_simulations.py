import sys
import time
import logging
import os 
import errno

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.grnn_utils import train_grnn_models, print_results
from src.data_utils import generate_data

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
    (10000, 100, 10), 
]

result_dir = '/Users/amber/Desktop/ece695_proj/result/grnn/'

if not os.path.isdir(result_dir):
    try:
        os.makedirs(result_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(result_dir):
            pass
        else:
            raise
            
for n, p, k in settings:
    for response_type in ['continuous']:
        print(f"#########################################")
        print(f"################ n={n}, p={p}, k={k}, type={response_type} ################")
        print(f"#########################################")

        try:
            logger.info(f"Starting simulation: n={n}, p={p}, k={k}, type={response_type}")
            X, y, beta = generate_data(
                n=n, 
                p=p, 
                k=k, 
                response_type=response_type,
                rho=0.5,
                seed=31415
            )
            dump_dir=os.path.join(result_dir, f"n={n}_p={p}_k={k}_type={response_type}_grnn_results.json")
            results = train_grnn_models(
                X=X, 
                y=y, 
                dump_dir=dump_dir, 
                test_size=0.25, 
                seed=695, 
                )
            print_results(results)
            logger.info("Simulation completed successfully")
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            continue

end_time = time.time()
duration = end_time - start_time
logger.info(f"Duration for running lassonet simulations: {duration:.6f} seconds")

