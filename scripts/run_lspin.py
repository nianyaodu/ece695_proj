
import time
import logging
import sys 
import os
import traceback

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.lspin_utils import run_single_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start_time = time.time()

settings = [
    ('MNIST', 'categorical'), 
    ('MNIST-Fashion', 'categorical'),
    ('MICE', 'categorical'),
    ('COIL', 'categorical'),
    ('ISOLET', 'categorical')
]

result_dir = '/Users/amber/Desktop/ece695_proj/result/lspin/'

for dataset, response_type in settings:
    print(f"#########################################")
    print(f"################ dataset={dataset}, response_type={response_type} ################")
    print(f"#########################################")
    try:
        logger.info(f"Starting processing for {dataset}")
        run_single_dataset(
            k=50,
            dataset=dataset,
            response_type=response_type,
            result_dir=result_dir,
            n_epochs=1000,
            seed=31415,
        )
        logger.info(f"Successfully completed {dataset}")
    except Exception as e:
        logger.error(f"Error processing {dataset}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        continue
end_time = time.time()
duration = end_time - start_time
logger.info(f"Duration for running lspin: {duration:.6f} seconds")

