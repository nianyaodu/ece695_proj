import os 
import time
import logging
import sys 
import errno 

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.grnn_utils import train_grnn_models, print_results
from src.data_utils import load_parkinsons_data, load_california_housing_data, load_boston_housing_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start_time = time.time()

result_dir = '/Users/amber/Desktop/ece695_proj/result/grnn/'

if not os.path.isdir(result_dir):
    try:
        os.makedirs(result_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(result_dir):
            pass
        else:
            raise
try:
    logger.info(f"Starting processing for grnn")
    # X, y = load_parkinsons_data()
    X, y = load_california_housing_data(test=True)
    dump_dir=os.path.join(result_dir, f"grnn_california_housing_results.json")
    results = train_grnn_models(
        X=X, 
        y=y, 
        dump_dir=dump_dir, 
        test_size=0.25, 
        seed=695, 
    )
    print_results(results)
    logger.info(f"Successfully completed grnn")
except Exception as e:
    logger.error(f"Error processing grnn: {str(e)}")

end_time = time.time()
duration = end_time - start_time
logger.info(f"Duration for running grnn: {duration:.6f} seconds")

