import os
import time
import logging
import sys
import errno
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_utils import load_parkinsons_data
from src.tabular_feature_selection.train_deep_model import main as train_main
import src.tabular_feature_selection.deep_lasso as deep_lasso
from src.tabular_feature_selection.deep_tabular.utils.testing import get_feat_importance_deeplasso
import src.tabular_feature_selection.deep_tabular.utils.data_tools as data_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup result directory
result_dir = '/Users/amber/Desktop/ece695_proj/result/deep_lasso/'

class ParkinsonsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], None, self.y[idx]

def get_dataloaders():
    """Create and return dataloaders for Parkinson's dataset"""
    logger.info("Preparing dataloaders...")
    
    X, y = load_parkinsons_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=695)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=695)
    
    train_dataset = ParkinsonsDataset(X_train, y_train)
    val_dataset = ParkinsonsDataset(X_val, y_val)
    test_dataset = ParkinsonsDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    
    return loaders, [], X.shape[1], 1

def print_results(stats, importances):
    """Print and save results"""
    logger.info("\nTraining Results:")
    logger.info(f"Test Stats: {stats['test_stats']}")
    logger.info(f"Validation Stats: {stats['val_stats']}")
    
    # Save importances
    importance_file = os.path.join(result_dir, 'feature_importances.txt')
    with open(importance_file, 'w') as f:
        for idx, importance in enumerate(importances):
            f.write(f"Feature {idx}: {importance:.4f}\n")
    logger.info(f"Feature importances saved to {importance_file}")

@hydra.main(config_path="config", config_name="train_model")
def main(cfg: DictConfig):
    if not os.path.isdir(result_dir):
        try:
            os.makedirs(result_dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(result_dir):
                pass
            else:
                raise

    start_time = time.time()

    try:
        logger.info("Starting Deep LASSO processing for Parkinson's dataset")
        
        # Override config settings
        cfg.dataset.task = "regression"
        cfg.dataset.name = "parkinsons"
        cfg.mode = "feature_selection"
        cfg.hyp.regularization = "deep_lasso"
        
        # Replace the get_dataloaders function in data_tools
        data_tools.get_dataloaders = get_dataloaders
        
        stats, importances = train_main(cfg)
        
        print_results(stats, importances)
        
        logger.info("Successfully completed Deep LASSO processing")
        
    except Exception as e:
        logger.error(f"Error processing Deep LASSO: {str(e)}")
        raise
    
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Duration for running Deep LASSO: {duration:.6f} seconds")
    
    return stats, importances

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in main program: {str(e)}")
        sys.exit(1)