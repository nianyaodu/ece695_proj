'''
https://github.com/riken-aip/pyHSICLasso/tree/master
https://github.com/lasso-net/lassonet/blob/master/experiments/run.py
'''
############## HSIC_lasso on 5 datasets ##############
import os
import errno
import pickle
import time
import logging
from typing import Tuple, Optional
import numpy as np
import sys 
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_utils import load_dataset, generate_data
from src.plot import plot_path
from src.loss_utils import calculate_relative_loss

import torch
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from lassonet import LassoNetClassifier, LassoNetRegressor
from lassonet.utils import eval_on_path

from pyHSICLasso import HSICLasso

def run_single_dataset(k: int,
                       dataset: str,   # "MNIST"
                       response_type: str, 
                       result_dir: str, 
                       n_epochs: int = 1000, 
                       seed: Optional[int] = None):
    """Run a single hsic_lassonet with dataset"""
    
    start_time = time.time()

    if not os.path.isdir(result_dir):
        try:
            os.makedirs(result_dir)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(result_dir):
                pass
            else:
                raise

    # Load dataset and split the data - here we use the same code as in lassonet to read in dataset 
    (X_train_valid, y_train_valid), (X_test, y_test) = load_dataset(dataset)
    
    if dataset == "Activity": 
        X_train_valid = X_train_valid.to_numpy().astype(np.float64)
        y_train_valid = y_train_valid.to_numpy().astype(np.int64)
        
        X_test = X_test.to_numpy().astype(np.float64) if X_test is not None else None
        y_test = y_test.to_numpy().astype(np.int64) if y_test is not None else None

    # Convert data types
    X_train_valid = X_train_valid.astype(np.float64)
    y_train_valid = y_train_valid.astype(np.int64)
    X_test = X_test.astype(np.float64)
    y_test = y_test.astype(np.int64)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_valid, y_train_valid, test_size=0.125, random_state=seed)

    default_batch_size = 256
    n_train = int(X_train.shape[0] * 0.64)  
    batch_size = min(default_batch_size, n_train)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    hsic_lasso = HSICLasso()
    if response_type == 'continuous':
        hsic_lasso.input(X_train, y_train)
        hsic_lasso.regression(k)
        
        y_pred_baseline = np.full_like(y_test, np.mean(y_train))
    else:
        unique_labels = np.unique(y_train_valid)
        num_classes = len(unique_labels)
        # Check if labels are consecutive integers starting from 0
        expected_labels = np.arange(num_classes)
        assert np.array_equal(unique_labels, expected_labels), "here we have categorical response, but labels are not consecutive integers starting from 0"
        
        print(f"X_train is {X_train}")
        print(f"y_train is {y_train}")
        hsic_lasso.input(X_train, y_train)
        hsic_lasso.classification(k)
        print(f"y_train.shape is {y_train.shape}")
        print(f"y_test.shape is {y_test.shape}")
        
        y_pred_baseline_proba = np.zeros((len(y_test), num_classes))
        
        for class_id in unique_labels: 
            prob = sum(y_train == class_id)/len(y_train)
            y_pred_baseline_proba[:, class_id] = prob
            
        # convert y_test into onehot vector 
        y_test_proba = np.zeros((len(y_test), num_classes))
        y_test_proba[np.arange(len(y_test)), y_test] = 1

    hsic_lasso.linkage(method="ward")  
    hsic_lasso.plot_heatmap(filepath=os.path.join(result_dir, f"{dataset}_type={response_type}_heatmap.png"))
    hsic_lasso.plot_dendrogram(filepath=os.path.join(result_dir, f"{dataset}_type={response_type}_dendrogram.png"))
    hsic_lasso.plot_path(filepath=os.path.join(result_dir, f"{dataset}_type={response_type}_path.png"))
    hsic_lasso.save_param(filename=os.path.join(result_dir, f"{dataset}_type={response_type}_param.csv"))
    
    # get the selected features from hsic_lasso
    selected_features = hsic_lasso.get_index()
    print(f"Number of selected features: {len(selected_features)}")
    print(f"X_train.shape is {X_train.shape}")
    average_sparsity = round(len(selected_features) / X_train.shape[1], 3)
    print(f"Average sparsity: {average_sparsity}")

    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Set the dimensions of the hidden layers
    hidden_dim = (X_test.shape[1] // 3,)
    # hidden_dim = (len(selected_features) // 3,)
    ModelClass = LassoNetRegressor if response_type == 'continuous' else LassoNetClassifier
    M = 0.0 if response_type == 'continuous' else 10
    
    # use LassoNetClassifier for training with the selected features 
    lasso_sparse = ModelClass(
        M=M,
        hidden_dims=hidden_dim,
        verbose=1,
        torch_seed=seed,
        random_state=seed,
        device=device,
        n_iters=n_epochs,
    )
    path_sparse = lasso_sparse.path(
        X_train_selected,
        y_train,
        X_val=X_val_selected,
        y_val=y_val,
        lambda_seq=[0.0, 0.001],
        return_state_dicts=True,
    )

    # Model evaluation
    if response_type == 'continuous':
        y_pred = lasso_sparse.predict(X_test_selected)
        relative_loss = calculate_relative_loss(y_test, y_pred, y_pred_baseline, response_type)
    else:
        y_pred = lasso_sparse.predict(X_test_selected)
        y_pred_proba = lasso_sparse.predict_proba(X_test_selected)
        relative_loss = calculate_relative_loss(y_test_proba, y_pred_proba, y_pred_baseline_proba, response_type)
    
    print(f"Relative loss: {relative_loss:.3f}")
    
    # Evaluate the model on the test data
    score = eval_on_path(lasso_sparse, path_sparse, X_test_selected, y_test)
    metric_name = "R2 score" if response_type == 'continuous' else "accuracy"
    print(f"Test {metric_name} in {dataset}, type={response_type}: {score}")
    if score[0] != score[1]:
        logging.warning(f"Two scores are not the same: {score}")

    # Save results
    results = {
        'relative_loss': relative_loss,
        'sparsity': average_sparsity,
        'score': score,
        'selected_features': selected_features
    }

    if response_type == 'continuous':
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE: {mse:.3f}")
        results['mse'] = mse
        results['r2_score'] = r2_score(y_test, y_pred)
    else:
        test_accuracy = np.mean(y_pred == y_test)
        results['accuracy'] = test_accuracy
            
    with open(os.path.join(result_dir, f"{dataset}_type={response_type}_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(result_dir, f"{dataset}_type={response_type}_lasso_sparse.pkl"), "wb") as f:
        pickle.dump(lasso_sparse, f)
    with open(os.path.join(result_dir, f"{dataset}_type={response_type}_path_scarse.pkl"), "wb") as f:
        pickle.dump(path_sparse, f)

    # visualizations 
    # save_path = os.path.join(result_dir, f'{dataset}_type={response_type}_path_analysis.png')
    # plot_path(model=lasso_sparse, X_test=X_test, y_test=y_test, save_path=save_path)


    # export the total running time 
    end_time = time.time()
    print(f"Duration for running {dataset}, type={response_type}: {end_time - start_time} seconds")


def run_single_simulation(n: int, 
                          p: int, 
                          k: int, 
                          response_type: str, 
                          result_dir: str,
                          n_epochs: int = 1000, 
                          seed: Optional[int] = None):
    """Run a single hsic_LassoNet with simulated data"""
    
    start_time = time.time()
    
    if not os.path.isdir(result_dir):
        try:
            os.makedirs(result_dir)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(result_dir):
                pass
            else:
                raise

    # generate simulate data
    X, y, true_beta = generate_data(n, p, k, response_type, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    print(f"X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}, X_test.shape: {X_test.shape}")
    print(f"y_train.shape: {y_train.shape}, y_val.shape: {y_val.shape}, y_test.shape: {y_test.shape}")
    
    default_batch_size = 256
    n_train = int(n * 0.64)  
    batch_size = min(default_batch_size, n_train)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    # Initialize model
    hsic_lasso = HSICLasso()
    if response_type == 'continuous':
        hsic_lasso.input(X_train, y_train)
        hsic_lasso.regression(k)
        
        y_pred_baseline = np.full_like(y_test, np.mean(y_train))
    else:
        unique_labels = np.unique(y_train)
        num_classes = len(unique_labels)
        # Check if labels are consecutive integers starting from 0
        expected_labels = np.arange(num_classes)
        assert np.array_equal(unique_labels, expected_labels), "here we have categorical response, but labels are not consecutive integers starting from 0"
        
        hsic_lasso.input(X_train, y_train)
        hsic_lasso.classification(k)
        
        y_pred_baseline_proba = np.zeros((len(y_test), num_classes))
        
        for class_id in unique_labels: 
            prob = sum(y_train == class_id)/len(y_train)
            y_pred_baseline_proba[:, class_id] = prob
            
        # convert y_test into onehot vector 
        y_test_proba = np.zeros((len(y_test), num_classes))
        y_test_proba[np.arange(len(y_test)), y_test] = 1
        
    # Save HSIC Lasso visualizations
    hsic_lasso.linkage(method="ward")
    hsic_lasso.plot_heatmap(filepath=os.path.join(result_dir, f"simulation_n={n}_p={p}_k={k}_type={response_type}_heatmap.png"))
    hsic_lasso.plot_dendrogram(filepath=os.path.join(result_dir, f"simulation_n={n}_p={p}_k={k}_type={response_type}_dendrogram.png"))
    hsic_lasso.plot_path(filepath=os.path.join(result_dir, f"simulation_n={n}_p={p}_k={k}_type={response_type}_path.png"))
    
    # Get selected features from hsic_lasso
    selected_features = hsic_lasso.get_index()
    print(f"Number of features selected by HSIC Lasso: {len(selected_features)}")
    average_sparsity=round(len(selected_features) / X_train.shape[1], 3)
    print(f"Average sparsity: {average_sparsity}")
    
    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Set the dimensions of the hidden layers
    hidden_dim = (X_test.shape[1] // 3,)
    ModelClass = LassoNetRegressor if response_type == 'continuous' else LassoNetClassifier
    M = 0.0 if response_type == 'continuous' else 10
    
    # use LassoNetClassifier for training with the selected features 
    lasso_sparse = ModelClass(
        M=M,
        hidden_dims=hidden_dim,
        verbose=1,
        torch_seed=seed,
        random_state=seed,
        device=device,
        n_iters=n_epochs,
    )
    path_sparse = lasso_sparse.path(
        X_train_selected,
        y_train,
        X_val=X_val_selected,
        y_val=y_val,
        lambda_seq=[0.0, 0.001],
        return_state_dicts=True,
    )

    # Model evaluation
    if response_type == 'continuous':
        y_pred = lasso_sparse.predict(X_test_selected)
        relative_loss = calculate_relative_loss(y_test, y_pred, y_pred_baseline, response_type)
        
        print(f"Mean of predictions: {np.mean(y_pred)}")
        print(f"Std of predictions: {np.std(y_pred)}")
        print(f"Mean of true values: {np.mean(y_test)}")
        print(f"Std of true values: {np.std(y_test)}")
    else:
        y_pred = lasso_sparse.predict(X_test_selected)
        y_pred_proba = lasso_sparse.predict_proba(X_test_selected)
        relative_loss = calculate_relative_loss(y_test_proba, y_pred_proba, y_pred_baseline_proba, response_type)
    
    print(f"Relative loss: {relative_loss:.3f}")
    
    # Evaluate the model on the test data
    score = eval_on_path(lasso_sparse, path_sparse, X_test_selected, y_test)
    metric_name = "R2 score" if response_type == 'continuous' else "accuracy"
    print(f"Test {metric_name} with n={n}, p={p}, k={k}, type={response_type}: {score}")
    if score[0] != score[1]:
        logging.warning(f"Two scores are not the same: {score}")
        
    # Save results
    results = {
        'relative_loss': relative_loss,
        'sparsity': average_sparsity,
        'score': score,
        'selected_features': selected_features
    }

    if response_type == 'continuous':
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE: {mse:.3f}")
        results['mse'] = mse
        results['r2_score'] = r2_score(y_test, y_pred)
    else:
        test_accuracy = np.mean(y_pred == y_test)
        results['accuracy'] = test_accuracy
            
    with open(os.path.join(result_dir, f"simulation_n={n}_p={p}_k={k}_type={response_type}_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(result_dir, f"simulation_n={n}_p={p}_k={k}_type={response_type}_lasso_sparse.pkl"), "wb") as f:
        pickle.dump(lasso_sparse, f)
    with open(os.path.join(result_dir, f"simulation_n={n}_p={p}_k={k}_type={response_type}_path_scarse.pkl"), "wb") as f:
        pickle.dump(path_sparse, f)
    
    # visualizations 
    # save_path = os.path.join(result_dir, f'simulation_n={n}_p={p}_k={k}_type={response_type}_path_analysis.png')
    # plot_path(model=lasso_sparse, X_test=X_test, y_test=y_test, save_path=save_path)

    # export the total running time 
    end_time = time.time()
    print(f"Duration for running in the setting n={n}, p={p}, k={k}, type={response_type}: {end_time - start_time} seconds")
    
    