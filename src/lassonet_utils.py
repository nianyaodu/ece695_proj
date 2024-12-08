'''
https://github.com/lasso-net/lassonet/blob/master/experiments/run.py
'''
############## lassonet on 5 datasets ##############
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


def run_single_dataset(k: int,
                       dataset: str,   # "MNIST"
                       response_type: str, 
                       result_dir: str, 
                       n_epochs: int = 1000, 
                       seed: Optional[int] = None):
    """Run a single lassonet with dataset"""
    
    start_time = time.time()

    if not os.path.isdir(result_dir):
        try:
            os.makedirs(result_dir)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(result_dir):
                pass
            else:
                raise

    # Load dataset and split the data
    (X_train_valid, y_train_valid), (X_test, y_test) = load_dataset(dataset)
    
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

    # Set the dimensions of the hidden layers
    hidden_dim = (X_test.shape[1] // 3,)
    ModelClass = LassoNetRegressor if response_type == 'continuous' else LassoNetClassifier
    M = 0.0 if response_type == 'continuous' else 10
    
    if response_type == 'continuous':
        y_pred_baseline = np.full_like(y_test, np.mean(y_train))
    else:
        unique_labels = np.unique(y_train)
        num_classes = len(unique_labels)
        # Check if labels are consecutive integers starting from 0
        expected_labels = np.arange(num_classes)
        assert np.array_equal(unique_labels, expected_labels), "here we have categorical response, but labels are not consecutive integers starting from 0"
        
        y_pred_baseline_proba = np.zeros((len(y_test), num_classes))
        
        for class_id in unique_labels: 
            prob = sum(y_train == class_id)/len(y_train)
            y_pred_baseline_proba[:, class_id] = prob
            
        # convert y_test into onehot vector 
        y_test_proba = np.zeros((len(y_test), num_classes))
        y_test_proba[np.arange(len(y_test)), y_test] = 1

        
    # Initialize the LassoNetClassifier model and compute the path
    lasso_model = ModelClass(
        M=M,
        hidden_dims=hidden_dim,
        verbose=1,
        torch_seed=seed,
        random_state=seed,
        device=device,
        n_iters=n_epochs,
        batch_size=batch_size,
    )
    path = lasso_model.path(X_train, y_train, X_val=X_val, y_val=y_val, return_state_dicts=True)

    # Select the features
    desired_save = next(save for save in path if save.selected.sum().item() <= k)
    SELECTED_FEATURES = desired_save.selected
    print(f"Number of selected features in {dataset}: {SELECTED_FEATURES.sum().item()}")
    average_sparsity = round(SELECTED_FEATURES.sum().item() / X_train.shape[1], 3)
    print(f"Average sparsity: {average_sparsity}")
    
    # added for activity data
    if dataset == "Activity":
        SELECTED_FEATURES = SELECTED_FEATURES.cpu().numpy()
        selected_features = np.array(SELECTED_FEATURES, dtype=bool)
        selected_indices = np.where(selected_features)[0]
        X_train_selected = X_train.iloc[:, selected_indices]
        X_val_selected = X_val.iloc[:, selected_indices]
        X_test_selected = X_test.iloc[:, selected_indices]
        
    else: 
        # Select the features from the training, validation, and test data
        X_train_selected = X_train[:, SELECTED_FEATURES]
        X_val_selected = X_val[:, SELECTED_FEATURES]
        X_test_selected = X_test[:, SELECTED_FEATURES]

    # Initialize another LassoNetClassifier for retraining with the selected features
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
        if isinstance(y_test_proba, torch.Tensor):
            print("1")
            y_test_proba = y_test_proba.cpu().numpy()
        if isinstance(y_pred_proba, torch.Tensor):
            print("2")
            y_pred_proba = y_pred_proba.cpu().numpy()
        if isinstance(y_pred_baseline_proba, torch.Tensor):
            print("3")
            y_pred_baseline_proba = y_pred_baseline_proba.cpu().numpy()
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
        'relative_loss': float(relative_loss) if torch.is_tensor(relative_loss) else relative_loss,
        'sparsity': float(average_sparsity) if torch.is_tensor(average_sparsity) else average_sparsity,
        'score': [float(s) if torch.is_tensor(s) else s for s in score],
        'selected_features': SELECTED_FEATURES.cpu().numpy().tolist() if torch.is_tensor(SELECTED_FEATURES) else SELECTED_FEATURES.tolist()
    }
    
    if response_type == 'continuous':
        mse = mean_squared_error(y_test, y_pred)
        results['mse'] = float(mse) if torch.is_tensor(mse) else mse
        results['r2_score'] = float(r2_score(y_test, y_pred))
    else:
        test_accuracy = np.mean(y_pred == y_test)
        results['accuracy'] = float(test_accuracy) if torch.is_tensor(test_accuracy) else test_accuracy

    # save the results
    with open(os.path.join(result_dir, f"{dataset}_type={response_type}_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    with open(os.path.join(result_dir, f"{dataset}_type={response_type}_lasso_model.pkl"), "wb") as f:
        pickle.dump(lasso_model, f)
    with open(os.path.join(result_dir, f"{dataset}_type={response_type}_path.pkl"), "wb") as f:
        pickle.dump(path, f)
    with open(os.path.join(result_dir, f"{dataset}_type={response_type}_lasso_sparse.pkl"), "wb") as f:
        pickle.dump(lasso_sparse, f)
    with open(os.path.join(result_dir, f"{dataset}_type={response_type}_path_scarse.pkl"), "wb") as f:
        pickle.dump(path_sparse, f)
    

    # visualizations 
    save_path = os.path.join(result_dir, f'{dataset}_type={response_type}_path_analysis.png')
    plot_path(model=lasso_model, X_test=X_test, y_test=y_test, save_path=save_path)

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
    """Run a single LassoNet with simulated data"""
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
    hidden_dim = (p // 3,)
    ModelClass = LassoNetRegressor if response_type == 'continuous' else LassoNetClassifier
    M = 0.0 if response_type == 'continuous' else 10
    
    if response_type == 'continuous':
        y_pred_baseline = np.full_like(y_test, np.mean(y_train))
    else:
        unique_labels = np.unique(y_train)
        num_classes = len(unique_labels)
        # Check if labels are consecutive integers starting from 0
        expected_labels = np.arange(num_classes)
        assert np.array_equal(unique_labels, expected_labels), "here we have categorical response, but labels are not consecutive integers starting from 0"
        
        y_pred_baseline_proba = np.zeros((len(y_test), num_classes))
        
        for class_id in unique_labels: 
            prob = sum(y_train == class_id)/len(y_train)
            y_pred_baseline_proba[:, class_id] = prob
            
        # convert y_test into onehot vector 
        y_test_proba = np.zeros((len(y_test), num_classes))
        y_test_proba[np.arange(len(y_test)), y_test] = 1
      
    lasso_model = ModelClass(
        M=M,
        hidden_dims=hidden_dim,
        verbose=1,
        torch_seed=seed,
        random_state=seed,
        device=device,
        n_iters=n_epochs,
        batch_size=batch_size,
    )
    path = lasso_model.path(X_train, y_train, X_val=X_val, y_val=y_val, return_state_dicts=True)
    
    # Select the features
    desired_save = next(save for save in path if save.selected.sum().item() <= k)
    SELECTED_FEATURES = desired_save.selected
    print(f"Number of selected features in the setting n={n}, p={p}, k={k}: {SELECTED_FEATURES.sum().item()}")
    print(f"X_train.shape is {X_train.shape}")
    print(f"len(SELECTED_FEATURES) is {len(SELECTED_FEATURES)}")
    print(f"SELECTED_FEATURES is {SELECTED_FEATURES}")
    print(f"SELECTED_FEATURES.sum() is {SELECTED_FEATURES.sum()}")
    print(f"SELECTED_FEATURES.sum().item() is {SELECTED_FEATURES.sum().item()}")
    average_sparsity = round(SELECTED_FEATURES.sum().item() / X_train.shape[1], 3)
    print(f"Average sparsity: {average_sparsity}")

    # Select the features from the training, validation, and test data
    X_train_selected = X_train[:, SELECTED_FEATURES]
    X_val_selected = X_val[:, SELECTED_FEATURES]
    X_test_selected = X_test[:, SELECTED_FEATURES]

    # Initialize another LassoNetClassifier for retraining with the selected features
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
    print(f"Test {metric_name} in n={n}, p={p}, k={k}, type={response_type}: {score}")
    if score[0] != score[1]:
        logging.warning(f"Two scores are not the same: {score}")

    # Save results
    results = {
        'relative_loss': float(relative_loss) if torch.is_tensor(relative_loss) else relative_loss,
        'sparsity': float(average_sparsity) if torch.is_tensor(average_sparsity) else average_sparsity,
        'score': float(score) if torch.is_tensor(score) else score,
        'selected_features': SELECTED_FEATURES.cpu().numpy().tolist() if torch.is_tensor(SELECTED_FEATURES) else SELECTED_FEATURES.tolist()
    }
    
    if response_type == 'continuous':
        mse = mean_squared_error(y_test, y_pred)
        results['mse'] = float(mse) if torch.is_tensor(mse) else mse
        results['r2_score'] = float(r2_score(y_test, y_pred))
    else:
        test_accuracy = np.mean(y_pred == y_test)
        results['accuracy'] = float(test_accuracy) if torch.is_tensor(test_accuracy) else test_accuracy

    # save the results
    with open(os.path.join(result_dir, f"simulation_n={n}_p={p}_k={k}_type={response_type}_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(result_dir, f"simulation_n={n}_p={p}_k={k}_type={response_type}_lasso_model.pkl"), "wb") as f:
        pickle.dump(lasso_model, f)
    with open(os.path.join(result_dir, f"simulation_n={n}_p={p}_k={k}_type={response_type}_path.pkl"), "wb") as f:
        pickle.dump(path, f)
    with open(os.path.join(result_dir, f"simulation_n={n}_p={p}_k={k}_type={response_type}_lasso_sparse.pkl"), "wb") as f:
        pickle.dump(lasso_sparse, f)
    with open(os.path.join(result_dir, f"simulation_n={n}_p={p}_k={k}_type={response_type}_path_scarse.pkl"), "wb") as f:
        pickle.dump(path_sparse, f)
    
    # visualizations 
    save_path = os.path.join(result_dir, f'simulation_n={n}_p={p}_k={k}_type={response_type}_path_analysis.png')
    plot_path(model=lasso_model, X_test=X_test, y_test=y_test, save_path=save_path)

    # export the total running time 
    end_time = time.time()
    print(f"Duration for running in the setting n={n}, p={p}, k={k}, type={response_type}: {end_time - start_time} seconds")
    