'''
https://github.com/federhub/pyGRNN/blob/master/Tutorial/Tutorial_PyGRNN.ipynb
'''
import numpy as np
import pandas as pd
import os 
import sys 
import errno 
import pickle 

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.loss_utils import calculate_relative_loss

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error as MSE, r2_score

from pyGRNN import GRNN, feature_selection as FS

def analyze_features(X, y, feature_names=None):
    """Perform feature selection using pyGRNN's Isotropic selector"""
    
    print("\nPerforming feature analysis...")
    if feature_names is None:
        feature_names = [f'V{i+1}' for i in range(X.shape[1])]
        
    print("\nPerforming feature analysis 1")
    '''
    Example 1: use Isotropic Selector to explore data dependencies in the input 
    space by analyzing the relatedness between features 
    '''
    IsotropicSelector = FS.Isotropic_selector(bandwidth='rule-of-thumb')
    IsotropicSelector.relatidness(X, feature_names=feature_names)
    # IsotropicSelector.plot_(feature_names = feature_names)
    
    '''
    Example 2: use Isotropic Selector to perform an exhaustive search; a rule-of-thumb
    is used to select the optimal bandwidth for each subset of features
    '''
    # IsotropicSelector.es(X, y.ravel(), feature_names=feature_names)
    
    '''
    Example 3: use Isotropic Selector to perform a complete analysis of the input
    space, recongising relevant, redundant, irrelevant features
    '''
    selected_features = IsotropicSelector.feat_selection(X, y.ravel(), 
                                                         feature_names=feature_names, 
                                                         strategy='ffs')
    
    return {
        'selector': IsotropicSelector,
        'selected_features': selected_features
    }
    
def train_grnn_models(X, y, dump_dir, test_size=0.25, seed=695):
    """
    Train both Isotropic and Anisotropic GRNN models
    """
    X_numpy = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    y_numpy = y.reshape(-1, 1) if len(y.shape) == 1 else y
    
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    feature_analysis = analyze_features(X_numpy, y_numpy.ravel(), feature_names)
    selector = feature_analysis['selector']
    
    best_set = selector.best_inSpaceIndex
    
    average_sparsity = len(best_set) / X_numpy.shape[1]
    print(f"number of features selected is {len(best_set)}")
    print(f"\n average sparsity is: {average_sparsity:.3f}")
    
    X_selected = X_numpy[:, best_set]
    print(f"\nSelected {len(best_set)} features:")
                 
    # Scale the features and target
    scaler_X = preprocessing.MinMaxScaler()
    scaler_y = preprocessing.MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X_selected)
    y_scaled = scaler_y.fit_transform(y_numpy)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=seed)
    
    y_baseline = np.full_like(y_test, np.mean(y_train))
    
    # model 1: Isotropic GRNN with Grid Search
    IGRNN = GRNN()
    params_IGRNN = {
        'kernel': ["RBF"],
        'sigma': list(np.arange(0.1, 4, 0.01)),
        'calibration': ['None']
    }
    
    grid_IGRNN = GridSearchCV(
        estimator=IGRNN,
        param_grid=params_IGRNN,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1
    )
    
    print("Training Isotropic GRNN...")
    grid_IGRNN.fit(X_train, y_train.ravel())
    best_igrnn = grid_IGRNN.best_estimator_
    y_pred_igrnn = best_igrnn.predict(X_test)
    
    # back to original scale 
    y_pred_igrnn_original = scaler_y.inverse_transform(y_pred_igrnn.reshape(-1, 1))
    y_test_original = scaler_y.inverse_transform(y_test)
    
    mse_IGRNN = MSE(y_test_original, y_pred_igrnn_original)
    r2_IGRNN = r2_score(y_test_original, y_pred_igrnn_original)
    
    relative_loss_igrnn = calculate_relative_loss(
        y_test_original,
        y_pred_igrnn_original,
        scaler_y.inverse_transform(y_baseline.reshape(-1, 1)),
        response_type='continuous'
    )
    print(f"relative loss for igrnn is {relative_loss_igrnn}")
        
    # model 2: Anisotropic GRNN
    print("\nTraining Anisotropic GRNN...")
    AGRNN = GRNN(calibration="gradient_search")
    AGRNN.fit(X_train, y_train.ravel())
    y_pred_agrnn = AGRNN.predict(X_test)
    
    # back to original scale 
    y_pred_agrnn_original = scaler_y.inverse_transform(y_pred_agrnn.reshape(-1, 1))
    
    mse_AGRNN = MSE(y_test_original, y_pred_agrnn_original)
    r2_AGRNN = r2_score(y_test_original, y_pred_agrnn_original)
    
    relative_loss_agrnn = calculate_relative_loss(
        y_test_original,
        y_pred_agrnn_original,
        scaler_y.inverse_transform(y_baseline.reshape(-1, 1)),
        response_type='continuous'
    )
    print(f"relative loss for agrnn is {relative_loss_agrnn}")
    
    # Create a results dictionary with only serializable data
    results = {
        'feature_selection': {
            'num_selected_features': len(best_set),
            'sparsity': average_sparsity
        },
        'IGRNN': {
            'best_params': grid_IGRNN.best_params_,
            'mse': float(mse_IGRNN),
            'r2': float(r2_IGRNN),
            'relative_loss': float(relative_loss_igrnn)
        },
        'AGRNN': {
            'sigma': float(AGRNN.sigma) if not np.all(np.isnan(AGRNN.sigma)) else -999.0,
            'mse': float(mse_AGRNN) if not np.all(np.isnan(AGRNN.sigma)) else -999.0,
            'r2': float(r2_AGRNN) if not np.all(np.isnan(AGRNN.sigma)) else -999.0,
            'relative_loss': float(relative_loss_agrnn) if not np.all(np.isnan(AGRNN.sigma)) else -999.0
        }
    }
    
    # Save results to JSON
    import json
    with open(dump_dir, 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

def print_results(results):
    """Print the performance metrics for both models"""
    print("\nResults Summary:")
    print("===============")
    print(f"\nFeature Selection:")
    print(f"Sparsity: {results['feature_selection']['sparsity']:.3f}")
    print(f"Selected {len(results['feature_selection']['selected_features'])} features")
    
    print("\nIsotropic GRNN:")
    print(f"Best parameters: {results['IGRNN']['best_params']}")
    print(f"MSE: {results['IGRNN']['mse']:.3f}")
    print(f"R2 Score: {results['IGRNN']['r2']:.3f}")
    print(f"Relative Loss: {results['IGRNN']['relative_loss']:.3f}")
    
    print("\nAnisotropic GRNN:")
    print(f"MSE: {results['AGRNN']['mse']:.3f}")
    print(f"R2 Score: {results['AGRNN']['r2']:.3f}")
    print(f"Relative Loss: {results['AGRNN']['relative_loss']:.3f}")