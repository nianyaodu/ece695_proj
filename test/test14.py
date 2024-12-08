'''
https://github.com/lasso-net/lassonet/blob/master/examples/diabetes.py
'''
############## test 14 - test regression tasks using lassonet #############

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale

from lassonet import LassoNetRegressor, plot_path

import numpy as np 

def calculate_relative_loss(y_true, y_pred, y_pred_baseline, response_type='continuous'):
    """
    Calculate relative loss compared to intercept-only model
    
    :param y_true: true labels. if not continuous response, then this should be onehot vector. 
    :param y_pred: model predictions. if not continuous response, then this should be onehot vector. 
    :param y_pred_baseline: baseline (intercept-only) model predictions. if not continuous response, then this should be onehot vector. 
    :param response_type: 'continuous' or 'categorical'
    
    :return relative_loss: relative loss of the model
    """
    eps = 1e-15  # avoid log(0)
    
    if response_type == 'continuous':
        loss_mean = np.mean((y_true - y_pred) ** 2)
        loss_baseline_mean = np.mean((y_true - y_pred_baseline) ** 2)
        
        loss_sum = np.sum((y_true - y_pred) ** 2)
        loss_baseline_sum = np.sum((y_true - y_pred_baseline) ** 2)
        
    else:
        print(f"calculating cross entropy loss for categorical response")
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_pred_baseline = np.clip(y_pred_baseline, eps, 1 - eps)
        
        loss = -np.mean(y_true * np.log(y_pred))
        loss_baseline = -np.mean(y_true * np.log(y_pred_baseline))
            
    print(f"loss is {loss}")
    print(f"loss_baseline is {loss_baseline}")
    
    relative_loss_mean = loss_mean / loss_baseline_mean if loss_baseline_mean != 0 else float('inf')
    relative_loss_mean = loss_sum / loss_baseline_sum if loss_baseline_sum != 0 else float('inf')
    
    return relative_loss

dataset = load_diabetes()
X = dataset.data
y = dataset.target
_, true_features = X.shape
# add dummy feature
X = np.concatenate([X, np.random.randn(*X.shape)], axis=1)
feature_names = list(dataset.feature_names) + ["fake"] * true_features

# standardize
X = StandardScaler().fit_transform(X)
y = scale(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LassoNetRegressor(
    M=1, 
    hidden_dims=(50,),
    verbose=True,
)
path = model.path(X_train, y_train, return_state_dicts=True)

# plot_path(model, X_test, y_test)

# plt.savefig("diabetes.png")

# plt.clf()

# n_features = X.shape[1]
# importances = model.feature_importances_.numpy()
# order = np.argsort(importances)[::-1]
# importances = importances[order]
# ordered_feature_names = [feature_names[i] for i in order]
# color = np.array(["g"] * true_features + ["r"] * (n_features - true_features))[order]


# plt.subplot(211)
# plt.bar(
#     np.arange(n_features),
#     importances,
#     color=color,
# )
# plt.xticks(np.arange(n_features), ordered_feature_names, rotation=90)
# colors = {"real features": "g", "fake features": "r"}
# labels = list(colors.keys())
# handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
# plt.legend(handles, labels)
# plt.ylabel("Feature importance")

# _, order = np.unique(importances, return_inverse=True)

# plt.subplot(212)
# plt.bar(
#     np.arange(n_features),
#     order + 1,
#     color=color,
# )
# plt.xticks(np.arange(n_features), ordered_feature_names, rotation=90)
# plt.legend(handles, labels)
# plt.ylabel("Feature order")

# plt.savefig("diabetes-bar.png")

# redefine selected features and use previous y 

desired_save = next(save for save in path if save.selected.sum().item() <= 20)
SELECTED_FEATURES = desired_save.selected

X_train_selected = X_train[:, SELECTED_FEATURES]
X_test_selected = X_test[:, SELECTED_FEATURES]
    
lasso_sparse = LassoNetRegressor(
    M=1,
    hidden_dims=(50,),
    verbose=True,
)
path_sparse = lasso_sparse.path(
    X_train_selected,
    y_train,
    lambda_seq=[0.0, 0.001],
    return_state_dicts=True,
)

y_pred_baseline = np.full_like(y_test, np.mean(y_train))
y_pred = lasso_sparse.predict(X_test_selected)
relative_loss = calculate_relative_loss(y_test, y_pred, y_pred_baseline, "continuous")
        
    