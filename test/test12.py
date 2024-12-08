'''
https://github.com/federhub/pyGRNN
'''
############## test 12 - test experiment using pygrnn ##############

import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from pyGRNN import GRNN
# Loading the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(preprocessing.minmax_scale(X),
                                                    preprocessing.minmax_scale(y.reshape((-1, 1))),
                                                    test_size=0.25)

# Example 1: use Isotropic GRNN with a Grid Search Cross validation to select the optimal bandwidth
IGRNN = GRNN()
params_IGRNN = {'kernel':["RBF"],
                'sigma' : list(np.arange(0.1, 4, 0.01)),
                'calibration' : ['None']
                 }
grid_IGRNN = GridSearchCV(estimator=IGRNN,
                          param_grid=params_IGRNN,
                          scoring='neg_mean_squared_error',
                          cv=5,
                          verbose=1
                          )
grid_IGRNN.fit(X_train, y_train.ravel())
best_model = grid_IGRNN.best_estimator_
y_pred = best_model.predict(X_test)
mse_IGRNN = MSE(y_test, y_pred)

# Example 2: use Anisotropic GRNN with Limited-Memory BFGS algorithm to select the optimal bandwidths
AGRNN = GRNN(calibration="gradient_search")
AGRNN.fit(X_train, y_train.ravel())
sigma=AGRNN.sigma 
y_pred = AGRNN.predict(X_test)
mse_AGRNN = MSE(y_test, y_pred)
