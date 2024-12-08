'''
https://github.com/jcyang34/lspin/blob/dev/Demo1.ipynb
'''
############## test 11 - test experiment using lspin (synthenic data) - demo1 ##############
import numpy as np
import scipy
import sys
import os
import optuna
import time 
import errno 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import cm,colors

from lspin_model import Model
from lspin_model import convertToOneHot, DataSet_meta

start_time = time.time()

result_dir = '/Users/amber/Desktop/ece695_proj/test/test11_result/'

if not os.path.isdir(result_dir):
    try:
        os.makedirs(result_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(result_dir):
            pass
        else:
            raise
        
np.random.seed(34)

Xs1 = np.random.normal(loc=1,scale=0.5,size=(300,5))
Ys1 = -2*Xs1[:,0]+1*Xs1[:,1]-0.5*Xs1[:,2]

Xs2 = np.random.normal(loc=-1,scale=0.5,size=(300,5))
Ys2 = -0.5*Xs2[:,2]+1*Xs2[:,3]-2*Xs2[:,4]

X_data = np.concatenate((Xs1,Xs2),axis=0)
Y_data = np.concatenate((Ys1.reshape(-1,1),Ys2.reshape(-1,1)),axis=0)

Y_data = Y_data-Y_data.min()
Y_data=Y_data/Y_data.max()

# The ground truth group label of each sample
case_labels = np.concatenate((np.array([1]*300),np.array([2]*300)))

Y_data = np.concatenate((Y_data,case_labels.reshape(-1,1)),axis=1)

# 10% for validation, 10% for test 
X_train,X_remain,yc_train,yc_remain = train_test_split(X_data,Y_data,train_size=0.8,shuffle=True,random_state=34)
X_valid,X_test,yc_valid,yc_test = train_test_split(X_remain,yc_remain,train_size=0.5,shuffle=True,random_state=34)

# Only 10 samples used for training
X_train,_,yc_train,_ = train_test_split(X_train,yc_train,train_size=10,shuffle=True,random_state=34)

print("Sample sizes:")
print(X_train.shape[0],X_valid.shape[0],X_test.shape[0])

y_train = yc_train[:,0].reshape(-1,1)
y_valid = yc_valid[:,0].reshape(-1,1)
y_test = yc_test[:,0].reshape(-1,1)

train_label = yc_train[:,1]
valid_label = yc_valid[:,1]
test_label= yc_test[:,1]

Counter(train_label)

Counter(valid_label)

dataset = DataSet_meta(**{'_data':X_train, '_labels':y_train,'_meta':y_train,
                '_valid_data':X_valid, '_valid_labels':y_valid,'_valid_meta':y_valid,
                '_test_data':X_test, '_test_labels':y_test,'_test_meta':y_test})

# reference ground truth feature matrix (training/test)
ref_feat_mat_train = np.array([[1,1,1,0,0] if label == 1 else [0,0,1,1,1] for label in train_label])
ref_feat_mat_test = np.array([[1,1,1,0,0] if label == 1 else [0,0,1,1,1] for label in test_label])

# run llspin 
# hyper-parameter specification
model_params = {     
    "input_node" : X_train.shape[1],       # input dimension for the prediction network
    "hidden_layers_node" : [100,100,10,1], # number of nodes for each hidden layer of the prediction net
    "output_node" : 1,                     # number of nodes for the output layer of the prediction net
    "feature_selection" : True,            # if using the gating net
    "gating_net_hidden_layers_node": [10], # number of nodes for each hidden layer of the gating net
    "display_step" : 500                   # number of epochs to output info
}
model_params['activation_pred']= 'none' # linear prediction
model_params['activation_gating'] = 'tanh'

model_params['batch_normalization'] = 'False'

training_params = {
    'batch_size':X_train.shape[0]
} 
# objective function for optuna hyper-parameter optimization
def llspin_objective(trial):  
    global model
    
    # hyper-parameter to optimize: lambda, learning rate, number of epochs
    model_params['lam'] = trial.suggest_loguniform('lam',1e-3,1e-2)
    training_params['lr'] = trial.suggest_loguniform('learning_rate', 1e-2, 2e-1)
    training_params['num_epoch'] = trial.suggest_categorical('num_epoch', [2000,5000,10000,15000])

    # specify the model with these parameters and train the model
    model = Model(**model_params)
    train_acces, train_losses, val_acces, val_losses = model.train(dataset=dataset,**training_params)

    print("In trial:---------------------")
    val_prediction = model.test(X_valid)[0]
    mse = mean_squared_error(y_valid.reshape(-1),val_prediction.reshape(-1))
    print("validation mse: {}".format(mse))
    
    loss= mse
            
    return loss
        
def callback(study,trial):
    global best_model
    if study.best_trial == trial:
        best_model = model
        
# optimize the model via Optuna and obtain the best model with smallest validation mse
best_model = None
model = None
study = optuna.create_study(pruner=None)
study.optimize(llspin_objective, n_trials=100, callbacks=[callback])

# get the training gate matrix
gate_mat_train = best_model.get_prob_alpha(X_train)

best_lr = study.best_params['learning_rate']
best_epoch = study.best_params['num_epoch']
best_lam = study.best_params['lam']

# test the best model
y_pred_llspin = best_model.test(X_test)[0]
            
print("Trial Finished*************")
print("Best model's lambda: {}".format(best_lam))
print("Best model's learning rate: {}".format(best_lr))
print("Best model's num of epochs: {}".format(best_epoch))
print("Test mse : {}".format(mean_squared_error(y_test.reshape(-1),y_pred_llspin.reshape(-1))))
print("Test r2 : {}".format(r2_score(y_test.reshape(-1),y_pred_llspin.reshape(-1))))

# comparing training gates to groundtruth 
cmap = cm.Blues
bounds=[0,0.5,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

title_size = 30
xtick_size = 20
ytick_size = 20
xlabel_size = 35
ylabel_size = 35
colorbar_tick_size = 20
title_pad = 10

fig, axes = plt.subplots(1, 2,sharex=False, sharey=True,figsize=(8, 6))

sorted_order = np.concatenate((np.where(train_label == 1)[0],np.where(train_label == 2)[0]))

im1 = axes[0].imshow(ref_feat_mat_train[sorted_order,:].astype(int),aspect='auto',cmap=cmap, norm=norm)
axes[0].set_title("Ground Truth",fontsize=title_size,fontweight="bold",pad=title_pad)
axes[0].set_ylabel("Sample Index",fontsize=ylabel_size)
axes[0].set_yticks([1,3,5,7,9])
axes[0].set_yticklabels([2,4,6,8,10],fontsize=ytick_size)
axes[0].set_xticks(list(range(5)))
axes[0].set_xticklabels(list(range(1,6)),fontsize=xtick_size)
axes[0].set_xlabel("Feature Index",fontsize=xlabel_size,labelpad=-5)

cbar = fig.colorbar(im1,ax=axes[0], cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 1])
cbar.ax.tick_params(labelsize=colorbar_tick_size)

im2 = axes[1].imshow(gate_mat_train[sorted_order,:],aspect='auto',cmap=cmap)
axes[1].set_title("LLSPIN Gates",fontsize=title_size,fontweight="bold",pad=title_pad)
axes[1].set_yticks([1,3,5,7,9])
axes[1].set_yticklabels([2,4,6,8,10],fontsize=ytick_size)
axes[1].set_xticks(list(range(5)))
axes[1].set_xticklabels(list(range(1,6)),fontsize=xtick_size)
axes[1].set_xlabel("Feature Index",fontsize=xlabel_size,labelpad=-5)

cbar = fig.colorbar(im2,ax=axes[1])
cbar.ax.tick_params(labelsize=colorbar_tick_size)

plt.tight_layout()
plt.savefig(os.path.join(result_dir, "training_vs_grounftruth.png"))

# export the total running time 
end_time = time.time()
print(f"Duration: {end_time - start_time} seconds")
