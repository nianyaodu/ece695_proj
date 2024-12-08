import numpy as np 

def calculate_relative_loss(y_true, y_pred, y_pred_baseline, response_type='continuous'):
    """
    Calculate relative loss compared to intercept-only model
    
    :param y_true: true labels. if not continuous response, then this should be onehot vector. but with probabiity on the entry. 
    :param y_pred: model predictions. if not continuous response, then this should be onehot vector. 
    :param y_pred_baseline: baseline (intercept-only) model predictions. if not continuous response, then this should be onehot vector. 
    :param response_type: 'continuous' or 'categorical'
    
    :return relative_loss: relative loss of the model
    """
    eps = 1e-15  # avoid log(0)
    
    if response_type == 'continuous':
        # check if y_true.shape is the same as y_pred.shapp, y_pred_baseline.shape 
        if y_true.shape != y_pred.shape: 
            y_pred = y_pred.reshape(-1) # the only different one should be y_pred
        
        assert y_true.shape == y_pred.shape, "when calculating relative loss, y_true and y_pred have differnet shapes"
        assert y_pred.shape == y_pred_baseline.shape, "when calculating relative loss, y_true and y_pred have differnet shapes"
        
        loss = np.mean((y_true - y_pred) ** 2)
        loss_baseline = np.mean((y_true - y_pred_baseline) ** 2)
    else:
        print(f"calculating cross entropy loss for categorical response")
        assert y_true.shape == y_pred.shape, "when calculating relative loss, y_true and y_pred have differnet shapes"
        assert y_pred.shape == y_pred_baseline.shape, "when calculating relative loss, y_true and y_pred have differnet shapes"
        
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_pred_baseline = np.clip(y_pred_baseline, eps, 1 - eps)
        
        loss = -np.mean(y_true * np.log(y_pred))
        loss_baseline = -np.mean(y_true * np.log(y_pred_baseline))
            
    print(f"loss is {loss}")
    print(f"loss_baseline is {loss_baseline}")
    
    relative_loss = loss / loss_baseline if loss_baseline != 0 else float('inf')
    return relative_loss