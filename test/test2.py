'''
https://github.com/lasso-net/lassonet/blob/master/examples/mnist_classif.py
'''
############## test 2 ##############
import os 
import errno

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lassonet import LassoNetClassifier

plot_dir = '/Users/amber/Desktop/ece695_proj/test2_result/'

if not os.path.isdir(plot_dir):
    try:
        os.makedirs(plot_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(plot_dir):
            pass
        else:
            raise
        
X, y = fetch_openml(name="mnist_784", return_X_y=True)
filter = y.isin(["5", "6"])
X = X[filter].values / 255
y = LabelEncoder().fit_transform(y[filter])

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LassoNetClassifier(M=30, verbose=True)
path = model.path(X_train, y_train, return_state_dicts=True)

img = model.feature_importances_.reshape(28, 28)

plt.title("Feature importance to discriminate 5 and 6")
plt.imshow(img)
plt.colorbar()
plt.savefig(os.path.join(plot_dir, "mnist-classification-importance.png"))

n_selected = []
accuracy = []
lambda_ = []

for save in path:
    model.load(save.state_dict)
    y_pred = model.predict(X_test)
    n_selected.append(save.selected.sum())
    accuracy.append(accuracy_score(y_test, y_pred))
    lambda_.append(save.lambda_)

to_plot = [160, 220, 300]

for i, save in zip(n_selected, path):
    if not to_plot:
        break
    if i > to_plot[-1]:
        continue
    to_plot.pop()
    plt.clf()
    plt.title(f"Linear model with {i} features")
    weight = save.state_dict["skip.weight"]
    img = (weight[1] - weight[0]).reshape(28, 28)
    plt.imshow(img)
    plt.colorbar()
    plt.savefig(os.path.join(plot_dir, f"mnist-classification-{i}.png"))

fig = plt.figure(figsize=(12, 12))

plt.subplot(311)
plt.grid(True)
plt.plot(n_selected, accuracy, ".-")
plt.xlabel("number of selected features")
plt.ylabel("classification accuracy")

plt.subplot(312)
plt.grid(True)
plt.plot(lambda_, accuracy, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("classification accuracy")

plt.subplot(313)
plt.grid(True)
plt.plot(lambda_, n_selected, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("number of selected features")

plt.savefig(os.path.join(plot_dir, "mnist-classification-training.png"))

print(f"DONE")