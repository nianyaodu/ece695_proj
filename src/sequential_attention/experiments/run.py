'''
https://github.com/google-research/google-research/blob/master/sequential_attention/sequential_attention/experiments/run.py
'''

# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Feature selection experiments."""

import json
import os
import pathlib
import random

from absl import app
from absl import flags
import numpy as np
from sequential_attention.experiments.datasets.dataset import get_dataset
from sequential_attention.experiments.models.mlp_lly import LiaoLattyYangModel
from sequential_attention.experiments.models.mlp_omp import OrthogonalMatchingPursuitModel
from sequential_attention.experiments.models.mlp_sa import SequentialAttentionModel
from sequential_attention.experiments.models.mlp_seql import SequentialLASSOModel
from sequential_attention.experiments.models.mlp_sparse import SparseModel
import tensorflow as tf

from sklearn.model_selection import train_test_split

from src.loss_utils import calculate_relative_loss

os.environ["TF_DETERMINISTIC_OPS"] = "1"

FLAGS = flags.FLAGS

# Experiment parameters
flags.DEFINE_integer("seed", 2023, "Random seed")
flags.DEFINE_enum(
    "data_name",
    "mnist",
    ["mnist", "fashion", "isolet", "mice", "coil", "activity", "synthetic"],
    "Data name",
)
flags.DEFINE_string(
    "data_path",
    None,
    "Path to synthetic data file"
)

flags.DEFINE_string(
    "model_dir",
    "./model_dir",
    "Checkpoint directory for feature selection model",
)

# Feature selection hyperparameters
flags.DEFINE_integer(
    "num_selected_features", 50, "Number of features to select"
)
flags.DEFINE_enum("algo", "sa", ["sa", "lly", "seql", "gl", "omp"], "Algorithm")
flags.DEFINE_integer(
    "num_inputs_to_select_per_step", 1, "Number of features to select at a time"
)

# Hyperparameters
flags.DEFINE_float(
    "val_ratio", 0.125, "How much of the training data to split for validation."
)
flags.DEFINE_list("deep_layers", "67", "Layers in MLP model")
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_integer("num_epochs", 20, "Number of epochs")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate")
flags.DEFINE_integer("decay_steps", 250, "Decay steps")
flags.DEFINE_float("decay_rate", 1.0, "Decay rate")
flags.DEFINE_float("alpha", 0, "Leaky ReLU alpha")
flags.DEFINE_bool("enable_batch_norm", False, "Enable batch norm")
flags.DEFINE_float("group_lasso_scale", 0.01, "Group LASSO scale")

# Finer control if needed
flags.DEFINE_integer("num_epochs_select", -1, "Number of epochs to fit")
flags.DEFINE_integer("num_epochs_fit", -1, "Number of epochs to select")

flags.DEFINE_list('hidden_layers', ['67'], 'Hidden layer sizes') # added 

ALGOS = {
    "sa": SequentialAttentionModel,
    "lly": LiaoLattyYangModel,
    "seql": SequentialLASSOModel,
    "gl": SequentialLASSOModel,
    "omp": OrthogonalMatchingPursuitModel,
}

def run_trial(
    batch_size=256,
    num_epochs_select=250,
    num_epochs_fit=250,
    learning_rate=0.0002,
    decay_steps=100,
    decay_rate=1.0,
):
  """Run a feature selection experiment with a given set of hyperparameters."""
  if FLAGS.data_name == "synthetic":
      print(f"here we are dealing with synthetic data (currently all synthetic data are binary classification tasks)")
      FLAGS.deep_layers = ["67"]
      FLAGS.hidden_layers = ["67"]
  
      data = np.load(FLAGS.data_path)
      X, y = data["X"], data["y"]
      
      X = X.astype(np.float32)
      y = y.astype(np.int32)
      y = tf.one_hot(y.astype(np.int32), depth=2).numpy() # convert response to one hot vector
      
      FLAGS.num_selected_features = min(FLAGS.num_selected_features, X.shape[1])
      x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=FLAGS.seed)
      x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=FLAGS.val_ratio, random_state=FLAGS.seed)
      
      # Create TF datasets
      ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
      ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
      ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
      
      datasets = {
          "ds_train": ds_train,
          "ds_val": ds_val,
          "ds_test": ds_test,
          "x_train": x_train,
          "y_train": y_train,
          "is_classification": True, # False, 
          "num_classes": 2, # None, 
          "num_features": X.shape[1]
      }
  else:
      datasets = get_dataset(FLAGS.data_name, FLAGS.val_ratio, batch_size)
    
  ds_train = datasets["ds_train"]
  ds_val = datasets["ds_val"]
  ds_test = datasets["ds_test"]
  is_classification = datasets["is_classification"]
  num_classes = datasets["num_classes"]
  num_features = datasets["num_features"]
  num_train_steps_select = num_epochs_select * len(ds_train)
  loss_fn = (
      tf.keras.losses.CategoricalCrossentropy()
      if is_classification
      else tf.keras.losses.MeanAbsoluteError()
  )

  model_dir = pathlib.Path(FLAGS.model_dir)
  model_dir_select = model_dir / "select"
  model_dir_fit = model_dir / "fit"
  model_dir_select.mkdir(exist_ok=True, parents=True)
  model_dir_fit.mkdir(exist_ok=True, parents=True)

  layer_sequence = [int(i) for i in FLAGS.hidden_layers] if FLAGS.hidden_layers else [67, 32]

  mlp_args = {
      "layer_sequence": [int(i) for i in FLAGS.hidden_layers], # deep_layers edited 
      "is_classification": is_classification,
      "num_classes": num_classes,
      "learning_rate": learning_rate,
      "decay_steps": decay_steps,
      "decay_rate": decay_rate,
      "alpha": FLAGS.alpha,
      "batch_norm": FLAGS.enable_batch_norm,
  }
  fs_args = {
      "num_inputs": num_features,
      "num_inputs_to_select": FLAGS.num_selected_features,
  }
  if FLAGS.algo == "sa":
    fs_args["num_inputs_to_select_per_step"] = (
        FLAGS.num_inputs_to_select_per_step
    )
    fs_args["num_train_steps"] = num_train_steps_select
  if FLAGS.algo == "seql":
    fs_args["num_train_steps"] = num_train_steps_select
    fs_args["group_lasso_scale"] = FLAGS.group_lasso_scale
  if FLAGS.algo == "gl":
    fs_args["num_inputs_to_select_per_step"] = FLAGS.num_selected_features
    fs_args["num_train_steps"] = num_train_steps_select
    fs_args["group_lasso_scale"] = FLAGS.group_lasso_scale
  if FLAGS.algo == "omp":
    fs_args["num_train_steps"] = num_train_steps_select
  if FLAGS.algo == "lly":
    del fs_args["num_inputs_to_select"]

  ########### Feature Selection ##########
  print("Starting selecting features...")

  if FLAGS.algo in ALGOS:
    args = {**mlp_args, **fs_args}
    if not mlp_args["layer_sequence"]:
        raise ValueError("Layer sequence cannot be empty. Please check FLAGS.hidden_layers")
    mlp_select = ALGOS[FLAGS.algo](**args)
    mlp_select.compile(loss=loss_fn, metrics=["accuracy"])
    mlp_select.fit(
        ds_train, validation_data=ds_val, epochs=num_epochs_select, verbose=2
    )

    ########### Get Features ##########
    if FLAGS.algo == "sa":
      selected_features = mlp_select.seqatt.selected_features
      _, selected_indices = tf.math.top_k(
          selected_features, k=FLAGS.num_selected_features
      )
      selected_indices = selected_indices.numpy()
    elif FLAGS.algo == "lly":
      x_train = datasets["x_train"]
      attention_logits = mlp_select.lly(tf.convert_to_tensor(x_train))
      _, selected_indices = tf.math.top_k(
          attention_logits, k=FLAGS.num_selected_features
      )
      selected_indices = selected_indices.numpy()
    elif FLAGS.algo in ["gl", "seql"]:
      selected_indices = (
          mlp_select.seql.selected_features_history.numpy().tolist()
      )
    elif FLAGS.algo == "omp":
      selected_indices = (
          mlp_select.omp.selected_features_history.numpy().tolist()
      )
    assert (
        len(selected_indices) == FLAGS.num_selected_features
    ), f"Selected: {selected_indices}"

  print("Finished selecting features...")

  selected_features = tf.math.reduce_sum(
      tf.one_hot(selected_indices, num_features, dtype=tf.int32), 0
  ).numpy()
  with open(model_dir_select / "selected_features.txt", "w") as fp:
    fp.write(",".join([str(i) for i in selected_indices]))
  tf.print("Selected", tf.reduce_sum(selected_features), "features")
  tf.print("Selected mask:", selected_features, summarize=-1)
  selected_features = tf.where(selected_features)[:, 0].numpy().tolist()
  selected_features = ",".join([str(i) for i in selected_features])
  print("Selected indices:", selected_features)

  selected_features = [int(i) for i in selected_features.split(",")]
  selected_features = tf.math.reduce_sum(
      tf.one_hot(selected_features, num_features, dtype=tf.float32), 0
  )

  ########### Model Training ##########

  print("Starting retraining...")

  mlp_fit = SparseModel(selected_features=selected_features, **mlp_args)
  mlp_fit.compile(loss=loss_fn, metrics=["accuracy"])
  mlp_fit.fit(
      ds_train, validation_data=ds_val, epochs=num_epochs_fit, verbose=2
  )

  print("Finished retraining...")
  ########### Evaluation ##########
  ########### added: to calculate relative loss and average sparsity ##########
  if FLAGS.data_name == "synthetic": 
    print("since this is synthetic data (binary classification), we can calculating relative loss and average sparsity")
  
    y_test_pred = mlp_fit.predict(ds_test) # this returns prediced probability for each class; shape is (n, 2)
    
    # baseline predictions (mean of training data)
    y_train = np.concatenate([y for x, y in ds_train], axis=0) # shape is (n, 2)
    unique_labels = np.unique(y_train)
    num_classes = len(unique_labels)
    expected_labels = np.arange(num_classes) # Check if labels are consecutive integers starting from 0
    assert np.array_equal(unique_labels, expected_labels), "here we have categorical response, but labels are not consecutive integers starting from 0"
    y_test = np.concatenate([y for x, y in ds_test], axis=0)
    class_0_prob = np.sum(y_train[:, 0]) / y_train.shape[0]
    class_1_prob = np.sum(y_train[:, 1]) / y_train.shape[0]
    y_pred_baseline_proba = np.zeros((len(y_test), num_classes))
    y_pred_baseline_proba[:, 0] = class_0_prob
    y_pred_baseline_proba[:, 1] = class_1_prob
    
    y_test = np.concatenate([y for x, y in ds_test], axis=0) # this will serve as y_true; shape is (n, 2)
    relative_loss = calculate_relative_loss(y_test, y_test_pred, y_pred_baseline_proba, 'categorical')
    print(f"relative loss is {relative_loss}")
    
    results = dict()
    results_val = mlp_fit.evaluate(ds_val, return_dict=True)
    results_test = mlp_fit.evaluate(ds_test, return_dict=True)
    results["val"] = round(results_val["accuracy"], 4)
    results["test"] = round(results_test["accuracy"], 4)
    results["relative_loss"] = round(relative_loss, 4)
    
  else:
    print("since this is applied data (multiclass classification), we can calculating relative loss and average sparsity")
  
    y_test_pred = mlp_fit.predict(ds_test) # this returns prediced probability for each class 
    print(f"y_test_pred is {y_test_pred}") 
    print(f"y_test_pred.shape is {y_test_pred.shape}")
    
    # baseline predictions (mean of training data)
    print(f"ds_train is {ds_train}")
    print(f"ds_train.shape is {ds_train}")
    y_train = np.concatenate([y for x, y in ds_train], axis=0)
    print(f"y_train is {y_train}")
    print(f"y_train.shape is {y_train.shape}")
    # unique_labels = y_train.shape[1]
    num_classes = y_train.shape[1]
    print(f"num_classes is {num_classes}")
    # expected_labels = np.arange(num_classes) # Check if labels are consecutive integers starting from 0
    # assert np.array_equal(unique_labels, expected_labels), "here we have categorical response, but labels are not consecutive integers starting from 0"
    y_test = np.concatenate([y for x, y in ds_test], axis=0)
    y_pred_baseline_proba = np.zeros((len(y_test), num_classes))
    for class_id in np.arange(num_classes): 
        print(f"class_id is {class_id}")
        print(f"y_train.shape[0] is {y_train.shape[0]}")
        print(f"y_train[:, class_id] is {y_train[:, int(class_id)]}")
        print(f"np.sum(y_train[:, class_id]) is {np.sum(y_train[:, int(class_id)])}")
        prob = np.sum(y_train[:, int(class_id)]) / y_train.shape[0]
        print("5")
        y_pred_baseline_proba[:, int(class_id)] = prob
        print("6")
    print(f"y_pred_baseline_proba is {y_pred_baseline_proba}")
    print(f"y_pred_baseline_proba.shape is {y_pred_baseline_proba.shape}")
    
    y_test = np.concatenate([y for x, y in ds_test], axis=0) # this will serve as y_true
    y_test = y_test.astype(int) # the original entries are float
    print(f"y_test is {y_test}")
    print(f"y_test.shape is {y_test.shape}")
    # # convert y_test into onehot vector 
    # print(f"len(y_test) is {len(y_test)}")
    # print(f"num_classes is {num_classes}")
    # y_test_proba = np.zeros((len(y_test), num_classes))
    # y_test_proba[np.arange(len(y_test)), y_test] = 1
    # print(f"after convering, y_test is {y_test}")
    # print(f"after converting, y_test.shape is {y_test.shape}")
        
    relative_loss = calculate_relative_loss(y_test, y_test_pred, y_pred_baseline_proba, 'categorical')
    print(f"relative loss is {relative_loss}")
  
    results = dict()
    results_val = mlp_fit.evaluate(ds_val, return_dict=True)
    results_test = mlp_fit.evaluate(ds_test, return_dict=True)
    results["val"] = round(results_val["accuracy"], 4)
    results["test"] = round(results_test["accuracy"], 4)
    results["relative_loss"] = round(relative_loss, 4)
  
  # Calculate average sparsity
  num_nonzero = tf.reduce_sum(selected_features).numpy()
  total_features = len(selected_features)
  sparsity = num_nonzero / total_features
  print(f"sparsity is {sparsity}")
  results["sparsity"] = round(sparsity, 4)

  with open(model_dir_fit / "results.json", "w") as fp:
    json.dump(results, fp)

  print(results)

  return results["val"]


def main(args):
  del args  # Not used.

  os.environ["PYTHONHASHSEED"] = str(FLAGS.seed)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)

  tf.keras.backend.clear_session()

  print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
  num_epochs_select = FLAGS.num_epochs
  num_epochs_fit = FLAGS.num_epochs
  if FLAGS.num_epochs_select > 0:
    num_epochs_select = FLAGS.num_epochs_select
  if FLAGS.num_epochs_fit > 0:
    num_epochs_fit = FLAGS.num_epochs_fit
  run_trial(
      batch_size=FLAGS.batch_size,
      num_epochs_select=num_epochs_select,
      num_epochs_fit=num_epochs_fit,
      learning_rate=FLAGS.learning_rate,
      decay_steps=FLAGS.decay_steps,
      decay_rate=FLAGS.decay_rate,
  )


if __name__ == "__main__":
  app.run(main)