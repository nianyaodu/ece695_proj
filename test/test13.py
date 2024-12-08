'''
https://github.com/google-research/google-research/blob/master/sequential_attention/sequential_attention/sequential_attention.py
'''
############## test 13 - test experiment using sequential_attention feature selection ##############

import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

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

"""Sequential Attention for Feature Selection.

https://arxiv.org/abs/2209.14881
"""

import tensorflow as tf


class SequentialAttention(tf.Module):
  """SequentialAttention module."""

  def __init__(
      self,
      num_candidates,
      num_candidates_to_select,
      num_candidates_to_select_per_step=1,
      start_percentage=0.1,
      stop_percentage=1.0,
      name='sequential_attention',
      reset_weights=True,
      **kwargs,
  ):
    """Creates a new SequentialAttention module."""

    super(SequentialAttention, self).__init__(name=name, **kwargs)

    assert num_candidates_to_select % num_candidates_to_select_per_step == 0, (
        'num_candidates_to_select must be a multiple of '
        'num_candidates_to_select_per_step.'
    )

    with self.name_scope:
      self._num_candidates = num_candidates
      self._num_candidates_to_select_per_step = (
          num_candidates_to_select_per_step
      )
      self._num_steps = (
          num_candidates_to_select // num_candidates_to_select_per_step
      )
      self._start_percentage = start_percentage
      self._stop_percentage = stop_percentage
      self._reset_weights = reset_weights

      init_attention_weights = tf.random.normal(
          shape=[num_candidates], stddev=0.00001, dtype=tf.float32
      )
      self._attention_weights = tf.Variable(
          initial_value=lambda: init_attention_weights,
          dtype=tf.float32,
          name='attention_weights',
      )

      self.selected_features = tf.Variable(
          tf.zeros(shape=[num_candidates], dtype=tf.float32),
          trainable=False,
          name='selected_features',
      )

  @tf.Module.with_name_scope
  def __call__(self, training_percentage):
    """Calculates attention weights for all candidates.

    Args:
      training_percentage: Percentage of training process that has been done.
        This input argument should be between 0 and 1 and should be montonically
        increasing.

    Returns:
      A vector of attention weights of size self._num_candidates. All the
      weights
      are between 0 and 1 and sum to 1.
    """
    percentage = (training_percentage - self._start_percentage) / (
        self._stop_percentage - self._start_percentage
    )
    curr_index = tf.cast(
        tf.math.floor(percentage * self._num_steps), dtype=tf.float32
    )
    curr_index = tf.math.minimum(curr_index, self._num_steps - 1.0)

    should_train = tf.less(curr_index, 0.0)

    num_selected = tf.math.reduce_sum(self.selected_features)
    should_select = tf.greater_equal(curr_index, num_selected)
    _, new_indices = tf.math.top_k(
        self._softmax_with_mask(
            self._attention_weights, 1.0 - self.selected_features
        ),
        k=self._num_candidates_to_select_per_step,
    )
    new_indices = self._k_hot_mask(new_indices, self._num_candidates)
    new_indices = tf.cond(
        should_select,
        lambda: new_indices,
        lambda: tf.zeros(self._num_candidates),
    )
    select_op = self.selected_features.assign_add(new_indices)
    init_attention_weights = tf.random.normal(
        shape=[self._num_candidates], stddev=0.00001, dtype=tf.float32
    )
    should_reset = tf.logical_and(should_select, self._reset_weights)
    new_weights = tf.cond(
        should_reset,
        lambda: init_attention_weights,
        lambda: self._attention_weights,
    )
    reset_op = self._attention_weights.assign(new_weights)

    with tf.control_dependencies([select_op, reset_op]):
      candidates = 1.0 - self.selected_features
      softmax = self._softmax_with_mask(self._attention_weights, candidates)
      return tf.cond(
          should_train,
          lambda: tf.ones(self._num_candidates),
          lambda: softmax + self.selected_features,
      )

  @tf.Module.with_name_scope
  def _k_hot_mask(self, indices, depth, dtype=tf.float32):
    return tf.math.reduce_sum(tf.one_hot(indices, depth, dtype=dtype), 0)

  @tf.Module.with_name_scope
  def _softmax_with_mask(self, logits, mask):
    shifted_logits = logits - tf.math.reduce_max(logits)
    exp_shifted_logits = tf.math.exp(shifted_logits)
    masked_exp_shifted_logits = tf.multiply(exp_shifted_logits, mask)
    return tf.math.divide_no_nan(
        masked_exp_shifted_logits, tf.math.reduce_sum(masked_exp_shifted_logits)
    )
    
class SequentialAttentionMNIST:
    def __init__(
        self,
        num_features_to_select=100,
        num_features_per_step=10,
        start_percentage=0.1,
        stop_percentage=1.0,
        hidden_dims=(256, 128),
        learning_rate=0.001,
        batch_size=256,
        num_epochs=50
    ):
        self.num_features = 784  # MNIST image size (28x28)
        self.num_classes = 10
        self.num_features_to_select = num_features_to_select
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Initialize Sequential Attention module
        self.attention = SequentialAttention(
            num_candidates=self.num_features,
            num_candidates_to_select=num_features_to_select,
            num_candidates_to_select_per_step=num_features_per_step,
            start_percentage=start_percentage,
            stop_percentage=stop_percentage
        )
        
        # Build classifier model
        self.model = self._build_classifier(hidden_dims, learning_rate)
    
    def _build_classifier(self, hidden_dims, learning_rate):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.num_features,)))
        
        # Hidden layers
        for units in hidden_dims:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(0.3))
        
        # Output layer
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def load_and_preprocess_data(self):
        # Load MNIST dataset
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
        
        # Flatten images and normalize
        X_train_full = X_train_full.reshape(-1, 784).astype('float32') / 255.0
        X_test = X_test.reshape(-1, 784).astype('float32') / 255.0
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=0.2, random_state=42
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train(self, X_train, y_train, X_val, y_val):
        print("Starting training...")
        start_time = time.time()
        
        n_steps = self.num_epochs
        for epoch in range(n_steps):
            # Calculate current training percentage
            training_percentage = (epoch + 1) / n_steps
            
            # Get attention weights
            attention_weights = self.attention(training_percentage)
            
            # Apply feature selection using attention weights
            X_train_selected = X_train * tf.cast(attention_weights > 0.5, tf.float32)
            X_val_selected = X_val * tf.cast(attention_weights > 0.5, tf.float32)
            
            # Train classifier for one epoch
            history = self.model.fit(
                X_train_selected, y_train,
                validation_data=(X_val_selected, y_val),
                batch_size=self.batch_size,
                epochs=1,
                verbose=1
            )
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                val_acc = history.history['val_accuracy'][0]
                num_selected = tf.reduce_sum(tf.cast(attention_weights > 0.5, tf.int32))
                print(f"Epoch {epoch + 1}/{n_steps}, "
                      f"Val Accuracy: {val_acc:.4f}, "
                      f"Selected Features: {num_selected}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
    
    def evaluate(self, X_test, y_test):
        # Get final attention weights
        attention_weights = self.attention(1.0)
        
        # Apply feature selection
        X_test_selected = X_test * tf.cast(attention_weights > 0.5, tf.float32)
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test_selected, y_test, verbose=0)
        
        # Get selected features
        selected_features = tf.cast(attention_weights > 0.5, tf.int32).numpy()
        num_selected = np.sum(selected_features)
        
        print("\nFinal Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Number of Selected Features: {num_selected}")
        
        return test_accuracy, selected_features

def run_experiment():
    # Initialize model
    model = SequentialAttentionMNIST(
        num_features_to_select=100,  # Final number of features to select
        num_features_per_step=10,    # Number of features to select per step
        start_percentage=0.1,        # When to start feature selection
        stop_percentage=0.9,         # When to stop feature selection
        hidden_dims=(256, 128),      # Hidden layer dimensions
        learning_rate=0.001,         # Learning rate
        batch_size=256,              # Batch size
        num_epochs=50               # Number of epochs
    )
    
    # Load and preprocess data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = model.load_and_preprocess_data()
    
    # Train model
    model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    test_accuracy, selected_features = model.evaluate(X_test, y_test)
    
    # Visualize selected features as an image
    feature_importance = np.zeros((28, 28))
    feature_importance = selected_features.reshape(28, 28)
    
    return test_accuracy, feature_importance

if __name__ == "__main__":
    test_accuracy, feature_importance = run_experiment()