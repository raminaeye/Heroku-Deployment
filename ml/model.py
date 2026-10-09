"""Model definition, training, evaluation metrics, and inference.

The model is a feedforward neural network built with tf.keras. Hyperparameters
are read from params.yaml so experiments can change them without touching code.
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
import tensorflow as tf
import numpy as np
import yaml
import os
# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # load model hyperparameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    batch_size = params["batch_size"]
    dim = X_train.shape[1]
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(dim,)),
    tf.keras.layers.Dropout(params["dropout1"]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(params["n_units_l1"], activation='relu'),
    tf.keras.layers.Dropout(params["dropout2"]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(params["n_units_l2"], activation='relu'),
    tf.keras.layers.Dense(params["n_units_l3"], activation='relu'),
    tf.keras.layers.Dense(params["n_units_l4"], activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
    adam = tf.keras.optimizers.Adam(
    learning_rate=params["lr"]
)
    model.compile(optimizer=adam,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size = batch_size, epochs=params["epochs"])
    
    model.save("model/TFmodel_v1.h5")
    
    return model 


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : tensorflow .h5
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return np.argmax(preds,axis=-1)
