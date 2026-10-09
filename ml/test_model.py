"""Unit tests for the trained model and its input data.

Run from the repository root:

    pytest ml/test_model.py

The tests load the cleaned census data and the trained model artifacts
(model/TFmodel_v1.h5, data/xTrain.joblib), so run `dvc pull` first if those
files are missing.
"""

import pytest
import pandas as pd
import tensorflow as tf
import os
import joblib

# Tests assume the working directory is the repository root.
ROOT_DIR = os.path.abspath(os.curdir)


@pytest.fixture
def dataTest():
    """Load the cleaned census dataset."""
    df = pd.read_csv(os.path.join(ROOT_DIR, 'data/census_clean.csv'))
    return df


def test_data_shape(dataTest):
    """Data is assumed to have no null values; dropping them changes nothing."""
    assert dataTest.shape == dataTest.dropna().shape, "Dropping null changes shape."


def test_model_exist():
    """The trained model file can be loaded."""
    model = tf.keras.models.load_model(os.path.join(ROOT_DIR, 'model/TFmodel_v1.h5'))
    assert model is not None, "Model does not exist"


def test_model_predicts():
    """The model can predict on the saved encoded training data."""
    try:
        model = tf.keras.models.load_model(os.path.join(ROOT_DIR, 'model/TFmodel_v1.h5'))
        encoded_data = joblib.load(os.path.join(ROOT_DIR, 'data/xTrain.joblib'))
        model.predict(encoded_data)
    except Exception:
        assert False, "Model could not predict on encoded data"


def test_model_prediction_shape(dataTest):
    """Predictions have one probability column per class (two classes)."""
    try:
        model = tf.keras.models.load_model(os.path.join(ROOT_DIR, 'model/TFmodel_v1.h5'))
        encoded_data = joblib.load(os.path.join(ROOT_DIR, 'data/xTrain.joblib'))
        prediction = model.predict(encoded_data)
        assert prediction.shape[1] == 2
    except Exception:
        assert False
