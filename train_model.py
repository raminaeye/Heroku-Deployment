# Script to train the machine learning model end to end:
# load the cleaned census data, split it, preprocess it, train the model,
# save the artifacts (encoded data, encoder, label binarizer, model), and
# evaluate performance overall and on slices of the test set.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model
from ml.evaluate_model import performance
import joblib
import os
ROOT_DIR = os.path.abspath(os.curdir)

# Load in the data. The notebook writes the index column, so drop it.
data = pd.read_csv(ROOT_DIR + '/data/census_clean.csv')
data = data.drop(["Unnamed: 0"], axis=1)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)

# Save encoded data 
joblib.dump(X_train,(ROOT_DIR+'/data/xTrain.joblib'))
joblib.dump(encoder,(ROOT_DIR+'/data/encoder.joblib'))
joblib.dump(lb,(ROOT_DIR+'/data/lb.joblib'))

# Train and save a model.
model = train_model(X_train,y_train)

# Evaluating the model performance on slices of the test dataset 
evaluation = performance(model, cat_features, test, encoder, lb)