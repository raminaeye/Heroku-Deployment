"""FastAPI service for census salary-range prediction.

Loads the trained TensorFlow model and the one-hot encoder, then serves two
endpoints:

* ``GET /`` - welcome/health message.
* ``POST /prediction`` - accepts one census record as JSON and returns
  "<=50K" or ">50K".

Run locally with:

    uvicorn src.main:app --host=0.0.0.0 --port=${PORT:-5000}

On Heroku, the DVC-tracked artifacts (model, encoders) are pulled from the
remote at startup before the model is loaded.
"""

from fastapi import FastAPI
import tensorflow as tf
import os
import sys
ROOT_DIR = os.path.abspath(os.pardir)
sys.path.insert(0, ROOT_DIR)

curr_DIR = os.path.abspath(os.curdir)
sys.path.insert(0, curr_DIR)

from pydantic import BaseModel, Field
from ml.model import inference
import numpy as np
import logging
import uvicorn
import joblib


# Instantiate the app.
app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename='log_main.log', filemode='w')


# Pull DVC-tracked artifacts on Heroku before loading anything.
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


try:
    # Load encoder and binarizer. When the app runs from src/, the artifacts
    # live one directory up; otherwise they live in the current directory.
    encoder = joblib.load(os.path.join(ROOT_DIR, 'data/encoder.joblib'))
    model = tf.keras.models.load_model(os.path.join(ROOT_DIR, 'model/TFmodel_v1.h5'))
except Exception:
    ROOT_DIR = os.path.abspath(os.curdir)
    encoder = joblib.load(os.path.join(ROOT_DIR, 'data/encoder.joblib'))
    model = tf.keras.models.load_model(os.path.join(ROOT_DIR, 'model/TFmodel_v1.h5'))


class csvFile(BaseModel):
    """One census record, as expected in the POST /prediction request body.

    CSV column names use hyphens, which are not valid Python identifiers, so
    the hyphenated names are mapped in via pydantic field aliases.
    """

    age: int = Field(..., example=25)
    workclass: str = Field(..., example="Never-married")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Divorced")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    """Return the API welcome message (used as a health check)."""
    logger.info("get method")
    return "Welcome to Project 4 for Salary Range Prediction!"


# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/prediction")
async def predict(input_: csvFile):
    """Predict "<=50K" or ">50K" from a posted census record.

    The categorical fields are encoded with the fitted training encoder and
    concatenated with the continuous fields in the same order used during
    training, then passed through the trained model.
    """
    logger.info("post method")

    X_categorical = [input_.workclass, input_.education, input_.marital_status, input_.occupation, input_.relationship, input_.race, input_.sex, input_.native_country]

    X_continuous = [input_.age, input_.fnlgt, input_.education_num, input_.capital_gain, input_.capital_loss, input_.hours_per_week]

    XCatEncoded = list(encoder.transform([X_categorical])[0])

    X = np.concatenate([[X_continuous], [XCatEncoded]], axis=1)

    Y = list(X[0])

    logger.info(Y)
    logger.info(np.shape(Y))

    pred = inference(model, [Y]).ravel()
    logger.info("prediction " + str(pred))

    if pred[0] == 0:
        return "<=50K"
    else:
        return ">50K"


if __name__ == "__main__":
    uvicorn.run(app)
