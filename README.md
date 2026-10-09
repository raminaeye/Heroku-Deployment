# Census Salary Prediction API on Heroku

A machine learning project that predicts whether a person's income is above or below $50K from US census data, and serves the prediction through a REST API deployed on Heroku.

The project covers the full ML pipeline: exploratory data analysis and data cleaning, preprocessing with scikit-learn, training a feedforward neural network with TensorFlow, slice-based model evaluation, and deployment as a FastAPI service. Data, encoders, and the trained model are versioned with DVC and stored in S3. GitHub Actions runs lint and tests on every push.

Deployed app: https://raminapi4.herokuapp.com/ (see `model_card.md` for model details)

## Repository structure

```
├── src/main.py            # FastAPI app: GET / (welcome) and POST /prediction (inference)
├── src/test_docs.py       # API tests using FastAPI's TestClient
├── ml/
│   ├── data.py            # preprocessing: one-hot encoding and label binarization
│   ├── model.py           # TensorFlow model definition, training, metrics, inference
│   ├── evaluate_model.py  # overall metrics plus per-slice evaluation
│   └── test_model.py      # unit tests for data and model artifacts
├── train_model.py         # end-to-end training script (split, preprocess, train, evaluate)
├── eda.ipynb              # exploratory analysis and data cleaning notebook
├── params.yaml            # model hyperparameters (units, dropout, learning rate, epochs)
├── fbeta.json             # overall F-beta score from the latest training run
├── slice_output.txt       # per-slice evaluation log
├── model_card.md          # model documentation: architecture, metrics, ethical notes
├── data/                  # DVC-tracked CSVs, encoders, label binarizer
├── model/                 # DVC-tracked trained TensorFlow model (TFmodel_v1.h5)
├── requirements.txt       # pinned Python dependencies
├── Procfile               # Heroku process: uvicorn src.main:app
├── runtime.txt            # Python version for the Heroku dyno
├── Aptfile                # apt packages installed on the Heroku dyno
├── sample_request.py      # sample GET/POST requests against a local server
├── heroku.py              # sample POST request against the deployed Heroku app
├── sanitycheck.py         # heuristic checker for the API test suite
└── screenshots/           # deployment screenshots
```

## How it works

1. `eda.ipynb` loads the raw UCI Census Income data, profiles it, strips stray spaces from column names and values, drops duplicates, and saves `data/census_clean.csv`.
2. `train_model.py` splits the cleaned data 80/20, one-hot encodes the 8 categorical features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`) and binarizes the `salary` label, then trains the network defined in `ml/model.py`.
3. `ml/evaluate_model.py` computes precision, recall, and F-beta on the test set and on slices of every categorical feature, so weaknesses on under-represented groups are visible.
4. `src/main.py` serves predictions. A POST body carries one census record (hyphenated CSV column names map to Python fields via pydantic aliases); the record is encoded with the saved training encoder and classified as `<=50K` or `>50K`.

## Setup

Python 3.8 is recommended. Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

The data and model artifacts are tracked with DVC. To download them:

```bash
dvc pull
```

This needs AWS credentials with access to the S3 remote configured for this project.

## Training the model

```bash
python train_model.py
```

This trains with the hyperparameters in `params.yaml` and saves the model to `model/TFmodel_v1.h5`, the encoder and label binarizer to `data/`, and the overall F-beta to `fbeta.json`. Slice metrics are printed to the console (a sample run is captured in `slice_output.txt`).

## Running the API locally

```bash
uvicorn src.main:app --host=0.0.0.0 --port=8000
```

Then try it with the sample script:

```bash
python sample_request.py
```

### API reference

**GET /** returns a welcome message (useful as a health check):

```bash
curl http://127.0.0.1:8000/
```

**POST /prediction** accepts one census record as JSON and returns `"<=50K"` or `">50K"`:

```bash
curl -X POST http://127.0.0.1:8000/prediction \
  -H "Content-Type: application/json" \
  -d '{"age": 31, "workclass": "Private", "fnlgt": 45781, "education": "Masters",
       "education-num": 14, "marital-status": "Never-married", "occupation": "Prof-specialty",
       "relationship": "Not-in-family", "race": "White", "sex": "Female",
       "capital-gain": 14084, "capital-loss": 0, "hours-per-week": 50,
       "native-country": "United-States"}'
```

## Running the tests

```bash
pytest
```

This runs the model/data unit tests in `ml/test_model.py` and the API tests in `src/test_docs.py`. Run the sanity checker to verify the API test suite covers response codes, response bodies, and both prediction outcomes:

```bash
python sanitycheck.py
```

## Deploying to Heroku

Deploying requires a Heroku account and the Heroku CLI. The outline is:

1. Create the app: `heroku create --app <your-app-name>`
2. Add buildpacks for apt and Python:
   ```bash
   heroku buildpacks:add --index 1 heroku-community/apt
   heroku buildpacks:add --index 2 heroku/python
   ```
3. Give the dyno S3 access so it can pull the DVC artifacts:
   ```bash
   heroku config:set AWS_ACCESS_KEY_ID=<your-key> AWS_SECRET_ACCESS_KEY=<your-secret>
   ```
4. Push and deploy: `git push heroku master`

On startup the app runs `dvc pull` (see `src/main.py`) to fetch the model, encoder, and data from the S3 remote, then serves with `uvicorn src.main:app` as declared in the `Procfile`. To hit the deployed app:

```bash
HEROKU_API_URL=https://<your-app-name>.herokuapp.com python heroku.py
```

## What you learn from this project

* Cleaning a real-world messy CSV with pandas before it ever reaches a model.
* Preprocessing with scikit-learn (one-hot encoding, label binarization) in a training/inference-safe way.
* Training and evaluating a TensorFlow feedforward network, including slice-based evaluation to surface fairness issues.
* Versioning data and models with DVC on S3 so experiments are reproducible.
* Serving a model with FastAPI, including pydantic models for request validation.
* CI with GitHub Actions and continuous deployment to Heroku.

## Data source

The census data comes from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/census+income
