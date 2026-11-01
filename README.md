# Census Income Prediction API

A FastAPI service that predicts whether a person's income is above $50K from US census data, with a TensorFlow model, slice evaluation, tests and a Heroku setup. Built for a Udacity MLOps project.

## What's inside

- `train_model.py` trains and evaluates the model
- `ml/` holds preprocessing, model and tests
- `src/main.py` is the API
- `model_card.md` describes the model

## How to run

```bash
pip install -r requirements.txt
python train_model.py
uvicorn src.main:app
```

The original DVC bucket is gone, so build `data/census_clean.csv` from the UCI Census Income data with `eda.ipynb` before training.
