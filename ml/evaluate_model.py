"""Evaluation helpers: overall metrics and slice-based performance.

Slice evaluation reports precision/recall/F-beta for each category of each
categorical feature, so weaknesses on under-represented groups are visible.
"""

from ml.data import process_data
from ml.model import compute_model_metrics, inference
import logging
import json
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename='eval_log.log', filemode='w')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def performance(model, cat_features, test, encoder, lb):
    """Evaluate a machine learning model overall and on data slices.

    Writes the overall F-beta score to fbeta.json, prints the overall
    precision/recall/F-beta, and logs slices whose F-beta is unusually low
    (< 0.2) or high (> 0.7).

    Inputs
    ------
    model : tf.keras.Model
        Trained machine learning model.
    cat_features : list[str]
        Names of the categorical features to slice on.
    test : pd.DataFrame
        Test set, including the "salary" label column.
    encoder : sklearn.preprocessing.OneHotEncoder
        Fitted encoder from training.
    lb : sklearn.preprocessing.LabelBinarizer
        Fitted label binarizer from training.

    Returns
    -------
    performance_slices : dict
        Maps each category with an outlying F-beta to its
        {'precision', 'recall', 'fbeta'} metrics.
    """
    # compute average performances on all slices first
    X_test, y_test = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
    preds = inference(model,X_test)
    pr, recall, fbeta = compute_model_metrics(y_test,preds)
    
    with open("./fbeta.json","w") as f:
        json.dump({"fbeta":fbeta},f)
    
    print(" Precision is {} Recall is {} and F-Beta Score is {}".format(pr,recall,fbeta))
    
    performance_slices = {}
    # Evaluate Model on slices of test data 
    for cat_ in cat_features:
        _categories = list(set(test[cat_]))
        for cat__ in _categories:
            test_1 = test[test[cat_] ==cat__]
            if len(test_1) > 1:
                # Proces the test data with the process_data function.
                X_test, y_test = process_data(test_1, categorical_features=cat_features, label="salary", training=False,encoder=encoder, lb=lb)
                preds = inference(model,X_test)
                pr, recall, fbeta = compute_model_metrics(y_test,preds)
                if fbeta < 0.2 or fbeta > 0.7:
                    performance_slices[cat__] = {'precision': pr, 'recall': recall, 'fbeta': fbeta}
                    logger.info("Main Category {}".format(cat_))
                    logger.info("Category {}".format(cat__))
                    logger.info("Precision {}".format(pr))
                    logger.info("recall {}".format(recall))
                    logger.info("fbeta {}".format(fbeta))
                    logger.info("-------------------------------")

    return performance_slices