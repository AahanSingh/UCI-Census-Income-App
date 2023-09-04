import pickle
import json

import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# TODO: implement hyperparameter tuning.


def save_model(save_path: str, model):
    with open(f'{save_path}/model.pkl', 'wb') as file:
        pickle.dump(model, file)


def load_model(model_path: str):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame):
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
    clf = RandomForestClassifier(max_depth=10, n_estimators=50)
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

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
    model : Random forest model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def get_unique(X, column):
    return sorted(X[column].unique())


def get_data_slice(X, column):
    unique = get_unique(X, column)
    for i in unique:
        yield (i, X.query(f"{column} == '{i}'"))
