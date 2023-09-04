from pathlib import Path
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model
from ml.model import save_model
from ml.model import compute_model_metrics
from ml.model import inference, get_data_slice

original_stdout = sys.stdout


def compute_metrics_on_slice(model, test, columns, slice_columns, encoder, lb):
    for c in slice_columns:
        print(c)
        with open("slice_output.txt", "a") as f:
            sys.stdout = f
            print(c)
            sys.stdout = original_stdout
        res = {}
        for val, data_slice in get_data_slice(test, c):
            X_test, y_test, _, _ = process_data(
                data_slice, columns, "salary", False, encoder, lb)
            preds = inference(
                model,
                X_test)
            precision, recall, fbeta = compute_model_metrics(
                y_test,
                preds
            )
            res[val] = {}
            res[val]["precision"] = precision
            res[val]["recall"] = recall
            res[val]["fbeta"] = fbeta
        print(pd.DataFrame.from_dict(res))
        print("\n\n")
        with open("slice_output.txt", "a") as f:
            sys.stdout = f
            print(pd.DataFrame.from_dict(res))
            print("\n\n")
            sys.stdout = original_stdout


def run(save_path, data_path):
    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    df = pd.read_csv(data_path)
    new_cols = {
        "education-num": "education_num",
        "marital-status": "marital_status",
        "capital-gain": "capital_gain",
        "capital-loss": "capital_loss",
        "hours-per-week": "hours_per_week",
        "native-country": "native_country"
    }
    df.rename(columns = new_cols, inplace=True)
    train, test = train_test_split(df, test_size=0.20)
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    print("TRAINING MODEL")
    trained_model = train_model(X_train, y_train)
    save_model(save_path, trained_model)
    with open(f'{save_path}/encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)
    print(f"\n\nModel saved at {save_path}")

    print("MODEL METRICS:")
    print("Overall metrics:")
    X_test, y_test, _, _ = process_data(
        test, cat_features, "salary", False, encoder, lb)
    preds = inference(
        trained_model,
        X_test)
    precision, recall, fbeta = compute_model_metrics(
        y_test,
        preds
    )
    print(f"\tPrecision: {precision}\tRecall: {recall}\tfbeta: {fbeta}")
    print("\n\nSLICE WISE METRICS:")
    with open("slice_output.txt", "a") as f:
        sys.stdout = f
        print("MODEL METRICS:")
        print("Overall metrics:")
        print(f"\tPrecision: {precision}\tRecall: {recall}\tfbeta: {fbeta}")
        print("\n\nSLICE WISE METRICS:")
        sys.stdout = original_stdout

    compute_metrics_on_slice(
        trained_model, test, cat_features, [
            "marital_status", "sex"], encoder, lb)


if __name__ == "__main__":
    save_path = f"{Path(__file__).parent.parent}/model/"
    data_path = f"{Path(__file__).parent.parent}/data/census-clean.csv"
    run(save_path, data_path)
