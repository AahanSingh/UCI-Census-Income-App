from pathlib import Path
import pytest
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.ml.model import get_unique
from src.ml.model import get_data_slice
from src.ml.model import train_model
from src.ml.data import process_data


@pytest.fixture
def dataset():
    dirname = Path(__file__).parent.parent
    datapath = f"{dirname}/data/census-clean.csv"
    df = pd.read_csv(datapath)
    new_cols = {
        "education-num": "education_num",
        "marital-status": "marital_status",
        "capital-gain": "capital_gain",
        "capital-loss": "capital_loss",
        "hours-per-week": "hours_per_week",
        "native-country": "native_country"
    }
    df.rename(columns = new_cols, inplace=True)
    return df


@pytest.fixture
def target():
    return "salary"


@pytest.fixture
def features():
    return [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]


@pytest.fixture
def X(dataset, features):
    return dataset[features]


@pytest.fixture
def Y(dataset, target):
    return dataset[target]


def test_get_unique(X):
    column = "education"
    unique_values = ['10th',
                     '11th',
                     '12th',
                     '1st-4th',
                     '5th-6th',
                     '7th-8th',
                     '9th',
                     'Assoc-acdm',
                     'Assoc-voc',
                     'Bachelors',
                     'Doctorate',
                     'HS-grad',
                     'Masters',
                     'Preschool',
                     'Prof-school',
                     'Some-college']
    unique_values = set(unique_values)
    assert unique_values == set(get_unique(X, column))


def test_get_data_slice(X):
    column = "sex"
    expected = [
        X.query("sex == 'Female'"),
        X.query("sex == 'Male'")
    ]
    res = [(val, i) for val, i in get_data_slice(X, column)]
    assert res[0][0] == "Female"
    assert res[1][0] == "Male"
    assert expected[0].equals(res[0][1])
    assert expected[1].equals(res[1][1])


def test_train_model(dataset, features, target):
    train, test = train_test_split(dataset, test_size=0.20)

    X, y, encoder, lb = process_data(
        X=train, categorical_features=features, label=target, training=True)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
