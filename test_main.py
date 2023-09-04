import pytest
import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.fixture
def valid_input():
    return {
        "age": 35,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }


@pytest.fixture
def valid_input2():
    return {
        "age": 26,
        "workclass": "State-gov",
        "fnlgt": 20000,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 0,
        "native-country": "Germany"
    }


@pytest.fixture
def invalid_input1():
    return {
        "age": 35,
        "workclass": 10101010,
        "fnlgt": "test"
    }


@pytest.fixture
def invalid_input2():
    return {
        "age": 43,
        "workclass": 40000,
        "fnlgt": 292175,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Divorced",
        "occupation": "Exec-managerial",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }


def test_greet_function():
    result = client.get("/")
    assert result.status_code == 200
    assert result.json() == "Greetings, welcome to the salary predictor"


def test_valid_input_data1(valid_input):
    result = client.post("/", json=valid_input)
    assert result.status_code == 200
    # Check that the predicted salary is correct
    assert result.json() == {"predicted_salary": "<=50K"}


def test_valid_input_data2(valid_input2):
    result = client.post("/", json=valid_input2)
    assert result.status_code == 200
    # Check that the predicted salary is correct
    assert result.json() == {"predicted_salary": ">50K"}


def test_invalid_input_data(invalid_input1):
    result = client.post("/", data=invalid_input1)
    assert result.status_code == 422


def test_invalid_input_data2(invalid_input2):
    result = client.post("/", data=invalid_input2)
    assert result.status_code == 422
