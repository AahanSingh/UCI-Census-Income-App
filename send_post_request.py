import requests
import json

def send_post_request():
    url = "https://uci-salary-predictor.onrender.com/"
    json_data = {
        "age": 26,
        "workclass": "State-gov",
        "fnlgt": 20000,
        "education": "Bachelors",
        "education-num": 0,
        "marital-status": "string",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 0,
        "native-country": "United-States"
    }
    response = requests.post(url, json=json_data)
    print(response.text)

send_post_request()
