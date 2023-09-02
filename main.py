from fastapi import FastAPI
# Import Union since our Item object will have tags that can be strings or
# a list.
from typing import Union
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
import pandas as pd

from src.ml.model import load_model, inference
from src.ml.data import process_data

# Declare the data object with its components and their type.


class InputData(BaseModel):
    age: int = 26
    workclass: str = "State-gov"
    fnlgt: int = 20000
    education: str = "Bachelors"
    education_num: int = Field(alias='education-num', examples=[13])
    marital_status: str = Field(
        alias='marital-status',
        examples=[
            "Married",
            "Divorced"])
    occupation: str = "Adm-clerical"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Female"
    capital_gain: int = Field(alias='capital-gain', examples=[2174])
    capital_loss: int = Field(alias='capital-loss', examples=[0])
    hours_per_week: int = Field(alias='hours-per-week', examples=[40])
    native_country: str = Field(
        alias='native-country',
        examples=["United-States"])


# Initialize FastAPI instance
app = FastAPI()
model = load_model("model/model.pkl")
encoder = load_model("model/encoder.pkl")
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
label_map = {0: "<=50K", 1: ">50K"}


@app.get("/")
async def greet():
    return "Greetings, welcome to the salary predictor"


@app.post("/")
async def infer(data: InputData):
    data = dict(data)
    for key, val in data.items():
        data[key] = [val]
    print(data)
    df = pd.DataFrame.from_dict(data)
    input_data, _, _, _ = process_data(
        df, cat_features, training=False, encoder=encoder)
    preds = inference(model, input_data)
    return {"predicted_salary": label_map[preds[0]]}
