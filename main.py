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
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')


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
