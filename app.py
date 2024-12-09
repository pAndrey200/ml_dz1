from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

def isNaN(num):
    return num != num

def update_df(df):
    df['mileage'] = df['mileage'].apply(lambda x : (float(x[:-5]) if x[:1] != '0' else np.NaN) if isNaN(x) == False else x)
    df['max_power'] = df['max_power'].apply(lambda x : (float(x[:-4]) if x!= '0' and x != ' bhp' else np.NaN) if isNaN(x) == False else x)
    df['engine'] = df['engine'].apply(lambda s : float(s[:-3]) if isNaN(s) == False else s)
    df.drop(columns=['torque'], inplace=True)
    df[['engine', 'seats']] = df[['engine', 'seats']].astype(int)
    df.drop(columns=['name'], inplace=True)
    categorical_features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    coloumns = encoder.get_feature_names_out(categorical_features)
    encoded_columns = encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_columns.toarray(),columns=coloumns)
    df = df.drop(columns=categorical_features)
    df = pd.concat([df, encoded_df], axis=1)
    return df

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])
    df = update_df(df)
    prediction = model.predict(df)[0]
    return prediction

@app.post("/predict_items")
async def predict_file(file: UploadFile = File(...)) -> str:
    df = pd.read_csv(file.file)
    df_res = df.copy()
    df = update_df(df)
    predictions = model.predict(df)
    df_res["predicted_price"] = predictions
    output = io.StringIO()
    df_res.to_csv(output, index=False)
    output.seek(0)
    response = StreamingResponse(output, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predicted_prices.csv"
    return response