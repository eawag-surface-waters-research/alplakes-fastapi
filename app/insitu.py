import os
import pandas as pd
from typing import List
from datetime import datetime, date
from typing import Dict, List
from pydantic import BaseModel, validator
from fastapi import HTTPException

from app import functions

class Metadata(BaseModel):
    key: str
    measurements: int
    start_date: date
    end_date: date
    month_coverage: List[int]

class ResponseModel(BaseModel):
    time: List[datetime]
    variable: functions.VariableKeyModel1D
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

def get_insitu_secchi_metadata(filesystem):
    folder = os.path.join(filesystem, "media", "insitu", "secchi")
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail="No insitu secchi depth data available")
    out = []
    for file in os.listdir(folder):
        try:
            df = pd.read_csv(os.path.join(folder, file))
            df["Time"] = pd.to_datetime(df["Time"], utc=True).dt.to_pydatetime()
            months = sorted(df['Time'].dt.month.unique().tolist())
            out.append({
                "key": file.split(".")[0],
                "measurements": len(df),
                "start_date": df["Time"].min().strftime("%Y-%m-%d"),
                "end_date": df["Time"].max().strftime("%Y-%m-%d"),
                "month_coverage": months
            })
        except:
            print("Failed to parse: {}".format(file))
    return out


def get_insitu_secchi_lake(filesystem, lake):
    file_path = os.path.join(filesystem, "media", "insitu", "secchi", "{}.csv".format(lake))
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Lake {} not available, please see the list of available lakes on "
                                                    "the metadata endpoint".format(lake))
    df = pd.read_csv(file_path)
    df["Time"] = pd.to_datetime(df["Time"], utc=True).dt.to_pydatetime()
    df = df.dropna()
    out = {"time": df["Time"].to_list(),
           "lat": df["Latitude"].to_list(),
           "lng": df["Longitude"].to_list(),
           "variable": {"data": df["Secchi depth [m]"].to_list(), "unit": "m", "description": "Secchi depth"}}
    return out
