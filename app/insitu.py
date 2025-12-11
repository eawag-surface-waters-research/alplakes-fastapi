import os
import json
import requests
import pandas as pd
from typing import List
from datetime import datetime, date, timezone, timedelta
from typing import Dict, List, Union
from pydantic import BaseModel, field_validator
from fastapi import HTTPException

from app import functions

class Metadata(BaseModel):
    key: str
    measurements: int
    start_date: date
    end_date: date
    month_coverage: List[int]

class ResponseModel(functions.TimeBaseModel):
    time: List[datetime]
    variable: functions.VariableKeyModel1D
    @field_validator('time')
    @classmethod
    def validate_timezone(cls, value):
        if isinstance(value, list):
            for v in value:
                if v.tzinfo is None:
                    raise ValueError('time must have a timezone')
        elif value.tzinfo is None:
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
            df["Time"] = pd.to_datetime(df["Time"], utc=True)
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
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    df = df.dropna(subset=['Secchi depth [m]'])
    out = {"time": df["Time"].to_list(),
           "lat": df["Latitude"].to_list(),
           "lng": df["Longitude"].to_list(),
           "variable": {"data": df["Secchi depth [m]"].to_list(), "unit": "m", "description": "Secchi depth"}}
    return out


class ResponseModelTemperatureMeta(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    depth: Union[str, float]
    category: str
    source: str
    url: str
    data_available: bool
    start_date: str
    end_date: str


def get_temperature_metadata(filesystem, station_id):
    out = {"id": station_id}
    station_dir = os.path.join(filesystem, "media/lake-scrape/temperature/", station_id)
    response = requests.get(
        "https://alplakes-eawag.s3.eu-central-1.amazonaws.com/insitu/summary/water_temperature.geojson")
    response.raise_for_status()
    stations_data = response.json()
    data = next((s for s in stations_data["features"] if s.get('id') == station_id), None)
    if data is None:
        raise HTTPException(status_code=400, detail="Station ID {} not recognised".format(station_id))
    out["name"] = data["properties"]["label"]
    out["lat"] = data["geometry"]["coordinates"][1]
    out["lng"] = data["geometry"]["coordinates"][0]
    out["depth"] = data["properties"]["depth"]
    out["category"] = data["properties"]["icon"]
    out["source"] = data["properties"]["source"]
    out["url"] = data["properties"]["url"]
    out["data_available"] = False
    out["start_date"] = ""
    out["end_date"] = ""
    if os.path.exists(station_dir):
        out["data_available"] = True
        files = os.listdir(station_dir)
        files = [os.path.join(station_dir, f) for f in files if f.endswith(".csv")]
        files.sort()
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        df = df.sort_values(by='time')
        time = pd.to_datetime(df["time"], unit='s').dt.strftime('%Y-%m-%d')
        out["start_date"] = time.iloc[0]
        out["end_date"] = time.iloc[-1]
    return out

def get_temperature_measured(filesystem, station_id, start_date, end_date):
    station_dir = os.path.join(filesystem, "media/lake-scrape/temperature", station_id)
    if not os.path.exists(station_dir):
        raise HTTPException(status_code=400, detail="Data not available for {}".format(station_id))
    files = os.listdir(station_dir)
    files.sort()
    files = [os.path.join(station_dir, f) for f in files if
             int(start_date[:4]) <= int(f.split(".")[0]) <= int(end_date[:4]) and f.endswith(".csv")]
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df["time"] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.dropna(how='all')
    start = pd.to_datetime(datetime.strptime(start_date, '%Y%m%d').replace(tzinfo=timezone.utc))
    end = pd.to_datetime(datetime.strptime(end_date, '%Y%m%d').replace(tzinfo=timezone.utc) + timedelta(days=1))
    selected = df[(df['time'] >= start) & (df['time'] < end)]
    if len(selected) == 0:
        raise HTTPException(status_code=400,
                            detail="No data available between {} and {}".format(start_date, end_date))
    output = {"time": list(selected["time"]), "variable": {"data": list(selected["value"]), "unit": "degC", "description": "Water Temperature"}}
    return output
