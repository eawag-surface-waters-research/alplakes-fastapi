import os
import json
import requests
import numpy as np
import pandas as pd
import xarray as xr
from enum import Enum
from datetime import datetime, timedelta, timezone, date
from fastapi import HTTPException
from typing import Dict, List, Union, Any
from pydantic import BaseModel, validator
from app import functions

class VariableKeyModelMeteoMeta(BaseModel):
    unit: str
    description: str
    start_date: date
    end_date: date

class ResponseModelMeteoMeta(BaseModel):
    id: str
    source: str
    name: str
    elevation: float
    lat: float
    lng: float
    variables: Dict[str, VariableKeyModelMeteoMeta]
    data_available: bool

class VariableKeyModelMeteo(BaseModel):
    unit: str
    description: str
    data: List[Union[float, None]]

class ResponseModelMeteo(BaseModel):
    time: List[datetime]
    variables: Dict[str, VariableKeyModelMeteo]
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value


def get_meteodata_station_metadata(filesystem, station_id):
    variables_convert = {
        "GS_10": "global_radiation",
        "TT_10": "air_temperature",
        "RF_10": "relative_humidity",
        "DD_10": "wind_direction",
        "FF_10": "wind_speed",
        "RWS_10": "precipitation"}
    variables_dict = functions.meteostation_variables()
    out = {"id": station_id}
    station_dir = os.path.join(filesystem, "media/dwd/meteodata", station_id)
    stations_file = os.path.join(filesystem, "media/dwd/meteodata/stations.json")
    if not os.path.exists(stations_file):
        response = requests.get(
            "https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/dwd/dwd_meteodata.json")
        response.raise_for_status()
        stations_data = response.json()
        with open(stations_file, 'w') as f:
            json.dump(stations_data, f)
    else:
        with open(stations_file, 'r') as f:
            stations_data = json.load(f)
    data = next((s for s in stations_data["features"] if s.get('id') == station_id), None)
    out["source"] = "Mistral"
    if data is None:
        raise HTTPException(status_code=400, detail="Station ID {} not recognised".format(station_id))
    out["name"] = data["properties"]["station_name"]
    out["elevation"] = float(data["properties"]["altitude"])
    out["lat"] = data["geometry"]["coordinates"][1]
    out["lng"] = data["geometry"]["coordinates"][0]
    out["variables"] = {}
    out["data_available"] = False
    if os.path.exists(station_dir):
        out["data_available"] = True
        files = os.listdir(station_dir)
        files = [os.path.join(station_dir, f) for f in files if f.endswith(".csv")]
        files.sort()
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors='coerce').notna()
        df = df.loc[:, df.any()]
        variables = list(df.columns[1:])
        for p in variables:
            if variables_convert[p] in variables_dict:
                d = variables_dict[variables_convert[p]]
                d["start_date"] = datetime.fromisoformat(str(min(df.loc[df[p], 'time']))).replace(
                    tzinfo=timezone.utc).strftime("%Y-%m-%d")
                d["end_date"] = datetime.fromisoformat(str(max(df.loc[df[p], 'time']))).replace(
                    tzinfo=timezone.utc).strftime("%Y-%m-%d")
                out["variables"][variables_convert[p]] = d
    return out

def get_meteodata_measured(filesystem, station_id, variables, start_date, end_date):
    variables_convert = {
        "GS_10": "global_radiation",
        "TT_10": "air_temperature",
        "RF_10": "relative_humidity",
        "DD_10": "wind_direction",
        "FF_10": "wind_speed",
        "RWS_10": "precipitation"}
    variables_adjust = {
        "global_radiation": functions.ten_minute_joules_cm_to_watts_m
    }
    variables_dict = functions.meteostation_variables()
    station_dir = os.path.join(filesystem, "media/dwd/meteodata", station_id)
    if not os.path.exists(station_dir):
        raise HTTPException(status_code=400, detail="Data not available for {}".format(station_id))

    files = os.listdir(station_dir)
    files.sort()
    files = [os.path.join(station_dir, f) for f in files if
             int(start_date[:4]) <= int(f.split(".")[0]) <= int(end_date[:4]) and f.endswith(".csv")]
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df = df.rename(columns=variables_convert)
    for v in variables:
        if v not in df.columns:
            raise HTTPException(status_code=400,
                                detail="Variable {} not available at station {}".format(v, station_id))
    df["time"] = pd.to_datetime(df['time'], utc=True)
    df[variables] = df[variables].apply(lambda x: pd.to_numeric(x, errors='coerce').round(1))
    df = df.dropna(how='all')

    for v in variables:
        if v in variables_adjust:
            df[v] = variables_adjust[v](df[v])
    start = datetime.strptime(start_date, '%Y%m%d').replace(tzinfo=timezone.utc).isoformat()
    end = (datetime.strptime(end_date, '%Y%m%d').replace(tzinfo=timezone.utc) + timedelta(days=1)).isoformat()
    selected = df[(df['time'] >= start) & (df['time'] < end)]
    if len(selected) == 0:
        raise HTTPException(status_code=400,
                            detail="No data available between {} and {}".format(start_date, end_date))
    output = {"time": list(selected["time"]), "variables": {}}
    for v in variables:
        output["variables"][v] = {"data": functions.filter_variable(list(selected[v])),
                                  "unit": variables_dict[v]["unit"],
                                  "description": variables_dict[v]["description"]}
    return output