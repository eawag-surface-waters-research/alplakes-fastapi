import os
import json
import requests
import pandas as pd
from enum import Enum
from typing import Dict, List
from pydantic import BaseModel, validator
from datetime import datetime, timedelta, timezone, date
from fastapi import HTTPException
from fastapi.responses import FileResponse

from app import functions

class VariableKeyModelMeta(BaseModel):
    unit: str
    description: str
    start_date: date
    end_date: date

class ResponseModelMeta(BaseModel):
    id: str
    source: str
    name: str
    elevation: float
    ch1903plus: List[float]
    lat: float
    lng: float
    variables: Dict[str, VariableKeyModelMeta]

class ResponseModel(BaseModel):
    time: List[datetime]
    variables: Dict[str, functions.VariableKeyModel1D]
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

def get_hydrodata_station_metadata(filesystem, station_id):
    parameter_types = [
        {"substring": "pegel", "unit": "m a.s.l", "description": "Water level above sea level"},
        {"substring": "abfluss", "unit": "m3", "description": "Water discharge"},
        {"substring": "phwert", "unit": "", "description": "Water pH"},
        {"substring": "wassertemperatur", "unit": "degC", "description": "Water temperature"},
        {"substring": "sauerstoff", "unit": "mg/l", "description": "Dissolved oxygen concentration"},
        {"substring": "leitfaehigkeit", "unit": "µS/cm", "description": "Water conductivity in microSiemens/cm"},
        {"substring": "truebung", "unit": "NTU", "description": "Water turbidity"},
    ]
    station_id = int(station_id)
    out = {"id": station_id}
    station_dir = os.path.join(filesystem, "media/bafu/hydrodata/stations", str(station_id))
    stations_file = os.path.join(filesystem, "media/bafu/hydrodata/stations/stations.json")
    if not os.path.exists(stations_file):
        response = requests.get(
            "https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/bafu/bafu_hydrodata.json")
        response.raise_for_status()
        stations_data = response.json()
        with open(stations_file, 'w') as f:
            json.dump(stations_data, f)
    else:
        with open(stations_file, 'r') as f:
            stations_data = json.load(f)
    data = next((s for s in stations_data["features"] if int(s["properties"]["id"]) == station_id), None)
    if data is None:
        raise HTTPException(status_code=400, detail="Data not available for {}".format(station_id))
    out["name"] = data["properties"]["name"]
    out["source"] = "Bafu"
    out["elevation"] = float(data["properties"]["Station elevation"].split(" ")[0])
    x, y = data["properties"]["ch1903"]
    out["ch1903plus"] = [x + 2000000, y + 1000000]
    out["lat"] = data["properties"]["wgs84"][0]
    out["lng"] = data["properties"]["wgs84"][1]
    out["variables"] = {}
    if not os.path.exists(station_dir):
        raise HTTPException(status_code=400, detail="Data not available for {}".format(station_id))
    for parameter in os.listdir(station_dir):
        d = {}
        properties = next((p for p in parameter_types if p["substring"] in parameter.lower()), None)
        if not properties == None:
            d["unit"] = properties["unit"]
            d["description"] = properties["description"]
        files = os.listdir(os.path.join(station_dir, parameter))
        files = [os.path.join(station_dir, parameter, f) for f in files if f.endswith(".csv")]
        files.sort()
        df = pd.read_csv(files[0])
        start_date = pd.to_datetime(df["Time"].iloc[0], utc=True).strftime("%Y-%m-%d")
        df = pd.read_csv(files[-1])
        end_date = pd.to_datetime(df["Time"].iloc[-1], utc=True).strftime("%Y-%m-%d")
        d["start_date"] = start_date
        d["end_date"] = end_date
        out["variables"][parameter] = d
    return out


class ResampleOptions(str, Enum):
    hourly = "hourly"
    daily = "daily"


def get_hydrodata_measured(filesystem, station_id, parameter, start_date, end_date, resample=None):
    parameter_types = [
        {"substring": "pegel", "unit": "m a.s.l", "description": "Water level above sea level"},
        {"substring": "abfluss", "unit": "m3", "description": "Water discharge"},
        {"substring": "phwert", "unit": "", "description": "Water pH"},
        {"substring": "wassertemperatur", "unit": "degC", "description": "Water temperature"},
        {"substring": "sauerstoff", "unit": "mg/l", "description": "Dissolved oxygen concentration"},
        {"substring": "leitfaehigkeit", "unit": "µS/cm", "description": "Water conductivity in microSiemens/cm"},
        {"substring": "truebung", "unit": "NTU", "description": "Water turbidity"},
    ]
    station_dir = os.path.join(filesystem, "media/bafu/hydrodata/stations", str(station_id))
    if not os.path.exists(station_dir):
        raise HTTPException(status_code=400,
                            detail="No data available for station id: {}".format(station_id))
    if not os.path.exists(os.path.join(station_dir, parameter)):
        raise HTTPException(status_code=400,
                            detail='Parameter "{}" not available for station {}, please select from: {}'.format(parameter, station_id, ", ".join(os.listdir(station_dir))))
    files = os.listdir(os.path.join(station_dir, parameter))
    files.sort()
    files = [os.path.join(station_dir, parameter, f) for f in files if int(start_date[:4]) <= int(f.split(".")[0].split("_")[-1]) <= int(end_date[:4]) and f.endswith(".csv")]
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No data available for requested time period.")
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    parameter_column_name = df.columns[1]
    df["time"] = pd.to_datetime(df['Time'], utc=True)
    df[parameter_column_name] = pd.to_numeric(df[parameter_column_name], errors='coerce').round(1)
    df.dropna(subset=[parameter_column_name], inplace=True)
    start = datetime.strptime(start_date, '%Y%m%d').replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, '%Y%m%d').replace(tzinfo=timezone.utc) + timedelta(days=1)
    df = df[(df['time'] >= start) & (df['time'] < end)]
    resample_options = {"hourly": "H", "daily": "D"}
    if resample is not None:
        df.set_index('time', inplace=True)
        df = df.resample(resample_options[resample]).mean(numeric_only=True)
        df = df.reset_index()
    df.dropna(subset=[parameter_column_name], inplace=True)
    if len(df) == 0:
        raise HTTPException(status_code=400,
                            detail="Not data available between {} and {}".format(start_date, end_date))
    properties = next((p for p in parameter_types if p["substring"] in parameter.lower()), None)
    d = {"data": list(df[parameter_column_name])}
    if not properties == None:
        d["unit"] = properties["unit"]
        d["description"] = properties["description"]
    return {"time": list(df["time"]), "variables": {parameter: d}}


class HydrodataPredicted(str, Enum):
    unofficial = "unofficial"
    official = "official"


def get_hydrodata_predicted(filesystem, status, station_id, model):
    file = os.path.join(filesystem, "media/bafu/hydrodata", "pqprevi-" + status, "Pqprevi_{}_{}.txt".format(model, station_id))
    if not os.path.exists(file):
        raise HTTPException(status_code=400,
                            detail="Prediction not available for model {} at station {}.".format(model, station_id))
    return FileResponse(file)


def metadata_hydrodata_total_lake_inflow(filesystem):
    output = []
    folder = os.path.join(filesystem, "media/bafu/hydrodata/TotalInflowLakes")
    lakes = os.listdir(folder)
    for lake in lakes:
        output.append({"lake": lake, "parameters": os.listdir(os.path.join(folder, lake))})
    return output


def get_hydrodata_total_lake_inflow(filesystem, lake, parameter, start_date, end_date):
    lake_dir = os.path.join(filesystem, "media/bafu/hydrodata/TotalInflowLakes", str(lake))
    if not os.path.exists(lake_dir):
        raise HTTPException(status_code=400,
                            detail="No data available for lake: {}".format(lake))
    if not os.path.exists(os.path.join(lake_dir, parameter)):
        raise HTTPException(status_code=400,
                            detail='Parameter "{}" not available for {}, please select from: {}'.format(
                                parameter, lake, ", ".join(os.listdir(lake_dir))))
    folder = os.path.join(lake_dir, parameter)
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    files = [os.path.join(folder, "{}_{}_{}.csv".format(lake, parameter,
                                                             (start_date + timedelta(days=x)).strftime("%Y-%m-%d")))
             for x in range((end_date - start_date).days + 1)]
    bad_files = []
    for file in files:
        if not os.path.isfile(file):
            bad_files.append(file.split("/")[-1].split(".")[0][-10:])
    if len(bad_files) > 0:
        raise HTTPException(status_code=400,
                            detail="Data not available for {} ({}) for the following dates: {}".format(
                                lake, parameter, ", ".join(bad_files)))

    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    return df.to_json()
