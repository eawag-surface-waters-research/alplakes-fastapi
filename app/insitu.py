import os
import json
import requests
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
from fastapi.responses import FileResponse


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
                "first": df["Time"].min(),
                "latest": df["Time"].max(),
                "months": months
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
    out = {"time": df["Time"].to_list(),
           "lat": df["Latitude"].to_list(),
           "lng": df["Longitude"].to_list(),
           "secchi": {"data": df["Secchi depth [m]"].to_list(), "unit": "m", "description": "Secchi depth"}}
    return out
