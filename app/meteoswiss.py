import os
import json
import numpy as np
import xarray as xr
from enum import Enum
from datetime import datetime, timedelta
from app.functions import NumpyArrayEncoder
from fastapi import HTTPException


class CosmoForecast(str, Enum):
    VNXQ94 = "VNXQ94"
    VNXZ32 = "VNXZ32"


def verify_cosmo_forecast(model, variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng):
    return True


def get_cosmo_forecast(filesystem, model, variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng):
    file = os.path.join(filesystem, "media/meteoswiss/cosmo", model, "{}.{}0000.nc".format(model, forecast_date))
    if not os.path.isfile(file):
        raise HTTPException(status_code=400, detail="Data not available for COSMO {} for the following date: {}".format(model, forecast_date))
    output = {}
    with xr.open_mfdataset(file) as ds:
        bad_variables = []
        for var in variables:
            if var not in ds.variables.keys():
                bad_variables.append(var)
        if len(bad_variables) > 0:
            raise HTTPException(status_code=400,
                                detail="{} are bad variables for COSMO {}. Please select from: {}".format(
                                    ", ".join(bad_variables), model, ", ".join(ds.keys())))

        output["time"] = np.array(ds.variables["time"].values, dtype=str)
        x, y = np.where(((ds.variables["lat_1"] >= ll_lat) & (ds.variables["lat_1"] <= ur_lat) & (ds.variables["lon_1"] >= ll_lng) & (ds.variables["lon_1"] <= ur_lng)))
        x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
        output["lat"] = ds.variables["lat_1"][x_min:x_max, y_min:y_max].values
        output["lng"] = ds.variables["lon_1"][x_min:x_max, y_min:y_max].values
        for var in variables:
            if var in ds.variables.keys():
                if ds.variables[var].dims == ('time', 'epsd_1', 'y_1', 'x_1'):
                    data = ds.variables[var][:, 0, x_min:x_max, y_min:y_max].values
                elif len(ds.variables[var].dims) == 5:
                    data = ds.variables[var][:, 0, 0, x_min:x_max, y_min:y_max].values
                else:
                    data = []
                output[var] = np.where(np.isnan(data), None, data)
            else:
                output[var] = []
    return json.loads(json.dumps(output, cls=NumpyArrayEncoder))


class CosmoReanalysis(str, Enum):
    VNJK21 = "VNJK21"
    VNXQ34 = "VNXQ34"


def verify_cosmo_reanalysis(model, variables, start_time, end_time, ll_lat, ll_lng, ur_lat, ur_lng):
    return True


def get_cosmo_reanalysis(filesystem, model, variables, start_time, end_time, ll_lat, ll_lng, ur_lat, ur_lng):
    folder = os.path.join(filesystem, "media/meteoswiss/cosmo", model)
    start_time = datetime.fromtimestamp(start_time)
    end_time = datetime.fromtimestamp(end_time)
    files = [os.path.join(folder, "{}.{}0000.nc".format(model, (start_time+timedelta(days=x)).strftime("%Y%m%d")))
             for x in range((end_time-start_time).days + 1)]
    bad_files = []
    for file in files:
        if not os.path.isfile(file):
            bad_files.append(file.split("/")[-1].split(".")[1][:8])
    if len(bad_files) > 0:
        raise HTTPException(status_code=400, detail="Data not available for COSMO {} for the following dates: {}".format(model, ", ".join(bad_files)))
    output = {}
    with xr.open_mfdataset(files) as ds:
        bad_variables = []
        for var in variables:
            if var not in ds.variables.keys():
                bad_variables.append(var)
        if len(bad_variables) > 0:
            raise HTTPException(status_code=400, detail="{} are bad variables for COSMO {}. Please select from: {}".format(", ".join(bad_variables), model, ", ".join(ds.keys())))
        output["time"] = np.array(ds.variables["time"].values, dtype=str)
        x, y = np.where(((ds.variables["lat_1"] >= ll_lat) & (ds.variables["lat_1"] <= ur_lat) & (
                    ds.variables["lon_1"] >= ll_lng) & (ds.variables["lon_1"] <= ur_lng)))
        x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
        output["lat"] = ds.variables["lat_1"][x_min:x_max, y_min:y_max].values
        output["lng"] = ds.variables["lon_1"][x_min:x_max, y_min:y_max].values
        for var in variables:
            if var in ds.variables.keys():
                if ds.variables[var].dims == ('time', 'epsd_1', 'y_1', 'x_1'):
                    data = ds.variables[var][:, 0, x_min:x_max, y_min:y_max].values
                elif len(ds.variables[var].dims) == 5:
                    data = ds.variables[var][:, 0, 0, x_min:x_max, y_min:y_max].values
                else:
                    data = []
                output[var] = np.where(np.isnan(data), None, data)
            else:
                output[var] = []
    return json.loads(json.dumps(output, cls=NumpyArrayEncoder))
