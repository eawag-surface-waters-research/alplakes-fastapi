import os
import numpy as np
import pandas as pd
import xarray as xr
from enum import Enum
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
from app.functions import daterange


def get_cosmo_metadata(filesystem):
    models = [{"model": "VNXQ34", "description": "Cosmo-1e 1 day deterministic"},
              {"model": "VNJK21", "description": "Cosmo-1e 1 day ensemble forecast"},
              {"model": "VNXQ94", "description": "Cosmo-1e 33 hour ensemble forecast"},
              {"model": "VNXZ32", "description": "Cosmo-2e 5 day ensemble forecast"}]

    for model in models:
        files = os.listdir(os.path.join(filesystem, "media/meteoswiss/cosmo", model["model"]))
        if len(files) > 0:
            files.sort()
            combined = '_'.join(files)
            missing_dates = []

            start_date = datetime.strptime(files[0].split(".")[1], '%Y%m%d%H%M')
            end_date = datetime.strptime(files[-1].split(".")[1], '%Y%m%d%H%M')

            for d in daterange(start_date, end_date):
                if d.strftime('%Y%m%d%H%M') not in combined:
                    if model["model"] == "VNXQ34":
                        missing_dates.append((d - timedelta(days=1)).strftime("%Y-%m-%d"))
                    else:
                        missing_dates.append(d.strftime("%Y-%m-%d"))

            if model["model"] == "VNXQ34":
                start_date = start_date - timedelta(days=1)
                end_date = end_date - timedelta(days=1)

            model["start_date"] = start_date.strftime("%Y-%m-%d")
            model["end_date"] = end_date.strftime("%Y-%m-%d")
            model["missing_dates"] = missing_dates
        else:
            model["start_date"] = "NA"
            model["end_date"] = "NA"
    return models


class CosmoForecast(str, Enum):
    VNXZ32 = "VNXZ32"
    VNXQ94 = "VNXQ94"


def verify_cosmo_area_forecast(model, variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng):
    return True


def verify_cosmo_point_forecast(model, variables, forecast_date, lat, lng):
    return True


def get_cosmo_area_forecast(filesystem, model, variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng):
    file = os.path.join(filesystem, "media/meteoswiss/cosmo", model, "{}.{}0000.nc".format(model, forecast_date))
    if not os.path.isfile(file):
        raise HTTPException(status_code=400,
                            detail="Data not available for COSMO {} for the following date: {}".format(model,
                                                                                                       forecast_date))
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

        output["time"] = np.array(ds.variables["time"].values, dtype=str).tolist()
        x, y = np.where(((ds.variables["lat_1"] >= ll_lat) & (ds.variables["lat_1"] <= ur_lat) & (
                    ds.variables["lon_1"] >= ll_lng) & (ds.variables["lon_1"] <= ur_lng)))

        if len(x) == 0:
            raise HTTPException(status_code=400,
                                detail="Data not available for COSMO {} for the requsted area.".format(model))

        x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
        output["lat"] = ds.variables["lat_1"][x_min:x_max, y_min:y_max].values.tolist()
        output["lng"] = ds.variables["lon_1"][x_min:x_max, y_min:y_max].values.tolist()
        for var in variables:
            if var in ds.variables.keys():
                if len(ds.variables[var].dims) == 3:
                    data = ds.variables[var][:, x_min:x_max, y_min:y_max].values
                elif len(ds.variables[var].dims) == 4:
                    data = ds.variables[var][:, 0, x_min:x_max, y_min:y_max].values
                elif len(ds.variables[var].dims) == 5:
                    data = ds.variables[var][:, 0, 0, x_min:x_max, y_min:y_max].values
                else:
                    data = []
                output[var] = {"name": ds.variables[var].attrs["long_name"],
                               "unit": ds.variables[var].attrs["units"],
                               "data": np.where(np.isnan(data), None, data).tolist()}
            else:
                output[var] = []
    return output


def get_cosmo_point_forecast(filesystem, model, variables, forecast_date, lat, lng):
    file = os.path.join(filesystem, "media/meteoswiss/cosmo", model, "{}.{}0000.nc".format(model, forecast_date))
    if not os.path.isfile(file):
        raise HTTPException(status_code=400,
                            detail="Data not available for COSMO {} for the following date: {}".format(model,
                                                                                                       forecast_date))
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

        output["time"] = np.array(ds.variables["time"].values, dtype=str).tolist()

        dist = ((ds.variables["lat_1"] - lat) ** 2 + (ds.variables["lon_1"] - lng) ** 2) ** 0.5
        xy = np.unravel_index(dist.argmin(), dist.shape)
        x, y = xy[0], xy[1]
        output["lat"] = float(ds.variables["lat_1"][x, y].values)
        output["lng"] = float(ds.variables["lon_1"][x, y].values)

        for var in variables:
            if var in ds.variables.keys():
                if len(ds.variables[var].dims) == 3:
                    data = ds.variables[var][:, x, y].values
                elif len(ds.variables[var].dims) == 4:
                    data = ds.variables[var][:, 0, x, y].values
                elif len(ds.variables[var].dims) == 5:
                    data = ds.variables[var][:, 0, 0, x, y].values
                else:
                    data = []
                output[var] = {"name": ds.variables[var].attrs["long_name"],
                               "unit": ds.variables[var].attrs["units"],
                               "data": np.where(np.isnan(data), None, data).tolist()}
            else:
                output[var] = []
    return output


class CosmoReanalysis(str, Enum):
    VNJK21 = "VNJK21"
    VNXQ34 = "VNXQ34"


def verify_cosmo_area_reanalysis(model, variables, start_time, end_time, ll_lat, ll_lng, ur_lat, ur_lng):
    return True


def verify_cosmo_point_reanalysis(model, variables, start_time, end_time, lat, lng):
    return True


def get_cosmo_area_reanalysis(filesystem, model, variables, start_date, end_date, ll_lat, ll_lng, ur_lat, ur_lng):
    # For reanalysis files the date on the file is one day after the data in the file
    folder = os.path.join(filesystem, "media/meteoswiss/cosmo", model)
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    files = [os.path.join(folder, "{}.{}0000.nc".format(model, (start_date + timedelta(days=x)).strftime("%Y%m%d")))
             for x in range(1, (end_date - start_date).days + 2)]
    bad_files = []
    for file in files:
        if not os.path.isfile(file):
            actual_date = datetime.strptime(file.split("/")[-1].split(".")[1][:8], '%Y%m%d') - timedelta(days=1)
            bad_files.append(actual_date.strftime("%Y-%m-%d"))
    if len(bad_files) > 0:
        raise HTTPException(status_code=400,
                            detail="Data not available for COSMO {} for the following dates: {}".format(model,
                                                                                                        ", ".join(
                                                                                                            bad_files)))
    output = {}
    with xr.open_mfdataset(files) as ds:
        bad_variables = []
        for var in variables:
            if var not in ds.variables.keys():
                bad_variables.append(var)
        if len(bad_variables) > 0:
            raise HTTPException(status_code=400,
                                detail="{} are bad variables for COSMO {}. Please select from: {}".format(
                                    ", ".join(bad_variables), model, ", ".join(ds.keys())))
        output["time"] = np.array(ds.variables["time"].values, dtype=str).tolist()
        if len(ds.variables["lat_1"].shape) == 3:
            x, y = np.where(((ds.variables["lat_1"][0] >= ll_lat) & (ds.variables["lat_1"][0] <= ur_lat) & (
                ds.variables["lon_1"][0] >= ll_lng) & (ds.variables["lon_1"][0] <= ur_lng)))
        else:
            x, y = np.where(((ds.variables["lat_1"] >= ll_lat) & (ds.variables["lat_1"] <= ur_lat) & (
                ds.variables["lon_1"] >= ll_lng) & (ds.variables["lon_1"] <= ur_lng)))

        if len(x) == 0:
            raise HTTPException(status_code=400,
                                detail="Data not available for COSMO {} for the requested area.".format(model))

        x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
        if len(ds.variables["lat_1"].shape) == 3:
            output["lat"] = ds.variables["lat_1"][0, x_min:x_max, y_min:y_max].values.tolist()
            output["lng"] = ds.variables["lon_1"][0, x_min:x_max, y_min:y_max].values.tolist()
        else:
            output["lat"] = ds.variables["lat_1"][x_min:x_max, y_min:y_max].values.tolist()
            output["lng"] = ds.variables["lon_1"][x_min:x_max, y_min:y_max].values.tolist()
        for var in variables:
            if var in ds.variables.keys():
                if len(ds.variables[var].dims) == 3:
                    data = ds.variables[var][:, x_min:x_max, y_min:y_max].values
                elif len(ds.variables[var].dims) == 4:
                    data = ds.variables[var][:, 0, x_min:x_max, y_min:y_max].values
                else:
                    data = []
                output[var] = {"name": ds.variables[var].attrs["long_name"],
                               "unit": ds.variables[var].attrs["units"],
                               "data": np.where(np.isnan(data), None, data).tolist()}
            else:
                output[var] = []
    return output


def get_cosmo_point_reanalysis(filesystem, model, variables, start_date, end_date, lat, lng):
    # For reanalysis files the date on the file is one day after the data in the file
    folder = os.path.join(filesystem, "media/meteoswiss/cosmo", model)
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    files = [os.path.join(folder, "{}.{}0000.nc".format(model, (start_date + timedelta(days=x)).strftime("%Y%m%d")))
             for x in range(1, (end_date - start_date).days + 2)]
    bad_files = []
    for file in files:
        if not os.path.isfile(file):
            bad_files.append(file.split("/")[-1].split(".")[1][:8])
    if len(bad_files) > 0:
        raise HTTPException(status_code=400,
                            detail="Data not available for COSMO {} for the following dates: {}".format(model,
                                                                                                        ", ".join(
                                                                                                            bad_files)))
    output = {}
    with xr.open_mfdataset(files) as ds:
        bad_variables = []
        for var in variables:
            if var not in ds.variables.keys():
                bad_variables.append(var)
        if len(bad_variables) > 0:
            raise HTTPException(status_code=400,
                                detail="{} are bad variables for COSMO {}. Please select from: {}".format(
                                    ", ".join(bad_variables), model, ", ".join(ds.keys())))
        output["time"] = np.array(ds.variables["time"].values, dtype=str).tolist()

        dist = ((ds.variables["lat_1"] - lat) ** 2 + (ds.variables["lon_1"] - lng) ** 2) ** 0.5
        xy = np.unravel_index(dist.argmin(), dist.shape)
        x, y = xy[0], xy[1]
        if len(ds.variables["lat_1"].shape) == 3:
            output["lat"] = float(ds.variables["lat_1"][0, x, y].values)
            output["lng"] = float(ds.variables["lon_1"][0, x, y].values)
        else:
            output["lat"] = float(ds.variables["lat_1"][x, y].values)
            output["lng"] = float(ds.variables["lon_1"][x, y].values)
        for var in variables:
            if var in ds.variables.keys():
                if len(ds.variables[var].dims) == 3:
                    data = ds.variables[var][:, x, y].values
                elif len(ds.variables[var].dims) == 4:
                    data = ds.variables[var][:, 0, x, y].values
                else:
                    data = []
                output[var] = {"name": ds.variables[var].attrs["long_name"],
                               "unit": ds.variables[var].attrs["units"],
                               "data": np.where(np.isnan(data), None, data).tolist()}
            else:
                output[var] = []
    return output


class MeteodataParameters(str, Enum):
    pva200h0 = "pva200h0"
    gre000h0 = "gre000h0"
    tre200h0 = "tre200h0"
    rre150h0 = "rre150h0"
    fkl010h0 = "fkl010h0"
    dkl010h0 = "dkl010h0"
    nto000d0 = "nto000d0"


def verify_meteodata_measured(station_id, parameter, start_date, end_date):
    return True


def get_meteodata_measured(filesystem, station_id, parameter, start_date, end_date):
    station_id = station_id[:3].upper()
    station_dir = os.path.join(filesystem, "media/meteoswiss/meteodata", station_id)
    if not os.path.exists(station_dir):
        raise HTTPException(status_code=400, detail="Data not available for {}".format(station_id))

    files = os.listdir(station_dir)
    files.sort()
    files = [f for f in files if int(start_date[:4]) <= int(f.split(".")[1]) <= int(end_date[:4])]
    dfs = []
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(station_dir, file))
            dfs.append(df)
    if len(dfs) == 0:
        raise HTTPException(status_code=400,
                            detail="No data available between {} and {}".format(start_date, end_date))
    df = pd.concat(dfs, ignore_index=True)
    if parameter not in df.columns:
        raise HTTPException(status_code=400, detail="Parameter {} not available at station {}".format(parameter, station_id))
    df["time"] = pd.to_datetime(df['Date'], format='%Y%m%d%H', utc=True)
    start = datetime.strptime(start_date, '%Y%m%d').replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, '%Y%m%d').replace(tzinfo=timezone.utc) + timedelta(days=1)
    selected = df[(df['time'] >= start) & (df['time'] <= end)]
    if len(selected) == 0:
        raise HTTPException(status_code=400,
                            detail="Not data available between {} and {}".format(start_date, end_date))
    out = {"Time": list(selected["time"]), parameter: list(selected[parameter])}
    return out
