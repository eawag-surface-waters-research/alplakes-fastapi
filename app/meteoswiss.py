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


class CosmoForecast(str, Enum):
    VNXZ32 = "VNXZ32"
    VNXQ94 = "VNXQ94"

class IconForecast(str, Enum):
    icon_ch2_eps = "icon-ch2-eps"
    icon_ch1_eps = "icon-ch1-eps"

class ResponseModel2D(BaseModel):
    time: List[datetime]
    lat: List[List[float]]
    lng: List[List[float]]
    variables: Dict[str, functions.VariableKeyModel2D]
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

class ResponseModel1D(BaseModel):
    time: List[datetime]
    lat: float
    lng: float
    distance: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value

class Metadata(BaseModel):
    model: str
    description: str
    start_date: date
    end_date: date
    missing_dates: List[date]


def get_cosmo_metadata(filesystem):
    models = [{"model": "VNXQ34", "description": "Cosmo-1e 1 day deterministic"},
              {"model": "VNXQ94", "description": "Cosmo-1e 33 hour ensemble forecast"},
              {"model": "VNXZ32", "description": "Cosmo-2e 5 day ensemble forecast"}]
    out = []
    for model in models:
        files = os.listdir(os.path.join(filesystem, "media/meteoswiss/cosmo", model["model"]))
        if len(files) > 0:
            files.sort()
            combined = '_'.join(files)
            missing_dates = []

            start_date = datetime.strptime(files[0].split(".")[1], '%Y%m%d%H%M')
            end_date = datetime.strptime(files[-1].split(".")[1], '%Y%m%d%H%M')

            for d in functions.daterange(start_date, end_date):
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
            out.append(model)
    return out


def get_icon_metadata(filesystem):
    models = [{"model": "icon-ch1-eps", "description": "ICON-CH1-EPS 33 hour ensemble forecast"},
              {"model": "icon-ch2-eps", "description": "ICON-CH2-EPS 5 day ensemble forecast"},
              {"model": "kenda-ch1", "description": "KENDA-CH1 1 day deterministic reanalysis"}]

    for model in models:
        files = os.listdir(os.path.join(filesystem, "media/meteoswiss/icon", model["model"]))
        if len(files) > 0:
            files.sort()
            combined = '_'.join(files)
            missing_dates = []

            start_date = datetime.strptime(files[0][:10], '%Y_%m_%d')
            end_date = datetime.strptime(files[-1][:10], '%Y_%m_%d')

            for d in functions.daterange(start_date, end_date):
                if d.strftime('%Y_%m_%d') not in combined:
                    if model["model"] == "kenda-ch1":
                        missing_dates.append((d - timedelta(days=1)).strftime("%Y-%m-%d"))
                    else:
                        missing_dates.append(d.strftime("%Y-%m-%d"))

            if model["model"] == "kenda-ch1":
                start_date = start_date - timedelta(days=1)
                end_date = end_date - timedelta(days=1)

            model["start_date"] = start_date.strftime("%Y-%m-%d")
            model["end_date"] = end_date.strftime("%Y-%m-%d")
            model["missing_dates"] = missing_dates
        else:
            model["start_date"] = "NA"
            model["end_date"] = "NA"
    return models


def get_cosmo_area_forecast(filesystem, model, input_variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng):
    file = os.path.join(filesystem, "media/meteoswiss/cosmo", model, "{}.{}0000.nc".format(model, forecast_date))
    if not os.path.isfile(file):
        raise HTTPException(status_code=400,
                            detail="Data not available for COSMO {} for the following date: {}".format(model,
                                                                                                       forecast_date))
    output = {}
    with xr.open_mfdataset(file) as ds:
        bad_variables = []
        variables = []
        for var in input_variables:
            if var in ds.variables.keys():
                variables.append(var)
            elif var.replace("_MEAN", "") in ds.variables.keys():
                variables.append(var.replace("_MEAN", ""))
            elif var + "_MEAN" in ds.variables.keys():
                variables.append(var + "_MEAN")
            else:
                bad_variables.append(var)
        if len(bad_variables) > 0:
            raise HTTPException(status_code=400,
                                detail="{} are bad variables for COSMO {}. Please select from: {}".format(
                                    ", ".join(bad_variables), model, ", ".join(ds.keys())))

        output["time"] = meteoswiss_time_iso(ds.variables["time"])
        x, y = np.where(((ds.variables["lat_1"] >= ll_lat) & (ds.variables["lat_1"] <= ur_lat) & (
                ds.variables["lon_1"] >= ll_lng) & (ds.variables["lon_1"] <= ur_lng)))

        if len(x) == 0:
            raise HTTPException(status_code=400,
                                detail="Data not available for COSMO {} for the requested area.".format(model))

        x_min, x_max, y_min, y_max = min(x), max(x) + 1, min(y), max(y) + 1
        output["lat"] = ds.variables["lat_1"][x_min:x_max, y_min:y_max].values.tolist()
        output["lng"] = ds.variables["lon_1"][x_min:x_max, y_min:y_max].values.tolist()
        output["variables"] = {}
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
                output["variables"][var.replace("_MEAN", "")] = {"description": ds.variables[var].attrs["long_name"],
                               "unit": ds.variables[var].attrs["units"],
                               "data": np.where(np.isnan(data), None, data).tolist()}
    return output


def get_cosmo_point_forecast(filesystem, model, input_variables, forecast_date, lat, lng):
    file = os.path.join(filesystem, "media/meteoswiss/cosmo", model, "{}.{}0000.nc".format(model, forecast_date))
    if not os.path.isfile(file):
        raise HTTPException(status_code=400,
                            detail="Data not available for COSMO {} for the following date: {}".format(model,
                                                                                                       forecast_date))
    output = {}
    with xr.open_mfdataset(file) as ds:
        bad_variables = []
        variables = []
        for var in input_variables:
            if var in ds.variables.keys():
                variables.append(var)
            elif var.replace("_MEAN", "") in ds.variables.keys():
                variables.append(var.replace("_MEAN", ""))
            elif var + "_MEAN" in ds.variables.keys():
                variables.append(var + "_MEAN")
            else:
                bad_variables.append(var)
        if len(bad_variables) > 0:
            raise HTTPException(status_code=400,
                                detail="{} are bad variables for COSMO {}. Please select from: {}".format(
                                    ", ".join(bad_variables), model, ", ".join(ds.keys())))

        output["time"] = meteoswiss_time_iso(ds.variables["time"])
        if len(ds.variables["lat_1"].shape) == 3:
            lat_grid, lng_grid = ds.variables["lat_1"][0, :, :].values, ds.variables["lon_1"][0, :, :].values
        else:
            lat_grid, lng_grid = ds.variables["lat_1"].values, ds.variables["lon_1"].values
        x, y, distance = functions.get_closest_location(lat, lng, lat_grid, lng_grid)
        output["lat"] = float(lat_grid[x, y])
        output["lng"] = float(lng_grid[x, y])
        output["distance"] = {"data": distance, "unit": "m", "description": "Distance from requested location to center of closest grid point"}
        output["variables"] = {}
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
                output["variables"][var.replace("_MEAN", "")] = {"description": ds.variables[var].attrs["long_name"],
                               "unit": ds.variables[var].attrs["units"],
                               "data": np.where(np.isnan(data), None, data).tolist()}
    return output


def get_icon_area_forecast(filesystem, model, input_variables, forecast_date, ll_lat, ll_lng, ur_lat, ur_lng):
    formatted_forecast_date = datetime.strptime(forecast_date, "%Y%m%d").strftime("%Y_%m_%d")
    file = os.path.join(filesystem, "media/meteoswiss/icon", model, "{}_00_{}_eawag_lakes.nc".format(formatted_forecast_date, model))
    if not os.path.isfile(file):
        raise HTTPException(status_code=400,
                            detail="Data not available for ICON {} for the following date: {}".format(model,
                                                                                                       forecast_date))
    output = {}
    with xr.open_mfdataset(file) as ds:
        bad_variables = []
        variables = []
        for var in input_variables:
            if var.replace("_MEAN", "") in ds.variables.keys():
                variables.append(var.replace("_MEAN", ""))
            elif var + "_MEAN" in ds.variables.keys():
                variables.append(var + "_MEAN")
            else:
                bad_variables.append(var)
        if len(bad_variables) > 0:
            raise HTTPException(status_code=400,
                                detail="{} are bad variables for ICON {}. Please select from: {}".format(
                                    ", ".join(bad_variables), model, ", ".join(ds.keys())))
        output["time"] = meteoswiss_time_iso(ds.variables["time"])
        x, y = np.where(((ds.variables["lat_1"] >= ll_lat) & (ds.variables["lat_1"] <= ur_lat) & (
                ds.variables["lon_1"] >= ll_lng) & (ds.variables["lon_1"] <= ur_lng)))

        if len(x) == 0:
            raise HTTPException(status_code=400,
                                detail="Data not available for ICON {} for the requsted area.".format(model))

        x_min, x_max, y_min, y_max = min(x), max(x) + 1, min(y), max(y) + 1
        output["lat"] = ds.variables["lat_1"][x_min:x_max, y_min:y_max].values.tolist()
        output["lng"] = ds.variables["lon_1"][x_min:x_max, y_min:y_max].values.tolist()
        output["variables"] = {}
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
                output["variables"][var.replace("_MEAN", "")] = {"description": ds.variables[var].attrs["long_name"],
                               "unit": ds.variables[var].attrs["units"],
                               "data": np.where(np.isnan(data), None, data).tolist()}
    return output


def get_icon_point_forecast(filesystem, model, input_variables, forecast_date, lat, lng):
    formatted_forecast_date = datetime.strptime(forecast_date, "%Y%m%d").strftime("%Y_%m_%d")
    file = os.path.join(filesystem, "media/meteoswiss/icon", model, "{}_00_{}_eawag_lakes.nc".format(formatted_forecast_date, model))
    if not os.path.isfile(file):
        raise HTTPException(status_code=400,
                            detail="Data not available for ICON {} for the following date: {}".format(model,
                                                                                                       forecast_date))
    output = {}
    with xr.open_mfdataset(file) as ds:
        bad_variables = []
        variables = []
        for var in input_variables:
            if var.replace("_MEAN", "") in ds.variables.keys():
                variables.append(var.replace("_MEAN", ""))
            elif var + "_MEAN" in ds.variables.keys():
                variables.append(var + "_MEAN")
            else:
                bad_variables.append(var)
        if len(bad_variables) > 0:
            raise HTTPException(status_code=400,
                                detail="{} are bad variables for ICON {}. Please select from: {}".format(
                                    ", ".join(bad_variables), model, ", ".join(ds.keys())))
        output["time"] = meteoswiss_time_iso(ds.variables["time"])
        if len(ds.variables["lat_1"].shape) == 3:
            lat_grid, lng_grid = ds.variables["lat_1"][0, :, :].values, ds.variables["lon_1"][0, :, :].values
        else:
            lat_grid, lng_grid = ds.variables["lat_1"].values, ds.variables["lon_1"].values
        x, y, distance = functions.get_closest_location(lat, lng, lat_grid, lng_grid)
        output["lat"] = float(lat_grid[x, y])
        output["lng"] = float(lng_grid[x, y])
        output["distance"] = {"data": distance, "unit": "m",
                              "description": "Distance from requested location to center of closest grid point"}
        output["variables"] = {}
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
                output["variables"][var.replace("_MEAN", "")] = {"description": ds.variables[var].attrs["long_name"],
                               "unit": ds.variables[var].attrs["units"],
                               "data": np.where(np.isnan(data), None, data).tolist()}
    return output


class CosmoReanalysis(str, Enum):
    VNXQ34 = "VNXQ34"


def get_cosmo_area_reanalysis(filesystem, model, input_variables, start_date, end_date, ll_lat, ll_lng, ur_lat, ur_lng):
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
    try:
        with xr.open_mfdataset(files) as ds:
            bad_variables = []
            variables = []
            for var in input_variables:
                if var.replace("_MEAN", "") in ds.variables.keys():
                    variables.append(var.replace("_MEAN", ""))
                elif var + "_MEAN" in ds.variables.keys():
                    variables.append(var + "_MEAN")
                else:
                    bad_variables.append(var)
            if len(bad_variables) > 0:
                raise HTTPException(status_code=400,
                                    detail="{} are bad variables for COSMO {}. Please select from: {}".format(
                                        ", ".join(bad_variables), model, ", ".join(ds.keys())))
            output["time"] = meteoswiss_time_iso(ds.variables["time"])
            if len(ds.variables["lat_1"].shape) == 3:
                x, y = np.where(((ds.variables["lat_1"][0] >= ll_lat) & (ds.variables["lat_1"][0] <= ur_lat) & (
                        ds.variables["lon_1"][0] >= ll_lng) & (ds.variables["lon_1"][0] <= ur_lng)))
            else:
                x, y = np.where(((ds.variables["lat_1"] >= ll_lat) & (ds.variables["lat_1"] <= ur_lat) & (
                        ds.variables["lon_1"] >= ll_lng) & (ds.variables["lon_1"] <= ur_lng)))

            if len(x) == 0:
                raise HTTPException(status_code=400,
                                    detail="Requested area is outside of the COSMO coverage area, or is too small.".format(
                                        model))

            x_min, x_max, y_min, y_max = min(x), max(x) + 1, min(y), max(y) + 1
            if len(ds.variables["lat_1"].dims) == 3:
                output["lat"] = ds.variables["lat_1"][0, x_min:x_max, y_min:y_max].values.tolist()
                output["lng"] = ds.variables["lon_1"][0, x_min:x_max, y_min:y_max].values.tolist()
            else:
                output["lat"] = ds.variables["lat_1"][x_min:x_max, y_min:y_max].values.tolist()
                output["lng"] = ds.variables["lon_1"][x_min:x_max, y_min:y_max].values.tolist()
            output["variables"] = {}
            for var in variables:
                if var in ds.variables.keys():
                    if len(ds.variables[var].dims) == 3:
                        data = ds.variables[var][:, x_min:x_max, y_min:y_max].values
                    elif len(ds.variables[var].dims) == 4:
                        data = ds.variables[var][:, 0, x_min:x_max, y_min:y_max].values
                    else:
                        data = []
                    output["variables"][var.replace("_MEAN", "")] = {"description": ds.variables[var].attrs["long_name"],
                                   "unit": ds.variables[var].attrs["units"],
                                   "data": np.where(np.isnan(data), None, data).tolist()}
        return output
    except xr.MergeError as e:
        raise HTTPException(status_code=400,
                            detail="COSMO grid is not consistent between {} and {}, please access individual days.".format(
                                start_date, end_date))
    except Exception as e:
        raise


def get_cosmo_point_reanalysis(filesystem, model, input_variables, start_date, end_date, lat, lng):
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
    try:
        with xr.open_mfdataset(files) as ds:
            bad_variables = []
            variables = []
            for var in input_variables:
                if var in ds.variables.keys():
                    variables.append(var)
                elif var.replace("_MEAN", "") in ds.variables.keys():
                    variables.append(var.replace("_MEAN", ""))
                elif var + "_MEAN" in ds.variables.keys():
                    variables.append(var + "_MEAN")
                else:
                    bad_variables.append(var)
            if len(bad_variables) > 0:
                raise HTTPException(status_code=400,
                                    detail="{} are bad variables for COSMO {}. Please select from: {}".format(
                                        ", ".join(bad_variables), model, ", ".join(ds.keys())))
            output["time"] = meteoswiss_time_iso(ds.variables["time"])
            if len(ds.variables["lat_1"].shape) == 3:
                lat_grid, lng_grid = ds.variables["lat_1"][0, :, :].values, ds.variables["lon_1"][0, :, :].values
            else:
                lat_grid, lng_grid = ds.variables["lat_1"].values, ds.variables["lon_1"].values
            x, y, distance = functions.get_closest_location(lat, lng, lat_grid, lng_grid)
            output["lat"] = float(lat_grid[x, y])
            output["lng"] = float(lng_grid[x, y])
            output["distance"] = {"data": distance, "unit": "m", "description": "Distance from requested location to center of closest grid point"}
            output["variables"] = {}
            for var in variables:
                if var in ds.variables.keys():
                    if len(ds.variables[var].dims) == 3:
                        data = ds.variables[var][:, x, y].values
                    elif len(ds.variables[var].dims) == 4:
                        data = ds.variables[var][:, 0, x, y].values
                    else:
                        data = []
                    output["variables"][var.replace("_MEAN", "")] = {"description": ds.variables[var].attrs["long_name"],
                                   "unit": ds.variables[var].attrs["units"],
                                   "data": np.where(np.isnan(data), None, data).tolist()}
        return output
    except xr.MergeError as e:
        raise HTTPException(status_code=400,
                            detail="COSMO grid is not consistent between {} and {}, please access individual days.".format(
                                start_date, end_date))
    except Exception as e:
        raise


class IconReanalysis(str, Enum):
    kenda_ch1 = "kenda-ch1"


def get_icon_area_reanalysis(filesystem, model, input_variables, start_date, end_date, ll_lat, ll_lng, ur_lat, ur_lng):
    # For reanalysis files the date on the file is one day after the data in the file
    folder = os.path.join(filesystem, "media/meteoswiss/icon", model)
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    files = [os.path.join(folder, "{}_00_kenda-ch1_eawag_lakes.nc".format((start_date + timedelta(days=x)).strftime("%Y_%m_%d")))
             for x in range(1, (end_date - start_date).days + 2)]
    bad_files = []
    for file in files:
        if not os.path.isfile(file):
            actual_date = datetime.strptime(file.split("/")[-1][:10], '%Y_%m_%d') - timedelta(days=1)
            bad_files.append(actual_date.strftime("%Y-%m-%d"))
    if len(bad_files) > 0:
        raise HTTPException(status_code=400,
                            detail="Data not available for KENDA {} for the following dates: {}".format(model,
                                                                                                        ", ".join(
                                                                                                            bad_files)))
    output = {}
    try:
        with xr.open_mfdataset(files) as ds:
            bad_variables = []
            variables = []
            for var in input_variables:
                if var in ds.variables.keys():
                    variables.append(var)
                elif var.replace("_MEAN", "") in ds.variables.keys():
                    variables.append(var.replace("_MEAN", ""))
                elif var + "_MEAN" in ds.variables.keys():
                    variables.append(var + "_MEAN")
                else:
                    bad_variables.append(var)
            if len(bad_variables) > 0:
                raise HTTPException(status_code=400,
                                    detail="{} are bad variables for COSMO {}. Please select from: {}".format(
                                        ", ".join(bad_variables), model, ", ".join(ds.keys())))
            output["time"] = meteoswiss_time_iso(ds.variables["time"])
            if len(ds.variables["lat_1"].shape) == 3:
                x, y = np.where(((ds.variables["lat_1"][0] >= ll_lat) & (ds.variables["lat_1"][0] <= ur_lat) & (
                        ds.variables["lon_1"][0] >= ll_lng) & (ds.variables["lon_1"][0] <= ur_lng)))
            else:
                x, y = np.where(((ds.variables["lat_1"] >= ll_lat) & (ds.variables["lat_1"] <= ur_lat) & (
                        ds.variables["lon_1"] >= ll_lng) & (ds.variables["lon_1"] <= ur_lng)))

            if len(x) == 0:
                raise HTTPException(status_code=400,
                                    detail="Requested area is outside of the KENDA coverage area, or is too small.".format(
                                        model))

            x_min, x_max, y_min, y_max = min(x), max(x) + 1, min(y), max(y) + 1
            if len(ds.variables["lat_1"].dims) == 3:
                output["lat"] = ds.variables["lat_1"][0, x_min:x_max, y_min:y_max].values.tolist()
                output["lng"] = ds.variables["lon_1"][0, x_min:x_max, y_min:y_max].values.tolist()
            else:
                output["lat"] = ds.variables["lat_1"][x_min:x_max, y_min:y_max].values.tolist()
                output["lng"] = ds.variables["lon_1"][x_min:x_max, y_min:y_max].values.tolist()
            output["variables"] = {}
            for var in variables:
                if var in ds.variables.keys():
                    if len(ds.variables[var].dims) == 3:
                        data = ds.variables[var][:, x_min:x_max, y_min:y_max].values
                    elif len(ds.variables[var].dims) == 4:
                        data = ds.variables[var][:, 0, x_min:x_max, y_min:y_max].values
                    else:
                        data = []
                    output["variables"][var.replace("_MEAN", "")] = {"description": ds.variables[var].attrs["long_name"],
                                   "unit": ds.variables[var].attrs["units"],
                                   "data": np.where(np.isnan(data), None, data).tolist()}
        return output
    except xr.MergeError as e:
        raise HTTPException(status_code=400,
                            detail="KENDA grid is not consistent between {} and {}, please access individual days.".format(
                                start_date, end_date))
    except Exception as e:
        raise


def get_icon_point_reanalysis(filesystem, model, input_variables, start_date, end_date, lat, lng):
    # For reanalysis files the date on the file is one day after the data in the file
    folder = os.path.join(filesystem, "media/meteoswiss/icon", model)
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    files = [os.path.join(folder, "{}_00_kenda-ch1_eawag_lakes.nc".format((start_date + timedelta(days=x)).strftime("%Y_%m_%d"))) for x in range(1, (end_date - start_date).days + 2)]
    bad_files = []
    for file in files:
        if not os.path.isfile(file):
            actual_date = datetime.strptime(file.split("/")[-1][:10], '%Y_%m_%d') - timedelta(days=1)
            bad_files.append(actual_date.strftime("%Y-%m-%d"))
    if len(bad_files) > 0:
        raise HTTPException(status_code=400,
                            detail="Data not available for KENDA {} for the following dates: {}".format(model,
                                                                                                        ", ".join(
                                                                                                            bad_files)))
    output = {}
    try:
        with xr.open_mfdataset(files) as ds:
            bad_variables = []
            variables = []
            for var in input_variables:
                if var in ds.variables.keys():
                    variables.append(var)
                elif var.replace("_MEAN", "") in ds.variables.keys():
                    variables.append(var.replace("_MEAN", ""))
                elif var + "_MEAN" in ds.variables.keys():
                    variables.append(var + "_MEAN")
                else:
                    bad_variables.append(var)
            if len(bad_variables) > 0:
                raise HTTPException(status_code=400,
                                    detail="{} are bad variables for KENDA {}. Please select from: {}".format(
                                        ", ".join(bad_variables), model, ", ".join(ds.keys())))
            output["time"] = meteoswiss_time_iso(ds.variables["time"])
            if len(ds.variables["lat_1"].shape) == 3:
                lat_grid, lng_grid = ds.variables["lat_1"][0, :, :].values, ds.variables["lon_1"][0, :, :].values
            else:
                lat_grid, lng_grid = ds.variables["lat_1"].values, ds.variables["lon_1"].values
            x, y, distance = functions.get_closest_location(lat, lng, lat_grid, lng_grid)
            output["lat"] = float(lat_grid[x, y])
            output["lng"] = float(lng_grid[x, y])
            output["distance"] = {"data": distance, "unit": "m", "description": "Distance from requested location to center of closest grid point"}
            output["variables"] = {}
            for var in variables:
                if var in ds.variables.keys():
                    if len(ds.variables[var].dims) == 3:
                        data = ds.variables[var][:, x, y].values
                    elif len(ds.variables[var].dims) == 4:
                        data = ds.variables[var][:, 0, x, y].values
                    else:
                        data = []
                    output["variables"][var.replace("_MEAN", "")] = {"description": ds.variables[var].attrs["long_name"],
                                   "unit": ds.variables[var].attrs["units"],
                                   "data": np.where(np.isnan(data), None, data).tolist()}
        return output
    except xr.MergeError as e:
        raise HTTPException(status_code=400,
                            detail="KENDA grid is not consistent between {} and {}, please access individual days.".format(
                                start_date, end_date))
    except Exception as e:
        raise


class MeteodataParameters(str, Enum):
    pva200h0 = "pva200h0"
    gre000h0 = "gre000h0"
    tre200h0 = "tre200h0"
    rre150h0 = "rre150h0"
    fkl010h0 = "fkl010h0"
    dkl010h0 = "dkl010h0"
    nto000d0 = "nto000d0"

class VariableKeyModelMeteoMeta(BaseModel):
    period: str
    unit: str
    description: str
    start_date: date
    end_date: date

class ResponseModelMeteoMeta(BaseModel):
    id: str
    source: str
    name: str
    elevation: float
    ch1903plus: List[float]
    lat: float
    lng: float
    variables: Dict[str, VariableKeyModelMeteoMeta]
    data_available: bool

class VariableKeyModelMeteo(BaseModel):
    period: str
    unit: str
    description: str
    data: List[float]

class ResponseModelMeteo(BaseModel):
    time: List[datetime]
    variables: Dict[str, VariableKeyModelMeteo]
    @validator('time', each_item=True)
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError('time must have a timezone')
        return value


def get_meteodata_station_metadata(filesystem, station_id):
    slf_stations = [{
        "id": "SLFGAD",
        "properties": {"station_name": "Gschletteregg", "altitude": 2063.0},
        "geometry": {"coordinates": [2673270.0, 1177465.0]}
    }, {
        "id": "SLFGU2",
        "properties": {"station_name": "Homad", "altitude": 2115.0},
        "geometry": {"coordinates": [2665100, 1170100]}
    }]
    parameters_dict = {
        "pva200h0": {"unit": "hPa", "description": "Vapour pressure 2 m above ground", "period": "hourly mean"},
        "gre000h0": {"unit": "W/m²", "description": "Global radiation", "period": "hourly mean"},
        "tre200h0": {"unit": "°C", "description": "Air temperature 2 m above ground", "period": "hourly mean"},
        "rre150h0": {"unit": "mm", "description": "Precipitation", "period": "hourly total"},
        "fkl010h0": {"unit": "m/s", "description": "Wind speed scalar", "period": "hourly mean"},
        "dkl010h0": {"unit": "°", "description": "Wind direction", "period": "hourly mean"},
        "nto000d0": {"unit": "%", "description": "Cloud cover", "period": "daily mean"}}
    out = {"id": station_id}
    station_id = station_id.upper()
    station_dir = os.path.join(filesystem, "media/meteoswiss/meteodata", station_id)
    stations_file = os.path.join(filesystem, "media/meteoswiss/meteodata/stations.json")
    partners_file = os.path.join(filesystem, "media/meteoswiss/meteodata/partners.json")
    if not os.path.exists(stations_file):
        response = requests.get(
            "https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/meteoswiss/meteoswiss_meteodata.json")
        response.raise_for_status()
        stations_data = response.json()
        with open(stations_file, 'w') as f:
            json.dump(stations_data, f)
    else:
        with open(stations_file, 'r') as f:
            stations_data = json.load(f)
    if not os.path.exists(partners_file):
        response = requests.get(
            "https://alplakes-eawag.s3.eu-central-1.amazonaws.com/static/meteoswiss/meteoswiss_partner_meteostations.json")
        response.raise_for_status()
        partner_data = response.json()
        with open(partners_file, 'w') as f:
            json.dump(partner_data, f)
    else:
        with open(partners_file, 'r') as f:
            partner_data = json.load(f)
    data = next((s for s in stations_data["features"] if s.get('id') == station_id), None)
    out["source"] = "MeteoSwiss"
    if data is None:
        data = next((s for s in partner_data["features"] if s.get('id') == station_id), None)
        out["source"] = "MeteoSwiss Partner Station"
    if data is None:
        data = next((s for s in slf_stations if s.get('id') == station_id), None)
        out["source"] = "SLF Station (Non Meteoswiss Partner Station)"
    if data is None:
        raise HTTPException(status_code=400, detail="Station ID {} not recognised".format(station_id))
    out["name"] = data["properties"]["station_name"]
    out["elevation"] = float(data["properties"]["altitude"])
    out["ch1903plus"] = data["geometry"]["coordinates"]
    lat, lng = functions.ch1903_plus_to_latlng(out["ch1903plus"][0], out["ch1903plus"][1])
    out["lat"] = lat
    out["lng"] = lng
    out["variables"] = {}
    out["data_available"] = False
    if os.path.exists(station_dir):
        out["data_available"] = True
        files = os.listdir(station_dir)
        files = [os.path.join(station_dir, f) for f in files if f.endswith(".csv")]
        files.sort()
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        df[df.columns[2:]] = df[df.columns[2:]].apply(pd.to_numeric, errors='coerce').notna()
        df = df.loc[:, df.any()]
        parameters = list(df.columns[2:])
        for p in parameters:
            if p in parameters_dict:
                d = parameters_dict[p]
                d["start_date"] = datetime.strptime(str(min(df.loc[df[p], 'Date'])), "%Y%m%d%H").replace(
                    tzinfo=timezone.utc).strftime("%Y-%m-%d")
                d["end_date"] = datetime.strptime(str(max(df.loc[df[p], 'Date'])), "%Y%m%d%H").replace(
                    tzinfo=timezone.utc).strftime("%Y-%m-%d")
                out["variables"][p] = d
    return out


def get_meteodata_measured(filesystem, station_id, parameter, start_date, end_date):
    parameters_dict = {
        "pva200h0": {"unit": "hPa", "description": "Vapour pressure 2 m above ground", "period": "hourly mean"},
        "gre000h0": {"unit": "W/m²", "description": "Global radiation", "period": "hourly mean"},
        "tre200h0": {"unit": "°C", "description": "Air temperature 2 m above ground", "period": "hourly mean"},
        "rre150h0": {"unit": "mm", "description": "Precipitation", "period": "hourly total"},
        "fkl010h0": {"unit": "m/s", "description": "Wind speed scalar", "period": "hourly mean"},
        "dkl010h0": {"unit": "°", "description": "Wind direction", "period": "hourly mean"},
        "nto000d0": {"unit": "%", "description": "Cloud cover", "period": "daily mean"}}
    station_id = station_id.upper()
    station_dir = os.path.join(filesystem, "media/meteoswiss/meteodata", station_id)
    if not os.path.exists(station_dir):
        raise HTTPException(status_code=400, detail="Data not available for {}".format(station_id))

    files = os.listdir(station_dir)
    files.sort()
    files = [os.path.join(station_dir, f) for f in files if
             int(start_date[:4]) <= int(f.split(".")[1]) <= int(end_date[:4]) and f.endswith(".csv")]
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    if parameter not in df.columns:
        raise HTTPException(status_code=400,
                            detail="Parameter {} not available at station {}".format(parameter, station_id))
    df["time"] = pd.to_datetime(df['Date'], format='%Y%m%d%H', utc=True)
    df[parameter] = pd.to_numeric(df[parameter], errors='coerce').round(1)
    df.dropna(subset=[parameter], inplace=True)
    start = datetime.strptime(start_date, '%Y%m%d').replace(tzinfo=timezone.utc).isoformat()
    end = (datetime.strptime(end_date, '%Y%m%d').replace(tzinfo=timezone.utc) + timedelta(days=1)).isoformat()
    selected = df[(df['time'] >= start) & (df['time'] < end)]
    if len(selected) == 0:
        raise HTTPException(status_code=400,
                            detail="No data available between {} and {}".format(start_date, end_date))
    return {"time": list(selected["time"]),
            "variables": {parameter: {"data": list(selected[parameter]),
                                     "unit": parameters_dict[parameter]["unit"],
                                     "description": parameters_dict[parameter]["description"],
                                     "period": parameters_dict[parameter]["period"]}}}

def meteoswiss_time_iso(time_array):
    return [datetime.utcfromtimestamp(time.astype('datetime64[s]').astype('int')).replace(tzinfo=timezone.utc) for
     time in np.array(time_array.values, dtype='datetime64[ns]')]