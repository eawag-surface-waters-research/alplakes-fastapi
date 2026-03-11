import os
import json
import requests
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import zoom as ndimage_zoom
from enum import Enum
from datetime import datetime, timedelta, timezone, date
from fastapi import HTTPException
from typing import Dict, List, Union, Any
from pydantic import BaseModel, field_validator
from app import functions


class CosmoForecast(str, Enum):
    VNXZ32 = "VNXZ32"
    VNXQ94 = "VNXQ94"

class IconForecast(str, Enum):
    icon_ch2_eps = "icon-ch2-eps"
    icon_ch1_eps = "icon-ch1-eps"

class ResponseModel2D(functions.TimeBaseModel):
    time: List[datetime]
    lat: List[List[float]]
    lng: List[List[float]]
    variables: Dict[str, functions.VariableKeyModel2D]
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

class ResponseModel1D(functions.TimeBaseModel):
    time: List[datetime]
    lat: float
    lng: float
    distance: functions.VariableKeyModel1D
    variables: Dict[str, functions.VariableKeyModel1D]
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

def get_icon_layer_alplakes(filesystem, variable, start_date, end_date, ll_lat, ll_lng, ur_lat, ur_lng):
    """Return ICON data in Alplakes compact plain-text format for a bounding box and time range.

    Uses kenda-ch1 reanalysis for historical dates where files exist, filling remaining dates
    with the most recent icon-ch2-eps forecast file.
    """
    start_date = datetime.strptime(start_date, '%Y%m%d').date()
    end_date = datetime.strptime(end_date, '%Y%m%d').date()

    # Find available reanalysis files (kenda-ch1); file date is offset +1 day
    reanalysis_folder = os.path.join(filesystem, "media/meteoswiss/icon/kenda-ch1")
    reanalysis_files = []
    for x in range(1, (end_date - start_date).days + 2):
        fname = os.path.join(reanalysis_folder, "{}_00_kenda-ch1_eawag_lakes.nc".format(
            (start_date + timedelta(days=x)).strftime("%Y_%m_%d")))
        if os.path.isfile(fname):
            reanalysis_files.append(fname)

    # Find the most recent forecast file (icon-ch2-eps)
    forecast_file = None
    forecast_folder = os.path.join(filesystem, "media/meteoswiss/icon/icon-ch2-eps")
    if os.path.isdir(forecast_folder):
        fc_files = sorted([f for f in os.listdir(forecast_folder) if f.endswith(".nc")])
        if fc_files:
            forecast_file = os.path.join(forecast_folder, fc_files[-1])

    if not reanalysis_files and forecast_file is None:
        raise HTTPException(status_code=404, detail="No ICON data available for the requested period.")

    def get_grid(ds):
        if len(ds.variables["lat_1"].shape) == 3:
            return ds.variables["lat_1"][0].values, ds.variables["lon_1"][0].values
        return ds.variables["lat_1"].values, ds.variables["lon_1"].values

    # Compute the canonical bbox on the reanalysis (fine) grid so geometry is identical
    # regardless of source. For forecast-only, derive the equivalent reanalysis bbox from
    # the forecast indices using the known grid relationship: reanalysis index = 2 * forecast index.
    if reanalysis_files:
        with xr.open_dataset(reanalysis_files[0]) as ds_grid:
            lat_g_r, lng_g_r = get_grid(ds_grid)
        xa, ya = np.where((lat_g_r >= ll_lat) & (lat_g_r <= ur_lat) & (lng_g_r >= ll_lng) & (lng_g_r <= ur_lng))
        if len(xa) == 0:
            raise HTTPException(status_code=400, detail="Requested area is outside of the ICON coverage area, or is too small.")
        xn, xx = int(min(xa)), int(max(xa)) + 1
        yn, yx = int(min(ya)), int(max(ya)) + 1
        F_X = (lat_g_r.shape[0] + 1) // 2
        F_Y = (lat_g_r.shape[1] + 1) // 2
    else:
        with xr.open_dataset(forecast_file) as ds_grid:
            lat_g_f, lng_g_f = get_grid(ds_grid)
        xa, ya = np.where((lat_g_f >= ll_lat) & (lat_g_f <= ur_lat) & (lng_g_f >= ll_lng) & (lng_g_f <= ur_lng))
        if len(xa) == 0:
            raise HTTPException(status_code=400, detail="Requested area is outside of the ICON coverage area, or is too small.")
        xn_f0, xx_f0 = int(min(xa)), int(max(xa)) + 1
        yn_f0, yx_f0 = int(min(ya)), int(max(ya)) + 1
        xn, xx = 2 * xn_f0, 2 * (xx_f0 - 1) + 1
        yn, yx = 2 * yn_f0, 2 * (yx_f0 - 1) + 1
        F_X, F_Y = lat_g_f.shape[0], lat_g_f.shape[1]

    def resolve_variable(ds, var):
        if var.replace("_MEAN", "") in ds.variables.keys():
            return var.replace("_MEAN", "")
        elif var + "_MEAN" in ds.variables.keys():
            return var + "_MEAN"
        return None

    def extract_spatial(ds, var_name, xn, xx, yn, yx):
        ndims = len(ds.variables[var_name].dims)
        if ndims == 3:
            return ds.variables[var_name][:, xn:xx, yn:yx].values
        elif ndims == 4:
            return ds.variables[var_name][:, 0, xn:xx, yn:yx].values
        elif ndims == 5:
            return ds.variables[var_name][:, 0, 0, xn:xx, yn:yx].values
        return None

    def to_geometry_string(lat_sub, lng_sub):
        geometry = np.concatenate((lat_sub, lng_sub), axis=1)
        return '\n'.join(','.join('%0.8f' % v for v in row) for row in geometry).replace("nan", "")

    def upsample(data):
        """Bilinear upsample to (2n-1) per spatial axis.
        Handles 2D (X, Y) arrays (e.g. lat/lng) and 3D (T, X, Y) arrays (variable data)."""
        if data.ndim == 2:
            X, Y = data.shape
            return ndimage_zoom(data, (((2 * X) - 1) / X, ((2 * Y) - 1) / Y), order=1)
        T, X, Y = data.shape
        return ndimage_zoom(data, (1.0, ((2 * X) - 1) / X, ((2 * Y) - 1) / Y), order=1)

    # Forecast bbox derived from the canonical reanalysis bbox:
    # include the surrounding forecast cell on each edge so the upsampled result
    # covers the full reanalysis bbox, then crop back to exact size.
    xn_f = xn // 2
    xx_f = min(F_X, xx // 2 + 1)
    yn_f = yn // 2
    yx_f = min(F_Y, yx // 2 + 1)
    xs, ys = xn - 2 * xn_f, yn - 2 * yn_f  # crop offsets after upsample
    xsize, ysize = xx - xn, yx - yn

    def crop_upsample(d):
        u = upsample(d)
        return u[xs:xs + xsize, ys:ys + ysize] if d.ndim == 2 else u[:, xs:xs + xsize, ys:ys + ysize]

    combined_times = []
    combined_data = []

    # Process reanalysis
    if reanalysis_files:
        if variable == "geometry":
            return to_geometry_string(lat_g_r[xn:xx, yn:yx], lng_g_r[xn:xx, yn:yx])
        try:
            with xr.open_mfdataset(reanalysis_files) as ds:
                if variable == "UV":
                    u_name = resolve_variable(ds, "U")
                    v_name = resolve_variable(ds, "V")
                    if u_name is None or v_name is None:
                        raise HTTPException(status_code=400, detail="U and V variables not available in ICON data.")
                    u_data = extract_spatial(ds, u_name, xn, xx, yn, yx)
                    v_data = extract_spatial(ds, v_name, xn, xx, yn, yx)
                    data = np.concatenate([u_data, v_data], axis=-1) if u_data is not None and v_data is not None else None
                else:
                    var_name = resolve_variable(ds, variable)
                    if var_name is None:
                        raise HTTPException(status_code=400, detail="Variable {} not available in ICON data. Please select from: {}".format(variable, ", ".join(ds.keys())))
                    data = extract_spatial(ds, var_name, xn, xx, yn, yx)
                times = meteoswiss_time_iso(ds.variables["time"])
                if data is not None:
                    combined_times.extend(times)
                    combined_data.append(data)
        except HTTPException:
            raise
        except xr.MergeError:
            raise HTTPException(status_code=400, detail="KENDA grid is not consistent across the requested date range, please access individual days.")

    # Process forecast — derive bbox from canonical reanalysis indices, upsample and crop.
    if forecast_file is not None:
        with xr.open_dataset(forecast_file) as ds:
            if variable == "geometry":
                lat_g, lng_g = get_grid(ds)
                return to_geometry_string(crop_upsample(lat_g[xn_f:xx_f, yn_f:yx_f]), crop_upsample(lng_g[xn_f:xx_f, yn_f:yx_f]))
            if variable == "UV":
                u_name = resolve_variable(ds, "U")
                v_name = resolve_variable(ds, "V")
                forecast_data = None
                if u_name is not None and v_name is not None:
                    u_data = extract_spatial(ds, u_name, xn_f, xx_f, yn_f, yx_f)
                    v_data = extract_spatial(ds, v_name, xn_f, xx_f, yn_f, yx_f)
                    if u_data is not None and v_data is not None:
                        forecast_data = np.concatenate([crop_upsample(u_data), crop_upsample(v_data)], axis=-1)
                elif not combined_times:
                    raise HTTPException(status_code=400, detail="U and V variables not available in ICON data.")
            else:
                var_name_f = resolve_variable(ds, variable)
                forecast_data = extract_spatial(ds, var_name_f, xn_f, xx_f, yn_f, yx_f) if var_name_f is not None else None
                if forecast_data is not None:
                    forecast_data = crop_upsample(forecast_data)
                if var_name_f is None and not combined_times:
                    raise HTTPException(status_code=400, detail="Variable {} not available in ICON data.".format(variable))
            if forecast_data is not None:
                forecast_times = meteoswiss_time_iso(ds.variables["time"])
                last_r_time = combined_times[-1] if combined_times else None
                start_idx = next((i for i, t in enumerate(forecast_times) if last_r_time is None or t > last_r_time), None)
                if start_idx is not None:
                    combined_times.extend(forecast_times[start_idx:])
                    combined_data.append(forecast_data[start_idx:])

    if not combined_times:
        raise HTTPException(status_code=404, detail="No ICON data available for the requested period.")

    all_data = np.concatenate(combined_data, axis=0)

    if variable == "T_2M":
        all_data = all_data - 273.15

    start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=timezone.utc)

    fmt = "%0.5f" if variable == "UV" else "%0.2f"
    out = ""
    for t_idx, t in enumerate(combined_times):
        if t < start_dt or t >= end_dt:
            continue
        out += t.strftime("%Y%m%d%H%M") + "\n"
        out += '\n'.join(','.join(fmt % v for v in row) for row in all_data[t_idx]).replace("nan", "") + "\n"
    return out


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
    ch1903plus: List[float]
    lat: float
    lng: float
    variables: Dict[str, VariableKeyModelMeteoMeta]
    data_available: bool

class VariableKeyModelMeteo(BaseModel):
    unit: str
    description: str
    data: List[Union[float, None]]

class ResponseModelMeteo(functions.TimeBaseModel):
    time: List[datetime]
    variables: Dict[str, VariableKeyModelMeteo]
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
    variables_convert = {
        "pva200h0": "vapour_pressure",
        "gre000h0": "global_radiation",
        "tre200h0": "air_temperature",
        "rre150h0": "precipitation",
        "fkl010h0": "wind_speed",
        "dkl010h0": "wind_direction",
        "nto000d0": "cloud_cover"}
    variables_dict = functions.meteostation_variables()
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
        variables = list(df.columns[2:])
        for p in variables:
            if variables_convert[p] in variables_dict:
                d = variables_dict[variables_convert[p]]
                d["start_date"] = datetime.strptime(str(min(df.loc[df[p], 'Date'])), "%Y%m%d%H").replace(
                    tzinfo=timezone.utc).strftime("%Y-%m-%d")
                d["end_date"] = datetime.strptime(str(max(df.loc[df[p], 'Date'])), "%Y%m%d%H").replace(
                    tzinfo=timezone.utc).strftime("%Y-%m-%d")
                out["variables"][variables_convert[p]] = d
    return out


def get_meteodata_measured(filesystem, station_id, variables, start_date, end_date):
    variables_convert = {
        "pva200h0": "vapour_pressure",
        "gre000h0": "global_radiation",
        "tre200h0": "air_temperature",
        "rre150h0": "precipitation",
        "fkl010h0": "wind_speed",
        "dkl010h0": "wind_direction",
        "nto000d0": "cloud_cover"}
    variables_adjust = {}
    variables_dict = functions.meteostation_variables()
    station_id = station_id.upper()
    station_dir = os.path.join(filesystem, "media/meteoswiss/meteodata", station_id)
    if not os.path.exists(station_dir):
        raise HTTPException(status_code=400, detail="Data not available for {}".format(station_id))

    files = os.listdir(station_dir)
    files.sort()
    files = [os.path.join(station_dir, f) for f in files if
             int(start_date[:4]) <= int(f.split(".")[1]) <= int(end_date[:4]) and f.endswith(".csv")]
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df = df.rename(columns=variables_convert)
    for v in variables:
        if v not in df.columns:
            raise HTTPException(status_code=400,
                                detail="Variable {} not available at station {}".format(v, station_id))
    df["time"] = pd.to_datetime(df['Date'], format='%Y%m%d%H', utc=True)
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


def meteoswiss_time_iso(time_array):
    return [datetime.fromtimestamp(time.astype('datetime64[s]').astype('int'), tz=timezone.utc) for
            time in np.array(time_array.values, dtype='datetime64[ns]')]